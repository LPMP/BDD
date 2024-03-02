#include "bdd_solver.h"
#include "ILP_parser.h"
#include "OPB_parser.h"
#include "min_marginal_utils.h"
#include "incremental_mm_agreement_rounding_cuda.h"
#include "incremental_mm_agreement_rounding.hxx"
#include "wedelin_primal_heuristic.hxx"
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <memory>
#include <unordered_set>
#include <stdlib.h>
#include <stdexcept>
#include <filesystem>
#include <CLI/CLI.hpp>
#include "bdd_logging.h"
#include "time_measure_util.h"
#include "run_solver_util.h"
#include "mm_primal_decoder.h"
#include <string>
#include <regex>

namespace LPMP {

    std::unique_ptr<char*[]> convert_string_to_argv(const std::vector<std::string>& args)
    {
        std::unique_ptr<char*[]> argv = std::make_unique<char*[]>(args.size()+1);

        static constexpr const char* prog_name = "bdd_solver";
        argv[0] = const_cast<char*>(prog_name);
        for(size_t i=0; i<args.size(); ++i)
            argv[i+1] = const_cast<char*>(args[i].c_str());
        return argv; 
    }

    inline void init_logging(const bdd_solver_options &options)
    {
        bdd_log.to_console_ = !options.suppress_console_output;
        if (!options.log_file.empty())
            bdd_log.set_file_stream(options.log_file);
    } 

    void print_statistics(ILP_input& ilp, bdd_preprocessor& bdd_pre)
    {
        bdd_log << "[print_statistics] #variables = " << ilp.nr_variables() << "\n";
        bdd_log << "[print_statistics] #constraints = " << ilp.nr_constraints() << "\n";
        bdd_log << "[print_statistics] #BDDs = " << bdd_pre.nr_bdds() << "\n";
        const auto num_constraints = ilp.nr_constraints();
        std::vector<size_t> num_constraints_per_var(ilp.nr_variables(), 0);
        for (size_t c = 0; c < ilp.nr_constraints(); c++)
            for(const auto v : ilp.variables(c))
                num_constraints_per_var[v]++;
        bdd_log << "[print_statistics] minimum num. constraints per var = " << *std::min_element(num_constraints_per_var.begin(), num_constraints_per_var.end()) << "\n";
        bdd_log << "[print_statistics] maximum num. constraints per var = " << *std::max_element(num_constraints_per_var.begin(), num_constraints_per_var.end()) << "\n";
        bdd_log << "[print_statistics] mean num. constraints per var = " << std::accumulate(num_constraints_per_var.begin(), num_constraints_per_var.end(), 0.0) / num_constraints_per_var.size() << "\n";
        
    }

    ILP_input parse_ilp_file(const std::string& filename)
    {
        // determine whether file is in LP format or in opb one.
        if(filename.substr(filename.find_last_of(".") + 1) == "opb")
        {
            bdd_log << "[bdd solver] Parse opb file\n";
            return OPB_parser::parse_file(filename);
        }
        else if(filename.substr(filename.find_last_of(".") + 1) == "lp")
        {
            bdd_log << "[bdd solver] Parse lp file\n";
            return ILP_parser::parse_file(filename);
        }
        else // peek into file
        {
            throw std::runtime_error("peeking into files not implemented yet");
        }
    }

    ILP_input parse_ilp_string(const std::string& input)
    {
        // if file begins with * (i.e. opb comment) or with min: then it is an opb file
        std::regex comment_regex("^\\w*\\*");
        std::regex opb_min_regex("^\\w*min:");
        if(std::regex_search(input, comment_regex) || std::regex_search(input, opb_min_regex)) 
        {
            bdd_log << "[bdd solver] Parse opb string\n";
            return OPB_parser::parse_string(input); 
        }
        else
        {
            bdd_log << "[bdd solver] Parse lp string\n";
            return ILP_parser::parse_string(input); 
        }
    }

    void read_ILP(bdd_solver_options& options)
    {
        options.ilp = [&]()
        {
            if(!options.input_file.empty())
            {
                return parse_ilp_file(options.input_file);
            }
            else if(!options.input_string.empty())
            {
                return parse_ilp_string(options.input_string);
            }
            else
                return options.ilp;
        }();
    }

    bdd_solver_options::bdd_solver_options(int argc, char** argv)
    {
        // setup command line arguemnts
        CLI::App app("LPMP BDD solver");
        app.allow_extras();
 
        auto input_group = app.add_option_group("input", "input either from file or as string");
        auto input_file_arg = input_group->add_option("-i, --input_file", input_file, "ILP input file name")
            ->check(CLI::ExistingPath);

        auto input_string_arg = input_group->add_option("--input_string", input_string, "ILP input in string");

        input_group->require_option(1); // either as string or as filename

        app.add_flag("--logarithms", take_cost_logarithms, "");

        auto optimization_group = app.add_option_group("optimization type", "{maximization|minimization}");
        optimization_group->add_flag("--minimization", [this](const size_t count) { assert(count > 0); this->optimization = optimization_type::minimization; }, "minimize problem");
        optimization_group->add_flag("--maximization", [this](const size_t count) { assert(count > 0); this->optimization = optimization_type::maximization; }, "maximization problem");
        optimization_group->require_option(0,1);
        
        std::unordered_map<std::string, ILP_input::variable_order> variable_order_map{
            {"input", ILP_input::variable_order::input},
            {"bfs", ILP_input::variable_order::bfs},
            {"cuthill", ILP_input::variable_order::cuthill},
            {"mindegree", ILP_input::variable_order::mindegree}
        };

        app.add_option("-o, --order", var_order, "variable order")
            ->transform(CLI::CheckedTransformer(variable_order_map, CLI::ignore_case));

        app.add_option("-m, --max_iter", max_iter, "maximal number of iterations, default value = 10000")
            ->check(CLI::NonNegativeNumber);

        app.add_option("--tolerance", tolerance, "termination criterion: lower bound relative progress tolerance, default value = " + std::to_string(tolerance))
            ->check(CLI::Range(0.0, std::numeric_limits<double>::infinity()));

        app.add_option("--improvement_slope", improvement_slope, "termination criterion: improvement between iterations as compared to improvement after first iterations, default value = " + std::to_string(improvement_slope))
            ->check(CLI::Range(0.0, 1.0));

        app.add_option("-l, --time_limit", time_limit, "time limit in seconds, default value = 3600")
            ->check(CLI::PositiveNumber);

        std::unordered_map<std::string, bdd_solver_impl> bdd_solver_impl_map{
            {"mma",bdd_solver_impl::sequential_mma},
            {"sequential_mma",bdd_solver_impl::sequential_mma},
            {"parallel_mma",bdd_solver_impl::parallel_mma},
            {"mma_cuda",bdd_solver_impl::mma_cuda},
            {"cuda_mma",bdd_solver_impl::mma_cuda},
            {"hybrid_parallel_mma",bdd_solver_impl::hybrid_parallel_mma},
            {"lbfgs_cuda_mma", bdd_solver_impl::lbfgs_cuda_mma},
            {"lbfgs_parallel_mma", bdd_solver_impl::lbfgs_parallel_mma},
            {"subgradient", bdd_solver_impl::subgradient}
        };

        auto solver_group = app.add_option_group("solver", "solver either a BDD solver, output of statistics or export of LP solved by BDD relaxation");
        solver_group->add_option("-s, --solver", bdd_solver_impl_, "name of solver for the relaxation")
            ->transform(CLI::CheckedTransformer(bdd_solver_impl_map, CLI::ignore_case));

        auto solution_statistics_arg = app.add_flag("--solution_statistics", solution_statistics, "list min marginals and objective after solving dual problem");

        std::unordered_map<std::string, bdd_solver_precision> bdd_solver_precision_map{
            {"float",bdd_solver_precision::single_prec},
            {"single",bdd_solver_precision::single_prec},
            {"double",bdd_solver_precision::double_prec}
        };

        auto bdd_solver_precision_arg = app.add_option("--precision", bdd_solver_precision_, "floating point precision used in solver (default double)")
            ->transform(CLI::CheckedTransformer(bdd_solver_precision_map, CLI::ignore_case));

        app.add_option("--smoothing", smoothing, "smoothing, default value = 0 (no smoothing)")
                ->check(CLI::PositiveNumber);

        app.add_flag("--cuda_split_long_bdds", cuda_split_long_bdds, "split long BDDs into short ones, might make cuda mma faster for problems with a few long inequalities");
        app.add_flag("--cuda_split_long_bdds_with_implication_bdd", cuda_split_long_bdds_implication_bdd, "split long BDDs into short ones and additionally construct implication BDD");
        app.add_option("--cuda_split_long_bdds_length", cuda_split_long_bdds_length, "split long BDDs into shorter ones of the specified length");

        // LBFGS
        app.add_option("--lbfgs_history_size", lbfgs_history_size, "history size for LBFGS, default = " + std::to_string(lbfgs_history_size));
        app.add_option("--lbfgs_step_size", lbfgs_step_size, "initial step size for LBFGS, default = " + std::to_string(lbfgs_step_size));
        app.add_option("--lbfgs_required_relative_lb_increase", lbfgs_required_relative_lb_increase, "required relative increase in dual lower bound for LBFGS to apply an update, default = " + std::to_string(lbfgs_required_relative_lb_increase));
        app.add_option("--lbfgs_step_size_decrease_factor", lbfgs_step_size_decrease_factor, "decrease factor in line search for LBFGS, default = " + std::to_string(lbfgs_step_size_decrease_factor));
        app.add_option("--lbfgs_step_size_increase_factor", lbfgs_step_size_increase_factor, "increase factor in line search for LBFGS, default = " + std::to_string(lbfgs_step_size_increase_factor));

        auto primal_group = app.add_option_group("primal rounding", "method for obtaining a primal solution from the dual optimization");
        auto incremental_primal_arg = primal_group->add_flag("--incremental_primal", incremental_primal_rounding, "incremental primal rounding flag");
        auto wedelin_primal_arg = primal_group->add_flag("--wedelin_primal", wedelin_primal_rounding, "Wedelin primal rounding flag");
        primal_group->require_option(0,1); 

        auto incremental_rounding_param_group = app.add_option_group("incremental primal rounding parameters", "parameters for rounding a primal solution");
        incremental_rounding_param_group->needs(incremental_primal_arg);

        incremental_rounding_param_group->add_option("--incremental_initial_perturbation", incremental_initial_perturbation, "value for initial perturbation for obtaining primal solutions by incremental primal rounding")
            ->check(CLI::PositiveNumber);

        incremental_rounding_param_group->add_option("--incremental_perturbation_growth_rate", incremental_growth_rate, "growth rate for increasing the perturbation for obtaining primal solutions by incremental primal rounding")
            ->check(CLI::Range(1.0,std::numeric_limits<double>::max()));

        incremental_rounding_param_group->add_option("--incremental_primal_num_itr_lb", incremental_primal_num_itr_lb, "number of iterations of dual optimization during incremental primal rounding")
            ->check(CLI::Range(1,std::numeric_limits<int>::max()));

        incremental_rounding_param_group->add_option("--incremental_primal_rounding_num_itr", incremental_primal_rounding_num_itr, "maximum number of incremental primal rounding iterations")
            ->check(CLI::Range(1,std::numeric_limits<int>::max()));

        auto wedelin_rounding_param_group = app.add_option_group("Wedelin primal rounding parameters", "parameters for rounding a primal solution");
        wedelin_rounding_param_group->needs(wedelin_primal_arg);

        wedelin_rounding_param_group->add_option("--wedelin_kappa_min", wedelin_kappa_min, "starting value of perturbation strength")
            ->check(CLI::Range(0.0,1.0));
        wedelin_rounding_param_group->add_option("--wedelin_kappa_max", wedelin_kappa_max, "maximum value of perturbation strength")
            ->check(CLI::Range(0.0,1.0));
        wedelin_rounding_param_group->add_option("--wedelin_kappa_step", wedelin_kappa_step, "increment value for perturbation")
            ->check(CLI::Range(0.0,1.0));
        wedelin_rounding_param_group->add_option("--wedelin_alpha", wedelin_alpha, "increment exponent value for perturbation")
            ->check(CLI::PositiveNumber);
        wedelin_rounding_param_group->add_option("--wedelin_delta", wedelin_delta, "constant perturbation value")
            ->check(CLI::PositiveNumber);
        wedelin_rounding_param_group->add_option("--wedelin_theta", wedelin_theta, "exponential decay factor for perturbation")
            ->check(CLI::Range(0.0,1.0));


        auto tighten_arg = app.add_flag("--tighten", tighten, "tighten relaxation flag");
        
        solver_group->add_flag("--statistics", statistics, "statistics of the problem");

        solver_group->add_option("--export_bdd_lp", export_bdd_lp_file, "filename for export of LP of the BDD relaxation");

        solver_group->add_option("--export_lp", export_lp_file, "filename for export of LP");

        solver_group->add_option("--export_bdd_graph", export_bdd_graph_file, "filename for export of BDD representation in .dot format");

        solver_group->require_option(1); // either a solver or statistics

        auto export_difficult_core_param = app.add_option("--export_difficult_core", export_difficult_core, "filename for export of LP resulting from fixing all variables with zero or one min-marginal");
        auto export_difficult_core_param_group = app.add_option_group("Difficult core export parameters", "parameters for deciding which variables to exclude from the difficult core");
        export_difficult_core_param_group->needs(export_difficult_core_param);
        export_difficult_core_param_group->add_option("--export_difficult_core_th", export_difficult_core_th, "threshold for min-marginals for fixing variables")
            ->check(CLI::PositiveNumber);

        app.add_flag("--suppress_console_output", suppress_console_output, "do not print on the console");
        app.add_option("--log_file", log_file, "log output into file");

        app.parse(argc, argv); 

        assert(input_string.empty() || input_file.empty());
    }

    bdd_solver_options::bdd_solver_options(const std::vector<std::string>& args)
        : bdd_solver_options(args.size()+1, convert_string_to_argv(args).get())
    {}

    bdd_solver_options::bdd_solver_options(const std::string& input_file_path)
    {
        ilp = parse_ilp_file(input_file_path);
    }

    bdd_solver_options::bdd_solver_options(ILP_input& _ilp) : ilp(_ilp) {}

    /*
    bdd_solver::bdd_solver(int argc, char** argv)
        : bdd_solver(bdd_solver_options(argc, argv))
    {}

    bdd_solver::bdd_solver(const std::vector<std::string>& args)
        : bdd_solver(bdd_solver_options(args))
    {}
    */

    //bdd_solver::bdd_solver(bdd_solver_options&& opt)
    //    : options(opt)
    //{
    // throw std::runtime_error("not implemented");
    //}

    bdd_solver::bdd_solver(bdd_solver_options opt)
        : options(opt)
    {
        init_logging(opt);

        read_ILP(options);

        options.ilp.reorder(options.var_order);
        options.ilp.normalize();

        bdd_log << "[bdd solver] ILP has " << options.ilp.nr_variables() << " variables and " << options.ilp.nr_constraints() << " constraints\n";
        if(options.ilp.preprocess())
            bdd_log << "[bdd solver] ILP has " << options.ilp.nr_variables() << " variables and " << options.ilp.nr_constraints() << " constraints after preprocessing\n";
        else
        {
            bdd_log << "[bdd solver] The problem appears to be infeasible.\n";
            return;
        }

        const auto start_time = std::chrono::steady_clock::now();

        costs = options.ilp.objective();

        if(options.take_cost_logarithms)
        {
            bdd_log << "[bdd solver] Take logarithms of costs\n";
            for(size_t i=0; i<costs.size(); ++i)
            {
                assert(costs[i] > 0);
                costs[i] = std::log(costs[i]);
            }
        }

        if(options.optimization == bdd_solver_options::optimization_type::maximization)
        {
            bdd_log << "[bdd solver] Use negative costs due to maximization\n";
            for(size_t i=0; i<costs.size(); ++i)
                costs[i] = -costs[i];
        }

        const bool normalize_constraints = [&]() {
            if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::sequential_mma)
                return true;
            return false;
        }();

        bdd_preprocessor bdd_pre(options.ilp, normalize_constraints, options.cuda_split_long_bdds, options.cuda_split_long_bdds_implication_bdd, options.cuda_split_long_bdds_length);

        bdd_log << std::setprecision(10);

        if(options.statistics)
        {
            print_statistics(options.ilp, bdd_pre);
            exit(0);
        }
        else if(!options.export_bdd_lp_file.empty())
        {
            std::ofstream f;
            f.open(options.export_bdd_lp_file);
            bdd_pre.get_bdd_collection().write_bdd_lp(f, costs.begin(), costs.end());
            f.close(); 
            exit(0);
        }
        else if(!options.export_lp_file.empty())
        {
            std::ofstream f;
            f.open(options.export_lp_file);
            const std::string extension = std::filesystem::path(options.export_lp_file).extension();
            if (extension == ".lp")
                options.ilp.write_lp(f);
            else if (extension == ".opb")
                options.ilp.write_opb(f);
            else if (extension == ".mps")
                options.ilp.write_mps(f);
            else
                throw std::runtime_error("Cannot recognize file extension " + extension + " for exporting problem file");
            f.close(); 
            exit(0);
        }
        else if(!options.export_bdd_graph_file.empty())
        {

            // get ending and filename
            const auto [filename, extension] = [&]() -> std::tuple<std::string, std::string> {
                const auto idx = options.export_bdd_graph_file.rfind('.');
                if(idx != std::string::npos)
                {
                    const std::string filename = options.export_bdd_graph_file.substr(0, idx);
                    const std::string extension = options.export_bdd_graph_file.substr(idx+1);
                    assert(extension == "dot");
                    return {filename, extension};
                }
                else
                {
                    // no extension found
                    return {options.export_bdd_graph_file, "dot"};
                }
            }();

            for(size_t bdd_nr=0; bdd_nr<bdd_pre.get_bdd_collection().nr_bdds(); ++bdd_nr)
            {
                std::ofstream f;
                const std::string dot_file = filename + "_" + std::to_string(bdd_nr) + ".dot";
                const std::string png_file = filename + "_" + std::to_string(bdd_nr) + ".png";
                f.open(dot_file);
                bdd_pre.get_bdd_collection().export_graphviz(bdd_nr, f);
                f.close(); 
                const std::string convert_command = "dot -Tpng " + dot_file + " > " + png_file;
                system(convert_command.c_str());
            }
            exit(0);
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::sequential_mma)
        {
            if(options.smoothing == 0)
            {
                if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                    solver = std::move(bdd_mma<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                    solver = std::move(bdd_mma<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else
                    throw std::runtime_error("only float and double precision allowed");
                bdd_log << "[bdd solver] constructed sequential mma solver\n"; 
            } else {
                if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                    solver = std::move(bdd_mma_smooth<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                    solver = std::move(bdd_mma_smooth<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else
                    throw std::runtime_error("only float and double precision allowed");
                bdd_log << "[bdd solver] constructed sequential smooth mma solver\n"; 
            }
        } 
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::parallel_mma)
        {
            if(options.smoothing == 0)
            {
                if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                    solver = std::move(bdd_parallel_mma<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                    solver = std::move(bdd_parallel_mma<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else
                    throw std::runtime_error("only float and double precision allowed");
                bdd_log << "[bdd solver] constructed parallel mma solver\n"; 
            }
            else
            {
                if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                    solver = std::move(bdd_parallel_mma_smooth<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                    solver = std::move(bdd_parallel_mma_smooth<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
                else
                    throw std::runtime_error("only float and double precision allowed");
                bdd_log << "[bdd solver] constructed smooth parallel mma solver\n"; 
            }
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::mma_cuda)
        {
            if(options.smoothing != 0)
                throw std::runtime_error("no smoothing implemented for cuda mma");
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_cuda<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_cuda<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else
                throw std::runtime_error("only float and double precision allowed");
            bdd_log << "[bdd solver] constructed CUDA based mma solver\n"; 
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::hybrid_parallel_mma)
        {
            if(options.smoothing != 0)
                throw std::runtime_error("no smoothing implemented for hybrid parallel mma");
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_multi_parallel_mma<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_multi_parallel_mma<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else
                throw std::runtime_error("only float and double precision allowed");
            bdd_log << "[bdd solver] constructed CUDA based mma solver\n"; 
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::lbfgs_cuda_mma)
        {
            if(options.smoothing != 0)
                throw std::runtime_error("no smoothing implemented for LBFGS mma cuda");
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_lbfgs_cuda_mma<float>(
                    bdd_pre.get_bdd_collection(), costs.begin(), costs.end(),
                    options.lbfgs_history_size,
                    options.lbfgs_step_size, options.lbfgs_required_relative_lb_increase, 
                    options.lbfgs_step_size_decrease_factor, options.lbfgs_step_size_increase_factor));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_lbfgs_cuda_mma<double>(
                    bdd_pre.get_bdd_collection(), costs.begin(), costs.end(),
                    options.lbfgs_history_size,
                    options.lbfgs_step_size, options.lbfgs_required_relative_lb_increase, 
                    options.lbfgs_step_size_decrease_factor, options.lbfgs_step_size_increase_factor));
            else
                throw std::runtime_error("only float and double precision allowed");
            bdd_log << "[bdd solver] constructed LBFGS CUDA based mma solver\n"; 
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::lbfgs_parallel_mma)
        {
            if(options.smoothing != 0)
                throw std::runtime_error("no smoothing implemented for LBFGS parallel mma");
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_lbfgs_parallel_mma<float>(
                    bdd_pre.get_bdd_collection(), costs.begin(), costs.end(),
                    options.lbfgs_history_size,
                    options.lbfgs_step_size, options.lbfgs_required_relative_lb_increase, 
                    options.lbfgs_step_size_decrease_factor, options.lbfgs_step_size_increase_factor));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_lbfgs_parallel_mma<double>(
                    bdd_pre.get_bdd_collection(), costs.begin(), costs.end(),
                    options.lbfgs_history_size,
                    options.lbfgs_step_size, options.lbfgs_required_relative_lb_increase, 
                    options.lbfgs_step_size_decrease_factor, options.lbfgs_step_size_increase_factor));
            else
                throw std::runtime_error("only float and double precision allowed");
            bdd_log << "[bdd solver] constructed LBFGS parallel mma solver\n"; 
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::subgradient)
        {
            if(options.smoothing != 0)
                throw std::runtime_error("no smoothing implemented for subgradient");
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_subgradient<float>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_subgradient<double>(bdd_pre.get_bdd_collection(), costs.begin(), costs.end()));
            else
                throw std::runtime_error("only float and double precision allowed");
            bdd_log << "[bdd solver] constructed subgradient solver\n"; 
        }
        else
        {
            throw std::runtime_error("no solver nor output of statistics or export of lp selected");
        }

        // set smoothing
        if(options.smoothing != 0.0)
            std::visit([&](auto&& s) { 
                    if constexpr(
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_smooth<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_smooth<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma_smooth<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma_smooth<double>>
                            )
                    s.set_smoothing(options.smoothing);
                    else
                    throw std::runtime_error("smoothing not implemented for chosen solver");
                    }, *solver);

        // set constant
        if(options.ilp.constant() != 0.0)
            std::visit([&](auto&& s) { 
                    if constexpr(
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_smooth<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_smooth<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma_smooth<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma_smooth<double>>
                            )
                    s.add_to_constant(options.ilp.constant());
                    else
                    throw std::runtime_error("constants not implemented for chosen solver");
            }, *solver);

        auto setup_time = (double) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() / 1000;
        bdd_log << "[bdd solver] setup time = " << setup_time << " s" << "\n";
        options.time_limit -= setup_time;
    }

    void bdd_solver::solve()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        if(options.time_limit < 0)
        {
            bdd_log << "[bdd_solver] Time limit exceeded.\n";
            return;
        }
        std::visit([&](auto&& s) {

                run_solver(s, options.max_iter, options.tolerance, options.improvement_slope, options.time_limit);
                }, *solver);

        // TODO: improve, do periodic tightening
        if(options.tighten)
        {
            for(size_t tighten_iter=0; tighten_iter<10; ++tighten_iter)
            {
            tighten();
            std::visit([&](auto&& s) {
                    for(size_t iter=0; iter<10; ++iter)
                        s.iteration();
                    }, *solver);
            }
        }

        if(options.solution_statistics)
        {
            bdd_log << "[bdd solver] print solution statistics:\n";
            const auto var_perm = options.ilp.get_variable_permutation().inverse_permutation();
            assert(var_perm.size() == options.ilp.nr_variables());
            const auto mms = min_marginals();
            const auto mm_diffs = min_marginal_differences(mms, 0.0);
            assert(mms.size() == options.ilp.nr_variables());
            assert(mm_diffs.size() == options.ilp.nr_variables());

            for(size_t i=0; i<mm_diffs.size(); ++i)
            {
                bdd_log << options.ilp.get_var_name(var_perm[i]) << ", position = " << var_perm[i] << ", c = " << options.ilp.objective(var_perm[i]) << ", min marginal = " << mm_diffs[i] << "\n";
            }

            print_mm(mms);
        }

        if(options.export_difficult_core != "")
            export_difficult_core();
    }

    two_dim_variable_array<std::array<double,2>> bdd_solver::min_marginals()
    {
        const auto mms = std::visit([&](auto&& s) { 
                return s.min_marginals();
                }, *solver); 
        return permute_min_marginals(mms, options.ilp.get_variable_permutation());
    }

    std::tuple<std::vector<std::string>, std::vector<double>, std::vector<double>> bdd_solver::min_marginals_with_variable_names()
    {
        return export_min_marginals_with_names(min_marginals(), options.ilp.var_index_to_name());
    }

    std::vector<std::string> bdd_solver::variable_names()
    {
        return options.ilp.var_index_to_name();
    }

    std::tuple<double, std::vector<char>> bdd_solver::round()
    {
        if(options.incremental_primal_rounding)
        {
            bdd_log << "[incremental primal rounding] start rounding\n";
            const auto sol = std::visit([&](auto &&s)
                                        {
                    if constexpr( // CPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_lbfgs_parallel_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_lbfgs_parallel_mma<double>>
                            // TODO: remove for cuda rounding again //
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<float>>
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<double>>
                            //////////////////////////////////////////
                            )
                            {
                    return incremental_mm_agreement_rounding_iter(s, options.incremental_initial_perturbation, options.incremental_growth_rate, options.incremental_primal_num_itr_lb, options.incremental_primal_rounding_num_itr);
                            }
                    else if constexpr( // GPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_lbfgs_cuda_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_lbfgs_cuda_mma<double>>
                            )
                    {
                    return s.incremental_mm_agreement_rounding(options.incremental_initial_perturbation, options.incremental_growth_rate, options.incremental_primal_num_itr_lb, options.incremental_primal_rounding_num_itr);
                    }
                    else
                    {
                    throw std::runtime_error("solver not supported for incremental rounding");
                    return std::vector<char>{};
                    } },
                                        *solver);

            double obj = std::numeric_limits<double>::infinity();
            if (sol.size() >= options.ilp.nr_variables())
                obj = options.ilp.evaluate(sol.begin(), sol.begin() + options.ilp.nr_variables());
            bdd_log << "[incremental primal rounding] solution objective = " << obj << "\n";
            return {obj, sol};
        }
        else if(options.wedelin_primal_rounding)
        {
            bdd_log << "[Wedelin primal rounding] start rounding\n";
            const auto sol = std::visit([&](auto&& s) {
                    if constexpr( // CPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<double>>
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>>
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                            // TODO: remove for cuda rounding again //
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<float>>
                            //|| std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<double>>
                            //////////////////////////////////////////
                            )
                    return wedelin_rounding(s, options.ilp,
                            options.wedelin_theta,
                            options.wedelin_delta,
                            options.wedelin_kappa_min, options.wedelin_kappa_max,
                            options.wedelin_kappa_step, options.wedelin_alpha,
                            500);
                    else if constexpr( // GPU rounding
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda<double>>
                            )
                    {
                    throw std::runtime_error("Wedelin rounding not implemented for GPU solver yet");
                    }

                    {
                    throw std::runtime_error("solver not supported for incremental rounding");
                    return std::vector<char>{};
                    }
                    }, *solver);

            const double obj = options.ilp.evaluate(sol.begin(), sol.end());
            bdd_log << "[incremental primal rounding] solution objective = " << obj << "\n";
            return {obj, sol};

        }
        else // no rounding
        {
            return {std::numeric_limits<double>::infinity(), std::vector<char>{}};
        }
    }

    void bdd_solver::tighten()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;

        if(options.time_limit < 0)
        {
            bdd_log << "Time limit exceeded, aborting tightening.\n";
            return;
        }

        bdd_log << "Tighten...\n";
        std::visit([](auto&& s) {
            if constexpr(std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>>)
            s.tighten();
            else
                throw std::runtime_error("tighten not implemented");
            }, *solver);
    }

    void bdd_solver::fix_variable(const size_t var, const bool value)
    {
        std::visit([var, value](auto&& s) {
            if constexpr(
                    std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma<double>>
                    ||
                    std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                    )
            s.fix_variable(var, value);
            else
                throw std::runtime_error("fix variable not implemented");
            }, *solver);
    }

    void bdd_solver::fix_variable(const std::string& var, const bool value)
    {
        const size_t var_index = options.ilp.get_var_index(var);
        fix_variable(var_index, value);
    }

    double bdd_solver::lower_bound()
    {
        return std::visit([](auto&& s) {
                return s.lower_bound(); 
                }, *solver);
    }

    void bdd_solver::export_difficult_core()
    {
        mm_primal_decoder mms(min_marginals());
        std::unordered_set<size_t> one_fixations, zero_fixations;
        for(size_t i=0; i<mms.size(); ++i)
        {
            const auto mmt = mms.compute_mm_type(i);
            const auto mm_sum = mms.mm_sum(i);
            if(mmt == mm_type::one && mm_sum[1] + options.export_difficult_core_th <= mm_sum[0])
                one_fixations.insert(i);
            else if(mmt == mm_type::zero && mm_sum[0] + options.export_difficult_core_th <= mm_sum[1])
                zero_fixations.insert(i);
        }
        ILP_input reduced_ilp = options.ilp.reduce(zero_fixations, one_fixations);
        bdd_log << "[bdd solver] Difficult core has " << reduced_ilp.nr_variables() << " variables and " << reduced_ilp.constraints().size() << " constraints left\n";

        std::ofstream f;
        f.open(options.export_difficult_core);
        const std::string extension = std::filesystem::path(options.export_difficult_core).extension();
        if(extension == ".lp")
            reduced_ilp.write_lp(f);
        else if(extension == ".opb")
            reduced_ilp.write_opb(f);
        else if(extension == ".mps")
            reduced_ilp.write_mps(f);
        else
            throw std::runtime_error("Cannot recognize file extension " + extension + " for difficult core export file");
        f.close(); 

    }

}
