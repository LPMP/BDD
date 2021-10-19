#include "bdd_solver.h"
#include "ILP_parser.h"
#include "OPB_parser.h"
#include "min_marginal_utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <memory>
#include <stdlib.h>
#include <stdexcept>
#include <CLI/CLI.hpp>
#include "time_measure_util.h"

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

    void print_statistics(ILP_input& ilp, bdd_storage& stor)
    {
        std::cout << "[print_statistics] #variables = " << ilp.nr_variables() << "\n";
        std::cout << "[print_statistics] #constraints = " << ilp.nr_constraints() << "\n";
        std::cout << "[print_statistics] #BDDs = " << stor.nr_bdds() << "\n";
        const auto var_groups = stor.compute_variable_groups();
        std::cout << "[print_statistics] #var groups = " << var_groups.size() << "\n";
        std::cout << "[print_statistics] #average nr vars per group = " << stor.nr_variables() / double(var_groups.size()) << "\n";
    }

    bdd_solver_options::bdd_solver_options(int argc, char** argv)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;

        // setup command line arguemnts
        CLI::App app("LPMP BDD solver");
        app.allow_extras();
 
        auto input_group = app.add_option_group("input", "input either from file or as string");
        //std::string input_file;
        auto input_file_arg = input_group->add_option("-i, --input_file", input_file, "ILP input file name")
            ->check(CLI::ExistingPath);

        //std::string lp_input_string;
        auto lp_input_string_arg = input_group->add_option("--lp_input_string", lp_input_as_string, "ILP input in string");

        //std::string opb_input_as_string;
        auto opb_input_string_arg = input_group->add_option("--opb_input_string", opb_input_as_string, "OPB input in string");

        input_group->require_option(1); // either as string or as filename

        std::unordered_map<std::string, ILP_input::variable_order> variable_order_map{
            {"input", ILP_input::variable_order::input},
            {"bfs", ILP_input::variable_order::bfs},
            {"cuthill", ILP_input::variable_order::cuthill},
            {"mindegree", ILP_input::variable_order::mindegree}
        };

        //ILP_input::variable_order variable_order_ = ILP_input::variable_order::input;A
        app.add_option("-o, --order", var_order, "variable order")
            ->transform(CLI::CheckedTransformer(variable_order_map, CLI::ignore_case));

        app.add_option("-m, --max_iter", max_iter, "maximal number of iterations, default value = 10000")
            ->check(CLI::PositiveNumber);

        app.add_option("--tolerance", tolerance, "lower bound relative progress tolerance, default value = 1e-06")
            ->check(CLI::PositiveNumber);

        app.add_option("-l, --time_limit", time_limit, "time limit in seconds, default value = 3600")
            ->check(CLI::PositiveNumber);

        std::unordered_map<std::string, bdd_solver_impl> bdd_solver_impl_map{
            {"mma",bdd_solver_impl::sequential_mma},
            {"mma_vec",bdd_solver_impl::sequential_mma}, // legacy name
            {"sequential_mma",bdd_solver_impl::sequential_mma},
            {"decomposition_mma",bdd_solver_impl::decomposition_mma},
            {"parallel_mma",bdd_solver_impl::parallel_mma},
            {"mma_cuda",bdd_solver_impl::mma_cuda}
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

        auto bdd_solver_precision_arg = app.add_option("--precision", bdd_solver_precision_, "floating point precision used in solver")
            ->transform(CLI::CheckedTransformer(bdd_solver_precision_map, CLI::ignore_case));

        auto primal_group = app.add_option_group("primal rounding", "method for obtaining a primal solution from the dual optimization");
        auto diving_primal_arg = primal_group->add_flag("--diving_primal", diving_primal_rounding, "diving primal rounding flag");
        auto incremental_primal_arg = primal_group->add_flag("--incremental_primal", incremental_primal_rounding, "incremental primal rounding flag");
        primal_group->require_option(0,1); 

        auto primal_param_group = app.add_option_group("primal diving parameters", "parameters for rounding a primal solution");
        primal_param_group->needs(diving_primal_arg);

        //bdd_fix_options fixing_options_;
        using fix_order = bdd_fix_options::variable_order;
        using fix_value = bdd_fix_options::variable_value;

        std::unordered_map<std::string, fix_order> fixing_var_order_map{{"marg_abs",fix_order::marginals_absolute},{"marg_up",fix_order::marginals_up},{"marg_down",fix_order::marginals_down},{"marg_red",fix_order::marginals_reduction}};
        primal_param_group->add_option("--fixing_order", fixing_options_.var_order, "variable order for primal heuristic, default value = marg_up")
            ->transform(CLI::CheckedTransformer(fixing_var_order_map, CLI::ignore_case));

        std::unordered_map<std::string, fix_value> fixing_var_value_map{{"marg",fix_value::marginal},{"red",fix_value::reduction},{"one",fix_value::one},{"zero",fix_value::zero}};
        primal_param_group->add_option("--fixing_value", fixing_options_.var_value, "preferred variable value for primal heuristic, default value = marg")
            ->transform(CLI::CheckedTransformer(fixing_var_value_map, CLI::ignore_case));

        auto incremental_rounding_param_group = app.add_option_group("incremental primal rounding parameters", "parameters for rounding a primal solution");

        incremental_rounding_param_group->add_option("--incremental_initial_perturbation", incremental_initial_perturbation, "value for initial perturbation for obtaining primal solutions by incremental primal rounding")
            ->check(CLI::PositiveNumber);
       incremental_rounding_param_group->needs(incremental_primal_arg);

        incremental_rounding_param_group->add_option("--incremental_perturbation_growth_rate", incremental_growth_rate, "growth rate for increasing the perturbation for obtaining primal solutions by incremental primal rounding")
            ->check(CLI::Range(1.0,std::numeric_limits<double>::max()));


        auto tighten_arg = app.add_flag("--tighten", tighten, "tighten relaxation flag");
        
        solver_group->add_flag("--statistics", statistics, "statistics of the problem");

        solver_group->add_option("--export_bdd_lp", export_bdd_lp_file, "filename for export of LP of the BDD relaxation");

        solver_group->add_option("--export_bdd_graph", export_bdd_graph_file, "filename for export of BDD representation in .dot format");

        solver_group->require_option(1); // either a solver or statistics

        // TODO: replace with needs as for incremental rounding options
        /*
        auto decomposition_mma_group = app.add_option_group("options for decomposition mma");
        decomposition_mma_group
            decomposition_mma_group->add_option("--nr_threads", decomposition_mma_options_.nr_threads, "number of threads (up to available nr of available units) for simultaneous optimization of the Lagrange decomposition")
            ->required()
            ->check(CLI::Range(2, omp_get_max_threads()));

        decomposition_mma_group->add_flag("--force_thread_nr", decomposition_mma_options_.force_thread_nr , "force the number of threads be as specified, do not choose lower thread number even if subproblems become small");

        decomposition_mma_group->add_option("--parallel_message_passing_weight", decomposition_mma_options_.parallel_message_passing_weight, "weight for passing messages between threads")
            ->check(CLI::Range(0.0,1.0));

        decomposition_mma_group->needs
        */

        app.callback([this,&app]() {
                CLI::App solver_app;

                if(bdd_solver_impl_ == bdd_solver_impl::decomposition_mma)
                {
#ifdef _OPENMP
                    std::cout << "use decomposition mma solver\n";
                    solver_app.add_option("--nr_threads", decomposition_mma_options_.nr_threads, "number of threads (up to available nr of available units) for simultaneous optimization of the Lagrange decomposition")
                        ->required()
                        ->check(CLI::Range(2, omp_get_max_threads()));

                    solver_app.add_flag("--force_thread_nr", decomposition_mma_options_.force_thread_nr , "force the number of threads be as specified, do not choose lower thread number even if subproblems become small");

                    solver_app.add_option("--parallel_message_passing_weight", decomposition_mma_options_.parallel_message_passing_weight, "weight for passing messages between threads")
                        ->check(CLI::Range(0.0,1.0));


                    solver_app.parse(app.remaining_for_passthrough());
#else
                    std::cout << "No OpenMP found, decomposition_mma not supported\n";
                    throw std::runtime_error("OpenMP needed but not found");
#endif
                }
        });

        app.parse(argc, argv); 

        ilp = [&]() {
            if(!input_file.empty())
            {
                if(input_file.substr(input_file.find_last_of(".") + 1) == "opb")
                {
                    std::cout << "Parse opb file\n";
                    return OPB_parser::parse_file(input_file);
                }
                else
                {
                    std::cout << "Parse lp file\n";
                    return ILP_parser::parse_file(input_file);
                }
            }
            else if(!lp_input_as_string.empty())
            {
                // Possibly check if file is in lp or opb format
                return ILP_parser::parse_string(lp_input_as_string);
            }
            else if(!opb_input_as_string.empty())
            {
                return OPB_parser::parse_string(opb_input_as_string); 
            }
            else
                return ILP_input();
        }();
    }

    bdd_solver_options::bdd_solver_options(const std::vector<std::string>& args)
        : bdd_solver_options(args.size()+1, convert_string_to_argv(args).get())
    {}

    bdd_solver::bdd_solver(int argc, char** argv)
        : bdd_solver(bdd_solver_options(argc, argv))
    {}

    bdd_solver::bdd_solver(const std::vector<std::string>& args)
        : bdd_solver(bdd_solver_options(args))
    {}

    bdd_solver::bdd_solver(bdd_solver_options opt)
        : options(opt)
    {
        options.ilp.reorder(options.var_order);

        std::cout << "ILP has " << options.ilp.nr_variables() << " variables and " << options.ilp.nr_constraints() << " constraints\n";
        if(options.ilp.preprocess())
            std::cout << "ILP has " << options.ilp.nr_variables() << " variables and " << options.ilp.nr_constraints() << " constraints after preprocessing\n";
        else
        {
            std::cout << "The problem appears to be infeasible." << std::endl;
            return;
        }

        const auto start_time = std::chrono::steady_clock::now();

        costs = options.ilp.objective();

        // print variable order
        // for (size_t v = 0; v < ilp.nr_variables(); v++)
        //     std::cout << ilp.get_var_name(v) << std::endl;

        bdd_preprocessor bdd_pre(options.ilp);
        //bdd_pre.construct_bdd_collection(); // this is only needed if bdd collection is used in bdd preprocessor
        bdd_storage stor(bdd_pre);

        std::cout << std::setprecision(10);

        if(options.statistics)
        {
            print_statistics(options.ilp, stor);
            exit(0);
        }
        else if(!options.export_bdd_lp_file.empty())
        {
            std::ofstream f;
            f.open(options.export_bdd_lp_file);
            bdd_pre.get_bdd_collection().write_bdd_lp(f, options.ilp.objective().begin(), options.ilp.objective().end());
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
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_mma_vec<float>(bdd_pre.get_bdd_collection(), options.ilp.objective().begin(), options.ilp.objective().end()));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_mma_vec<double>(bdd_pre.get_bdd_collection(), options.ilp.objective().begin(), options.ilp.objective().end()));
            else
                throw std::runtime_error("only float and double precision allowed");
            std::cout << "constructed sequential mma solver\n"; 
        } 
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::decomposition_mma)
        {
            solver = std::move(decomposition_bdd_mma(stor, options.ilp.objective().begin(), options.ilp.objective().end(), options.decomposition_mma_options_));
            std::cout << "constructed decomposition mma solver\n";
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::parallel_mma)
        {
            if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::single_prec)
                solver = std::move(bdd_parallel_mma<float>(bdd_pre.get_bdd_collection(), options.ilp.objective().begin(), options.ilp.objective().end()));
            else if(options.bdd_solver_precision_ == bdd_solver_options::bdd_solver_precision::double_prec)
                solver = std::move(bdd_parallel_mma<double>(bdd_pre.get_bdd_collection(), options.ilp.objective().begin(), options.ilp.objective().end()));
            else
                throw std::runtime_error("only float and double precision allowed");
            std::cout << "constructed parallel mma solver\n"; 
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::mma_cuda)
        {
            solver = std::move(bdd_cuda(bdd_pre.get_bdd_collection(), options.ilp.objective().begin(), options.ilp.objective().end()));
            std::cout << "constructed CUDA based mma solver\n"; 
        }
        else
        {
            throw std::runtime_error("no solver nor output of statistics or export of lp selected");
        }

        if(options.diving_primal_rounding)
        {
            std::cout << options.fixing_options_.var_order << ", " << options.fixing_options_.var_value << "\n";
            primal_heuristic = std::move(bdd_fix(stor, options.fixing_options_));
            std::cout << "constructed primal heuristic\n";
        }

        auto setup_time = (double) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() / 1000;
        std::cout << "setup time = " << setup_time << " s";
        std::cout << "\n";
        options.time_limit -= setup_time;
    }

    ILP_input bdd_solver::get_ILP(const std::string& input_file, ILP_input::variable_order variable_order_)
    { 
        ILP_input ilp = ILP_parser::parse_file(input_file);
        ilp.reorder(variable_order_);
        return ilp;
    }

    bdd_storage bdd_solver::transfer_to_bdd_storage(bdd_preprocessor& bdd_pre)
    {
        bdd_storage stor(bdd_pre);
        return stor;
    }

    void initialize_solver(bdd_storage& stor)
    {

    }

    void bdd_solver::solve()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        if(options.time_limit < 0)
        {
            std::cout << "[bdd_solver] Time limit exceeded." << std::endl;
            return;
        }
        std::visit([&](auto&& s) {

                const auto start_time = std::chrono::steady_clock::now();
                double lb_prev = s.lower_bound();
                double lb_post = lb_prev;
                std::cout << "[bdd_solver] initial lower bound = " << lb_prev;
                auto time = std::chrono::steady_clock::now();
                std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
                std::cout << "\n";
                for(size_t iter=0; iter<options.max_iter; ++iter)
                {
                    s.iteration();
                    lb_prev = lb_post;
                    lb_post = s.lower_bound();
                    std::cout << "[bdd_solver] iteration " << iter << ", lower bound = " << lb_post;
                    time = std::chrono::steady_clock::now();
                    double time_spent = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
                     std::cout << ", time = " << time_spent << " s";
                     std::cout << "\n";
                     if (time_spent > options.time_limit)
                     {
                        std::cout << "[bdd_solver] Time limit reached." << std::endl;
                        break;
                     }
                     if (std::abs(lb_prev-lb_post) < std::abs(options.tolerance*lb_prev))
                     {
                         std::cout << "[bdd_solver] Relative progress less than tolerance (" << options.tolerance << ")\n";
                         break;
                     }
                }
                std::cout << "[bdd_solver] final lower bound = " << s.lower_bound() << "\n"; 
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
            std::cout << "print solution statistics:\n";
            const auto var_perm = options.ilp.get_variable_permutation().inverse_permutation();
            assert(var_perm.size() == options.ilp.nr_variables());
            const auto mms = min_marginals();
            const auto mm_diffs = min_marginal_differences(mms, 0.0);
            assert(mms.size() == options.ilp.nr_variables());
            assert(mm_diffs.size() == options.ilp.nr_variables());

            for(size_t i=0; i<mm_diffs.size(); ++i)
            {
                std::cout << options.ilp.get_var_name(var_perm[i]) << ", position = " << var_perm[i] << ", c = " << options.ilp.objective(var_perm[i]) << ", min marginal = " << mm_diffs[i] << "\n";
            }

            print_mm(mms);
        }
    }

    two_dim_variable_array<std::array<double,2>> bdd_solver::min_marginals()
    {
        const auto mms = std::visit([&](auto&& s) { 
                return s.min_marginals();
                }, *solver); 
        return permute_min_marginals(mms, options.ilp.get_variable_permutation());
    }

    void bdd_solver::round()
    {
        if(options.diving_primal_rounding)
        {
            assert(options.diving_primal_rounding == bool(primal_heuristic));
            if(!options.diving_primal_rounding)
                return;

            MEASURE_FUNCTION_EXECUTION_TIME;

            if(options.time_limit < 0)
            {
                std::cout << "Time limit exceeded, aborting rounding." << std::endl;
                return;
            }

            if(!primal_heuristic)
            {
                std::cout << "no primal heuristic intialized\n";
                return;
            }

            std::cout << "Retrieving total min-marginals..." << std::endl;

            const std::vector<double> total_min_marginals = min_marginal_differences(min_marginals(), 0.0);
            bool success = primal_heuristic->round(total_min_marginals);

            if (!success)
                return;

            std::vector<char> primal_solution = primal_heuristic->primal_solution();
            assert(std::all_of(primal_solution.begin(), primal_solution.end(), [](char x){ return (x >= 0) && (x <= 1);}));
            assert(primal_solution.size() == costs.size());
            double upper_bound = std::inner_product(primal_solution.begin(), primal_solution.end(), costs.begin(), 0.0);
            std::cout << "Primal solution value: " << upper_bound << std::endl;
        }
        else if(options.incremental_primal_rounding)
        {
            std::cout << "[incremental primal rounding] start rounding\n";
            const auto sol = std::visit([&](auto&& s) {
                    if constexpr(
                            std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_vec<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_vec<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<float>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_parallel_mma<double>>
                            || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_cuda>
                            )
                    return incremental_mm_agreement_rounding_iter(s, options.incremental_initial_perturbation, options.incremental_growth_rate);
                    else
                    {
                    throw std::runtime_error("solver not supported for incremental rounding");
                    return std::vector<char>{};
                    }
                    }, *solver);

            const double obj = options.ilp.evaluate(sol.begin(), sol.end());
            std::cout << "[incremental primal rounding] solution objective = " << obj << "\n";
        }
    } 

    void bdd_solver::tighten()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;

        if(options.time_limit < 0)
        {
            std::cout << "Time limit exceeded, aborting tightening." << std::endl;
            return;
        }

        std::cout << "Tighten...\n";
        std::visit([](auto&& s) {
            if constexpr(std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_vec<float>>)
            s.tighten();
            else
                throw std::runtime_error("tighten not implemented");
            }, *solver);
    }

    void bdd_solver::fix_variable(const size_t var, const bool value)
    {
        std::visit([var, value](auto&& s) {
            if constexpr(
                    std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_vec<float>> || std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_vec<double>>
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
}
