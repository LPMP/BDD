#include "bdd_solver.h"
#include "ILP_parser.h"
#include "OPB_parser.h"
#include "min_marginal_utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <memory>
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
        std::cout << "#variables = " << ilp.nr_variables() << "\n";
        std::cout << "#constraints = " << ilp.nr_constraints() << "\n";
        std::cout << "#BDDs = " << stor.nr_bdds() << "\n";
        const auto var_groups = stor.compute_variable_groups();
        std::cout << "#var groups = " << var_groups.size() << "\n";
        std::cout << "#average nr vars per group = " << stor.nr_variables() / double(var_groups.size()) << "\n";
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
        auto lp_input_string_arg = input_group->add_option("--input_as_string", lp_input_as_string, "ILP input in string");

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

        //enum class bdd_solver_impl { mma, mma_srmp, mma_agg, decomposition_mma, anisotropic_mma, mma_vec } bdd_solver_impl_;
        std::unordered_map<std::string, bdd_solver_impl> bdd_solver_impl_map{
            {"mma",bdd_solver_impl::mma},
            {"decomposition_mma",bdd_solver_impl::decomposition_mma},
            {"mma_srmp",bdd_solver_impl::mma_srmp},
            {"mma_agg",bdd_solver_impl::mma_agg},
            {"anisotropic_mma",bdd_solver_impl::anisotropic_mma},
            {"mma_vec",bdd_solver_impl::mma_vec}
        };

        auto solver_group = app.add_option_group("solver", "solver either a BDD solver, output of statistics or export of LP solved by BDD relaxation");
        solver_group->add_option("-s, --solver", bdd_solver_impl_, "the name of solver for the relaxation")
            ->transform(CLI::CheckedTransformer(bdd_solver_impl_map, CLI::ignore_case));

        auto solution_statistics_arg = app.add_flag("--solution_statistics", solution_statistics, "list min marginals and, objective after solving dual problem");

        //bool primal_rounding = false;
        auto primal_arg = app.add_flag("-p, --primal", primal_rounding, "primal rounding flag");

        auto primal_param_group = app.add_option_group("primal parameters", "parameters for rounding a primal solution");
        primal_param_group->needs(primal_arg);

        //bdd_fix_options fixing_options_;
        using fix_order = bdd_fix_options::variable_order;
        using fix_value = bdd_fix_options::variable_value;

        std::unordered_map<std::string, fix_order> fixing_var_order_map{{"marg_abs",fix_order::marginals_absolute},{"marg_up",fix_order::marginals_up},{"marg_down",fix_order::marginals_down},{"marg_red",fix_order::marginals_reduction}};
        primal_param_group->add_option("--fixing_order", fixing_options_.var_order, "variable order for primal heuristic, default value = marg_up")
            ->transform(CLI::CheckedTransformer(fixing_var_order_map, CLI::ignore_case));

        std::unordered_map<std::string, fix_value> fixing_var_value_map{{"marg",fix_value::marginal},{"red",fix_value::reduction},{"one",fix_value::one},{"zero",fix_value::zero}};
        primal_param_group->add_option("--fixing_value", fixing_options_.var_value, "preferred variable value for primal heuristic, default value = marg")
            ->transform(CLI::CheckedTransformer(fixing_var_value_map, CLI::ignore_case));

        auto tighten_arg = app.add_flag("--tighten", tighten, "tighten relaxation flag");
        
        bool statistics = false;
        solver_group->add_flag("--statistics", statistics, "statistics of the problem");

        solver_group->add_option("--export_bdd_lp", export_bdd_lp_file, "filename for export of LP of the BDD relaxation");

        solver_group->require_option(1); // either a solver or statistics

        app.callback([this,&app]() {
                CLI::App solver_app;
                std::cout << "decomposition_mma callback\n";

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
        else if(!options.export_bdd_lp.empty())
        {
            bdd_pre.construct_bdd_collection(); // this is only needed if bdd collection is used in bdd preprocessor
            std::ofstream f;
            f.open(options.export_bdd_lp);
            bdd_pre.get_bdd_collection().write_bdd_lp(f, options.ilp.objective().begin(), options.ilp.objective().end());
            f.close();
            exit(0);
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::mma)
        {
            solver = std::move(bdd_mma(stor, options.ilp.objective().begin(), options.ilp.objective().end()));
            std::cout << "constructed mma solver\n";
        } 
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::mma_srmp)
        {
            solver = std::move(bdd_mma_srmp(stor, options.ilp.objective().begin(), options.ilp.objective().end()));
            std::cout << "constructed srmp mma solver\n";
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::mma_agg)
        {
            solver = std::move(bdd_mma_agg(stor, options.ilp.objective().begin(), options.ilp.objective().end()));
            std::cout << "constructed aggressive mma solver\n";
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::decomposition_mma)
        {
            solver = std::move(decomposition_bdd_mma(stor, options.ilp.objective().begin(), options.ilp.objective().end(), options.decomposition_mma_options_));
            std::cout << "constructed decomposition mma solver\n";
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::anisotropic_mma)
        {
            solver = std::move(bdd_mma_anisotropic(stor, options.ilp.objective().begin(), options.ilp.objective().end()));
            std::cout << "constructed anisotropic mma solver\n"; 
        }
        else if(options.bdd_solver_impl_ == bdd_solver_options::bdd_solver_impl::mma_vec)
        {
            solver = std::move(bdd_mma_vec(stor, options.ilp.objective().begin(), options.ilp.objective().end()));
            std::cout << "constructed vectorized mma solver\n"; 
        }
        else
        {
            assert(false);
        }

        if(options.primal_rounding)
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
        if(options.time_limit < 0)
        {
            std::cout << "Time limit exceeded." << std::endl;
            return;
        }
        std::visit([&](auto&& s) {
                s.solve(options.max_iter, options.tolerance, options.time_limit);
                }, *solver);

        // TODO: improve, do periodic tightening
        if(options.tighten)
        {
            for(size_t tighten_iter=0; tighten_iter<10; ++tighten_iter)
            {
            tighten();
            std::visit([&](auto&& s) {
                    s.solve(10, 1e-9, options.time_limit);
                    }, *solver);
            }
        }

        if(options.solution_statistics)
        {
            std::cout << "print solution statistics:\n";
            const auto mms = min_marginals();
            const auto mm_diffs = min_marginal_differences(mms, 0.0);
            assert(mms.size() == options.ilp.nr_variables());
            assert(mm_diffs.size() == options.ilp.nr_variables());

            for(size_t i=0; i<mm_diffs.size(); ++i)
            {
                std::cout << options.ilp.get_var_name(i) << ", c = " << options.ilp.objective(i) << ", min marginal = " << mm_diffs[i] << "\n";
            }
        }
    }

    two_dim_variable_array<std::array<double,2>> bdd_solver::min_marginals()
    {
        return std::visit([&](auto&& s) { 
                return s.min_marginals();
                }, *solver); 
    }

    void bdd_solver::round()
    {
        assert(options.primal_rounding == bool(primal_heuristic));
        if(!options.primal_rounding)
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

        // TODO: const?
        std::vector<double> total_min_marginals = min_marginal_differences(min_marginals(), 0.0);
        bool success = primal_heuristic->round(total_min_marginals);

        if (!success)
            return;

        std::vector<char> primal_solution = primal_heuristic->primal_solution();
        assert(std::all_of(primal_solution.begin(), primal_solution.end(), [](char x){ return (x >= 0) && (x <= 1);}));
        assert(primal_solution.size() == costs.size());
        double upper_bound = std::inner_product(primal_solution.begin(), primal_solution.end(), costs.begin(), 0.0);
        std::cout << "Primal solution value: " << upper_bound << std::endl;
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
            if constexpr(std::is_same_v<std::remove_reference_t<decltype(s)>, bdd_mma_vec>)
            s.tighten();
            else
                throw std::runtime_error("tighten not implemented");
            }, *solver);
    }

    double bdd_solver::lower_bound()
    {
        return std::visit([](auto&& s) {
                return s.lower_bound(); 
                }, *solver);

    }

}

