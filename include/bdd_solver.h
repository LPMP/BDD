#pragma once

#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "decomposition_bdd_mma.h"
#include "bdd_mma_vec.h"
#include "bdd_mma_smooth.h"
#include "bdd_cuda.h"
#include "bdd_parallel_mma.h"
#include "bdd_parallel_mma_smooth.h"
#include "bdd_fix.h"
#include "incremental_mm_agreement_rounding.hxx"
#include <variant> 
#include <optional>
#include <CLI/CLI.hpp>
#include "time_measure_util.h"

namespace LPMP {

    // glue together various phases of solving:
    // (i) obtain an ILP_input, reorder variables.
    // (ii) preprocess it.
    // (iii) transformation into QBDD format.
    // (iv) give the QBDDs to a specific bdd solver (i.e. mma, parallel_mma, ...).
    // (v) solve the dual.
    // (vi) try to obtain a primal solution.
    
    struct bdd_solver_options {
        bdd_solver_options(int argc, char** argv);
        bdd_solver_options(const std::vector<std::string>& args);
        template<typename FILE_INPUT_FUNCTION, typename STRING_INPUT_FUNCTION>
            bdd_solver_options(int argc, char** argv, FILE_INPUT_FUNCTION file_input_function, STRING_INPUT_FUNCTION string_input_function);
        bdd_solver_options() {};

        std::string input_file;
        std::string input_string;
        bool take_cost_logarithms = false;
        enum class optimization_type { minimization, maximization } optimization = optimization_type::minimization;
        ILP_input ilp;
        ILP_input::variable_order var_order = ILP_input::variable_order::input;

        // termination criteria //
        size_t max_iter = 1000;
        double tolerance = 1e-9;
        double improvement_slope = 1e-6;
        double time_limit = 3600;
        //////////////////////////

        enum class bdd_solver_impl { sequential_mma, decomposition_mma, mma_cuda, parallel_mma } bdd_solver_impl_;
        enum class bdd_solver_precision { single_prec, double_prec } bdd_solver_precision_ = bdd_solver_precision::single_prec;
        decomposition_mma_options decomposition_mma_options_;
        bool solution_statistics = false;

        double smoothing = 0;

        bool tighten = false;

        bool incremental_primal_rounding = false;
        double incremental_initial_perturbation = std::numeric_limits<double>::infinity();
        double incremental_growth_rate = 1.2;
        int incremental_primal_num_itr_lb = 500;

        bool diving_primal_rounding = false;
        bdd_fix_options fixing_options_;

        bool statistics = false;
        std::string export_bdd_lp_file = "";
        std::string export_bdd_graph_file = "";

        bool constraint_groups = true; // allow constraint groups to be formed e.g. from indicators in the input lp files
    };

    class bdd_solver {
        public:
            bdd_solver(bdd_solver_options opt);
            //bdd_solver(bdd_solver_options&& opt);
            //bdd_solver(int argc, char** argv);
            //bdd_solver(const std::vector<std::string>& args);

            void solve();
            double round();
            void tighten();
            double lower_bound();
            void fix_variable(const size_t var, const bool value);
            void fix_variable(const std::string& var, const bool value);
            two_dim_variable_array<std::array<double,2>> min_marginals();

        private:
            //bdd_preprocessor preprocess(ILP_input& ilp);
            bdd_solver_options options;
            using solver_type = std::variant<
                bdd_mma_vec<float>, bdd_mma_vec<double>, bdd_mma_smooth<float>, bdd_mma_smooth<double>,
                decomposition_bdd_mma,
                bdd_cuda<float>, bdd_cuda<double>,
                bdd_parallel_mma<float>, bdd_parallel_mma<double>, bdd_parallel_mma_smooth<float>, bdd_parallel_mma_smooth<double>
                    >;
            std::optional<solver_type> solver;
            std::vector<double> costs;
            std::optional<bdd_fix> primal_heuristic;
    };

    template<typename FILE_INPUT_FUNCTION, typename STRING_INPUT_FUNCTION>
        bdd_solver_options::bdd_solver_options(int argc, char** argv, FILE_INPUT_FUNCTION file_input_function, STRING_INPUT_FUNCTION string_input_function)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;

        // setup command line arguemnts
        CLI::App app("LPMP BDD solver");
        app.allow_extras();
 
        auto input_group = app.add_option_group("input", "input either from file or as string");
        auto input_file_arg = input_group->add_option("-i, --input_file", input_file, "ILP input file name")
            ->check(CLI::ExistingPath);

        auto input_string_arg = input_group->add_option("--input_string", input_string, "ILP input in string");

        input_group->require_option(1); // either as string or as filename

        app.add_flag("--logarithms", take_cost_logarithms, "");
        std::unordered_map<std::string, optimization_type> optimization_type_map{
            {"maximization", optimization_type::maximization},
            {"minimization", optimization_type::minimization}
        };
        app.add_option("--optimization", optimization, "minimization/maximization of objective")
            ->transform(CLI::CheckedTransformer(optimization_type_map, CLI::ignore_case));

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
            ->check(CLI::PositiveNumber);

        app.add_option("--improvement_slope", improvement_slope, "termination criterion: improvement between iterations as compared to improvement after first iterations, default value = " + std::to_string(improvement_slope))
            ->check(CLI::Range(0.0, 1.0));

        app.add_option("--constraint_groups", constraint_groups, "allow multiple constraints to be fused into one, default = true");

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

        app.add_option("--smoothing", smoothing, "smoothing, default value = 0 (no smoothing)")
                ->check(CLI::PositiveNumber);

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

        incremental_rounding_param_group->add_option("--incremental_primal_num_itr_lb", incremental_primal_num_itr_lb, "number of iterations of dual optimization during incremental primal rounding")
            ->check(CLI::Range(1,std::numeric_limits<int>::max()));

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
                    std::cout << "[bdd solver] use decomposition mma solver\n";
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

        assert(input_string.empty() || input_file.empty());

        ilp = [&]() {
            if(!input_file.empty())
            {
                return file_input_function(input_file);
            }
            else if(!input_string.empty())
            {
                return string_input_function(input_string);
            }
            else
                return ILP_input();
        }();

        ilp.normalize();
    }

}
