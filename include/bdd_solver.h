#pragma once

#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "bdd_mma.h"
#include "bdd_mma_smooth.h"
#include "bdd_cuda.h"
#include "bdd_parallel_mma.h"
#include "bdd_parallel_mma_smooth.h"
#include "bdd_multi_parallel_mma.h"
#include "bdd_lbfgs_parallel_mma.h"
#include "bdd_lbfgs_cuda_mma.h"
#include "bdd_subgradient.h"
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
        bdd_solver_options() {};
        bdd_solver_options(const std::string& ilp_file_path);
        bdd_solver_options(ILP_input& _ilp);

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

        enum class bdd_solver_impl { sequential_mma, mma_cuda, parallel_mma, hybrid_parallel_mma, lbfgs_cuda_mma, lbfgs_parallel_mma, subgradient } bdd_solver_impl_;
        enum class bdd_solver_precision { single_prec, double_prec } bdd_solver_precision_ = bdd_solver_precision::double_prec;
        bool solution_statistics = false;

        double smoothing = 0;

        bool tighten = false;

        // cuda solver options //
        bool cuda_split_long_bdds = false;
        bool cuda_split_long_bdds_implication_bdd = false;
        size_t cuda_split_long_bdds_length = std::numeric_limits<size_t>::max();
        /////////////////////////

        // lbfgs solver options //
        double lbfgs_step_size = 1e-6;
        size_t lbfgs_history_size = 5;
        double lbfgs_required_relative_lb_increase = 1e-6;
        double lbfgs_step_size_decrease_factor = 0.8;
        double lbfgs_step_size_increase_factor = 1.1;

        // incremental perturbation rounding //
        bool incremental_primal_rounding = false;
        double incremental_initial_perturbation = std::numeric_limits<double>::infinity();
        double incremental_growth_rate = 1.2;
        int incremental_primal_num_itr_lb = 500;
        int incremental_primal_rounding_num_itr = 500;
        //////////////////////////////////////

        // Wedelin rounding //
        // value ranges are taken from "Learning parameters of the Wedelin heuristic with application to crew and bus driver scheduling"
        bool wedelin_primal_rounding = false;
        double wedelin_alpha = 0.5; // [0,2]
        double wedelin_delta = 0.001; // 10^-3, 10^-1]
        double wedelin_theta = 0.8; // [0,1]
        double wedelin_kappa_min = 0.1; // [0,0.5]
        double wedelin_kappa_max = 0.95; // [0.6,1]
        double wedelin_kappa_step = 0.0001; // [10^-4, 10^-2]
        //////////////////////

        bool statistics = false;
        std::string export_bdd_lp_file = "";
        std::string export_lp_file = "";
        std::string export_bdd_graph_file = "";

        // export difficult part of the problems including zero and undecided min-marginals 
        std::string export_difficult_core = "";
        double export_difficult_core_th = 1e-6;

        // logging options
        bool suppress_console_output = false;
        std::string log_file;
    };

    class bdd_solver {
        public:
            bdd_solver(bdd_solver_options opt);
            //bdd_solver(bdd_solver_options&& opt);
            //bdd_solver(int argc, char** argv);
            //bdd_solver(const std::vector<std::string>& args);

            void solve();
            std::tuple<double, std::vector<char>> round();
            void tighten();
            double lower_bound();
            void fix_variable(const size_t var, const bool value);
            void fix_variable(const std::string& var, const bool value);
            two_dim_variable_array<std::array<double,2>> min_marginals();
            void export_difficult_core();

        private:
            //bdd_preprocessor preprocess(ILP_input& ilp);
            bdd_solver_options options;
            using solver_type = std::variant<
                bdd_mma<float>, bdd_mma<double>, bdd_mma_smooth<float>, bdd_mma_smooth<double>,
                bdd_cuda<float>, bdd_cuda<double>,
                bdd_parallel_mma<float>, bdd_parallel_mma<double>, bdd_parallel_mma_smooth<float>, bdd_parallel_mma_smooth<double>,
                bdd_multi_parallel_mma<float>, bdd_multi_parallel_mma<double>,
                bdd_lbfgs_parallel_mma<double>,
                bdd_lbfgs_parallel_mma<float>,
                bdd_lbfgs_cuda_mma<double>,
                bdd_lbfgs_cuda_mma<float>,
                bdd_subgradient<float>,
                bdd_subgradient<double>
                    >;
            std::optional<solver_type> solver;
            std::vector<double> costs;
    };

}
