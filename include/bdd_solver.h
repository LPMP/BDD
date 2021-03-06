#pragma once

#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "bdd_storage.h"
#include "bdd_mma.h"
#include "bdd_mma_srmp.h"
#include "bdd_mma_agg.h"
#include "decomposition_bdd_mma.h"
#include "bdd_mma_anisotropic.h"
#include "bdd_mma_vec.h"
#include "bdd_fix.h"
#include <variant> 
#include <optional>
#include <CLI/CLI.hpp>

namespace LPMP {

    // glue together various phases of solving:
    // (i) obtain an ILP_input, reorder variables.
    // (ii) preprocess it.
    // (iii) give the input to bdd_storage for transformation into the modified BDD format we use.
    // (iv) give the bdd_storage to a specific bdd solver (i.e. bdd_min_marginal_averaging, decomposition bdd etc.).
    // (v) solve the dual.
    // (vi) try to obtain a primal solution.
    
    struct bdd_solver_options {
        bdd_solver_options(int argc, char** argv);
        bdd_solver_options(const std::vector<std::string>& args);

        std::string input_file;
        std::string lp_input_as_string;
        std::string opb_input_as_string;
        ILP_input ilp;
        ILP_input::variable_order var_order = ILP_input::variable_order::input;

        size_t max_iter = 1000;
        double tolerance = 1e-6;
        double time_limit = 3600;
        enum class bdd_solver_impl { mma, mma_srmp, mma_agg, decomposition_mma, anisotropic_mma, mma_vec } bdd_solver_impl_;
        decomposition_mma_options decomposition_mma_options_;

        bool tighten = false;

        bool primal_rounding = false;
        bdd_fix_options fixing_options_;

        bool statistics = false;
        std::string export_bdd_lp;
    };

    class bdd_solver {
        public:
            bdd_solver(bdd_solver_options opt);
            bdd_solver(int argc, char** argv);
            bdd_solver(const std::vector<std::string>& args);
            //bdd_solver(bdd_solver_options&& opt);

            void solve();
            void round();
            void tighten();
            double lower_bound();
            std::vector<double> min_marginals();

        private:
            ILP_input get_ILP(const std::string& input, ILP_input::variable_order variable_order_);
            //bdd_preprocessor preprocess(ILP_input& ilp);
            bdd_storage transfer_to_bdd_storage(bdd_preprocessor& bp);
            void construct_solver(bdd_storage& bs);

            bdd_solver_options options;
            using solver_type = std::variant<bdd_mma, bdd_mma_srmp, bdd_mma_agg, decomposition_bdd_mma, bdd_mma_anisotropic, bdd_mma_vec>;
            std::optional<solver_type> solver;
            std::vector<double> costs;
            std::optional<bdd_fix> primal_heuristic;
    };

}
