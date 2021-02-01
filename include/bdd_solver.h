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

    class bdd_solver {
        public:
            bdd_solver(int argc, char** argv);
            bdd_solver(const std::vector<std::string>& args);

            void solve();
            void round();
            double lower_bound();

        private:
            ILP_input get_ILP(const std::string& input, ILP_input::variable_order variable_order_);
            //bdd_preprocessor preprocess(ILP_input& ilp);
            bdd_storage transfer_to_bdd_storage(bdd_preprocessor& bp);
            void construct_solver(bdd_storage& bs);


            using solver_type = std::variant<bdd_mma, bdd_mma_srmp, bdd_mma_agg, decomposition_bdd_mma, bdd_mma_anisotropic, bdd_mma_vec>;
            std::optional<solver_type> solver;
            size_t max_iter = 10000;
            double time_limit = 3600;
            double tolerance = 1e-06;
            std::vector<double> costs;
            std::optional<bdd_fix> primal_heuristic;
    };

}
