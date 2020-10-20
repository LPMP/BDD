#pragma once

#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "bdd_storage.h"
#include "bdd_solver_interface.h"

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

        private:
            ILP_input get_ILP(const std::string& input, ILP_input::variable_order variable_order_);
            //bdd_preprocessor preprocess(ILP_input& ilp);
            bdd_storage transfer_to_bdd_storage(bdd_preprocessor& bp);
            void construct_solver(bdd_storage& bs);
            void solve();
            void round();

            //ILP_input ilp_input_;
            //bdd_preprocessor bdd_preprocessor_;
            //bdd_storage bdd_storage_;
            bdd_solver_interface* bdd_solver_;

    };

}
