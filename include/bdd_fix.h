#pragma once

#include "bdd_storage.h"
#include <memory>

namespace LPMP {

    struct bdd_fix_options {
        enum variable_order { marginals_absolute = 0, marginals_up = 1, marginals_down = 2, marginals_reduction = 3};
        enum variable_value { marginal = 0, reduction = 1, one = 2, zero = 3};
        variable_order var_order;
        variable_value var_value;
    };

    class bdd_fix {
        public:
            bdd_fix(bdd_storage& stor);
            bdd_fix(bdd_storage& stor, bdd_fix_options opts);
            bdd_fix(bdd_fix&&);
            bdd_fix& operator=(bdd_fix&&);
            ~bdd_fix();
            bool round(const std::vector<double> total_min_marginals);
            std::vector<char> primal_solution();
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

};
