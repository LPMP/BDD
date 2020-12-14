#pragma once

#include "bdd_storage.h"
#include <memory>

namespace LPMP {

    class bdd_fix {
        public:
            bdd_fix(bdd_storage& stor);
            bdd_fix(bdd_storage& stor, const int var_order, const int var_value);
            bdd_fix(bdd_fix&&);
            bdd_fix& operator=(bdd_fix&&);
            ~bdd_fix();
            void set_var_order(const int var_value);
            void set_var_value(const int var_order);
            bool round(const std::vector<double> total_min_marginals);
            std::vector<char> primal_solution();
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

};
