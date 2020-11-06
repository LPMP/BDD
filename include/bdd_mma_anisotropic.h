#pragma once

#include "bdd_storage.h"
#include <memory>

namespace LPMP {

    class bdd_mma_anisotropic {
        public:
            bdd_mma_anisotropic(bdd_storage& stor);
            template<typename ITERATOR>
            bdd_mma_anisotropic(bdd_storage& stor, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_mma_anisotropic(bdd_mma_anisotropic&&);
            bdd_mma_anisotropic& operator=(bdd_mma_anisotropic&&);
            ~bdd_mma_anisotropic();
            void set_cost(const double c, const size_t var);
            double lower_bound();
            void iteration(); 
            void solve(const size_t max_iter);
            void backward_run(); 
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename ITERATOR>
        bdd_mma_anisotropic::bdd_mma_anisotropic(bdd_storage& stor, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_mma_anisotropic(stor)
        {
            assert(std::distance(cost_begin, cost_end) <= stor.nr_variables());
            size_t var = 0;
            for(auto cost_it=cost_begin; cost_it!=cost_end; ++cost_it)
                set_cost(*cost_it, var++);
            backward_run();
        }

};

