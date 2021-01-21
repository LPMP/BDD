#pragma once

#include "bdd_storage.h"
#include <memory>

namespace LPMP {

    class bdd_mma_vec {
        public:
            enum class averaging_type { classic, vec };

            bdd_mma_vec(bdd_storage& stor);
            template<typename ITERATOR>
            bdd_mma_vec(bdd_storage& stor, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_mma_vec(bdd_mma_vec&&);
            bdd_mma_vec& operator=(bdd_mma_vec&&);
            ~bdd_mma_vec();
            void set_cost(const double c, const size_t var);
            void set_avg_type(const averaging_type avg_type);
            double lower_bound();
            std::vector<double> total_min_marginals();
            void solve(const size_t max_iter, const double tolerance);
            void backward_run(); 
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename ITERATOR>
        bdd_mma_vec::bdd_mma_vec(bdd_storage& stor, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_mma_vec(stor)
        {
            assert(std::distance(cost_begin, cost_end) <= stor.nr_variables());
            size_t var = 0;
            for(auto cost_it=cost_begin; cost_it!=cost_end; ++cost_it)
                set_cost(*cost_it, var++);
            backward_run();
        }

};


