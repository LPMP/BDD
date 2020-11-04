#pragma once

#include "bdd_storage.h"
#include <memory>

namespace LPMP {

    class bdd_mma {
        public:
            enum class averaging_type { classic, srmp };

            bdd_mma(bdd_storage& stor);
            template<typename ITERATOR>
            bdd_mma(bdd_storage& stor, ITERATOR cost_begin, ITERATOR cost_end, averaging_type avg_type);
            bdd_mma(bdd_mma&&);
            bdd_mma& operator=(bdd_mma&&);
            ~bdd_mma();
            void set_cost(const double c, const size_t var);
            void set_avg_type(const averaging_type avg_type);
            double lower_bound();
            void iteration(); 
            void solve(const size_t max_iter);
            void backward_run(); 
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename ITERATOR>
        bdd_mma::bdd_mma(bdd_storage& stor, ITERATOR cost_begin, ITERATOR cost_end, bdd_mma::averaging_type avg_type)
        : bdd_mma(stor)
        {
            assert(std::distance(cost_begin, cost_end) <= stor.nr_variables());
            size_t var = 0;
            for(auto cost_it=cost_begin; cost_it!=cost_end; ++cost_it)
                set_cost(*cost_it, var++);
            set_avg_type(avg_type);
            backward_run();
        }

};
