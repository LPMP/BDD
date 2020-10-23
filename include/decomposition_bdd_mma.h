#pragma once

#include <memory>
#include "bdd_mma.h"

namespace LPMP {

    class decomposition_bdd_mma {
        public:

            struct options {
                bdd_mma::averaging_type avg_type;
                size_t nr_threads; 
                double parallel_message_passing_weight;
            };

            decomposition_bdd_mma(bdd_storage& bdd_storage_, options opt);
            template<typename ITERATOR>
                decomposition_bdd_mma(bdd_storage& stor, ITERATOR cost_begin, ITERATOR cost_end, options opt);
            decomposition_bdd_mma(decomposition_bdd_mma&&);

            decomposition_bdd_mma& operator=(decomposition_bdd_mma&&);

            ~decomposition_bdd_mma();
            void set_cost(const double c, const size_t var);
            void backward_run();
            void iteration();
            double lower_bound();

        private:
            struct impl;
            std::unique_ptr<impl> pimpl;

    };

    template<typename ITERATOR>
        decomposition_bdd_mma::decomposition_bdd_mma(bdd_storage& stor, ITERATOR cost_begin, ITERATOR cost_end, decomposition_bdd_mma::options opt)
        : decomposition_bdd_mma(stor, opt)
        {
            assert(std::distance(cost_begin, cost_end) <= stor.nr_variables());
            size_t var = 0;
            for(auto cost_it=cost_begin; cost_it!=cost_end; ++cost_it)
                set_cost(*cost_it, var++);
            backward_run();
        }

}
