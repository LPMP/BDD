#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <memory>

namespace LPMP {

    class bdd_cuda {
        public:
            bdd_cuda(BDD::bdd_collection& bdd_col);
            template<typename ITERATOR>
            bdd_cuda(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_cuda(bdd_cuda&&);
            bdd_cuda& operator=(bdd_cuda&&);
            ~bdd_cuda();
            void set_cost(const double c, const size_t var);
            template<typename ITERATOR>
                void set_costs(ITERATOR cost_begin, ITERATOR cost_end);
            double lower_bound();
            two_dim_variable_array<std::array<double,2>> min_marginals();
            void iteration();
            void backward_run(); 
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename ITERATOR>
        bdd_cuda::bdd_cuda(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_cuda(bdd_col)
        {
            set_costs(cost_begin, cost_end);
        }

    template<typename ITERATOR>
        void bdd_cuda::set_costs(ITERATOR cost_begin, ITERATOR cost_end)
        {
            // TODO: not fast!
            auto cost_it = cost_begin;
            for(size_t i=0; i<std::distance(cost_begin, cost_end); ++i, ++cost_it)
                set_cost(*cost_it, i); 

        }
};
