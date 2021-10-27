#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <memory>

namespace LPMP {

    template<typename REAL>
    class bdd_cuda {
        public:
            bdd_cuda(BDD::bdd_collection& bdd_col);
            template<typename ITERATOR>
            bdd_cuda(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_cuda(bdd_cuda&&);
            bdd_cuda& operator=(bdd_cuda&&);
            ~bdd_cuda();
            //void set_cost(const double c, const size_t var);
            template<typename ITERATOR>
                void update_costs(ITERATOR cost_lo_begin, ITERATOR cost_lo_end, ITERATOR cost_hi_begin, ITERATOR cost_hi_end);
            double lower_bound();
            two_dim_variable_array<std::array<double,2>> min_marginals();
            void iteration();
            void backward_run(); 

            std::vector<char> incremental_mm_agreement_rounding(const double init_delta, const double delta_grwoth_rate, const int num_itr_lb);
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename REAL>
        template<typename ITERATOR>
        bdd_cuda<REAL>::bdd_cuda(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_cuda(bdd_col)
        {
            update_costs(cost_begin, cost_begin, cost_begin, cost_end);
        }

};
