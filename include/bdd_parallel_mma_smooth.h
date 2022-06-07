#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <memory>

namespace LPMP {

    template<typename REAL>
        class bdd_parallel_mma_smooth {
            public:
                bdd_parallel_mma_smooth(BDD::bdd_collection& bdd_col);
                template<typename ITERATOR>
                    bdd_parallel_mma_smooth(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end);
                bdd_parallel_mma_smooth(bdd_parallel_mma_smooth&&);
                bdd_parallel_mma_smooth& operator=(bdd_parallel_mma_smooth&&);
                ~bdd_parallel_mma_smooth();
                void update_cost(const double lo_cost, const double hi_cost, const size_t var);
                template<typename COST_ITERATOR>
                    void update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end);
                void add_to_constant(const double c);
                double lower_bound();
                void iteration();
                void backward_run(); 
                two_dim_variable_array<std::array<double,2>> min_marginals();
                void fix_variable(const size_t var, const bool value);
                void set_smoothing(const double smoothing);
            private:

                class impl;
                std::unique_ptr<impl> pimpl;
        };

    template<typename REAL>
        template<typename ITERATOR>
        bdd_parallel_mma_smooth<REAL>::bdd_parallel_mma_smooth(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_parallel_mma_smooth(bdd_col)
        {
            size_t var = 0;
            update_costs(cost_begin, cost_begin, cost_begin, cost_end);
            backward_run();
        }

}

