#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <memory>

namespace LPMP {

    template<typename REAL>
    class bdd_parallel_mma {
        public:
            bdd_parallel_mma(BDD::bdd_collection& bdd_col);
            template<typename ITERATOR>
            bdd_parallel_mma(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_parallel_mma(bdd_parallel_mma&&);
            bdd_parallel_mma& operator=(bdd_parallel_mma&&);
            ~bdd_parallel_mma();

            template<typename ITERATOR>
                void update_costs(ITERATOR cost_lo_begin, ITERATOR cost_lo_end, ITERATOR cost_hi_begin, ITERATOR cost_hi_end);
            void add_to_constant(const double c);

            size_t nr_variables() const;
            size_t nr_bdds(const size_t var) const;
            double lower_bound();
            void iteration();
            void distribute_delta();
            void backward_run(); 
            two_dim_variable_array<std::array<double,2>> min_marginals();
            void fix_variable(const size_t var, const bool value);
            template<typename ITERATOR>
                void fix_variables(ITERATOR zero_fixations_begin, ITERATOR zero_fixations_end, ITERATOR one_fixations_begin, ITERATOR one_fixations_end);

            void tighten();
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename REAL>
    template<typename ITERATOR>
        bdd_parallel_mma<REAL>::bdd_parallel_mma(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_parallel_mma(bdd_col)
        {
            update_costs(cost_begin, cost_begin, cost_begin, cost_end);
            backward_run();
        }
};






