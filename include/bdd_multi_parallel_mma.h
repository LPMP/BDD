#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <array>
#include <memory>

namespace LPMP {

    template<typename REAL>
    class bdd_multi_parallel_mma {
        public:
            bdd_multi_parallel_mma(BDD::bdd_collection& bdd_col);
            template<typename ITERATOR>
                bdd_multi_parallel_mma(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_multi_parallel_mma(bdd_multi_parallel_mma&&);
            bdd_multi_parallel_mma& operator=(bdd_multi_parallel_mma&&);
            ~bdd_multi_parallel_mma();
            size_t nr_variables() const;
            size_t nr_bdds(const size_t var) const;
            void update_costs(const two_dim_variable_array<std::array<double,2>>& delta);
            template<typename COST_ITERATOR>
                void update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end);
            void add_to_constant(const double c);
            double lower_bound();
            void iteration();
            two_dim_variable_array<std::array<double,2>> min_marginals();
            template<typename ITERATOR>
                two_dim_variable_array<char> bdd_feasibility(ITERATOR sol_begin, ITERATOR sol_end);
            void fix_variable(const size_t var, const bool value);

        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename REAL>
        template<typename ITERATOR>
        bdd_multi_parallel_mma<REAL>::bdd_multi_parallel_mma(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_multi_parallel_mma(bdd_col)
        {
            update_costs(cost_begin, cost_begin, cost_begin, cost_end);
        }

}
