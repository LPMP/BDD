#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <memory>

namespace LPMP {

    class bdd_mma_vec {
        public:
            enum class averaging_type { classic, vec };

            bdd_mma_vec(BDD::bdd_collection& bdd_col);
            template<typename ITERATOR>
            bdd_mma_vec(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_mma_vec(bdd_mma_vec&&);
            bdd_mma_vec& operator=(bdd_mma_vec&&);
            ~bdd_mma_vec();
            void set_cost(const double c, const size_t var);
            void set_avg_type(const averaging_type avg_type);
            double lower_bound();
            void solve(const size_t max_iter, const double tolerance, const double time_limit);
            void backward_run(); 
            two_dim_variable_array<std::array<double,2>> min_marginals();

            void tighten();
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename ITERATOR>
        bdd_mma_vec::bdd_mma_vec(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_mma_vec(bdd_col)
        {
            size_t var = 0;
            for(auto cost_it=cost_begin; cost_it!=cost_end; ++cost_it, ++var)
                set_cost(*cost_it, var);
            backward_run();
        }

};


