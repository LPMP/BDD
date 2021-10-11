#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <memory>

namespace LPMP {

    template<typename REAL>
    class bdd_mma_vec {
        public:
            enum class averaging_type { classic, vec };

            bdd_mma_vec(BDD::bdd_collection& bdd_col);
            template<typename ITERATOR>
            bdd_mma_vec(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_mma_vec(bdd_mma_vec&&);
            bdd_mma_vec& operator=(bdd_mma_vec&&);
            ~bdd_mma_vec();
            void update_cost(const double c, const size_t var);
            template<typename COST_ITERATOR>
                void update_costs(COST_ITERATOR cost_begin, COST_ITERATOR cost_end);
            void set_avg_type(const averaging_type avg_type);
            double lower_bound();
            void iteration();
            void backward_run(); 
            two_dim_variable_array<std::array<double,2>> min_marginals();
            void fix_variable(const size_t var, const bool value);

            void tighten();
        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename REAL>
    template<typename ITERATOR>
        bdd_mma_vec<REAL>::bdd_mma_vec(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_mma_vec(bdd_col)
        {
            size_t var = 0;
            update_costs(cost_begin, cost_end);
            backward_run();
        }

    template<typename REAL>
        template<typename COST_ITERATOR>
        void bdd_mma_vec<REAL>::update_costs(COST_ITERATOR cost_begin, COST_ITERATOR cost_end)
        {
            size_t var = 0;
            for(auto cost_it=cost_begin; cost_it!=cost_end; ++cost_it, ++var)
                update_cost(*cost_it, var);
        }

};


