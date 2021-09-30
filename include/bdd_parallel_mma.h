#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <memory>

namespace LPMP {

    class bdd_parallel_mma {
        public:
            bdd_parallel_mma(BDD::bdd_collection& bdd_col);
            template<typename ITERATOR>
            bdd_parallel_mma(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end);
            bdd_parallel_mma(bdd_parallel_mma&&);
            bdd_parallel_mma& operator=(bdd_parallel_mma&&);
            ~bdd_parallel_mma();

            template<typename ITERATOR>
                void set_costs(ITERATOR cost_begin, ITERATOR cost_end);

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

    template<typename ITERATOR>
        bdd_parallel_mma::bdd_parallel_mma(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end)
        : bdd_parallel_mma(bdd_col)
        {
            set_costs(cost_begin, cost_end);
            backward_run();
        }

};



