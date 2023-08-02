#pragma once

#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <memory>

namespace LPMP {

    template<typename REAL>
    class bdd_lbfgs_parallel_mma {
        public:
            bdd_lbfgs_parallel_mma(BDD::bdd_collection& bdd_col, const int _history_size, 
                const double _init_step_size = 1e-6, const double _req_rel_lb_increase = 1e-6, 
                const double _step_size_decrease_factor = 0.8, const double _step_size_increase_factor = 1.1);
            template<typename ITERATOR>
            bdd_lbfgs_parallel_mma(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end,
                const int _history_size, const double _init_step_size = 1e-6, const double _req_rel_lb_increase = 1e-6, 
                const double _step_size_decrease_factor = 0.8, const double _step_size_increase_factor = 1.1);
            bdd_lbfgs_parallel_mma(bdd_lbfgs_parallel_mma&&);
            bdd_lbfgs_parallel_mma& operator=(bdd_lbfgs_parallel_mma&&);
            ~bdd_lbfgs_parallel_mma();
            template<typename ITERATOR>
                void update_costs(ITERATOR cost_lo_begin, ITERATOR cost_lo_end, ITERATOR cost_hi_begin, ITERATOR cost_hi_end);
            double lower_bound();
            two_dim_variable_array<std::array<double,2>> min_marginals();
            void iteration();
            void backward_run(); 

            std::vector<char> incremental_mm_agreement_rounding(const double init_delta, const double delta_grwoth_rate, const int num_itr_lb, const int num_rounds = 500);

        private:

            class impl;
            std::unique_ptr<impl> pimpl;
    };

    template<typename REAL>
        template<typename ITERATOR>
        bdd_lbfgs_parallel_mma<REAL>::bdd_lbfgs_parallel_mma(BDD::bdd_collection& bdd_col, ITERATOR cost_begin, ITERATOR cost_end,
            const int _history_size, const double _init_step_size, const double _req_rel_lb_increase, 
            const double _step_size_decrease_factor, const double _step_size_increase_factor)
        : bdd_lbfgs_parallel_mma(bdd_col, _history_size,
            _init_step_size, _req_rel_lb_increase,
            _step_size_decrease_factor, _step_size_increase_factor)
        {
            update_costs(cost_begin, cost_begin, cost_begin, cost_end);
        }

};
