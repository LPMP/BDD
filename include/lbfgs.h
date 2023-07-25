#pragma once
#include "bdd_collection/bdd_collection.h"
#include <vector>

namespace LPMP {

template<class SOLVER, typename VECTOR, typename REAL>
class lbfgs : public SOLVER
{
    public:
        lbfgs(const BDD::bdd_collection& bdd_col, const int _history_size, 
        const double _init_step_size = 1e-6, const double _req_rel_lb_increase = 1e-6, 
        const double _step_size_decrease_factor = 0.8, const double _step_size_increase_factor = 1.1);

        void iteration();

        void update_costs(const VECTOR& cost_delta_0, const VECTOR& cost_delta_1);

    private:
        void store_iterate();

        VECTOR compute_update_direction();

        void search_step_size_and_apply(const VECTOR& update);

        void flush_lbfgs_states(); 

        bool update_possible();

        void next_itr_without_storage();

        std::vector<VECTOR> s_history, y_history; // difference of x, grad_x f(x) resp.
        VECTOR prev_x, prev_grad_f;
        std::vector<REAL> rho_inv_history;
        const int m;
        double step_size;
        double init_lb_increase;
        bool init_lb_valid = false;
        const double required_relative_lb_increase, step_size_decrease_factor, step_size_increase_factor;
        int num_stored = 0;
        int num_unsuccessful_lbfgs_updates_ = 0;
        int next_insertion_index = 0;
        double initial_rho_inv = 0.0;

        bool prev_states_stored = false;
        bool initial_rho_inv_valid = false;
    };
}