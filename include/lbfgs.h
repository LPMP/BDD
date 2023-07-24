#pragma once
#include <vector>

namespace LPMP {

template<typename VECTOR, typename REAL>
class lbfgs {
    public:
        lbfgs(const size_t _num_variables, const int _history_size, 
        const double _init_lb_increase, const double _init_step_size = 1e-6, const double _req_rel_lb_increase = 1e-6, 
        const double _step_size_decrease_factor = 0.8, const double _step_size_increase_factor = 1.1);
        
        template<typename SOLVER>
        void iteration(SOLVER& s);
       
    private:
        template<typename SOLVER>
        void store_iterate(const SOLVER& s);

        VECTOR compute_update_direction();

        template<typename SOLVER>
        void search_step_size_and_apply(SOLVER& s, const VECTOR& update);

        void flush_states(); 

        bool update_possible();

        void next_itr_without_storage();

        std::vector<VECTOR> s_history, y_history; // difference of x, grad_x f(x) resp.
        VECTOR prev_x, prev_grad_f;
        std::vector<REAL> rho_inv_history;
        const size_t num_variables;
        const int m;
        double step_size;
        const double init_lb_increase, required_relative_lb_increase, step_size_decrease_factor, step_size_increase_factor;
        int num_stored = 0;
        int next_insertion_index = 0;
        double initial_rho_inv = 0.0;

        bool prev_states_stored = false;
        bool initial_rho_inv_valid = false;

    };
}