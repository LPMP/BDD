#include "time_measure_util.h"
#include "cuda_utils.h"
#include "lbfgs.h"
#include <thrust/for_each.h>
#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace LPMP {

    template<typename VECTOR, typename REAL>
    lbfgs<VECTOR, REAL>::lbfgs(const size_t _num_variables, const int _history_size, 
        const double _init_lb_increase, const double _init_step_size, const double _req_rel_lb_increase, 
        const double _step_size_decrease_factor, const double _step_size_increase_factor) : 
        num_variables(_num_variables), m(_history_size), 
        init_lb_increase(_init_lb_increase), step_size(_init_step_size), required_relative_lb_increase(_req_rel_lb_increase),
        step_size_decrease_factor(_step_size_decrease_factor), step_size_increase_factor(_step_size_increase_factor)
    {
        prev_x = VECTOR(num_variables);
        prev_grad_f = VECTOR(num_variables);
        rho_inv_history = std::vector<REAL>(m);
        s_history = std::vector<VECTOR>(m);
        y_history = std::vector<VECTOR>(m);
        std::cout<<"[lbfgs] Initialized LBFGS with history size: "<<m<<"\n";
    }

    template<typename VECTOR, typename REAL>
    template<typename SOLVER>
    void lbfgs<VECTOR, REAL>::store_iterate(const SOLVER& s)
    {
        // TODO: Provide following two functions in all solvers.
        // should return thrust::device_vector fror GPU and thrust::host_vector/std::vector for CPU solvers.
        VECTOR cur_x = s.net_solver_costs();
        VECTOR cur_grad_f = s.bdds_solution_vec();
        
        assert(cur_x.size() == prev_x.size());
        assert(cur_grad_f.size() == prev_x.size());
        if (!prev_states_stored)
        {
            prev_x = cur_x;
            prev_grad_f = cur_grad_f;
            prev_states_stored = true;
        }
        else
        {
            VECTOR cur_s(cur_x.size()); // compute x_k - x_{k-1}
            thrust::transform(cur_x.begin(), cur_x.end(), prev_x.begin(), cur_s.begin(), thrust::minus<REAL>());

            // compute grad_f_k - grad_f_{k-1}, but since we have maximization problem and lbfgs updates are derived for minimization so multiply gradients by -1.
            VECTOR cur_y(cur_grad_f.size());
            thrust::transform(prev_grad_f.begin(), prev_grad_f.end(), cur_grad_f.begin(), cur_y.begin(), thrust::minus<REAL>());

            REAL rho_inv = thrust::inner_product(cur_s.begin(), cur_s.end(), cur_y.begin(), (REAL) 0.0);
            if (!initial_rho_inv_valid)
            {
                initial_rho_inv = rho_inv;
                initial_rho_inv_valid = true;
            }
            if (rho_inv / initial_rho_inv > 1e-8) // otherwise, skip the iterate as curvature condition is not strongly satisfied.
            {
                rho_inv_history[next_insertion_index] = rho_inv;
                s_history[next_insertion_index] = cur_s;
                y_history[next_insertion_index] = cur_y;
                next_insertion_index = (next_insertion_index + 1) % m;
                num_stored = min(num_stored + 1, m);
            } // when skipping estimate of hessian will become out-of-date. However removing these updates as below gives worse results than not removing.
            else
            {
                num_stored = max(num_stored - 1, 0);
            }
            prev_x = cur_x;
            prev_grad_f = cur_grad_f;
        }
    }

    template<typename VECTOR, typename REAL>
    template<typename SOLVER>
    void lbfgs<VECTOR, REAL>::iteration(SOLVER& s)
    {
        // 1. Update LBFGS states:
        this->store_iterate(s);

        // 2. Check if enough history accumulated
       if (!this->update_possible())
            return;

        // 3. Compute LBFGS update direction. This can be infeasible.
        VECTOR grad_lbfgs = this->compute_update_direction();

        // 4. Make the update direction dual feasible by making it sum to zero for all primal variables.
        s.make_dual_feasible(grad_lbfgs.begin(), grad_lbfgs.end()); //TODO: Implement for all solvers
        
        // 5. Apply the update by choosing appropriate step size:
        this->apply_update(s, grad_lbfgs);
    }

    template<typename VECTOR, typename REAL>
    template<typename SOLVER>
    void lbfgs<VECTOR, REAL>::search_step_size_and_apply(SOLVER& s, const VECTOR& update)
    {    
        const REAL lb_pre = s.lower_bound();
        auto calculate_rel_change = [&]() {
            return (s.lower_bound() - lb_pre) / (1e-9 + this->init_lb_increase);
        };
        double prev_step_size = 0.0;
        auto apply_update = [&](const REAL new_step_size) 
        {
            double net_step_size = new_step_size - prev_step_size;
            if (net_step_size != 0.0)
                s.update_dual_costs_with_step_size(update.begin(), update.end(), net_step_size); // TODO: implement for each solver.
            prev_step_size = net_step_size;
        };

        size_t num_updates = 0;
        REAL curr_rel_change = 0.0;
        REAL best_step_size = 0.0;
        REAL best_rel_improvement = 0.0;
        do
        {
            apply_update(this->step_size);
            curr_rel_change = calculate_rel_change();
            if (best_rel_improvement < curr_rel_change)
            {
                best_rel_improvement = curr_rel_change;
                best_step_size = this->step_size;
            }
            if (curr_rel_change <= 0.0)
                this->step_size *= this->step_size_decrease_factor;
            else if (curr_rel_change < required_relative_lb_increase)
                this->step_size *= this->step_size_increase_factor;

            std::cout<<"[lbfgs] relative_change: "<<curr_rel_change<<", step size: "<<this->step_size<<"\n";
            if (num_updates > 5)
            {
                if (best_rel_improvement > required_relative_lb_increase / 10.0) //TODO: Have a separate parameter?
                    apply_update(best_step_size);
                else
                {
                    apply_update(0.0);
                    this->num_unsuccessful_lbfgs_updates_ += 1;
                }
                return;
            }
            num_updates++;
        } while(curr_rel_change < required_relative_lb_increase);
        if (num_updates == 1 && this->num_unsuccessful_lbfgs_updates_ == 0)
            this->step_size *= this->step_size_increase_factor;
        this->num_unsuccessful_lbfgs_updates_ = 0;
    }


    template<typename REAL>
    struct update_q
    {
        const REAL alpha;
        const REAL* y;
        REAL* q;
        __host__ __device__ void operator()(const int idx)
        {
            q[idx] -= alpha * y[idx];
        }
    };

    template<typename REAL>
    struct update_r
    {
        const REAL alpha;
        const REAL beta;
        const REAL* s;
        REAL* r;
        __host__ __device__ void operator()(const int idx)
        {
            r[idx] += s[idx] * (alpha - beta);
        }
    };

    template<typename VECTOR, typename REAL>
    VECTOR lbfgs<VECTOR, REAL>::compute_update_direction()
    {
        assert(this->update_possible());
        VECTOR direction(num_variables);

        const int n = s_history[0].size();

        std::vector<REAL> alpha_history;
        for (int count = 0; count < num_stored; count++)
        {
            int i = next_insertion_index - count - 1;
            if (i < 0)
                i = m + i;

            assert(i >= 0 && i < m);
            assert(s_history[i].size() == n);
            const REAL alpha = thrust::inner_product(s_history[i].begin(), s_history[i].end(), direction.begin(), (REAL) 0.0) / (rho_inv_history[i]);
            
            alpha_history.push_back(alpha);
            update_q<REAL> update_q_func({alpha, thrust::raw_pointer_cast(y_history[i].data()), thrust::raw_pointer_cast(direction.data())});

            thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + n, update_q_func);
        }

        REAL last_y_norm = thrust::inner_product(y_history.back().begin(), y_history.back().end(), y_history.back().begin(), (REAL) 0.0);
        REAL initial_H_diag_multiplier = rho_inv_history.back() / (1e-8 + last_y_norm);
        // Skip line 5 in Alg.1 and fuse with line 7 for first loop itr.
        for (int count = 0; count < num_stored; count++)
        {
            int i = next_insertion_index - num_stored + count;
            if (i < 0)
                i = m + i;

            assert(i >= 0 && i < m);
            assert(y_history[i].size() == n);

            REAL current_rho = 1 / (rho_inv_history[i]);
            if (count == 0)
                current_rho *= initial_H_diag_multiplier;
            const REAL beta = current_rho * thrust::inner_product(y_history[i].begin(), y_history[i].end(), direction.begin(), (REAL) 0.0);

            update_r<REAL> update_r_func({alpha_history[num_stored - count - 1], beta, thrust::raw_pointer_cast(s_history[i].data()), thrust::raw_pointer_cast(direction.data())});

            thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + n, update_r_func);
        }
        return direction;
    }

    template<typename VECTOR, typename REAL>
    void lbfgs<VECTOR, REAL>::flush_states()
    {
        num_stored = 0;
        next_insertion_index = 0;
        prev_states_stored = false;
        initial_rho_inv = 0.0;
        initial_rho_inv_valid = false;
    }

    template<typename VECTOR, typename REAL>
    void lbfgs<VECTOR, REAL>::next_itr_without_storage()
    {
        num_stored = max(num_stored - 1, 0);
    }

    template<typename VECTOR, typename REAL>
    bool lbfgs<VECTOR, REAL>::update_possible()
    {
        if (num_stored < m)
            return false;
        return true;
    }

    template class lbfgs<thrust::host_vector<float>, float>;
    template class lbfgs<thrust::device_vector<float>, float>;
    template class lbfgs<thrust::host_vector<double>, double>;
    template class lbfgs<thrust::device_vector<double>, double>;
}