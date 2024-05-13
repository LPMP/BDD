#pragma once

#include "bdd_solver/lbfgs.h"

namespace LPMP {

    template <class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::lbfgs(
        const BDD::bdd_collection &bdd_col, const std::vector<double> &costs_hi,
        const int _history_size,
        const double _init_step_size, const double _req_rel_lb_increase,
        const double _step_size_decrease_factor, const double _step_size_increase_factor)
        : SOLVER(bdd_col, costs_hi),
          m(_history_size),
          step_size(_init_step_size), required_relative_lb_increase(_req_rel_lb_increase),
          step_size_decrease_factor(_step_size_decrease_factor), step_size_increase_factor(_step_size_increase_factor)
    {
        bdd_log << "[lbfgs] Initialized LBFGS with"
                << "\n[lbfgs]\t\thistory size: " << m
                << "\n[lbfgs]\t\tinitial step size " << step_size
                << "\n[lbfgs]\t\trequired relative lb increase " << required_relative_lb_increase
                << "\n[lbfgs]\t\tstep size decrease factor " << step_size_decrease_factor
                << "\n[lbfgs]\t\tstep size increase factor " << step_size_increase_factor
                << "\n[lbfgs]\t\thistory size" << m << "\n";

        assert(step_size > 0.0);
        assert(step_size_decrease_factor > 0.0 && step_size_decrease_factor < 1.0);
        assert(step_size_increase_factor > 1.0);
        assert(required_relative_lb_increase > 0.0);
        assert(m > 1);

        prev_x = VECTOR(this->nr_layers());
        prev_grad_f = INT_VECTOR(this->nr_layers());
    }

    template <class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::lbfgs(
        const BDD::bdd_collection &bdd_col,
        const int _history_size,
        const double _init_step_size, const double _req_rel_lb_increase,
        const double _step_size_decrease_factor, const double _step_size_increase_factor)
        : lbfgs(bdd_col, std::vector<double>{}, _history_size, _init_step_size, _req_rel_lb_increase, _step_size_decrease_factor, _step_size_increase_factor)
    {}

    template <class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::store_iterate(const INT_VECTOR& cur_grad_f)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

        const VECTOR cur_x = this->net_solver_costs();
        
        assert(cur_x.size() == prev_x.size());
        assert(cur_grad_f.size() == prev_x.size());
        if (!prev_states_stored)
        {
            prev_x = cur_x;
            if constexpr(CUDA_SOLVER)
            {
#ifdef CUDACC
                thrust::copy(cur_grad_f.begin(), cur_grad_f.end(), prev_grad_f.begin());
#endif
            }
            else
            {
                std::copy(cur_grad_f.begin(), cur_grad_f.end(), prev_grad_f.begin());
            }
            prev_states_stored = true;
        }
        else
        {
            VECTOR cur_s(cur_x.size()); // compute x_k - x_{k-1}
            if constexpr(CUDA_SOLVER)
            {
#ifdef CUDACC
            thrust::transform(cur_x.begin(), cur_x.end(), prev_x.begin(), cur_s.begin(), _1 - _2);
#endif
            }
            else
            {
            std::transform(cur_x.begin(), cur_x.end(), prev_x.begin(), cur_s.begin(), std::minus<REAL>());
            }

            // compute grad_f_k - grad_f_{k-1}, but since we have maximization problem and lbfgs updates are derived for minimization so multiply gradients by -1.
            INT_VECTOR cur_y(cur_grad_f.size());

            REAL rho_inv;
            if constexpr (CUDA_SOLVER)
            {
#ifdef CUDACC
                thrust::transform(prev_grad_f.begin(), prev_grad_f.end(), cur_grad_f.begin(), cur_y.begin(), _1 - _2);
                rho_inv = thrust::inner_product(cur_s.begin(), cur_s.end(), cur_y.begin(), (REAL)0.0);
#endif
            }
            else
            {
                std::transform(prev_grad_f.begin(), prev_grad_f.end(), cur_grad_f.begin(), cur_y.begin(), std::minus<REAL>());
                rho_inv = std::inner_product(cur_s.begin(), cur_s.end(), cur_y.begin(), (REAL)0.0);
            }

            if (!initial_rho_inv_valid)
            {
                initial_rho_inv = rho_inv;
                initial_rho_inv_valid = true;
            }
            //if (rho_inv / initial_rho_inv > 1e-8) // otherwise, skip the iterate as curvature condition is not strongly satisfied.
            if (rho_inv > 1e-8) // otherwise, skip the iterate as curvature condition is not strongly satisfied.
            {
                history.push_back({});
                history.back().s = cur_s;
                history.back().y = cur_y;
                history.back().rho_inv = rho_inv;

                if(history.size() > m)
                {
                    history.pop_front();
                    assert(history.size() == m);
                }
            }
            else
            {
                prev_states_stored = false;
            }
            prev_x = cur_x;
            if constexpr(CUDA_SOLVER)
            {
#ifdef CUDACC
                thrust::copy(cur_grad_f.begin(), cur_grad_f.end(), prev_grad_f.begin());
#endif
            }
            else
            {
                std::copy(cur_grad_f.begin(), cur_grad_f.end(), prev_grad_f.begin());
            }
        }
    }

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::iteration()
    {
        if (lb_history.empty())
            lb_history.push_back(this->lower_bound());

        // Update LBFGS states:
        const auto cur_grad_f = this->bdds_solution_vec();
        this->store_iterate(cur_grad_f);

        // Check if enough history accumulated
        if (this->choose_solver() == solver_type::lbfgs)
        {
            lbfgs_iteration(cur_grad_f); 
            //mma_iteration();
        }
        else
            mma_iteration();

        lb_history.push_back(this->lower_bound());
    }

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::search_step_size_and_apply(const VECTOR& update)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        const REAL lb_pre = this->lower_bound();
    
        auto calculate_rel_change = [&]() {
            assert(lb_history.size() >= m);
            const double cur_lb_increase = this->lower_bound() - lb_pre;
            const double past_lb_increase = *(lb_history.rbegin()+m-2) - *(lb_history.rbegin()+m-1);
            //assert(cur_lb_increase >= 0.0);
            assert(past_lb_increase >= 0.0);
            const double ratio = cur_lb_increase / (1e-9 + past_lb_increase);
            // bdd_log << "[lbfgs] cur lb increase = " << cur_lb_increase << ", past lb increase = " << past_lb_increase << ", cur/past lb increase = " << ratio << "\n";
            return ratio;
            //return (this->lower_bound() - lb_pre) / (1e-9 + this->init_lb_increase);
        };

        double prev_step_size = 0.0;
        auto apply_update = [&](const REAL new_step_size) 
        {
            double net_step_size = new_step_size - prev_step_size;
            if (net_step_size != 0.0)
                this->gradient_step(update, net_step_size); // TODO: implement for each solver.
            prev_step_size = new_step_size;
        };

        size_t num_updates = 0;
        REAL curr_rel_change = 0.0;
        REAL best_step_size = 0.0;
        REAL best_rel_improvement = 0.0;
        do
        {
            apply_update(this->step_size);
            curr_rel_change = calculate_rel_change();
            // bdd_log << "[lbfgs] perform update step with step size " << this->step_size << ", curr_rel_change: "<<curr_rel_change<<"\n";
            if (best_rel_improvement < curr_rel_change)
            {
                best_rel_improvement = curr_rel_change;
                best_step_size = this->step_size;
            }

            if (curr_rel_change <= 0.0)
                this->step_size *= this->step_size_decrease_factor;
            else if (curr_rel_change < required_relative_lb_increase)
                this->step_size *= this->step_size_increase_factor;

            if (num_updates > 5)
            {
                if (best_rel_improvement > required_relative_lb_increase / 10.0) //TODO: Have a separate parameter?
                    apply_update(best_step_size);
                else
                {
                    bdd_log<<"[lbfgs] step size selection unsuccessful.\n";
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

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    VECTOR lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::compute_update_direction(const INT_VECTOR& cur_grad_f)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

        assert(this->lbfgs_update_possible());

        VECTOR direction(cur_grad_f.size());
        if constexpr(CUDA_SOLVER)
        {
#ifdef CUDACC
            thrust::copy(cur_grad_f.begin(), cur_grad_f.end(), direction.begin());
#endif
        }
        else
        {
            std::copy(cur_grad_f.begin(), cur_grad_f.end(), direction.begin());
        }

        assert(history.size() > 0);
        const size_t n = history.back().s.size();

        std::vector<REAL> alpha_history;
        for (int i = history.size()-1; i >= 0; i--)
        {
            REAL alpha;
            if constexpr (CUDA_SOLVER)
            {
#ifdef CUDACC
                REAL alpha = thrust::inner_product(history[i].s.begin(), history[i].s.end(), direction.begin(), (REAL)0.0) / (history[i].rho_inv);
#endif
            }
            else
            {
                REAL alpha = std::inner_product(history[i].s.begin(), history[i].s.end(), direction.begin(), (REAL)0.0) / (history[i].rho_inv);
            }

            alpha_history.push_back(alpha);

            if constexpr (CUDA_SOLVER)
            {
#ifdef CUDACC
                thrust::transform(history[i].y.begin(), history[i].y.end(), direction.begin(), direction.begin(), _2 - alpha * _1);
#endif
            }
            else
            {
                for (size_t j = 0; j < n; ++j)
                    direction[j] -= alpha * history[i].y[j];
            }
        }

        std::reverse(alpha_history.begin(), alpha_history.end());
        
        REAL last_y_norm;
        if constexpr (CUDA_SOLVER)
        {
#ifdef CUDACC
            last_y_norm = thrust::inner_product(history.back().y.begin(), history.back().y.end(), history.back().y.begin(), (REAL)0.0);
#endif
        }
        else
        {
            last_y_norm = std::inner_product(history.back().y.begin(), history.back().y.end(), history.back().y.begin(), (REAL)0.0);
        }
        REAL initial_H_diag_multiplier = history.back().rho_inv / (1e-8 + last_y_norm);
        // Skip line 5 in Alg.1 and fuse with line 7 for first loop itr.
        //for (int count = 0; count < history.size(); count++)
        for (int i = 0; i < history.size(); i++)
        {
            assert(history[i].y.size() == n);

            REAL current_rho = 1.0 / (history[i].rho_inv);
            if (i == 0)
                current_rho *= initial_H_diag_multiplier;
                if constexpr (CUDA_SOLVER)
                {
#ifdef CUDACC
                    const REAL beta = current_rho * thrust::inner_product(history[i].y.begin(), history[i].y.end(), direction.begin(), (REAL)0.0);
                    thrust::transform(history[i].s.begin(), history[i].s.end(), direction.begin(), direction.begin(), _2 + (alpha_history[i] - beta) * _1);
#endif
                }
                else
                {
                    const REAL beta = current_rho * std::inner_product(history[i].y.begin(), history[i].y.end(), direction.begin(), (REAL)0.0);
                    for (size_t j = 0; j < n; ++j)
                        direction[j] += (alpha_history[i] - beta) * history[i].s[j];
                }
        }
        return direction;
    }

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::flush_lbfgs_states()
    {
        num_unsuccessful_lbfgs_updates_ = 0;
        history.clear();
        prev_states_stored = false;
        initial_rho_inv = 0.0;
        initial_rho_inv_valid = false;
        //init_lb_valid = false;
    }

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::next_itr_without_storage()
    {
        history.pop_front();
    }

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    bool lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::lbfgs_update_possible() const
    {
        if (history.size() < m || num_unsuccessful_lbfgs_updates_ > 5) // || !init_lb_valid)
            return false;
        return true;
    }

    template <class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::update_costs(const std::vector<REAL> &cost_lo, const std::vector<REAL> &cost_hi)
    {
        flush_lbfgs_states();
        static_cast<SOLVER*>(this)->update_costs(cost_lo, cost_hi);
    }

//    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
//    template<typename ITERATOR>
//    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::update_costs(
//        ITERATOR cost_delta_0_begin, ITERATOR cost_delta_0_end,
//        ITERATOR cost_delta_1_begin, ITERATOR cost_delta_1_end)
//    {
//        flush_lbfgs_states();
//        static_cast<SOLVER*>(this)->update_costs(cost_delta_0_begin, cost_delta_0_end, cost_delta_1_begin, cost_delta_1_end);
//    }

#ifdef WITH_CUDA
    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::update_costs(const thrust::device_vector<REAL>& cost_0, const thrust::device_vector<REAL>& cost_1)
    {
        flush_lbfgs_states();
        if constexpr (CUDA_SOLVER)
            static_cast<SOLVER *>(this)->update_costs(cost_0, cost_1);
        else
            assert(false);
    }
#endif

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::mma_iteration()
    {
        const double lb_before = this->lower_bound();
        const auto pre = std::chrono::steady_clock::now();
        static_cast<SOLVER *>(this)->iteration();
        const auto post = std::chrono::steady_clock::now();
        const double lb_after = this->lower_bound();
        mma_lb_increase_per_time = (lb_after - lb_before) / std::chrono::duration<double>(post - pre).count();
        // bdd_log << "[lbfgs] mma lb increase over time = " << mma_lb_increase_per_time << "\n";
        mma_iterations++;
    }

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    void lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::lbfgs_iteration(const INT_VECTOR& cur_grad_f)
    {
        const double lb_before = this->lower_bound();
        const auto pre = std::chrono::steady_clock::now();

        // Compute LBFGS update direction. This can be infeasible.
        auto grad_lbfgs = this->compute_update_direction(cur_grad_f);

        // Make the update direction dual feasible by making it sum to zero for all primal variables.
        this->make_dual_feasible(grad_lbfgs); // TODO: Implement for all solvers

        // Apply the update by choosing appropriate step size:
        this->search_step_size_and_apply(grad_lbfgs);
        static_cast<SOLVER*>(this)->iteration();

        const auto post = std::chrono::steady_clock::now();
        const double lb_after = this->lower_bound();
        assert(lb_after >= lb_before - 1e-6);
        lbfgs_lb_increase_per_time = (lb_after - lb_before) / std::chrono::duration<double>(post - pre).count();
        // bdd_log << "[lbfgs] lbfgs pre lb = " << lb_before << ", after lb = " << lb_after << "\n";
        // bdd_log << "[lbfgs] lbfgs lb increase over time = " << lbfgs_lb_increase_per_time << "\n";
        lbfgs_iterations++;
    }

    template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
    typename lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::solver_type lbfgs<SOLVER, VECTOR, REAL, INT_VECTOR, CUDA_SOLVER>::choose_solver() const
    {
        if (!lbfgs_update_possible())
        {
            // bdd_log << "[lbfgs] Do mma iterations for collecting states\n";
            return solver_type::mma;
        }
        return solver_type::lbfgs;
    }

}