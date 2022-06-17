#pragma once

#include "bdd_cuda_parallel_mma.h"

namespace LPMP {

    template<typename REAL>
    class solver_state_cache
    {
        public:
            solver_state_cache(const int num_caches, const int num_iterations, const int num_layers)
            {
                num_iterations_ = num_iterations;
                num_layers_ = num_layers; 
                if (num_caches == 0 || num_iterations == 0)
                    return;
                cache_interval_ = max(ceil((float) num_iterations / num_caches), 1.0);
                max_cached_iteration_ = min(cache_interval_ * (num_caches - 1), num_iterations - 1);
                num_caches_ = 1 + max_cached_iteration_ / cache_interval_;
                
                if (num_caches_ == 0)
                    return;

                lo_costs_cache_ = std::vector<std::vector<REAL>>(num_caches_);
                hi_costs_cache_ = std::vector<std::vector<REAL>>(num_caches_);
                def_mm_cache_ = std::vector<std::vector<REAL>>(num_caches_);
            }
            void check_and_set_cache(const int itr, 
                                    const thrust::device_ptr<const REAL> lo_costs_ptr,
                                    const thrust::device_ptr<const REAL> hi_costs_ptr,
                                    const thrust::device_ptr<const REAL> def_mm_ptr);
            int check_and_get_cache(const int itr,
                                    thrust::device_ptr<REAL> lo_costs_ptr,
                                    thrust::device_ptr<REAL> hi_costs_ptr,
                                    thrust::device_ptr<REAL> def_mm_ptr);
            int max_cached_iteration() const {return max_cached_iteration_; }
        private:
            int num_caches_, num_iterations_, num_layers_;
            int max_cached_iteration_;
            int cache_interval_ = -1;
            std::vector<std::vector<REAL>> lo_costs_cache_, hi_costs_cache_, def_mm_cache_;
    };

    template<typename REAL>
    class bdd_cuda_learned_mma : public bdd_cuda_parallel_mma<REAL> {
        public:

            bdd_cuda_learned_mma() {}
            bdd_cuda_learned_mma(const BDD::bdd_collection& bdd_col);

            // dist_weights points to disribution weights and mm_diff_out_ptr will contain deferred min-marginal differences.
            // The size of underlying arrays should be corresponding to the size of hi_cost_ i.e. = this->nr_layers(). 
            // Returns the number of iterations solver actually due to improvement_slope > 0.
            // Uses omega_vec instead of omega_scalar if given. 
            int iterations(const thrust::device_ptr<const REAL> dist_weights, 
                            const int num_itr, 
                            const REAL omega_scalar, // will not be used if omega_vec != nullptr
                            const double improvement_slope = 1e-6,
                            thrust::device_ptr<REAL> sol_avg = nullptr,
                            thrust::device_ptr<REAL> lb_first_diff_avg = nullptr,
                            thrust::device_ptr<REAL> lb_second_diff_avg = nullptr,
                            const int compute_history_for_itr = 0,
                            const REAL history_avg_beta = 0.9,
                            const thrust::device_ptr<const REAL> omega_vec = nullptr);

            // Assumes that deffered_mm_diff_ contains the mm's containing the values before iterations() was called.
            void grad_iterations(
                const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost, Outputs in-place to compute grad. lo_cost before iterations.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost, Outputs in-place to compute grad. hi_cost before iterations.
                thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff., Outputs in-place to compute grad. w.r.t deferred min-marginals used in iterations.
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).
                thrust::device_ptr<REAL> grad_omega,    // Output: contains grad w.r.t omega. Should be of size 1 if omega_vec is nullptr otherwise size of omega_vec.
                const REAL omega_scalar, // will not be used if omega_vec != nullptr
                const int track_grad_after_itr,      // First runs the solver for track_grad_after_itr many iterations without tracking gradients and then backpropagates through only last track_grad_for_num_itr many itrs.
                const int track_grad_for_num_itr,     // See prev. argument.
                const int num_caches,
                const thrust::device_ptr<const REAL> omega_vec = nullptr
            );

            // Assumes that deffered_mm_diff_ contains the mm's which were distributed in distribute_delta().
            void grad_distribute_delta(
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output after distributing delta.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output after distributing delta.
                thrust::device_ptr<REAL> grad_deff_mm // Output: contains grad w.r.t deff. min-marginal differences.
            );

            // Useful for cases when min_marginals_cuda() was called during forward pass to compute all min marginals (e.g. to compute loss).
            void grad_mm_diff_all_hops(
                thrust::device_ptr<REAL> incoming_grad_mm,
                thrust::device_ptr<REAL> grad_lo_cost_out, // Outputs in-place to compute grad. lo_cost before all min-marginal difference computation.
                thrust::device_ptr<REAL> grad_hi_cost_out // Outputs in-place to compute grad. hi_cost before all min-marginal difference computation.
                );

            // During primal rounding calling update_costs(lo_pert, hi_pert) changes the dual costs, the underlying primal objective vector also changes.
            // Here we compute gradients of such pertubation operation assuming that distribution of (lo_pert, hi_pert) was done with isoptropic weights.
            void grad_cost_perturbation(
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output after adding primal pertubation and Outputs in-place to compute grad. lo_cost before it.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output after adding primal pertubation and Outputs in-place to compute grad. hi_cost before it.
                thrust::device_ptr<REAL> grad_lo_pert_out, // Output: contains grad w.r.t pertubation in lo costs, assumes the memory is already allocated (= nr_variables()).
                thrust::device_ptr<REAL> grad_hi_pert_out // Output: contains grad w.r.t pertubation in hi costs, assumes the memory is already allocated (= nr_variables()).
            );

            void grad_lower_bound_per_bdd(
                thrust::device_ptr<REAL> grad_lb_per_bdd, // Input: incoming grad w.r.t lower bound per BDD.
                thrust::device_ptr<REAL> grad_lo_cost_out, // Gradients w.r.t lo costs
                thrust::device_ptr<REAL> grad_hi_cost_out // Gradients w.r.t hi costs
            );

            void set_initial_lb_change(const double val) {
                if (!std::isfinite(initial_lb_change_))
                    initial_lb_change_ = val; 
            }

            double get_initial_lb_change() const { return initial_lb_change_; }

        private:
            // last_hop should be equal to nr_hops() - 1 for complete forward pass.
            void forward_iteration_learned_mm_dist(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_ptr, const REAL omega_scalar, const thrust::device_ptr<const REAL> omega_vec = nullptr);

            // similar to above except that last_hop should be equal to 0 for complete backward pass.
            void backward_iteration_learned_mm_dist(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_ptr, const REAL omega_scalar, const thrust::device_ptr<const REAL> omega_vec = nullptr); 

            // Compute gradient of forward_iteration_learned_mm_dist.
            // Assumes solver state is set to state before forward_iteration_learned_mm_dist was called. 
            void grad_forward_iteration_learned_mm_dist(
                thrust::device_ptr<REAL> deferred_min_marg_diff, // deferred min-marginals used in forward pass, output will contains deferred min-marginals after forward iteration (not useful)
                const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost current iteration and Outputs in-place to compute grad. lo_cost before iteration.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost current iteration and Outputs in-place to compute grad. hi_cost before iteration.
                thrust::device_ptr<REAL> grad_cost_from_root, // Input: incoming grad w.r.t cost_from_root (size = nr_bdd_nodes()), Outputs in-place to compute grad before iteration.
                thrust::device_ptr<REAL> grad_cost_from_terminal, // Input: incoming grad w.r.t cost_from_terminal (size = nr_bdd_nodes()), Outputs in-place to compute grad before iteration.
                thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. current iteration and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iteration.
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).
                const REAL omega_scalar, // will not be used if omega_vec != nullptr
                thrust::device_ptr<REAL> grad_omega,    // Output: contains grad w.r.t omega (size = 1) if omega_vec = nullptr otherwise size should be same size of omega_vec.
                const thrust::device_ptr<const REAL> omega_vec = nullptr
                );

            // Compute gradient of backward_iteration_learned_mm_dist.
            // Assumes solver state is set to state before backward_iteration_learned_mm_dist was called. 
            void grad_backward_iteration_learned_mm_dist(
                thrust::device_ptr<REAL> deferred_min_marg_diff, // deferred min-marginals used in forward pass, output will contains deferred min-marginals after backward iteration (not useful)
                const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost current iteration and Outputs in-place to compute grad. lo_cost before iteration.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost current iteration and Outputs in-place to compute grad. hi_cost before iteration.
                thrust::device_ptr<REAL> grad_cost_from_root, // Input: incoming grad w.r.t cost_from_root (size = nr_bdd_nodes()), Outputs in-place to compute grad before iteration.
                thrust::device_ptr<REAL> grad_cost_from_terminal, // Input: incoming grad w.r.t cost_from_terminal (size = nr_bdd_nodes()) iteration and Outputs in-place to compute grad before iteration.
                thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. current iteration and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iteration.
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).
                const REAL omega_scalar, // will not be used if omega_vec != nullptr
                thrust::device_ptr<REAL> grad_omega,    // Output: contains grad w.r.t omega (size = 1) if omega_vec = nullptr otherwise size should be same size of omega_vec.
                const thrust::device_ptr<const REAL> omega_vec = nullptr
                );

            // Compute gradient for dual update of a hop and min-marginal calculation.
            // Where the update is performed only for all dual variables only in the current hop (i.e., block).
            // Rest of the hi / lo costs remain unchanged.
            // The functions takes as input dL / d (hi / lo costs new), dL / d (current mm difference), dL / d (cost_from_root / cost_from_terminal) and updates in-place.
            void grad_hop_update_learned_mm_dist(
                const thrust::device_ptr<const REAL> before_update_lo_cost,
                const thrust::device_ptr<const REAL> before_update_hi_cost,
                thrust::device_ptr<REAL> cur_min_marg_diff, // current min-marginals which were subtracted in present iteration.
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost and Outputs in-place to compute grad. lo_cost before hop update.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost and Outputs in-place to compute grad. hi_cost before hop update.
                thrust::device_ptr<REAL> grad_cost_from_root, // grad w.r.t cost_from_terminal (size = nr_bdd_nodes()).
                thrust::device_ptr<REAL> grad_cost_from_terminal, // grad w.r.t cost_from_terminal (size = nr_bdd_nodes()).
                thrust::device_ptr<REAL> grad_mm,     // Input: incoming grad w.r.t min-marginal differences current hop update. Is overwritten by accumulated grad_mm (not useful).
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers())
                thrust::device_ptr<REAL> grad_delta_lo, // To accumulate gradients w.r.t delta lo (should be intialized by zero for first hop)
                thrust::device_ptr<REAL> grad_delta_hi, // To accumulate gradients w.r.t delta hi (should be intialized by zero for first hop)
                const thrust::device_ptr<const REAL> dist_weights,            // distribution weights used in the forward pass.
                const int hop_index, 
                const REAL omega_scalar, // will not be used if omega_vec != nullptr
                thrust::device_ptr<REAL> grad_omega,    // Output: contains grad w.r.t omega (size = 1) if omega_vec = nullptr otherwise size should be same size of omega_vec.
                const thrust::device_ptr<const REAL> omega_vec = nullptr
                );

            void compute_grad_cost_from_root(thrust::device_ptr<REAL> grad_cost_from_root,  // incoming gradient of hop_index + 1 root costs is used to compute grad for hop_index root costs.
                                            thrust::device_ptr<REAL> grad_lo_cost,          // accumulates gradient for hop_index
                                            thrust::device_ptr<REAL> grad_hi_cost,          // accumulates gradient for hop_index
                                            const int hop_index);

            void compute_grad_cost_from_terminal(thrust::device_ptr<REAL> grad_cost_from_terminal,  // incoming gradient of hop_index terminal costs is used to compute grad for hop_index + 1 terminal costs.
                                                thrust::device_ptr<REAL> grad_lo_cost,          // accumulates gradient for hop_index
                                                thrust::device_ptr<REAL> grad_hi_cost,          // accumulates gradient for hop_index
                                                const int hop_index);

            void grad_mm_diff_of_hop(const thrust::device_ptr<const REAL> before_update_lo_cost, // min-marginal computation was done on input costs (not updated) costs.
                                    const thrust::device_ptr<const REAL> before_update_hi_cost,
                                    const thrust::device_ptr<const REAL> cur_min_marg_diff, // current min-marginals which were subtracted in present iteration.
                                    thrust::device_ptr<REAL> incoming_grad_mm_diff_hop, // gradient of min-marginal diff. output is backpropagates through multiplication by omega .
                                    thrust::device_ptr<REAL> grad_lo_cost,
                                    thrust::device_ptr<REAL> grad_hi_cost,
                                    thrust::device_ptr<REAL> grad_cost_from_root,
                                    thrust::device_ptr<REAL> grad_cost_from_terminal,
                                    thrust::device_ptr<REAL> grad_omega,    // Output: contains grad w.r.t omega (size = 1) if omega_vec = nullptr otherwise size should be same size of omega_vec.
                                    const int hop_index, 
                                    const REAL omega_scalar, // will not be used if omega_vec != nullptr
                                    const thrust::device_ptr<const REAL> omega_vec = nullptr,
                                    const bool backprop_omega = true
                                    );
            double initial_lb_change_ = std::numeric_limits<double>::infinity();
    };
}
