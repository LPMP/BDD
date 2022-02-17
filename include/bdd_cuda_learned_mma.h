#pragma once

#include "bdd_cuda_parallel_mma.h"

namespace LPMP {

    template<typename REAL>
    class bdd_cuda_learned_mma : public bdd_cuda_parallel_mma<REAL> {
        public:

            bdd_cuda_learned_mma() {}
            bdd_cuda_learned_mma(const BDD::bdd_collection& bdd_col);

            // dist_weights points to disribution weights and mm_diff_out_ptr will contain deferred min-marginal differences.
            // The size of underlying arrays should be corresponding to the size of hi_cost_ i.e. = this->nr_layers(). 
            void iterations(const thrust::device_ptr<const REAL> dist_weights, const int num_itr, const REAL omega);

            // Assumes that deffered_mm_diff_ contains the mm's containing the values before iterations() was called.
            void grad_iterations(
                const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost, Outputs in-place to compute grad. lo_cost before iterations.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost, Outputs in-place to compute grad. hi_cost before iterations.
                thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff., Outputs in-place to compute grad. w.r.t deferred min-marginals used in iterations.
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).
                thrust::device_ptr<REAL> grad_omega,    // Output: contains grad w.r.t omega (size = 1).
                const REAL omega,
                const int track_grad_after_itr,      // First runs the solver for track_grad_after_itr many iterations without tracking gradients and then backpropagates through only last track_grad_for_num_itr many itrs.
                const int track_grad_for_num_itr     // See prev. argument.
            );

            // Assumes that deffered_mm_diff_ contains the mm's which were distributed in distribute_delta().
            void grad_distribute_delta(
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output after distributing delta.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output after distributing delta.
                thrust::device_ptr<REAL> grad_deff_mm // Output: contains grad w.r.t deff. min-marginal differences.
            );

            // Useful for cases when min_marginals_cuda() was called during forward pass to compute all min marginals (e.g. to compute loss).
            void grad_mm_diff_all_hops(
                const thrust::device_ptr<const REAL> incoming_grad_mm,
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

        private:
            // last_hop should be equal to nr_hops() - 1 for complete forward pass.
            void forward_iteration_learned_mm_dist(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_ptr, const REAL omega);

            // similar to above except that last_hop should be equal to 0 for complete backward pass.
            void backward_iteration_learned_mm_dist(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_ptr, const REAL omega); 

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
                const REAL omega,
                thrust::device_ptr<REAL> grad_omega    // Output: contains grad w.r.t omega (size = 1).
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
                const REAL omega,
                thrust::device_ptr<REAL> grad_omega    // Output: contains grad w.r.t omega (size = 1).
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
                const int hop_index, const REAL omega,
                thrust::device_ptr<REAL> grad_omega    // Output: contains grad w.r.t omega (size = 1).
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
                                    const thrust::device_ptr<const REAL> incoming_grad_mm_diff_hop, // gradient of (min-marginal diff. weighted by omega).
                                    thrust::device_ptr<REAL> grad_lo_cost,
                                    thrust::device_ptr<REAL> grad_hi_cost,
                                    thrust::device_ptr<REAL> grad_cost_from_root,
                                    thrust::device_ptr<REAL> grad_cost_from_terminal,
                                    thrust::device_ptr<REAL> grad_omega,
                                    const int hop_index, const REAL omega,
                                    bool compute_grad_omega = true);
    };
}