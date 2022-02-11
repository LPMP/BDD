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
            void iterations(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_out_ptr, const int num_itr, const REAL omega);
            void distribute_delta(const thrust::device_ptr<const REAL> dist_weights);

            void grad_iterations(
                const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output from iterations and Outputs in-place to compute grad. lo_cost before iterations.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output from iterations and Outputs in-place to compute grad. hi_cost before iterations.
                thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. which were output from iterations and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iterations.
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).
                thrust::device_ptr<REAL> grad_omega,    // Output: contains grad w.r.t omega (size = 1).
                const REAL omega,
                const int track_grad_after_itr,      // First runs the solver for track_grad_after_itr many iterations without tracking gradients and then backpropagates through only last track_grad_for_num_itr many itrs.
                const int track_grad_for_num_itr     // See prev. argument.
            );

            void grad_distribute_delta(
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output after distributing delta and Outputs in-place to compute grad. lo_cost before it.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output after distributing delta and Outputs in-place to compute grad. hi_cost before it.
                thrust::device_ptr<REAL> grad_dist_weights_out // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).
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

        private:
            // last_hop should be equal to nr_hops() - 1 for complete forward pass.
            void forward_iteration_learned_mm_dist(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_out_ptr, const int last_hop, const REAL omega);

            // similar to above except that last_hop should be equal to 0 for complete backward pass.
            void backward_iteration_learned_mm_dist(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_out_ptr, const int last_hop, const REAL omega); 

            // Compute gradient of forward_iteration_learned_mm_dist.
            // Assumes solver state is set to state before forward_iteration_learned_mm_dist was called. 
            void grad_forward_iteration_learned_mm_dist(
                const thrust::device_ptr<const REAL> deferred_min_marg_diff, // deferred min-marginals used in forward pass.
                const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output from current iteration and Outputs in-place to compute grad. lo_cost before iteration.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output from current iteration and Outputs in-place to compute grad. hi_cost before iteration.
                thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. which were output from current iteration and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iteration.
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).
                const REAL omega,
                thrust::device_ptr<REAL> grad_omega    // Output: contains grad w.r.t omega (size = 1).
                );

            // Compute gradient of backward_iteration_learned_mm_dist.
            // Assumes solver state is set to state before backward_iteration_learned_mm_dist was called. 
            void grad_backward_iteration_learned_mm_dist(
                const thrust::device_ptr<const REAL> deferred_min_marg_diff, // deferred min-marginals used in forward pass.
                const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output from current iteration and Outputs in-place to compute grad. lo_cost before iteration.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output from current iteration and Outputs in-place to compute grad. hi_cost before iteration.
                thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. which were output from current iteration and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iteration.
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).
                const REAL omega,
                thrust::device_ptr<REAL> grad_omega    // Output: contains grad w.r.t omega (size = 1).
                );

            // Compute gradient for all computations done for a hop i.e., min-marginal calculation and hi / lo costs update.
            // Where the update is performed only for all dual variables only in the current hop (i.e., block).
            // Rest of the hi / lo costs remain unchanged.
            // The functions takes as input dL / d (hi / lo costs new) and dL / d (current mm difference) and updates in-place.
            void grad_hop_update_learned_mm_dist(
                thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output from current hop update and Outputs in-place to compute grad. lo_cost before hop update.
                thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output from current hop update and Outputs in-place to compute grad. hi_cost before hop update.
                thrust::device_ptr<REAL> grad_mm,     // Input: incoming grad w.r.t min-marginal differences which were output from current hop update. Is overwritten by accumulated grad_mm (not useful).
                thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers())
                thrust::device_ptr<REAL> grad_delta_lo, // To accumulate gradients w.r.t delta lo (should be intialized by zero for first hop)
                thrust::device_ptr<REAL> grad_delta_hi, // To accumulate gradients w.r.t delta hi (should be intialized by zero for first hop)
                const thrust::device_ptr<const REAL> dist_weights,            // distribution weights used in the forward pass.
                const int hop_index, const REAL omega,
                thrust::device_ptr<REAL> grad_omega    // Output: contains grad w.r.t omega (size = 1).
                );

            // Computes gradient of min-marginal difference computation for the given hop. So given dL / d mm_diff as input, it computes
            // dL / d(hi - lo costs) = (dL / dmm)^T. (dmm / d (hi - lo costs)). The output has same dimensions as 
            // hi / lo costs. Gradient contains zero for terminal nodes. 
            thrust::device_vector<REAL> grad_mm_diff_of_hop(const thrust::device_ptr<const REAL> incoming_grad_mm_hop, const int hop_index, const REAL omega);

    };
}