#include "bdd_cuda_learned_mma.h"
#include "cuda_utils.h"

namespace LPMP {
    template<typename REAL>
    bdd_cuda_learned_mma<REAL>::bdd_cuda_learned_mma(const BDD::bdd_collection& bdd_col) : bdd_cuda_parallel_mma<REAL>(bdd_col) 
    { }

    template<typename REAL>
    __global__ void forward_step_learned_mm_dist(const int cur_num_bdd_nodes, const int start_offset,
                                                        const int* const __restrict__ lo_bdd_node_index, 
                                                        const int* const __restrict__ hi_bdd_node_index, 
                                                        const int* const __restrict__ bdd_node_to_layer_map, 
                                                        const int* const __restrict__ primal_variable_index, 
                                                        const REAL* const __restrict__ delta_lo_sum,
                                                        const REAL* const __restrict__ delta_hi_sum,
                                                        const REAL* const __restrict__ mm_diff,
                                                        const REAL* const __restrict__ dist_weights,
                                                        const REAL* const __restrict__ lo_cost_in,
                                                        const REAL* const __restrict__ hi_cost_in,
                                                        REAL* __restrict__ lo_cost_out,
                                                        REAL* __restrict__ hi_cost_out,
                                                        REAL* __restrict__ cost_from_root)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_node_idx = start_index + start_offset; bdd_node_idx < cur_num_bdd_nodes + start_offset; bdd_node_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_node_idx];
            if (next_lo_node < 0) // will matter when one row contains multiple BDDs, otherwise the terminal nodes are at the end anyway.
                continue; // nothing needs to be done for terminal node.
            
            const int layer_idx = bdd_node_to_layer_map[bdd_node_idx];
            const REAL cur_mm_diff_hi_lo = mm_diff[layer_idx]; 
            const REAL dist_w = dist_weights[layer_idx];
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const REAL cur_lo_cost = lo_cost_in[layer_idx] + min(cur_mm_diff_hi_lo, 0.0f) + dist_w * delta_lo_sum[cur_primal_idx];
            const REAL cur_c_from_root = cost_from_root[bdd_node_idx];

            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root + cur_lo_cost);

            const REAL cur_hi_cost = hi_cost_in[layer_idx] + min(-cur_mm_diff_hi_lo, 0.0f) + dist_w * delta_hi_sum[cur_primal_idx];
            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + cur_hi_cost);

            lo_cost_out[layer_idx] = cur_lo_cost;
            hi_cost_out[layer_idx] = cur_hi_cost;
        }
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::forward_iteration_learned_mm_dist(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_out_ptr, const int last_hop, const REAL omega)
    {
        assert(last_hop < this->nr_hops());
        if(!this->backward_state_valid_) // TODO: This can be optimized during grad. calculation by only running for unsolved hops.
            this->backward_run(false); //For the first iteration need to have costs from terminal. 
        
        // Clear states.
        this->flush_costs_from_root();
        this->flush_mm(mm_diff_out_ptr);
        for (int s = 0; s <= last_hop; s++)
        {
            this->min_marginals_from_directional_costs(s, omega, mm_diff_out_ptr);

            const int num_nodes_processed = s > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[s - 1] : 0;
            const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            const int blockCount = ceil(cur_num_bdd_nodes / (float) NUM_THREADS);

            forward_step_learned_mm_dist<<<blockCount, NUM_THREADS>>>(cur_num_bdd_nodes, num_nodes_processed,
                                                            thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                            thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                            thrust::raw_pointer_cast(this->delta_lo_.data()),
                                                            thrust::raw_pointer_cast(this->delta_hi_.data()),
                                                            thrust::raw_pointer_cast(mm_diff_out_ptr),
                                                            thrust::raw_pointer_cast(dist_weights),
                                                            thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                            thrust::raw_pointer_cast(this->lo_cost_out_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_out_.data()),
                                                            thrust::raw_pointer_cast(this->cost_from_root_.data()));
        }
        if (last_hop == this->nr_hops() - 1) // if was complete forward pass.
        {
            this->forward_state_valid_ = true;
            this->compute_delta(mm_diff_out_ptr);
        }
        else
        {   // Since the solver was run incompletely so copy the not updated dual variables to output.
            const int start_offset = this->cum_nr_layers_per_hop_dist_[last_hop];
            thrust::copy(this->lo_cost_.begin() + start_offset, this->lo_cost_.end(), this->lo_cost_out_.begin());
            thrust::copy(this->hi_cost_.begin() + start_offset, this->hi_cost_.end(), this->hi_cost_out_.begin());
        }
        thrust::swap(this->lo_cost_, this->lo_cost_out_);
        thrust::swap(this->hi_cost_, this->hi_cost_out_);
        this->flush_backward_states();
    }

    template<typename REAL>
    __global__ void backward_step_learned_mm_dist(const int cur_num_bdd_nodes, const int start_offset,
                                            const int* const __restrict__ lo_bdd_node_index, 
                                            const int* const __restrict__ hi_bdd_node_index, 
                                            const int* const __restrict__ bdd_node_to_layer_map, 
                                            const int* const __restrict__ primal_variable_index, 
                                            const REAL* const __restrict__ delta_lo_sum,
                                            const REAL* const __restrict__ delta_hi_sum,
                                            const REAL* const __restrict__ mm_diff,
                                            const REAL* const __restrict__ dist_weights,
                                            const REAL* const __restrict__ lo_cost_in,
                                            const REAL* const __restrict__ hi_cost_in,
                                            REAL* __restrict__ lo_cost_out,
                                            REAL* __restrict__ hi_cost_out,
                                            REAL* __restrict__ cost_from_terminal)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_node_idx = start_index + start_offset; bdd_node_idx < cur_num_bdd_nodes + start_offset; bdd_node_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_node_idx];
            if (next_lo_node < 0)
                continue; // nothing needs to be done for terminal node.
            
            const int layer_idx = bdd_node_to_layer_map[bdd_node_idx];
            const REAL cur_mm_diff_hi_lo = mm_diff[layer_idx]; 

            const REAL dist_w = dist_weights[layer_idx];
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const REAL cur_lo_cost = lo_cost_in[layer_idx] + min(cur_mm_diff_hi_lo, 0.0f) + dist_w * delta_lo_sum[cur_primal_idx];
            const REAL cur_hi_cost = hi_cost_in[layer_idx] + min(-cur_mm_diff_hi_lo, 0.0f) + dist_w * delta_hi_sum[cur_primal_idx];
            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];

            // Update costs from terminal:
            cost_from_terminal[bdd_node_idx] = min(cur_hi_cost + cost_from_terminal[next_hi_node], cur_lo_cost + cost_from_terminal[next_lo_node]);

            lo_cost_out[layer_idx] = cur_lo_cost;
            hi_cost_out[layer_idx] = cur_hi_cost;
        }
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::backward_iteration_learned_mm_dist(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_out_ptr, const int last_hop, const REAL omega)
    {
        assert(last_hop < this->nr_hops());
        assert(this->forward_state_valid_);

        this->flush_mm(mm_diff_out_ptr);
        for (int s = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 2; s >= last_hop; s--)
        {
            this->min_marginals_from_directional_costs(s, omega, mm_diff_out_ptr);

            const int start_offset = s > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[s - 1] : 0;

            const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[s] - start_offset;
            const int blockCount = ceil(cur_num_bdd_nodes / (REAL) NUM_THREADS);

            backward_step_learned_mm_dist<<<blockCount, NUM_THREADS>>>(cur_num_bdd_nodes, start_offset,
                                                            thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                            thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                            thrust::raw_pointer_cast(this->delta_lo_.data()),
                                                            thrust::raw_pointer_cast(this->delta_hi_.data()),
                                                            thrust::raw_pointer_cast(mm_diff_out_ptr),
                                                            thrust::raw_pointer_cast(dist_weights),
                                                            thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                            thrust::raw_pointer_cast(this->lo_cost_out_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_out_.data()),
                                                            thrust::raw_pointer_cast(this->cost_from_terminal_.data()));
        }
        if (last_hop == 0) // if was complete backward pass.
        {
            this->backward_state_valid_ = true;
            this->compute_delta(mm_diff_out_ptr);
        }
        else
        {   // Since the solver was run incompletely so copy the not updated dual variables to output.
            const int end_offset = this->cum_nr_layers_per_hop_dist_[last_hop - 1];
            thrust::copy(this->lo_cost_.begin(), this->lo_cost_.begin() + end_offset, this->lo_cost_out_.begin());
            thrust::copy(this->hi_cost_.begin(), this->hi_cost_.begin() + end_offset, this->hi_cost_out_.begin());
        }
        thrust::swap(this->lo_cost_, this->lo_cost_out_);
        thrust::swap(this->hi_cost_, this->hi_cost_out_);
        this->flush_forward_states();
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::iterations(const thrust::device_ptr<const REAL> dist_weights, thrust::device_ptr<REAL> mm_diff_out_ptr, const int num_itr, const REAL omega)
    {
        for(int itr = 0; itr < num_itr; itr++)
        {
            forward_iteration_learned_mm_dist(dist_weights, mm_diff_out_ptr, this->nr_hops() - 1, omega);
            backward_iteration_learned_mm_dist(dist_weights, mm_diff_out_ptr, 0, omega);
        }
    }

    template<typename REAL>
    struct distribute_delta_weighted_func {
        const int* primal_index;
        const REAL* delta_lo;
        const REAL* delta_hi;
        const REAL* dist_weights;
        REAL* lo_costs;
        REAL* hi_costs;
        __host__ __device__ void operator()(const int i)
        {
            const int current_primal_index = primal_index[i];
            if (current_primal_index == INT_MAX)
                return; // terminal node.

            lo_costs[i] += dist_weights[i] * delta_lo[current_primal_index];
            hi_costs[i] += dist_weights[i] * delta_hi[current_primal_index];
        }
    };

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::distribute_delta(const thrust::device_ptr<const REAL> dist_weights)
    {
        assert(!this->delta_normalized_);
        distribute_delta_weighted_func<REAL> func_dist_delta({
                                            thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                            thrust::raw_pointer_cast(this->delta_lo_.data()),
                                            thrust::raw_pointer_cast(this->delta_hi_.data()),
                                            thrust::raw_pointer_cast(dist_weights),
                                            thrust::raw_pointer_cast(this->lo_cost_.data()),
                                            thrust::raw_pointer_cast(this->hi_cost_.data())});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), func_dist_delta);
        this->flush_forward_states();
        this->flush_backward_states();

        thrust::fill(this->delta_lo_.begin(), this->delta_lo_.end(), 0.0f);
        thrust::fill(this->delta_hi_.begin(), this->delta_hi_.end(), 0.0f);
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_iterations(
            const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
            thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output from iterations and Outputs in-place to compute grad. lo_cost before iterations.
            thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output from iterations and Outputs in-place to compute grad. hi_cost before iterations.
            thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. which were output from iterations and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iterations.
            thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto the current point.
            const double omega,
            const int track_grad_after_itr,      // First runs the solver for track_grad_after_itr many iterations without tracking gradients and then backpropagates through only last track_grad_for_num_itr many itrs.
            const int track_grad_for_num_itr     // See prev. argument.
        )
    {
        thrust::fill(grad_dist_weights_out, grad_dist_weights_out + this->nr_layers(), 0.0);
        iterations(dist_weights, this->mm_diff_.data(), track_grad_after_itr, omega);
        const auto initial_costs = this->get_solver_costs();

        for(int itr = track_grad_for_num_itr - 1; itr >= 0; itr--)
        {
            // Reset solver to original state.
            if (itr < track_grad_for_num_itr - 1)
                this->set_solver_costs(initial_costs);
    
            // To compute grad for iteration itr, first take the solver to state of iteration itr - 1.
            if (itr > 0)
                iterations(dist_weights, this->mm_diff_.data(), itr - 1, omega);

            const auto cur_costs = this->get_solver_costs();

            // First backprop through backward iteration which requires running forward iteration to get to the required state.
            forward_iteration_learned_mm_dist(dist_weights, this->mm_diff_.data(), this->nr_hops() - 1, omega);
            grad_backward_iteration_learned_mm_dist(this->mm_diff_.data(), dist_weights, grad_lo_cost, grad_hi_cost, grad_mm, grad_dist_weights_out, omega);
            // std::cout<<"grad_lo_min: "<<*thrust::min_element(grad_lo_cost, grad_lo_cost + this->nr_layers());
            // std::cout<<", grad_lo_max: "<<*thrust::max_element(grad_lo_cost, grad_lo_cost + this->nr_layers())<<"\n";

            this->set_solver_costs(cur_costs);
            grad_forward_iteration_learned_mm_dist(this->mm_diff_.data(), dist_weights, grad_lo_cost, grad_hi_cost, grad_mm, grad_dist_weights_out, omega);
            // std::cout<<"grad_lo_min: "<<*thrust::min_element(grad_lo_cost, grad_lo_cost + this->nr_layers());
            // std::cout<<", grad_lo_max: "<<*thrust::max_element(grad_lo_cost, grad_lo_cost + this->nr_layers())<<"\n\n";
        }
    }

    template<typename REAL>
    struct scale_grad_mm {
        const int* bdd_index;
        const REAL* incoming_grad_mm_diff;
        REAL* grad_out;
        const REAL omega;
        __host__ __device__ void operator()(const int layer_index)
        {
            const int current_bdd_index = bdd_index[layer_index];
            const REAL w = incoming_grad_mm_diff[current_bdd_index] * omega;
            grad_out[layer_index] *= w;
        }
    };

    // computing gradients w.r.t lambda = hi_cost - lo_cost. Gradient w.r.t hi_cost and lo_cost can be computed outside this function.
    template<typename REAL>
    thrust::device_vector<REAL> bdd_cuda_learned_mma<REAL>::grad_mm_diff_of_hop(const thrust::device_ptr<const REAL> incoming_grad_mm_hop, const int hop_index, const REAL omega)
    {
        thrust::device_vector<REAL> grad_lambda; 
        const int start_offset = hop_index > 0 ? this->cum_nr_layers_per_hop_dist_[hop_index - 1] : 0;
        const int end_offset = this->cum_nr_layers_per_hop_dist_[hop_index];
        const int num_layers_hop = end_offset - start_offset; // incoming_grad_mm should point to a vector of size num_layers_hop.
        assert(hop_index < this->cum_nr_layers_per_hop_dist_.size() - 1); // last hop contains terminal nodes so its mm's are not computed in forward pass anyway.
        {
            thrust::device_vector<REAL> orig_costs(num_layers_hop);

            auto grad_mm_hi_or_lo = [&](auto arc_cost_begin, auto arc_cost_end) -> thrust::device_vector<REAL> {
                // Set lo (or hi) costs of hop to INF so that when we compute opt. solution per bdd, then it automatically uses hi (or lo) arcs.
                thrust::copy(arc_cost_begin, arc_cost_end, orig_costs.begin()); 
                thrust::fill(arc_cost_begin, arc_cost_end, CUDART_INF_F_HOST);
                this->flush_forward_states();
                this->flush_backward_states();
                const thrust::device_vector<REAL> per_bdd_sol = this->bdds_solution_cuda();
                assert(std::isfinite(this->lower_bound()));
                thrust::copy(orig_costs.begin(), orig_costs.end(), arc_cost_begin); // Revert the operation of settings costs to INF.
                this->flush_forward_states();
                this->flush_backward_states();
                return per_bdd_sol;
            };

            thrust::device_vector<REAL> per_bdd_sol_lo = grad_mm_hi_or_lo(this->hi_cost_.begin() + start_offset, this->hi_cost_.begin() + end_offset);
            thrust::device_vector<REAL> per_bdd_sol_hi = grad_mm_hi_or_lo(this->lo_cost_.begin() + start_offset, this->lo_cost_.begin() + end_offset);
            grad_lambda = thrust::device_vector<REAL>(per_bdd_sol_lo.size());
            thrust::transform(per_bdd_sol_hi.begin(), per_bdd_sol_hi.end(), per_bdd_sol_lo.begin(), grad_lambda.begin(), thrust::minus<REAL>());
        }
        thrust::device_vector<REAL> rearranged_grad_mm_bdd_index(this->nr_bdds(), 0.0); // arrange grad_mm_hop so that grad value for each BDD can be found at the BDD index.
        thrust::scatter(incoming_grad_mm_hop, incoming_grad_mm_hop + num_layers_hop, this->bdd_index_.begin() + start_offset, rearranged_grad_mm_bdd_index.begin());

        // Now multiply each BDD solution by corresponding gradient dL / d mm_diff.
        scale_grad_mm<REAL> func({thrust::raw_pointer_cast(this->bdd_index_.data()), 
                                thrust::raw_pointer_cast(rearranged_grad_mm_bdd_index.data()), 
                                thrust::raw_pointer_cast(grad_lambda.data()),
                                omega});
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->bdd_index_.size(), func);
        return grad_lambda;
    }
    
    // computing gradients w.r.t lambda = hi_cost - lo_cost. Gradient w.r.t hi_cost and lo_cost can be computed outside this function.
    template<typename REAL>
    std::tuple<thrust::device_vector<REAL>, thrust::device_vector<REAL>> bdd_cuda_learned_mma<REAL>::grad_mm_diff_all_hops(const thrust::device_ptr<const REAL> incoming_grad_mm, const REAL omega)
    {
        thrust::device_vector<REAL> grad_hi_cost(this->hi_cost_.size(), 0.0);
        const int num_hops = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        for (int hop_index = 0; hop_index < num_hops; hop_index++) // Compute vJp by populating few rows of the Jacobian, multiplying by part of v and then accumulating into the output.
        {
            const int start_offset = hop_index > 0 ? this->cum_nr_layers_per_hop_dist_[hop_index - 1] : 0;
            const thrust::device_vector<REAL> grad_hi_cost_hop = grad_mm_diff_of_hop(incoming_grad_mm + start_offset, hop_index, omega);
            assert(grad_hi_cost_hop.size() == grad_hi_cost.size());
            thrust::transform(grad_hi_cost.begin(), grad_hi_cost.end(), grad_hi_cost_hop.begin(), grad_hi_cost.begin(), thrust::plus<REAL>());
        }
        thrust::device_vector<REAL> grad_lo_cost(this->lo_cost_.size());
        // grad_lo_cost is just opposite of grad_hi_cost:
        thrust::copy(thrust::make_transform_iterator(grad_hi_cost.begin(), thrust::negate<REAL>()), thrust::make_transform_iterator(grad_hi_cost.end(), thrust::negate<REAL>()), grad_lo_cost.begin());
        return {grad_lo_cost, grad_hi_cost};
    }

    template<typename REAL>
    struct grad_dual_update_func {
        const int* primal_index;
        const REAL* cur_mm_diff;
        const REAL* dist_weights;
        REAL* grad_lo_cost;
        REAL* grad_hi_cost;
        REAL* grad_cur_mm_diff;
        REAL* grad_delta_lo;
        REAL* grad_delta_hi;
        const unsigned long num_vars;
        __host__ __device__ void operator()(const int i)
        {
            const int primal = primal_index[i];
            if (primal >= num_vars)
                return; 
            const REAL current_mm = cur_mm_diff[i];
            if (current_mm >= 0)
                grad_cur_mm_diff[i] -= grad_hi_cost[i];
            else
                grad_cur_mm_diff[i] += grad_lo_cost[i];
            atomicAdd(&grad_delta_lo[primal], grad_lo_cost[i] * dist_weights[i]);
            atomicAdd(&grad_delta_hi[primal], grad_hi_cost[i] * dist_weights[i]);
        }
    };

    template<typename REAL>
    struct grad_dist_weights_func {
        const int* primal_index;
        const REAL* grad_lo_cost;
        const REAL* grad_hi_cost;
        const REAL* delta_lo;
        const REAL* delta_hi;
        REAL* grad_dist_weights;
        const unsigned long num_vars;
        __host__ __device__ void operator()(const int i)
        {
            const int primal = primal_index[i];
            if (primal >= num_vars)
                return; 
            grad_dist_weights[i] += delta_lo[primal] * grad_lo_cost[i] + delta_hi[primal] * grad_hi_cost[i];
        }
    };

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_hop_update_learned_mm_dist(
            thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output from current hop update and Outputs in-place to compute grad. lo_cost before hop update.
            thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output from current hop update and Outputs in-place to compute grad. hi_cost before hop update.
            thrust::device_ptr<REAL> grad_mm,     // Input: incoming grad w.r.t min-marginal differences which were output from current hop update. Is overwritten by accumulated grad_mm (not useful).
            thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto current point.
            thrust::device_ptr<REAL> grad_delta_lo, // To accumulate gradients w.r.t delta lo (should be intialized by zero for first hop)
            thrust::device_ptr<REAL> grad_delta_hi, // To accumulate gradients w.r.t delta hi (should be intialized by zero for first hop)
            const thrust::device_ptr<const REAL> dist_weights,            // Input: distribution weights used in the forward pass.
            const int hop_index, const double omega)
    {
        const int start_offset = hop_index > 0 ? this->cum_nr_layers_per_hop_dist_[hop_index - 1] : 0;
        const int end_offset = this->cum_nr_layers_per_hop_dist_[hop_index];
        const int num_layers_hop = end_offset - start_offset;

        // 1.1 Backprop through hop update to accumulate grad w.r.t mm difference, grad w.r.t delta lo and hi:
        // This needs access to current min-marginal difference:
        this->min_marginals_from_directional_costs(hop_index, omega);

        grad_dual_update_func<REAL> func_grad_dual_update({
                                                thrust::raw_pointer_cast(this->primal_variable_index_.data()  + start_offset),
                                                thrust::raw_pointer_cast(this->mm_diff_.data() + start_offset),
                                                thrust::raw_pointer_cast(dist_weights + start_offset),
                                                thrust::raw_pointer_cast(grad_lo_cost + start_offset),
                                                thrust::raw_pointer_cast(grad_hi_cost + start_offset),
                                                thrust::raw_pointer_cast(grad_mm + start_offset),
                                                thrust::raw_pointer_cast(grad_delta_lo),
                                                thrust::raw_pointer_cast(grad_delta_hi),
                                                this->nr_variables()
                                            });
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + num_layers_hop, func_grad_dual_update);

        // Backprop through mm computation:
        const thrust::device_vector<REAL> grad_lambda_hop_mm = grad_mm_diff_of_hop(grad_mm + start_offset, hop_index, omega);
        // After this the input grad_mm can be overwritten with updated gradients to send back.

        // Compute grad w.r.t dist_weights.
        grad_dist_weights_func<REAL> func_grad_dist_w({
                                                thrust::raw_pointer_cast(this->primal_variable_index_.data() + start_offset),
                                                thrust::raw_pointer_cast(grad_lo_cost + start_offset),
                                                thrust::raw_pointer_cast(grad_hi_cost + start_offset),
                                                thrust::raw_pointer_cast(this->delta_lo_.data()),
                                                thrust::raw_pointer_cast(this->delta_hi_.data()),
                                                thrust::raw_pointer_cast(grad_dist_weights_out + start_offset),
                                                this->nr_variables()
                                            });
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + num_layers_hop, func_grad_dist_w);

        // After this the input grad_lo_cost, grad_hi_cost can be overwritten with updated gradients to send back.
        // Add contribution to grad_lo_cost, grad_hi_cost by the step of mm computation. (Jacobian of hi(lo) costs w.r.t prev. hi(lo) costs is identity):
        thrust::transform(grad_hi_cost, grad_hi_cost + this->nr_layers(), grad_lambda_hop_mm.begin(), grad_hi_cost, thrust::plus<REAL>());
        thrust::transform(grad_lo_cost, grad_lo_cost + this->nr_layers(), grad_lambda_hop_mm.begin(), grad_lo_cost, thrust::minus<REAL>());
    }

    template<typename REAL>
    struct grad_def_min_marg_func {
        const int* primal_index;
        const REAL* def_min_marg_diff;
        const REAL* grad_delta_lo;
        const REAL* grad_delta_hi;
        REAL* grad_def_min_marg;
        const unsigned long num_vars;
        __host__ __device__ void operator()(const int i)
        {
            const int primal = primal_index[i];
            if (primal >= num_vars)
                return;
            const REAL def_mm = def_min_marg_diff[i];
            if (def_mm >= 0)
                grad_def_min_marg[i] = 1.0 * grad_delta_hi[primal];
            else
                grad_def_min_marg[i] = -1.0 * grad_delta_lo[primal];
        }
    };

    // Compute gradient of forward_iteration_learned_mm_dist.
    // Assumes solver state is set to state before forward_iteration_learned_mm_dist was called. 
    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_forward_iteration_learned_mm_dist(
        const thrust::device_ptr<const REAL> deferred_min_marg_diff, // deferred min-marginals used in forward pass.
        const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
        thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output from current iteration and Outputs in-place to compute grad. lo_cost before iteration.
        thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output from current iteration and Outputs in-place to compute grad. hi_cost before iteration.
        thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. which were output from current iteration and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iteration.
        thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto the current point.
        const double omega)
    {
        const auto initial_costs = this->get_solver_costs();

        thrust::device_vector<REAL> grad_delta_lo(this->delta_lo_.size(), 0.0);
        thrust::device_vector<REAL> grad_delta_hi(this->delta_hi_.size(), 0.0);

        for (int s = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 2; s >= 0; s--)
        {
            // Reset solver to original state.
            if (s < this->cum_nr_bdd_nodes_per_hop_dist_.size() - 2)
                this->set_solver_costs(initial_costs);
    
            // To compute grad for hop s, first take the solver to state of hop s - 1.
            if (s > 0)
                forward_iteration_learned_mm_dist(dist_weights, this->mm_diff_.data(), s - 1, omega);

            grad_hop_update_learned_mm_dist(grad_lo_cost, grad_hi_cost, grad_mm, grad_dist_weights_out, grad_delta_lo.data(), grad_delta_hi.data(), dist_weights, s, omega);
        }

        // Deferred min-marginals were used to compute delta_lo and delta_hi, perform backprop through this op.
        grad_def_min_marg_func<REAL> func_grad_def_mm({thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                    thrust::raw_pointer_cast(deferred_min_marg_diff),
                                                    thrust::raw_pointer_cast(grad_delta_lo.data()),
                                                    thrust::raw_pointer_cast(grad_delta_lo.data()),
                                                    thrust::raw_pointer_cast(grad_mm)});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), func_grad_def_mm);
    }

    // Compute gradient of backward_iteration_learned_mm_dist.
    // Assumes solver state is set to state before backward_iteration_learned_mm_dist was called. 
    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_backward_iteration_learned_mm_dist(
        const thrust::device_ptr<const REAL> deferred_min_marg_diff, // deferred min-marginals used in forward pass.
        const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
        thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output from current iteration and Outputs in-place to compute grad. lo_cost before iteration.
        thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output from current iteration and Outputs in-place to compute grad. hi_cost before iteration.
        thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. which were output from current iteration and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iteration.
        thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto the current point.
        const double omega)
    {
        const auto initial_costs = this->get_solver_costs();

        thrust::device_vector<REAL> grad_delta_lo(this->delta_lo_.size(), 0.0);
        thrust::device_vector<REAL> grad_delta_hi(this->delta_hi_.size(), 0.0);

        for (int s = 0; s < this->cum_nr_bdd_nodes_per_hop_dist_.size() - 1; s++)
        {
            // Reset solver to original state.
            if (s > 0)
                this->set_solver_costs(initial_costs);
    
            // To compute grad for hop s, first take the solver to state of hop s + 1.
            if (s < this->cum_nr_bdd_nodes_per_hop_dist_.size() - 2)
            {
                this->forward_run();
                backward_iteration_learned_mm_dist(dist_weights, this->mm_diff_.data(), s + 1, omega);
            }

            grad_hop_update_learned_mm_dist(grad_lo_cost, grad_hi_cost, grad_mm, grad_dist_weights_out, grad_delta_lo.data(), grad_delta_hi.data(), dist_weights, s, omega);
        }

        // Deferred min-marginals were used to compute delta_lo and delta_hi, perform backprop through this op.
        grad_def_min_marg_func<REAL> func_grad_def_mm({thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                    thrust::raw_pointer_cast(deferred_min_marg_diff),
                                                    thrust::raw_pointer_cast(grad_delta_lo.data()),
                                                    thrust::raw_pointer_cast(grad_delta_hi.data()),
                                                    thrust::raw_pointer_cast(grad_mm),
                                                    this->nr_variables()});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), func_grad_def_mm);
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_distribute_delta(
        thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output after distributing delta.
        thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output after distributing delta.
        thrust::device_ptr<REAL> grad_dist_weights_out // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto current point.
    )
    {
        // Nothing to do for grad_lo_cost, grad_hi_cost since Jacobian is identity. Only need to compute grad_dist_weights_out. 
        grad_dist_weights_func<REAL> func_grad_dist_w({
                            thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                            thrust::raw_pointer_cast(grad_lo_cost),
                            thrust::raw_pointer_cast(grad_hi_cost),
                            thrust::raw_pointer_cast(this->delta_lo_.data()),
                            thrust::raw_pointer_cast(this->delta_hi_.data()),
                            thrust::raw_pointer_cast(grad_dist_weights_out),
                            this->nr_variables()
                        });
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), func_grad_dist_w);
    }

    template<typename REAL>
    struct normalize_by_num_bdds {
        __host__ __device__ void operator()(const thrust::tuple<REAL&, REAL&, int> t) const
        {
            const int num_bdds = thrust::get<2>(t);
            REAL& hi_cost = thrust::get<0>(t);
            hi_cost /= num_bdds;
            REAL& lo_cost = thrust::get<1>(t);
            lo_cost /= num_bdds;
        }
    };

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_cost_pertubation(
        thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output after adding primal pertubation and Outputs in-place to compute grad. lo_cost before it.
        thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output after adding primal pertubation and Outputs in-place to compute grad. hi_cost before it.
        thrust::device_ptr<REAL> grad_lo_pert_out, // Output: contains grad w.r.t pertubation in lo costs, assumes the memory is already allocated (= nr_variables()).
        thrust::device_ptr<REAL> grad_hi_pert_out // Output: contains grad w.r.t pertubation in hi costs, assumes the memory is already allocated (= nr_variables()).
    )
    {
        auto first_val = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_permutation_iterator(grad_lo_cost, this->primal_variable_sorting_order_.begin()),
            thrust::make_permutation_iterator(grad_hi_cost, this->primal_variable_sorting_order_.begin())));

        auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(grad_lo_pert_out, grad_hi_pert_out));

        thrust::equal_to<int> binary_pred;
        auto new_end = thrust::reduce_by_key(this->primal_variable_index_sorted_.begin(), this->primal_variable_index_sorted_.end() - this->nr_bdds_, first_val, 
                            thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_sum());
        assert(thrust::distance(first_out_val, new_end.second) == this->delta_hi_.size());

        // Normalize by number of BDDs (assumes isotropic cost distribution during forward pass).
        auto first = thrust::make_zip_iterator(thrust::make_tuple(grad_lo_pert_out, grad_hi_pert_out, this->num_bdds_per_var_.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(grad_lo_pert_out + this->nr_variables(), grad_hi_pert_out + this->nr_variables(), this->num_bdds_per_var_.end()));
        thrust::for_each(first, last, normalize_by_num_bdds<REAL>());
    }

    template class bdd_cuda_learned_mma<float>;
    template class bdd_cuda_learned_mma<double>;
}
