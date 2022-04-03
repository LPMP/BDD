#include "bdd_cuda_learned_mma.h"
#include "cuda_utils.h"
#include <time.h>
#include <thrust/random.h>

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
    void bdd_cuda_learned_mma<REAL>::forward_iteration_learned_mm_dist(
                                const thrust::device_ptr<const REAL> dist_weights, 
                                thrust::device_ptr<REAL> mm_diff_ptr, 
                                const REAL omega_scalar, 
                                const thrust::device_ptr<const REAL> omega_vec)
    {
        if(!this->backward_state_valid_)
            this->backward_run(false); //For the first iteration need to have costs from terminal. 
        
        this->compute_delta(mm_diff_ptr);
        this->flush_mm(mm_diff_ptr);
        // Clear states.
        this->flush_costs_from_root();
        for (int s = 0; s < this->nr_hops(); s++)
        {
            const int start_offset_layer = s > 0 ? this->cum_nr_layers_per_hop_dist_[s - 1]: 0;
            this->min_marginals_from_directional_costs(s, omega_scalar, mm_diff_ptr + start_offset_layer, 
                                                        this->lo_cost_.data(), this->hi_cost_.data(), omega_vec);

            const int num_nodes_processed = s > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[s - 1] : 0;
            const int cur_num_bdd_nodes = this->nr_bdd_nodes(s);
            const int blockCount = ceil(cur_num_bdd_nodes / (float) NUM_THREADS_CUDA);

            forward_step_learned_mm_dist<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, num_nodes_processed,
                                                            thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                            thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                            thrust::raw_pointer_cast(this->delta_lo_.data()),
                                                            thrust::raw_pointer_cast(this->delta_hi_.data()),
                                                            thrust::raw_pointer_cast(mm_diff_ptr),
                                                            thrust::raw_pointer_cast(dist_weights),
                                                            thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                            thrust::raw_pointer_cast(this->lo_cost_out_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_out_.data()),
                                                            thrust::raw_pointer_cast(this->cost_from_root_.data()));
        }
        this->forward_state_valid_ = true;
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
    void bdd_cuda_learned_mma<REAL>::backward_iteration_learned_mm_dist(
                                    const thrust::device_ptr<const REAL> dist_weights, 
                                    thrust::device_ptr<REAL> mm_diff_ptr, 
                                    const REAL omega_scalar, 
                                    const thrust::device_ptr<const REAL> omega_vec)
    {
        assert(this->forward_state_valid_);

        this->compute_delta(mm_diff_ptr);
        this->flush_mm(mm_diff_ptr);
        for (int s = this->nr_hops() - 1; s >= 0; s--)
        {
            const int start_offset_layer = s > 0 ? this->cum_nr_layers_per_hop_dist_[s - 1]: 0;
            this->min_marginals_from_directional_costs(s, omega_scalar, mm_diff_ptr + start_offset_layer, 
                                                        this->lo_cost_.data(), this->hi_cost_.data(), omega_vec);

            const int start_offset = s > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[s - 1] : 0;
            const int cur_num_bdd_nodes = this->nr_bdd_nodes(s);
            const int blockCount = ceil(cur_num_bdd_nodes / (REAL) NUM_THREADS_CUDA);

            backward_step_learned_mm_dist<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, start_offset,
                                                            thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                            thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                            thrust::raw_pointer_cast(this->delta_lo_.data()),
                                                            thrust::raw_pointer_cast(this->delta_hi_.data()),
                                                            thrust::raw_pointer_cast(mm_diff_ptr),
                                                            thrust::raw_pointer_cast(dist_weights),
                                                            thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                            thrust::raw_pointer_cast(this->lo_cost_out_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_out_.data()),
                                                            thrust::raw_pointer_cast(this->cost_from_terminal_.data()));
        }
        this->backward_state_valid_ = true;
        thrust::swap(this->lo_cost_, this->lo_cost_out_);
        thrust::swap(this->hi_cost_, this->hi_cost_out_);
        this->flush_forward_states();
    }

    template<typename REAL>
    int bdd_cuda_learned_mma<REAL>::iterations(
                                const thrust::device_ptr<const REAL> dist_weights, 
                                const int num_itr, 
                                const REAL omega_scalar, 
                                const double improvement_slope,
                                const thrust::device_ptr<const REAL> omega_vec)
    {
        const double lb_initial = this->lower_bound();
        double lb_first_iter = std::numeric_limits<double>::max();
        double lb_prev = lb_initial;
        double lb_post = lb_prev;
        int itr = 0;
        for(itr = 0; itr < num_itr; itr++)
        {
            forward_iteration_learned_mm_dist(dist_weights, this->deffered_mm_diff_.data(), omega_scalar, omega_vec);
            backward_iteration_learned_mm_dist(dist_weights, this->deffered_mm_diff_.data(), omega_scalar, omega_vec);
            lb_prev = lb_post;
            lb_post = this->lower_bound();
            if(itr == 0)
                lb_first_iter = lb_post;
            if (std::abs(lb_prev - lb_post) < improvement_slope * std::abs(lb_initial - lb_first_iter))
                break;
        }
        return itr;
    }

    template<typename REAL>
    void solver_state_cache<REAL>::check_and_set_cache(const int itr, 
                                                const thrust::device_ptr<const REAL> lo_costs_ptr,
                                                const thrust::device_ptr<const REAL> hi_costs_ptr,
                                                const thrust::device_ptr<const REAL> def_mm_ptr)
    {
        assert(itr < num_iterations_);
        if (num_caches_ == 0) return;
        if (itr % cache_interval_ != 0) return; 
        const int cache_index = itr / cache_interval_;
        if (cache_index >= num_caches_) return;
        lo_costs_cache_[cache_index] = std::vector<REAL>(num_layers_);
        hi_costs_cache_[cache_index] = std::vector<REAL>(num_layers_);
        def_mm_cache_[cache_index] = std::vector<REAL>(num_layers_);

        thrust::copy(lo_costs_ptr, lo_costs_ptr + num_layers_, lo_costs_cache_[cache_index].begin());
        thrust::copy(hi_costs_ptr, hi_costs_ptr + num_layers_, hi_costs_cache_[cache_index].begin());
        thrust::copy(def_mm_ptr, def_mm_ptr + num_layers_, def_mm_cache_[cache_index].begin());
    }

    template<typename REAL>
    int solver_state_cache<REAL>::check_and_get_cache(const int itr, 
                                                thrust::device_ptr<REAL> lo_costs_ptr,
                                                thrust::device_ptr<REAL> hi_costs_ptr,
                                                thrust::device_ptr<REAL> def_mm_ptr)
    {
        if (num_caches_ == 0)
            return 0;

        const int cache_lower_index = min(itr / cache_interval_, num_caches_ - 1);
        thrust::copy(lo_costs_cache_[cache_lower_index].begin(), lo_costs_cache_[cache_lower_index].end(), lo_costs_ptr);
        thrust::copy(hi_costs_cache_[cache_lower_index].begin(), hi_costs_cache_[cache_lower_index].end(), hi_costs_ptr);
        thrust::copy(def_mm_cache_[cache_lower_index].begin(), def_mm_cache_[cache_lower_index].end(), def_mm_ptr);
        return cache_lower_index * cache_interval_; // Return the iteration index for which cache is valid for.
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_iterations(
            const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
            thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost, Outputs in-place to compute grad. lo_cost before iterations.
            thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost, Outputs in-place to compute grad. hi_cost before iterations.
            thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff., Outputs in-place to compute grad. w.r.t deferred min-marginals used in iterations.
            thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto the current point.
            thrust::device_ptr<REAL> grad_omega,    // Output: contains grad w.r.t omega (size = 1).
            const REAL omega_scalar,
            const int track_grad_after_itr,      // First runs the solver for track_grad_after_itr many iterations without tracking gradients and then backpropagates through only last track_grad_for_num_itr many itrs.
            const int track_grad_for_num_itr,     // See prev. argument.
            const int num_caches,
            const thrust::device_ptr<const REAL> omega_vec)
    {
        if (track_grad_for_num_itr == 0)
            return;

        // add_noise<REAL>(this->lo_cost_.data(), this->nr_layers(), (REAL) 1e-6);
        // add_noise<REAL>(this->hi_cost_.data(), this->nr_layers(), (REAL) 1e-6);

        thrust::fill(grad_dist_weights_out, grad_dist_weights_out + this->nr_layers(), 0.0);
        if (!omega_vec.get())
            thrust::fill(grad_omega, grad_omega + 1, 0.0); // omega_scalar is used.
        else
            thrust::fill(grad_omega, grad_omega + this->nr_layers(), 0.0); // omega_vec is used. grad_omega should allocate required space beforehand.

        thrust::device_vector<REAL> grad_cost_from_root(this->cost_from_root_.size(), 0.0);
        thrust::device_vector<REAL> grad_cost_from_terminal(this->cost_from_terminal_.size(), 0.0);

        iterations(dist_weights, track_grad_after_itr, omega_scalar, 0.0, omega_vec);
        solver_state_cache<REAL> costs_cache(max(num_caches, 1), track_grad_for_num_itr, this->nr_layers()); // Atleast cache the starting point through max(num_caches, 1).

        // Populate cache.
        for(int solver_itr = 0; solver_itr <= costs_cache.max_cached_iteration(); solver_itr++)
        {
            // Cache the input for solver_itr. So if solver_itr = 0, it caches the input costs.
            costs_cache.check_and_set_cache(solver_itr, this->lo_cost_.data(), this->hi_cost_.data(), this->deffered_mm_diff_.data());
            iterations(dist_weights, min(1, costs_cache.max_cached_iteration() - solver_itr), omega_scalar, 0.0, omega_vec);
        }

        for(int itr = track_grad_for_num_itr - 1; itr >= 0; itr--)
        {
            // To compute grad for iteration itr, first take the solver to state which was input to iteration itr.
            const int cache_itr_index = costs_cache.check_and_get_cache(itr, this->lo_cost_.data(), this->hi_cost_.data(), this->deffered_mm_diff_.data());
            assert(cache_itr_index <= itr);
            iterations(dist_weights, itr - cache_itr_index, omega_scalar, 0.0, omega_vec); // run solver for 'itr - cache_itr_index' many more iterations.
    
            // save costs and mm for later in GPU memory.
            const auto cur_costs = this->get_solver_costs();

            // First backprop through backward iteration which requires running forward iteration to get to the required state.
            forward_iteration_learned_mm_dist(dist_weights, this->deffered_mm_diff_.data(), omega_scalar, omega_vec);

            // backward_iteration mapped (lo_costs, hi_costs, dist_weights, deferred mms, cost from root) -> (new_lo_costs, new_hi_costs, new mms, costs from terminal)
            grad_backward_iteration_learned_mm_dist(this->deffered_mm_diff_.data(), dist_weights, 
                                                    grad_lo_cost, grad_hi_cost, 
                                                    grad_cost_from_root.data(), grad_cost_from_terminal.data(), 
                                                    grad_mm, grad_dist_weights_out, omega_scalar, grad_omega, omega_vec);
            // backward_iteration produced terminal costs whose gradients are now accumulated into their predecessors.
            // So zero-out terminal costs gradients for accumulation from forward_iteration.
            thrust::fill(grad_cost_from_terminal.begin(), grad_cost_from_terminal.end(), 0.0); 

            this->set_solver_costs(cur_costs);
            // forward_iteration mapped (lo_costs, hi_costs, dist_weights, deferred mms, cost from terminal) -> (new_lo_costs, new_hi_costs, new mms, costs from root)
            grad_forward_iteration_learned_mm_dist(this->deffered_mm_diff_.data(), dist_weights, 
                                                grad_lo_cost, grad_hi_cost, 
                                                grad_cost_from_root.data(), grad_cost_from_terminal.data(), 
                                                grad_mm, grad_dist_weights_out, omega_scalar, grad_omega, omega_vec);

            thrust::fill(grad_cost_from_root.begin(), grad_cost_from_root.end(), 0.0);
        }
        if (track_grad_for_num_itr > 0) // Now backpropagate gradients of terminal costs back to hi and lo costs.
            for (int hop_index = 0; hop_index < this->nr_hops(); hop_index++) // Inverse direction as that of backward_run().
                compute_grad_cost_from_terminal(grad_cost_from_terminal.data(), grad_lo_cost, grad_hi_cost, hop_index);
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_mm_diff_all_hops(
        thrust::device_ptr<REAL> incoming_grad_mm,
        thrust::device_ptr<REAL> grad_lo_cost_out, // Outputs in-place to compute grad. lo_cost before all min-marginal difference computation.
        thrust::device_ptr<REAL> grad_hi_cost_out // Outputs in-place to compute grad. hi_cost before all min-marginal difference computation.
        )
    {
        // Make costs from root and terminal valid w.r.t current state.
        this->forward_run();
        this->backward_run(false);

        thrust::device_vector<REAL> grad_cost_from_root(this->cost_from_root_.size(), 0.0);
        thrust::device_vector<REAL> grad_cost_from_terminal(this->cost_from_terminal_.size(), 0.0);

        thrust::fill(grad_lo_cost_out, grad_lo_cost_out + this->nr_layers(), 0.0);
        thrust::fill(grad_hi_cost_out, grad_hi_cost_out + this->nr_layers(), 0.0);
        
        thrust::device_vector<REAL> grad_omega(1, 0.0);

        // Backprop through min-marginal computation from arc costs and root, terminal costs:
        for (int hop_index = 0; hop_index < this->nr_hops(); hop_index++)
            grad_mm_diff_of_hop(this->lo_cost_.data(), this->hi_cost_.data(), this->deffered_mm_diff_.data(), incoming_grad_mm, grad_lo_cost_out, grad_hi_cost_out, 
                                grad_cost_from_root.data(), grad_cost_from_terminal.data(), 
                                grad_omega.data(), hop_index, 1.0, nullptr, false);

        // mm gradients are backpropagated into arc costs and costs from root/terminal. Now backprop through costs from root/terminal calculation.
        for (int hop_index = 0; hop_index < this->nr_hops(); hop_index++) // Inverse direction as that of backward_run().
            compute_grad_cost_from_terminal(grad_cost_from_terminal.data(), grad_lo_cost_out, grad_hi_cost_out, hop_index);

        for (int hop_index = this->nr_hops() - 1; hop_index >= 0; hop_index--) // Inverse direction as that of forward_run().
            compute_grad_cost_from_root(grad_cost_from_root.data(), grad_lo_cost_out, grad_hi_cost_out, hop_index);
    }

    template<typename REAL>
    struct grad_dual_update_func {
        const int* primal_index;
        const REAL* cur_mm_diff;
        const REAL* dist_weights;
        const REAL* grad_lo_cost;
        const REAL* grad_hi_cost;
        const REAL* delta_lo;
        const REAL* delta_hi;
        REAL* grad_cur_mm_diff;
        REAL* grad_delta_lo;
        REAL* grad_delta_hi;
        REAL* grad_dist_weights;
        const unsigned long num_vars;
        __host__ __device__ void operator()(const int i)
        {
            const int primal = primal_index[i];
            if (primal >= num_vars)
                return; 
            const REAL current_mm = cur_mm_diff[i];
            const REAL current_grad_lo_cost = grad_lo_cost[i];
            const REAL current_grad_hi_cost = grad_hi_cost[i];
            if (current_mm >= 0)
                grad_cur_mm_diff[i] -= current_grad_hi_cost;
            else
                grad_cur_mm_diff[i] += current_grad_lo_cost;

            atomicAdd(&grad_delta_lo[primal], current_grad_lo_cost * dist_weights[i]);
            atomicAdd(&grad_delta_hi[primal], current_grad_hi_cost * dist_weights[i]);

            grad_dist_weights[i] += delta_lo[primal] * current_grad_lo_cost + delta_hi[primal] * current_grad_hi_cost;
        }
    };

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_hop_update_learned_mm_dist(
            const thrust::device_ptr<const REAL> before_update_lo_cost,
            const thrust::device_ptr<const REAL> before_update_hi_cost,
            thrust::device_ptr<REAL> cur_min_marg_diff, // current min-marginals which were subtracted in present iteration.
            thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost and Outputs in-place to compute grad. lo_cost before hop update.
            thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost and Outputs in-place to compute grad. hi_cost before hop update.
            thrust::device_ptr<REAL> grad_cost_from_root, // grad w.r.t cost_from_terminal (size = nr_bdd_nodes()).
            thrust::device_ptr<REAL> grad_cost_from_terminal, // grad w.r.t cost_from_terminal (size = nr_bdd_nodes()).
            thrust::device_ptr<REAL> grad_mm,     // Input: incoming grad w.r.t min-marginal differences of current hop update. Is overwritten by accumulated grad_mm.
            thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto current point.
            thrust::device_ptr<REAL> grad_delta_lo, // To accumulate gradients w.r.t delta lo (should be intialized by zero for first hop)
            thrust::device_ptr<REAL> grad_delta_hi, // To accumulate gradients w.r.t delta hi (should be intialized by zero for first hop)
            const thrust::device_ptr<const REAL> dist_weights,            // Input: distribution weights used in the forward pass.
            const int hop_index, const REAL omega_scalar,
            thrust::device_ptr<REAL> grad_omega,
            const thrust::device_ptr<const REAL> omega_vec)
    {
        const int start_offset = hop_index > 0 ? this->cum_nr_layers_per_hop_dist_[hop_index - 1] : 0;
        const int end_offset = this->cum_nr_layers_per_hop_dist_[hop_index];
        const int num_layers_hop = end_offset - start_offset;

        grad_dual_update_func<REAL> func_grad_dual_update({
                                                thrust::raw_pointer_cast(this->primal_variable_index_.data()  + start_offset),
                                                thrust::raw_pointer_cast(cur_min_marg_diff + start_offset),
                                                thrust::raw_pointer_cast(dist_weights + start_offset),
                                                thrust::raw_pointer_cast(grad_lo_cost + start_offset),
                                                thrust::raw_pointer_cast(grad_hi_cost + start_offset),
                                                thrust::raw_pointer_cast(this->delta_lo_.data()),
                                                thrust::raw_pointer_cast(this->delta_hi_.data()),
                                                thrust::raw_pointer_cast(grad_mm + start_offset),
                                                thrust::raw_pointer_cast(grad_delta_lo),
                                                thrust::raw_pointer_cast(grad_delta_hi),
                                                thrust::raw_pointer_cast(grad_dist_weights_out + start_offset),
                                                this->nr_variables()
                                            });
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + num_layers_hop, func_grad_dual_update);

        // Backprop through mm computation:
        grad_mm_diff_of_hop(before_update_lo_cost, before_update_hi_cost, cur_min_marg_diff, 
                            grad_mm, grad_lo_cost, grad_hi_cost, 
                            grad_cost_from_root, grad_cost_from_terminal, 
                            grad_omega, hop_index, omega_scalar, omega_vec);
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
            {
                grad_def_min_marg[i] = 0.0;
            }
            else
            {
                const REAL def_mm = def_min_marg_diff[i];
                if (def_mm >= 0)
                    grad_def_min_marg[i] = 1.0 * grad_delta_hi[primal];
                else
                    grad_def_min_marg[i] = -1.0 * grad_delta_lo[primal];
            }
        }
    };

    // Compute gradient of forward_iteration_learned_mm_dist.
    // Assumes solver state is set to state before forward_iteration_learned_mm_dist was called. 
    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_forward_iteration_learned_mm_dist(
        thrust::device_ptr<REAL> deferred_min_marg_diff, // deferred min-marginals used in forward pass, output will min-marginals hi. (not useful)
        const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
        thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost current iteration and Outputs in-place to compute grad. lo_cost before iteration.
        thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost current iteration and Outputs in-place to compute grad. hi_cost before iteration.
        thrust::device_ptr<REAL> grad_cost_from_root, // Input: incoming grad w.r.t cost_from_root (size = nr_bdd_nodes()), Outputs in-place to compute grad before iteration.
        thrust::device_ptr<REAL> grad_cost_from_terminal, // Input: incoming grad w.r.t cost_from_terminal (size = nr_bdd_nodes()), Outputs in-place to compute grad before iteration.
        thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. current iteration and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iteration.
        thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto the current point.
        const REAL omega_scalar,
        thrust::device_ptr<REAL> grad_omega,
        const thrust::device_ptr<const REAL> omega_vec)
    {
        thrust::device_vector<REAL> deffered_min_marg_diff_orig(deferred_min_marg_diff, deferred_min_marg_diff + this->nr_layers());

        // Reconstruct the states used in forward pass of this iteration.
        // deferred_min_marg_diff now contains current min-marginal differences
        // and deffered_min_marg_diff_orig contains the deferred ones.
        forward_iteration_learned_mm_dist(dist_weights, deferred_min_marg_diff, omega_scalar, omega_vec);
        // lo_cost_out_, hi_cost_out_ now contain dual costs before this.

        thrust::device_vector<REAL> grad_delta_lo(this->delta_lo_.size(), 0.0);
        thrust::device_vector<REAL> grad_delta_hi(this->delta_hi_.size(), 0.0);

        for (int s = this->nr_hops() - 1; s >= 0; s--)
        {
            compute_grad_cost_from_root(grad_cost_from_root, grad_lo_cost, grad_hi_cost, s);          
            grad_hop_update_learned_mm_dist(this->lo_cost_out_.data(), this->hi_cost_out_.data(), deferred_min_marg_diff, 
                                            grad_lo_cost, grad_hi_cost,
                                            grad_cost_from_root, grad_cost_from_terminal, 
                                            grad_mm, grad_dist_weights_out, 
                                            grad_delta_lo.data(), grad_delta_hi.data(), 
                                            dist_weights, s, omega_scalar, grad_omega, omega_vec);
        }

        // Deferred min-marginals were used to compute delta_lo and delta_hi, perform backprop through this op.
        grad_def_min_marg_func<REAL> func_grad_def_mm({thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                    thrust::raw_pointer_cast(deffered_min_marg_diff_orig.data()),
                                                    thrust::raw_pointer_cast(grad_delta_lo.data()),
                                                    thrust::raw_pointer_cast(grad_delta_hi.data()),
                                                    thrust::raw_pointer_cast(grad_mm),
                                                    this->nr_variables()});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), func_grad_def_mm);
    }

    // Compute gradient of backward_iteration_learned_mm_dist.
    // Assumes solver state is set to state before backward_iteration_learned_mm_dist was called. 
    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_backward_iteration_learned_mm_dist(
        thrust::device_ptr<REAL> deferred_min_marg_diff, // deferred min-marginals used in forward pass, output will min-marginals hi. (not useful)
        const thrust::device_ptr<const REAL> dist_weights, // distribution weights used in the forward pass.
        thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost current iteration and Outputs in-place to compute grad. lo_cost before iteration.
        thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost current iteration and Outputs in-place to compute grad. hi_cost before iteration.
        thrust::device_ptr<REAL> grad_cost_from_root, // Input: incoming grad w.r.t cost_from_root (size = nr_bdd_nodes()), Outputs in-place to compute grad before iteration.
        thrust::device_ptr<REAL> grad_cost_from_terminal, // Input: incoming grad w.r.t cost_from_terminal (size = nr_bdd_nodes()), Outputs in-place to compute grad before iteration.
        thrust::device_ptr<REAL> grad_mm, // Input: incoming grad w.r.t min-marg. diff. current iteration and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iteration.
        thrust::device_ptr<REAL> grad_dist_weights_out, // Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()) and contains valid gradients upto the current point.
        const REAL omega_scalar,
        thrust::device_ptr<REAL> grad_omega,
        const thrust::device_ptr<const REAL> omega_vec)
    {
        thrust::device_vector<REAL> deffered_min_marg_diff_orig(deferred_min_marg_diff, deferred_min_marg_diff + this->nr_layers());

        // Reconstruct the states used in forward pass of this iteration.
        // deferred_min_marg_diff now contains current min-marginal differences
        backward_iteration_learned_mm_dist(dist_weights, deferred_min_marg_diff, omega_scalar, omega_vec);
        // lo_cost_out_, hi_cost_out_ now contain dual costs before this.

        thrust::device_vector<REAL> grad_delta_lo(this->delta_lo_.size(), 0.0);
        thrust::device_vector<REAL> grad_delta_hi(this->delta_hi_.size(), 0.0);

        for (int s = 0; s < this->cum_nr_bdd_nodes_per_hop_dist_.size() - 1; s++)
        {
            compute_grad_cost_from_terminal(grad_cost_from_terminal, grad_lo_cost, grad_hi_cost, s);

            grad_hop_update_learned_mm_dist(this->lo_cost_out_.data(), this->hi_cost_out_.data(), deferred_min_marg_diff, 
                                            grad_lo_cost, grad_hi_cost,
                                            grad_cost_from_root, grad_cost_from_terminal, 
                                            grad_mm, grad_dist_weights_out, 
                                            grad_delta_lo.data(), grad_delta_hi.data(), 
                                            dist_weights, s, omega_scalar, grad_omega, omega_vec);
        }

        // Deferred min-marginals were used to compute delta_lo and delta_hi, perform backprop through this op.
        grad_def_min_marg_func<REAL> func_grad_def_mm({thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                    thrust::raw_pointer_cast(deffered_min_marg_diff_orig.data()),
                                                    thrust::raw_pointer_cast(grad_delta_lo.data()),
                                                    thrust::raw_pointer_cast(grad_delta_hi.data()),
                                                    thrust::raw_pointer_cast(grad_mm),
                                                    this->nr_variables()});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), func_grad_def_mm);
    }

    template<typename REAL>
    __global__ void grad_cost_from_root_indices_cuda(const int cur_num_bdd_nodes, const int start_offset, const int start_offset_next_hop,
                                                const int* const __restrict__ lo_bdd_node_index, 
                                                const int* const __restrict__ hi_bdd_node_index, 
                                                const int* const __restrict__ bdd_node_to_layer_map, 
                                                const REAL* const __restrict__ lo_cost,
                                                const REAL* const __restrict__ hi_cost,
                                                const REAL* const __restrict__ cost_from_root,
                                                int* next_hop_backprop_index)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_node_idx = start_index + start_offset; bdd_node_idx < cur_num_bdd_nodes + start_offset; bdd_node_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_node_idx];
            if (next_lo_node < 0)
                continue; // nothing needs to be done for terminal node.
            
            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
            const int layer_idx = bdd_node_to_layer_map[bdd_node_idx];
            const REAL cur_c_from_root = cost_from_root[bdd_node_idx];

            if(cur_c_from_root + lo_cost[layer_idx] <= cost_from_root[next_lo_node])
            {
                const bool next_lo_bot_sink = lo_bdd_node_index[next_lo_node] == BOT_SINK_INDICATOR_CUDA;
                if (!next_lo_bot_sink)
                    next_hop_backprop_index[next_lo_node - start_offset_next_hop] = bdd_node_idx; // without atomic for randomization.
            }

            if(cur_c_from_root + hi_cost[layer_idx] <= cost_from_root[next_hi_node])
            {
                const bool next_hi_bot_sink = lo_bdd_node_index[next_hi_node] == BOT_SINK_INDICATOR_CUDA;
                if (!next_hi_bot_sink)
                    next_hop_backprop_index[next_hi_node - start_offset_next_hop] = bdd_node_idx; // without atomic for randomization.
            }
        }
    }

    template<typename REAL>
    __global__ void propagate_grad_cost_from_root_cuda(const int cur_num_bdd_nodes, const int start_offset, const int start_offset_next_hop,
                                                const int* const __restrict__ lo_bdd_node_index, 
                                                const int* const __restrict__ hi_bdd_node_index, 
                                                const int* const __restrict__ bdd_node_to_layer_map, 
                                                const int* const __restrict__ next_hop_backprop_index,
                                                REAL* __restrict__ grad_cost_from_root,
                                                REAL* __restrict__ grad_lo_cost,
                                                REAL* __restrict__ grad_hi_cost)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_node_idx = start_index + start_offset; bdd_node_idx < cur_num_bdd_nodes + start_offset; bdd_node_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_node_idx];
            if (next_lo_node < 0)
                continue; // nothing needs to be done for terminal node.

            const int layer_idx = bdd_node_to_layer_map[bdd_node_idx];
            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
            const bool next_lo_bot_sink = lo_bdd_node_index[next_lo_node] == BOT_SINK_INDICATOR_CUDA;
            const bool next_hi_bot_sink = lo_bdd_node_index[next_hi_node] == BOT_SINK_INDICATOR_CUDA;

            if (!next_lo_bot_sink && next_hop_backprop_index[next_lo_node - start_offset_next_hop] == bdd_node_idx)
            {
                const REAL incoming_grad = grad_cost_from_root[next_lo_node];
                assert(isfinite(incoming_grad));
                grad_cost_from_root[bdd_node_idx] += incoming_grad;
                atomicAdd(&grad_lo_cost[layer_idx], incoming_grad);
            }
            if (!next_hi_bot_sink && next_hop_backprop_index[next_hi_node - start_offset_next_hop] == bdd_node_idx)
            {
                const REAL incoming_grad = grad_cost_from_root[next_hi_node];
                assert(isfinite(incoming_grad));
                grad_cost_from_root[bdd_node_idx] += incoming_grad;
                atomicAdd(&grad_hi_cost[layer_idx], incoming_grad);
            }
        }
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::compute_grad_cost_from_root(
        thrust::device_ptr<REAL> grad_cost_from_root,  // incoming gradient of hop_index + 1 root costs is used to compute grad for hop_index root costs.
        thrust::device_ptr<REAL> grad_lo_cost,          // accumulates gradient for hop_index
        thrust::device_ptr<REAL> grad_hi_cost,          // accumulates gradient for hop_index
        const int hop_index)
    {
        assert(hop_index < this->nr_hops());
        const int start_offset = hop_index > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1] : 0;
        const int cur_num_bdd_nodes = this->nr_bdd_nodes(hop_index);
        const int blockCount = ceil(cur_num_bdd_nodes / (float) NUM_THREADS_CUDA);

        thrust::device_vector<int> next_hop_backprop_index(cur_num_bdd_nodes, -1);

        grad_cost_from_root_indices_cuda<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, start_offset, start_offset + cur_num_bdd_nodes,
                                                            thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                            thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                            thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                            thrust::raw_pointer_cast(this->cost_from_root_.data()),
                                                            thrust::raw_pointer_cast(next_hop_backprop_index.data()));

        propagate_grad_cost_from_root_cuda<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, start_offset, start_offset + cur_num_bdd_nodes,
                                                            thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                            thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                            thrust::raw_pointer_cast(next_hop_backprop_index.data()),
                                                            thrust::raw_pointer_cast(grad_cost_from_root),
                                                            thrust::raw_pointer_cast(grad_lo_cost),
                                                            thrust::raw_pointer_cast(grad_hi_cost));

    }

    template<typename REAL>
    __global__ void grad_cost_from_terminal_cuda(const int cur_num_bdd_nodes, const int start_offset, const unsigned int seed,
                                            const int* const __restrict__ lo_bdd_node_index, 
                                            const int* const __restrict__ hi_bdd_node_index, 
                                            const int* const __restrict__ bdd_node_to_layer_map, 
                                            const int* const __restrict__ primal_variable_index, 
                                            const REAL* const __restrict__ lo_cost,
                                            const REAL* const __restrict__ hi_cost,
                                            const REAL* const __restrict__ cost_from_terminal,
                                            REAL* __restrict__ grad_cost_from_terminal,
                                            REAL* __restrict__ grad_lo_cost,
                                            REAL* __restrict__ grad_hi_cost)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;

        for (int bdd_node_idx = start_index + start_offset; bdd_node_idx < cur_num_bdd_nodes + start_offset; bdd_node_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_node_idx];
            if (next_lo_node < 0) // will matter when one row contains multiple BDDs, otherwise the terminal nodes are at the end anyway.
                continue; // nothing needs to be done for terminal node.
            
            const int layer_idx = bdd_node_to_layer_map[bdd_node_idx];

            const REAL current_incoming_grad = grad_cost_from_terminal[bdd_node_idx];
            assert(isfinite(current_incoming_grad));
            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];

            const float lo_path_cost = lo_cost[layer_idx] + cost_from_terminal[next_lo_node];
            const float hi_path_cost = hi_cost[layer_idx] + cost_from_terminal[next_hi_node];
            bool backprop_lo = lo_path_cost < hi_path_cost;
            if (hi_path_cost == lo_path_cost)
            {
                thrust::default_random_engine rng;
                thrust::uniform_int_distribution<int> dist(0, 1);
                rng.discard(bdd_node_idx + seed);
                const int gradient_index = dist(rng);
                backprop_lo = gradient_index == 0;
            }
            if (backprop_lo)
            { // min is coming from lo arc:
                atomicAdd(&grad_cost_from_terminal[next_lo_node], current_incoming_grad);
                atomicAdd(&grad_lo_cost[layer_idx], current_incoming_grad);
            }
            else
            {
                atomicAdd(&grad_cost_from_terminal[next_hi_node], current_incoming_grad);
                atomicAdd(&grad_hi_cost[layer_idx], current_incoming_grad);
            }
        }
    }

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::compute_grad_cost_from_terminal(
        thrust::device_ptr<REAL> grad_cost_from_terminal,  // incoming gradient of hop_index terminal costs is used to compute grad for hop_index + 1 terminal costs.
        thrust::device_ptr<REAL> grad_lo_cost,          // accumulates gradient for hop_index
        thrust::device_ptr<REAL> grad_hi_cost,          // accumulates gradient for hop_index
        const int hop_index)
    {
        assert(hop_index < this->nr_hops());
        const int start_offset = hop_index > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1] : 0;
        const int cur_num_bdd_nodes = this->nr_bdd_nodes(hop_index);
        const int blockCount = ceil(cur_num_bdd_nodes / (float) NUM_THREADS_CUDA);
        grad_cost_from_terminal_cuda<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, start_offset, time(NULL),
                                                        thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                        thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                        thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                        thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                        thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                        thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                        thrust::raw_pointer_cast(this->cost_from_terminal_.data()),
                                                        thrust::raw_pointer_cast(grad_cost_from_terminal),
                                                        thrust::raw_pointer_cast(grad_lo_cost),
                                                        thrust::raw_pointer_cast(grad_hi_cost));
    }

    template<typename REAL>
    __global__ void grad_min_marginals_indices_cuda(const int cur_num_bdd_nodes, const int start_offset, const int start_offset_layer,
                                                    const int* const __restrict__ lo_bdd_node_index, 
                                                    const int* const __restrict__ hi_bdd_node_index, 
                                                    const int* const __restrict__ bdd_node_to_layer_map, 
                                                    const REAL* const __restrict__ lo_cost,
                                                    const REAL* const __restrict__ hi_cost,
                                                    const REAL* const __restrict__ cost_from_root,
                                                    const REAL* const __restrict__ cost_from_terminal,
                                                    const REAL* const __restrict__ mm_lo_local, 
                                                    const REAL* const __restrict__ mm_hi_local,
                                                    int* __restrict__ to_accumulate_lo,
                                                    int* __restrict__ to_accumulate_hi,
                                                    int* __restrict__ num_adds_lo,
                                                    int* __restrict__ num_adds_hi)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_node_idx = start_index + start_offset; bdd_node_idx < cur_num_bdd_nodes + start_offset; bdd_node_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_node_idx];
            if (next_lo_node < 0) // will matter when one row contains multiple BDDs, otherwise the terminal nodes are at the end anyway.
                continue; // nothing needs to be done for terminal node.

            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
            const REAL cur_c_from_root = cost_from_root[bdd_node_idx];
            const int layer_idx = bdd_node_to_layer_map[bdd_node_idx];
            const REAL current_lo_path_cost = cur_c_from_root + lo_cost[layer_idx] + cost_from_terminal[next_lo_node];
            if (current_lo_path_cost <= mm_lo_local[layer_idx - start_offset_layer])
            {
                atomicAdd(&num_adds_lo[layer_idx - start_offset_layer], 1);
                to_accumulate_lo[start_index] = 1;
            }
            const REAL current_hi_path_cost = cur_c_from_root + hi_cost[layer_idx] + cost_from_terminal[next_hi_node];
            if (current_hi_path_cost <= mm_hi_local[layer_idx - start_offset_layer])
            {
                atomicAdd(&num_adds_hi[layer_idx - start_offset_layer], 1);
                to_accumulate_hi[start_index] = 1;
            }
        }
    }

    struct pick_subgradient_grad_mm_nodes {
        int* num_adds;
        const int* primal_index;
        unsigned int N; // to generate different random number each time.
        const unsigned long num_vars;
        __host__ __device__ void operator()(const int i)
        {
            if (primal_index[i] >= num_vars)
                return;

            thrust::default_random_engine rng;
            const int maxv = num_adds[i];
            assert(maxv > 0);
            thrust::uniform_int_distribution<int> dist(0, maxv - 1);
            rng.discard(i + N);
            const int gradient_index = dist(rng);
            num_adds[i] = gradient_index;
        }
    };

    template<typename REAL>
    struct grad_mm_bdd_node_costs_func {
        const REAL omega_scalar;
        const int start_offset;
        const int start_offset_layer;
        const REAL* grad_mm_diff;
        const int* bdd_node_to_layer_map;
        const int* lo_bdd_node_index;
        const int* hi_bdd_node_index;
        const int* grad_index_lo;
        const int* grad_index_hi;
        const int* to_accumulate_lo;
        const int* to_accumulate_hi;
        REAL* grad_cost_from_root;
        REAL* grad_cost_from_terminal;
        __host__ __device__ void operator()(const int bdd_node_idx)
        {
            const int next_lo_node = lo_bdd_node_index[bdd_node_idx];
            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
            if (next_lo_node < 0)
                return; 

            const int layer_idx = bdd_node_to_layer_map[bdd_node_idx];
            const REAL incoming_grad = omega_scalar * grad_mm_diff[layer_idx];
            assert(isfinite(incoming_grad));

            const int local_node_idx = bdd_node_idx - start_offset;
            const int cur_acc_lo = to_accumulate_lo[local_node_idx];
            if (cur_acc_lo - 1 == grad_index_lo[layer_idx])
            {
                const bool acc_lo_due_first_node = local_node_idx == 0 || layer_idx != bdd_node_to_layer_map[bdd_node_idx - 1];
                const bool acc_lo_due_rest = local_node_idx != 0 && cur_acc_lo > to_accumulate_lo[local_node_idx - 1]; 

                if (acc_lo_due_first_node || acc_lo_due_rest)
                {
                    grad_cost_from_root[bdd_node_idx] -= incoming_grad;
                    atomicAdd(&grad_cost_from_terminal[next_lo_node], -incoming_grad);
                }
            }
            const int cur_acc_hi = to_accumulate_hi[local_node_idx];
            if (cur_acc_hi - 1 == grad_index_hi[layer_idx])
            {
                const bool acc_hi_due_first_node = local_node_idx == 0 || layer_idx != bdd_node_to_layer_map[bdd_node_idx - 1];
                const bool acc_hi_due_rest = local_node_idx != 0 && cur_acc_hi > to_accumulate_hi[local_node_idx - 1]; 

                if (acc_hi_due_first_node || acc_hi_due_rest)
                {
                    grad_cost_from_root[bdd_node_idx] += incoming_grad;
                    atomicAdd(&grad_cost_from_terminal[next_hi_node], incoming_grad);
                }
            }
        }
    };

    template<typename REAL>
    struct grad_mm_bdd_arcs_func {
        const REAL omega_scalar;
        const int start_offset_layer;
        const unsigned long num_vars;
        const int* primal_index;
        const REAL* grad_mm_diff;
        REAL* grad_lo_cost;
        REAL* grad_hi_cost;
        __host__ __device__ void operator()(const int layer_idx)
        {
            if (primal_index[layer_idx] >= num_vars) //terminal nodes.
                return;

            const REAL incoming_grad = omega_scalar * grad_mm_diff[layer_idx];
            assert(isfinite(incoming_grad));
            grad_lo_cost[layer_idx] -= incoming_grad;
            grad_hi_cost[layer_idx] += incoming_grad;
        }
    };

    template<typename REAL>
    struct grad_omega_func {
        const int* primal_index;
        const REAL* grad_mm;
        const REAL* mm_lo;
        const REAL* mm_hi;
        REAL* out;
        const unsigned long num_vars;
        __host__ __device__ void operator()(const int i)
        {
            if (primal_index[i] >= num_vars)
                out[i] = 0;
            else
                out[i] += grad_mm[i] * (mm_hi[i] - mm_lo[i]);
        }
    };

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_mm_diff_of_hop(
        const thrust::device_ptr<const REAL> before_update_lo_cost,
        const thrust::device_ptr<const REAL> before_update_hi_cost,
        thrust::device_ptr<REAL> memory_for_mm_hi,
        thrust::device_ptr<REAL> incoming_grad_mm_diff_hop,
        thrust::device_ptr<REAL> grad_lo_cost,
        thrust::device_ptr<REAL> grad_hi_cost,
        thrust::device_ptr<REAL> grad_cost_from_root,
        thrust::device_ptr<REAL> grad_cost_from_terminal,
        thrust::device_ptr<REAL> grad_omega,
        const int hop_index, const REAL omega_scalar, 
        const thrust::device_ptr<const REAL> omega_vec,
        const bool backprop_omega)
    {
        const int start_offset_layer = hop_index > 0 ? this->cum_nr_layers_per_hop_dist_[hop_index - 1]: 0;
        const int end_offset_layer = this->cum_nr_layers_per_hop_dist_[hop_index];

        thrust::fill(this->mm_lo_local_.begin(), this->mm_lo_local_.end(), CUDART_INF_F_HOST);
        thrust::fill(memory_for_mm_hi + start_offset_layer, memory_for_mm_hi + end_offset_layer, CUDART_INF_F_HOST);

        // Populate mm_hi and mm_lo separately w.r.t before update costs and do not convert to difference.
        this->min_marginals_from_directional_costs(hop_index, omega_scalar, memory_for_mm_hi + start_offset_layer, 
                                                    before_update_lo_cost, before_update_hi_cost, omega_vec, false);
        // memory_for_mm_hi + start_offset_layer and mm_lo_local_ contains mm hi and mm lo resp. for the current hop.

        const int cur_num_layers = end_offset_layer - start_offset_layer;

        if (omega_vec.get())
        {   // mm differences during forward pass were multiplied by vector valued omega thus scale the gradients by same factor:
            assert(omega_scalar == 1.0);
            thrust::transform(incoming_grad_mm_diff_hop + start_offset_layer, 
                            incoming_grad_mm_diff_hop + end_offset_layer, 
                            omega_vec + start_offset_layer, 
                            incoming_grad_mm_diff_hop + start_offset_layer, thrust::multiplies<REAL>());
        }

        const int num_nodes_processed = hop_index > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1] : 0;
        const int cur_num_bdd_nodes = this->nr_bdd_nodes(hop_index);
        const int blockCount = ceil(cur_num_bdd_nodes / (float) NUM_THREADS_CUDA);

        thrust::device_vector<int> num_adds_lo(cur_num_layers, 0);
        thrust::device_vector<int> num_adds_hi(cur_num_layers, 0);

        thrust::device_vector<int> to_accumulate_lo(cur_num_bdd_nodes, 0);
        thrust::device_vector<int> to_accumulate_hi(cur_num_bdd_nodes, 0);

        grad_min_marginals_indices_cuda<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, num_nodes_processed, start_offset_layer,
                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                thrust::raw_pointer_cast(before_update_lo_cost),
                                                thrust::raw_pointer_cast(before_update_hi_cost),
                                                thrust::raw_pointer_cast(this->cost_from_root_.data()),
                                                thrust::raw_pointer_cast(this->cost_from_terminal_.data()),
                                                thrust::raw_pointer_cast(this->mm_lo_local_.data()),
                                                thrust::raw_pointer_cast(memory_for_mm_hi + start_offset_layer),
                                                thrust::raw_pointer_cast(to_accumulate_lo.data()),
                                                thrust::raw_pointer_cast(to_accumulate_hi.data()),
                                                thrust::raw_pointer_cast(num_adds_lo.data()),
                                                thrust::raw_pointer_cast(num_adds_hi.data()));

        thrust::inclusive_scan_by_key(this->bdd_node_to_layer_map_.begin() + num_nodes_processed, 
                                    this->bdd_node_to_layer_map_.begin() + num_nodes_processed + cur_num_bdd_nodes,
                                    to_accumulate_lo.begin(), to_accumulate_lo.begin());

        thrust::inclusive_scan_by_key(this->bdd_node_to_layer_map_.begin() + num_nodes_processed, 
                                    this->bdd_node_to_layer_map_.begin() + num_nodes_processed + cur_num_bdd_nodes,
                                    to_accumulate_hi.begin(), to_accumulate_hi.begin());

        unsigned int seed = time(NULL);
        pick_subgradient_grad_mm_nodes pick_grad_lo({thrust::raw_pointer_cast(num_adds_lo.data()), 
                                                thrust::raw_pointer_cast(this->primal_variable_index_.data() + start_offset_layer),
                                                seed, this->nr_variables()});
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + num_adds_lo.size(), pick_grad_lo);

        seed = time(NULL);
        pick_subgradient_grad_mm_nodes pick_grad_hi({thrust::raw_pointer_cast(num_adds_hi.data()),
                                                thrust::raw_pointer_cast(this->primal_variable_index_.data() + start_offset_layer),
                                                seed, this->nr_variables()});
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + num_adds_hi.size(), pick_grad_hi);

        grad_mm_bdd_node_costs_func<REAL> grad_mm_bdd_node_costs({omega_scalar, num_nodes_processed, start_offset_layer,
                                                thrust::raw_pointer_cast(incoming_grad_mm_diff_hop),
                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(num_adds_lo.data()),
                                                thrust::raw_pointer_cast(num_adds_hi.data()),
                                                thrust::raw_pointer_cast(to_accumulate_lo.data()),
                                                thrust::raw_pointer_cast(to_accumulate_hi.data()),
                                                thrust::raw_pointer_cast(grad_cost_from_root),
                                                thrust::raw_pointer_cast(grad_cost_from_terminal)});

        thrust::for_each(thrust::make_counting_iterator<int>(0) + num_nodes_processed, 
                        thrust::make_counting_iterator<int>(0) + num_nodes_processed + cur_num_bdd_nodes,
                        grad_mm_bdd_node_costs);

        grad_mm_bdd_arcs_func<REAL> grad_mm_bdd_arcs({omega_scalar, start_offset_layer, this->nr_variables(),
                                                thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                thrust::raw_pointer_cast(incoming_grad_mm_diff_hop),
                                                thrust::raw_pointer_cast(grad_lo_cost),
                                                thrust::raw_pointer_cast(grad_hi_cost)});
        thrust::for_each(thrust::make_counting_iterator<int>(0) + start_offset_layer, thrust::make_counting_iterator<int>(0) + end_offset_layer, grad_mm_bdd_arcs);
                                            
        if (!backprop_omega)
            return; 

        if (!omega_vec.get()) // omega_scalar was used.
        {
            // Compute grad omega. 
            thrust::device_vector<REAL> grad_mm_multi_mm_diff(cur_num_layers, 0.0); // store the elements after pointwise multiplication
            grad_omega_func<REAL> grad_omega_calc({
                                            thrust::raw_pointer_cast(this->primal_variable_index_.data()  + start_offset_layer),
                                            thrust::raw_pointer_cast(incoming_grad_mm_diff_hop + start_offset_layer),
                                            thrust::raw_pointer_cast(this->mm_lo_local_.data()),
                                            thrust::raw_pointer_cast(memory_for_mm_hi + start_offset_layer),
                                            thrust::raw_pointer_cast(grad_mm_multi_mm_diff.data()),
                                            this->nr_variables()});
            thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + cur_num_layers, grad_omega_calc);
            grad_omega[0] += thrust::reduce(grad_mm_multi_mm_diff.begin(), grad_mm_multi_mm_diff.end());
        }
        else
        {
            grad_omega_func<REAL> grad_omega_calc({
                                            thrust::raw_pointer_cast(this->primal_variable_index_.data()  + start_offset_layer),
                                            thrust::raw_pointer_cast(incoming_grad_mm_diff_hop + start_offset_layer),
                                            thrust::raw_pointer_cast(this->mm_lo_local_.data()),
                                            thrust::raw_pointer_cast(memory_for_mm_hi + start_offset_layer),
                                            thrust::raw_pointer_cast(grad_omega + start_offset_layer),
                                            this->nr_variables()});
            thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + cur_num_layers, grad_omega_calc);
        }
    }
    
    template<typename REAL>
    struct grad_dist_deffered_mm_diff_func {
        const int* primal_index;
        const REAL* mm_diff;
        const REAL* grad_lo_cost;
        const REAL* grad_hi_cost;
        REAL* grad_deff_mm;
        const unsigned long num_vars;
        __host__ __device__ void operator()(const int i)
        {
            const int primal = primal_index[i];
            if (primal >= num_vars)
            {
                grad_deff_mm[i] = 0;
                return; 
            }
            const REAL cur_mm_diff = mm_diff[i];
            if (cur_mm_diff >= 0)
                grad_deff_mm[i] = grad_hi_cost[i];
            else
                grad_deff_mm[i] = -grad_lo_cost[i];
        }
    };

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_distribute_delta(
        thrust::device_ptr<REAL> grad_lo_cost, // Input: incoming grad w.r.t lo_cost which were output after distributing delta.
        thrust::device_ptr<REAL> grad_hi_cost, // Input: incoming grad w.r.t hi_cost which were output after distributing delta.
        thrust::device_ptr<REAL> grad_deff_mm // Output: contains grad w.r.t deff. min-marginal differences.
    )
    {
        // Nothing to do for grad_lo_cost, grad_hi_cost since Jacobian is identity. Only need to compute grad_dist_weights_out. 
        grad_dist_deffered_mm_diff_func<REAL> func_grad_dist({
                            thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                            thrust::raw_pointer_cast(this->deffered_mm_diff_.data()),
                            thrust::raw_pointer_cast(grad_lo_cost),
                            thrust::raw_pointer_cast(grad_hi_cost),
                            thrust::raw_pointer_cast(grad_deff_mm),
                            this->nr_variables()});
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), func_grad_dist);
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
    void bdd_cuda_learned_mma<REAL>::grad_cost_perturbation(
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

    template<typename REAL>
    struct scale_grad_lb {
        const int* primal_index;
        const int* bdd_index;
        const REAL* incoming_grad_lb;
        REAL* grad_lo;
        REAL* grad_hi;
        const size_t nr_vars;
        const bool account_for_constant;
        __host__ __device__ void operator()(const int layer_index)
        {
            const int primal = primal_index[layer_index];
            if (primal >= nr_vars)
            {
                grad_hi[layer_index] = 0.0;
                grad_lo[layer_index] = 0.0;
            }
            else
            {
                const int current_bdd_index = bdd_index[layer_index];
                const REAL grad_lb = incoming_grad_lb[current_bdd_index];
                const REAL current_grad_hi = grad_hi[layer_index];
                grad_hi[layer_index] = current_grad_hi * grad_lb;
                grad_lo[layer_index] = (1.0 - current_grad_hi) * grad_lb;
            }
        }
    };

    template<typename REAL>
    void bdd_cuda_learned_mma<REAL>::grad_lower_bound_per_bdd(
        thrust::device_ptr<REAL> grad_lb_per_bdd, // Input: incoming grad w.r.t lower bound per BDD.
        thrust::device_ptr<REAL> grad_lo_cost_out, // Gradients w.r.t lo costs
        thrust::device_ptr<REAL> grad_hi_cost_out // Gradients w.r.t hi costs
    )
    {
        this->bdds_solution_cuda(grad_hi_cost_out);
        // Now multiply each BDD solution by corresponding gradient dL / d lb_per_bdd.
        scale_grad_lb<REAL> func({thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                thrust::raw_pointer_cast(this->bdd_index_.data()), 
                                thrust::raw_pointer_cast(grad_lb_per_bdd), 
                                thrust::raw_pointer_cast(grad_lo_cost_out),
                                thrust::raw_pointer_cast(grad_hi_cost_out),
                                this->nr_variables()});
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), func);
    }

    template class bdd_cuda_learned_mma<float>;
    template class bdd_cuda_learned_mma<double>;
}
