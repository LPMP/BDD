#include "bdd_cuda_parallel_mma.h"
#include "cuda_utils.h"
#include <chrono>

namespace LPMP {

    template<typename REAL>
    bdd_cuda_parallel_mma<REAL>::bdd_cuda_parallel_mma(const BDD::bdd_collection& bdd_col) : bdd_cuda_base<REAL>(bdd_col)
    {
        init();
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::init()
    {
        mm_lo_local_ = thrust::device_vector<REAL>(*std::max_element(this->cum_nr_layers_per_hop_dist_.begin(), this->cum_nr_layers_per_hop_dist_.end())); // size of largest layer.

        // Copy from arc costs because it contains infinity for arcs to bot sink
        hi_cost_out_ = thrust::device_vector<REAL>(this->hi_cost_);
        lo_cost_out_ = thrust::device_vector<REAL>(this->lo_cost_);
    }

    template<typename REAL>
    struct compute_mm_diff_flush_mm_lo {
        REAL omega;
        __device__ void operator()(const thrust::tuple<REAL&, REAL&> t) const
        {
            REAL& mm_hi = thrust::get<0>(t);
            REAL& mm_lo = thrust::get<1>(t);
            if (!isfinite(mm_hi) || !isfinite(mm_lo))
                mm_hi = 0.0; // Set mm_difference to zero to not create NaNs in costs.
            else
                mm_hi = omega * (mm_hi - mm_lo);
            mm_lo = CUDART_INF_F;
        }
    };

    template<typename REAL>
    struct compute_mm_diff_flush_mm_lo_with_omega_vec {
        __device__ void operator()(const thrust::tuple<REAL&, REAL&, const REAL> t) const
        {
            REAL& mm_hi = thrust::get<0>(t);
            REAL& mm_lo = thrust::get<1>(t);
            const REAL omega = thrust::get<2>(t);
            if (!isfinite(mm_hi) || !isfinite(mm_lo))
                mm_hi = 0.0; // Set mm_difference to zero to not create NaNs in costs.
            else
                mm_hi = omega * (mm_hi - mm_lo);
            mm_lo = CUDART_INF_F;
        }
    };

    template<typename REAL>
    __global__ void min_marginals_from_directional_costs_cuda(const int cur_num_bdd_nodes, const int start_offset, const int start_offset_layer,
                                                            const int* const __restrict__ lo_bdd_node_index, 
                                                            const int* const __restrict__ hi_bdd_node_index, 
                                                            const int* const __restrict__ bdd_node_to_layer_map, 
                                                            const REAL* const __restrict__ lo_cost,
                                                            const REAL* const __restrict__ hi_cost,
                                                            const REAL* const __restrict__ cost_from_root,
                                                            const REAL* const __restrict__ cost_from_terminal,
                                                            REAL* __restrict__ mm_lo_local, REAL* __restrict__ mm_hi)
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

            atomicMin(&mm_lo_local[layer_idx - start_offset_layer], cur_c_from_root + lo_cost[layer_idx] + cost_from_terminal[next_lo_node]);
            atomicMin(&mm_hi[layer_idx], cur_c_from_root + hi_cost[layer_idx] + cost_from_terminal[next_hi_node]);
        }
    }

    // This function does not need lo_path_costs and hi_path_costs to compute min-marginals. Writes min-marginal differences in GPU array referenced by mm_diff_ptr at 
    // locations corresponding to hop_index.
    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::min_marginals_from_directional_costs(const int hop_index, const REAL omega_scalar, thrust::device_ptr<REAL> mm_diff_ptr, const thrust::device_ptr<const REAL> omega_vec)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        
        const int num_nodes_processed = hop_index > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1] : 0;
        const int end_node = this->cum_nr_bdd_nodes_per_hop_dist_[hop_index];
        const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[hop_index] - num_nodes_processed;
        const int blockCount = ceil(cur_num_bdd_nodes / (REAL) NUM_THREADS_CUDA);

        const int start_offset_layer = hop_index > 0 ? this->cum_nr_layers_per_hop_dist_[hop_index - 1]: 0;
        const int end_offset_layer = this->cum_nr_layers_per_hop_dist_[hop_index];
        const int cur_num_layers = end_offset_layer - start_offset_layer;

        min_marginals_from_directional_costs_cuda<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, num_nodes_processed, start_offset_layer,
                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                thrust::raw_pointer_cast(this->cost_from_root_.data()),
                                                thrust::raw_pointer_cast(this->cost_from_terminal_.data()),
                                                thrust::raw_pointer_cast(mm_lo_local_.data()),
                                                thrust::raw_pointer_cast(mm_diff_ptr));

        thrust::device_ptr<REAL> mm_lo_start = mm_lo_local_.data();

        if (!omega_vec.get()) // Uses omega as a scalar value.
        {
            auto first = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_ptr + start_offset_layer, mm_lo_start));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_ptr + end_offset_layer, mm_lo_start + cur_num_layers));
            thrust::for_each(first, last, compute_mm_diff_flush_mm_lo<REAL>({omega_scalar})); // Convert to min-marginal difference and set mm_lo_local_ to inf.
        }
        else    // Assumes a value per layer is given which is going to be used to damp min-marginal differences.
        {
            assert(omega_scalar == 1.0);
            auto first = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_ptr + start_offset_layer, mm_lo_start, omega_vec + start_offset_layer));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_ptr + end_offset_layer, mm_lo_start + cur_num_layers, omega_vec + end_offset_layer));
            thrust::for_each(first, last, compute_mm_diff_flush_mm_lo_with_omega_vec<REAL>()); // Convert to min-marginal difference and set mm_lo_local_ to inf.
        }
        #ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
        #endif
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::min_marginals_from_directional_costs(const int hop_index, const REAL omega)
    {
        min_marginals_from_directional_costs(hop_index, omega, this->deffered_mm_diff_.data());
    }


    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::iteration(const REAL omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        if(delta_lo_hi_.size() == 0)
            delta_lo_hi_ = thrust::device_vector<REAL>(this->nr_variables() * 2, 0.0);

        forward_mm(omega, delta_lo_hi_);
        normalize_delta(delta_lo_hi_);
        backward_mm(omega, delta_lo_hi_);
        normalize_delta(delta_lo_hi_);
    }

    // arguments:
    // in:
    // delta_lo_hi: in: values to add to current costs
    // mm_diff:
    // lo_cost_in, hi_cost_in: base costs before adding delta
    // lo_cost_out, hi_cost_out: in costs + delta averaged by mm_diff
    // out:
    // cost_from_root: 
    // analoguously for backward_step_with_solve
    template<typename REAL>
    __global__ void forward_step_with_solve(const int cur_num_bdd_nodes, const int start_offset,
                                const int* const __restrict__ lo_bdd_node_index, 
                                const int* const __restrict__ hi_bdd_node_index, 
                                const int* const __restrict__ bdd_node_to_layer_map, 
                                const int* const __restrict__ primal_variable_index, 
                                const REAL* const __restrict__ delta_lo_hi,
                                const REAL* const __restrict__ mm_diff,
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
            assert(isfinite(cur_mm_diff_hi_lo));
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const REAL cur_lo_cost = lo_cost_in[layer_idx] + min(cur_mm_diff_hi_lo, 0.0f) + delta_lo_hi[2 * cur_primal_idx];
            const REAL cur_c_from_root = cost_from_root[bdd_node_idx];

            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root + cur_lo_cost);

            const REAL cur_hi_cost = hi_cost_in[layer_idx] + min(-cur_mm_diff_hi_lo, 0.0f) + delta_lo_hi[2 * cur_primal_idx + 1];
            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + cur_hi_cost);

            assert(isfinite(cur_lo_cost));
            assert(isfinite(cur_hi_cost));
            lo_cost_out[layer_idx] = cur_lo_cost;
            hi_cost_out[layer_idx] = cur_hi_cost;
        }
    }

    template<typename REAL>
        void bdd_cuda_parallel_mma<REAL>::forward_mm(const REAL omega, thrust::device_vector<REAL>& delta_lo_hi)
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("cuda mma forward mma");
            if(!this->backward_state_valid_)
                this->backward_run(false); //For the first iteration need to have costs from terminal. 

            assert(delta_lo_hi.size() == 2 * this->nr_variables());

        // Clear states.
        this->flush_costs_from_root();
        flush_mm(this->deffered_mm_diff_.data());

        const int num_steps = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        for (int s = 0; s < num_steps; s++)
        {
            // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
            min_marginals_from_directional_costs(s, omega);

            const int num_nodes_processed = s > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[s - 1] : 0;
            const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            const int blockCount = ceil(cur_num_bdd_nodes / (float) NUM_THREADS_CUDA);

            // 2. Subtract from hi_costs, update costs from root.
            forward_step_with_solve<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, num_nodes_processed,
                                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                                thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                                // in
                                                                thrust::raw_pointer_cast(delta_lo_hi.data()),
                                                                thrust::raw_pointer_cast(this->deffered_mm_diff_.data()),
                                                                thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                                thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                                // out
                                                                thrust::raw_pointer_cast(this->lo_cost_out_.data()),
                                                                thrust::raw_pointer_cast(this->hi_cost_out_.data()),
                                                                thrust::raw_pointer_cast(this->cost_from_root_.data()));
        }
        thrust::swap(this->lo_cost_, lo_cost_out_);
        thrust::swap(this->hi_cost_, hi_cost_out_);

        compute_delta(this->deffered_mm_diff_.data(), delta_lo_hi.data());

        this->forward_state_valid_ = true;
        this->flush_backward_states();

        #ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
        #endif
    }

    template<typename REAL>
    __global__ void backward_step_with_solve(const int cur_num_bdd_nodes, const int start_offset, 
                                            const int* const __restrict__ lo_bdd_node_index, 
                                            const int* const __restrict__ hi_bdd_node_index, 
                                            const int* const __restrict__ bdd_node_to_layer_map, 
                                            const int* const __restrict__ primal_variable_index, 
                                            const REAL* const __restrict__ delta_lo_hi,
                                            const REAL* const __restrict__ mm_diff,
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
            assert(isfinite(cur_mm_diff_hi_lo));
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const REAL cur_hi_cost = hi_cost_in[layer_idx] + (min(-cur_mm_diff_hi_lo, 0.0f)) + (delta_lo_hi[2 * cur_primal_idx + 1]);
            const REAL cur_lo_cost = lo_cost_in[layer_idx] + (min(cur_mm_diff_hi_lo, 0.0f)) + (delta_lo_hi[2 * cur_primal_idx]);

            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];

            // Update costs from terminal:
            cost_from_terminal[bdd_node_idx] = min(cur_hi_cost + cost_from_terminal[next_hi_node], cur_lo_cost + cost_from_terminal[next_lo_node]);
            assert(isfinite(cur_lo_cost));
            assert(isfinite(cur_hi_cost));

            lo_cost_out[layer_idx] = cur_lo_cost;
            hi_cost_out[layer_idx] = cur_hi_cost;
        }
    }

    template<typename REAL>
        void bdd_cuda_parallel_mma<REAL>::backward_mm(const REAL omega, thrust::device_vector<REAL>& delta_lo_hi)
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("cuda mma backward mma");
            assert(this->forward_state_valid_); 

            assert(delta_lo_hi.size() == 2 * this->nr_variables());

            flush_mm(this->deffered_mm_diff_.data());

            for (int s = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 2; s >= 0; s--)
            {
                // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
                min_marginals_from_directional_costs(s, omega);

                const int start_offset = s > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[s - 1] : 0;

                const int cur_num_bdd_nodes = this->nr_bdd_nodes(s);
                const int blockCount = ceil(cur_num_bdd_nodes / (REAL) NUM_THREADS_CUDA);

                // 2. Subtract from hi_costs, update costs from terminal.
                backward_step_with_solve<<<blockCount, NUM_THREADS_CUDA>>>(cur_num_bdd_nodes, start_offset,
                        thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                        thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                        thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                        thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                        thrust::raw_pointer_cast(delta_lo_hi.data()),
                        thrust::raw_pointer_cast(this->deffered_mm_diff_.data()),
                        thrust::raw_pointer_cast(this->lo_cost_.data()),
                        thrust::raw_pointer_cast(this->hi_cost_.data()),
                        thrust::raw_pointer_cast(lo_cost_out_.data()),
                        thrust::raw_pointer_cast(hi_cost_out_.data()),
                        thrust::raw_pointer_cast(this->cost_from_terminal_.data()));
            }
            thrust::swap(this->lo_cost_, lo_cost_out_);
            thrust::swap(this->hi_cost_, hi_cost_out_);

            compute_delta(this->deffered_mm_diff_.data(), delta_lo_hi.data());

            this->flush_forward_states();
            this->backward_state_valid_ = true;

#ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
#endif
        }

    template<typename REAL> struct pos_part
    {
        __host__ __device__ REAL operator()(const REAL x) { return max(x, (REAL) 0); }
    };

    template<typename REAL> struct abs_neg_part
    {
        __host__ __device__ REAL operator()(const REAL x) { return -min(x, (REAL) 0); }
    };

    template<typename REAL>
    struct compute_delta_atomic {
        const int* primal_index;
        const REAL* mm_to_distribute;
        REAL* delta_lo_hi;
        const unsigned long num_vars;
        __host__ __device__ void operator()(const int i)
        {
            const int cur_primal = primal_index[i];
            if (cur_primal < num_vars)
            {
                const REAL cur_mm = mm_to_distribute[i];
                if (cur_mm > 0)
                    atomicAdd(&delta_lo_hi[2 * cur_primal + 1], cur_mm);
                else if (cur_mm < 0)
                    atomicAdd(&delta_lo_hi[2 * cur_primal], -cur_mm);
            }
        }
    };

    // TODO: use device_vector?
    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::compute_delta(
            const thrust::device_ptr<const REAL> mm_to_distribute,
            thrust::device_ptr<REAL> delta_lo_hi) const
    {

        thrust::fill(delta_lo_hi, delta_lo_hi + 2 * this->nr_variables(), 0.0);

        compute_delta_atomic<REAL> compute_delta_atomic_func({
                                            thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                            thrust::raw_pointer_cast(mm_to_distribute),
                                            thrust::raw_pointer_cast(delta_lo_hi),
                                            this->nr_variables()});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + this->nr_layers(), compute_delta_atomic_func);

        // auto first_val = thrust::make_zip_iterator(thrust::make_tuple(
        //     thrust::make_permutation_iterator(thrust::make_transform_iterator(mm_to_distribute, pos_part<REAL>()), this->primal_variable_sorting_order_.begin()),
        //     thrust::make_permutation_iterator(thrust::make_transform_iterator(mm_to_distribute, abs_neg_part<REAL>()), this->primal_variable_sorting_order_.begin())));

        // auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(delta_hi, delta_lo));

        // thrust::equal_to<int> binary_pred;
        // auto new_end = thrust::reduce_by_key(this->primal_variable_index_sorted_.begin(), this->primal_variable_index_sorted_.end() - this->nr_bdds_, first_val, 
        //                     thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_sum());
        //assert(thrust::distance(first_out_val, new_end.second) == delta_hi.size());
        // thrust::reduce_by_key(thrust::make_permutation_iterator(this->primal_variable_index_.begin(), primal_variable_sorting_order_.begin()),
        //                     thrust::make_permutation_iterator(this->primal_variable_index_.end(), primal_variable_sorting_order_.end()), first_val, 
        //                     thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_sum()); // Uses less memory but slower.
    }

    template<typename REAL>
    struct normalize_delta_st {
        const int* num_bdds_per_primal_var;
        REAL* delta_lo_hi;
        __host__ __device__ void operator()(const int i) const
        {
            const int norm = num_bdds_per_primal_var[i / 2];
            delta_lo_hi[i] /= norm;
        }
    };

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::normalize_delta(thrust::device_vector<REAL>& delta_lo_hi) const
    {
        normalize_delta_st<REAL> normalize_delta_func({
                                    thrust::raw_pointer_cast(this->num_bdds_per_var_.data()),
                                    thrust::raw_pointer_cast(delta_lo_hi.data())});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + 2 * this->num_bdds_per_var_.size(), normalize_delta_func);
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::flush_mm(thrust::device_ptr<REAL> mm_diff_ptr)
    {   // Makes min marginals INF so that they can be populated again by in-place minimization
        thrust::fill(mm_lo_local_.begin(), mm_lo_local_.end(), CUDART_INF_F_HOST);
        thrust::fill(mm_diff_ptr, mm_diff_ptr + this->nr_layers(), CUDART_INF_F_HOST);
    }

    template class bdd_cuda_parallel_mma<float>;
    template class bdd_cuda_parallel_mma<double>;
}
