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
        delta_lo_ = thrust::device_vector<REAL>(this->nr_variables());
        delta_hi_ = thrust::device_vector<REAL>(this->nr_variables());
        mm_lo_local_ = thrust::device_vector<REAL>(*std::max_element(this->cum_nr_layers_per_hop_dist_.begin(), this->cum_nr_layers_per_hop_dist_.end())); // size of largest layer.
        mm_diff_ = thrust::device_vector<REAL>(this->hi_cost_.size());
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
                                                            REAL* __restrict__ mm_lo_local, REAL* __restrict__ mm_hi_local)
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
            atomicMin(&mm_hi_local[layer_idx - start_offset_layer], cur_c_from_root + hi_cost[layer_idx] + cost_from_terminal[next_hi_node]);
        }
    }

    // This function does not need lo_path_costs and hi_path_costs to compute min-marginals.
    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::min_marginals_from_directional_costs(const int hop_index, const REAL omega, REAL* const mm_diff_ptr_with_start_offset)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        
        const int num_nodes_processed = hop_index > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1] : 0;
        const int end_node = this->cum_nr_bdd_nodes_per_hop_dist_[hop_index];
        const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[hop_index] - num_nodes_processed;
        const int blockCount = ceil(cur_num_bdd_nodes / (REAL) NUM_THREADS);

        const int start_offset_layer = hop_index > 0 ? this->cum_nr_layers_per_hop_dist_[hop_index - 1]: 0;
        const int end_offset_layer = this->cum_nr_layers_per_hop_dist_[hop_index];
        const int cur_num_layers = end_offset_layer - start_offset_layer;

        min_marginals_from_directional_costs_cuda<<<blockCount, NUM_THREADS>>>(cur_num_bdd_nodes, num_nodes_processed, start_offset_layer,
                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                thrust::raw_pointer_cast(this->cost_from_root_.data()),
                                                thrust::raw_pointer_cast(this->cost_from_terminal_.data()),
                                                thrust::raw_pointer_cast(mm_lo_local_.data()),
                                                mm_diff_ptr_with_start_offset);

        thrust::device_ptr<REAL> mm_lo_start(mm_lo_local_.data());
        thrust::device_ptr<REAL> mm_diff_start = thrust::device_pointer_cast(mm_diff_ptr_with_start_offset);

        auto first = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_start, mm_lo_start));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_start + cur_num_layers, mm_lo_start + cur_num_layers));

        thrust::for_each(first, last, compute_mm_diff_flush_mm_lo<REAL>({omega})); // Convert to min-marginal difference and set mm_lo_local_ to inf.

        #ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
        #endif
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::min_marginals_from_directional_costs(const int hop_index, const REAL omega)
    {
        const int start_offset_layer = hop_index > 0 ? this->cum_nr_layers_per_hop_dist_[hop_index - 1]: 0;
        min_marginals_from_directional_costs(hop_index, omega, thrust::raw_pointer_cast(mm_diff_.data() + start_offset_layer));
    }


    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::iteration()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        forward_iteration(0.5);
        backward_iteration(0.5);
    }

    template<typename REAL>
    __global__ void forward_step_with_solve(const int cur_num_bdd_nodes, const int start_offset,
                                const int* const __restrict__ lo_bdd_node_index, 
                                const int* const __restrict__ hi_bdd_node_index, 
                                const int* const __restrict__ bdd_node_to_layer_map, 
                                const int* const __restrict__ primal_variable_index, 
                                const REAL* const __restrict__ delta_lo,
                                const REAL* const __restrict__ delta_hi,
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
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const REAL cur_lo_cost = lo_cost_in[layer_idx] + min(cur_mm_diff_hi_lo, 0.0f) + delta_lo[cur_primal_idx];
            const REAL cur_c_from_root = cost_from_root[bdd_node_idx];

            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root + cur_lo_cost);

            const REAL cur_hi_cost = hi_cost_in[layer_idx] + min(-cur_mm_diff_hi_lo, 0.0f) + delta_hi[cur_primal_idx];
            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + cur_hi_cost);

            lo_cost_out[layer_idx] = cur_lo_cost;
            hi_cost_out[layer_idx] = cur_hi_cost;
        }
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::forward_iteration(const REAL omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        if(!this->backward_state_valid_)
            this->backward_run(false); //For the first iteration need to have costs from terminal. 
        
        // Clear states.
        this->flush_costs_from_root();
        flush_mm();

        const int num_steps = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        int num_nodes_processed = 0;
        for (int s = 0; s < num_steps; s++)
        {
            // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
            min_marginals_from_directional_costs(s, omega);

            const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            const int blockCount = ceil(cur_num_bdd_nodes / (float) NUM_THREADS);

            // 2. Subtract from hi_costs, update costs from root.
            forward_step_with_solve<<<blockCount, NUM_THREADS>>>(cur_num_bdd_nodes, num_nodes_processed,
                                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                                thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_.data()),
                                                                thrust::raw_pointer_cast(mm_diff_.data()),
                                                                thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                                thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                                thrust::raw_pointer_cast(this->lo_cost_out_.data()),
                                                                thrust::raw_pointer_cast(this->hi_cost_out_.data()),
                                                                thrust::raw_pointer_cast(this->cost_from_root_.data()));
            num_nodes_processed += cur_num_bdd_nodes;
        }
        thrust::swap(this->lo_cost_, lo_cost_out_);
        thrust::swap(this->hi_cost_, hi_cost_out_);
        compute_delta();
        normalize_delta();

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
                                            const REAL* const __restrict__ delta_lo,
                                            const REAL* const __restrict__ delta_hi,
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
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const REAL cur_hi_cost = hi_cost_in[layer_idx] + (min(-cur_mm_diff_hi_lo, 0.0f)) + (delta_hi[cur_primal_idx]);
            const REAL cur_lo_cost = lo_cost_in[layer_idx] + (min(cur_mm_diff_hi_lo, 0.0f)) + (delta_lo[cur_primal_idx]);

            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];

            // Update costs from terminal:
            cost_from_terminal[bdd_node_idx] = min(cur_hi_cost + cost_from_terminal[next_hi_node], cur_lo_cost + cost_from_terminal[next_lo_node]);

            lo_cost_out[layer_idx] = cur_lo_cost;
            hi_cost_out[layer_idx] = cur_hi_cost;
        }
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::backward_iteration(const REAL omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        assert(this->forward_state_valid_); 

        flush_mm();

        for (int s = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 2; s >= 0; s--)
        {
            // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
            min_marginals_from_directional_costs(s, omega);

            const int start_offset = s > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[s - 1] : 0;

            const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[s] - start_offset;
            const int blockCount = ceil(cur_num_bdd_nodes / (REAL) NUM_THREADS);

            // 2. Subtract from hi_costs, update costs from terminal.
            backward_step_with_solve<<<blockCount, NUM_THREADS>>>(cur_num_bdd_nodes, start_offset,
                                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                                thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_.data()),
                                                                thrust::raw_pointer_cast(mm_diff_.data()),
                                                                thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                                thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                                thrust::raw_pointer_cast(lo_cost_out_.data()),
                                                                thrust::raw_pointer_cast(hi_cost_out_.data()),
                                                                thrust::raw_pointer_cast(this->cost_from_terminal_.data()));
        }
        thrust::swap(this->lo_cost_, lo_cost_out_);
        thrust::swap(this->hi_cost_, hi_cost_out_);
        compute_delta();
        normalize_delta();

        this->flush_forward_states();
        this->backward_state_valid_ = true;

        #ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
        #endif
    }

    template<typename REAL>
    struct distribute_delta_func {
        const REAL* delta_lo;
        const REAL* delta_hi;
        __host__ __device__ void operator()(const thrust::tuple<int, REAL&, REAL&> t) const
        {
            const int primal_index = thrust::get<0>(t);
            if (primal_index == INT_MAX)
                return; // terminal node.

            REAL& lo_cost = thrust::get<1>(t);
            REAL& hi_cost = thrust::get<2>(t);
            lo_cost += delta_lo[primal_index];
            hi_cost += delta_hi[primal_index];
        }
    };

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::distribute_delta()
    {
        assert(this->primal_variable_index_.size() == this->lo_cost_.size());
        assert(this->primal_variable_index_.size() == this->hi_cost_.size());
        assert(this->delta_lo_.size() == this->num_bdds_per_var_.size());
        assert(this->delta_hi_.size() == this->num_bdds_per_var_.size());
        assert(this->delta_hi_.size() == this->nr_vars_);

        auto first = thrust::make_zip_iterator(thrust::make_tuple(this->primal_variable_index_.begin(), this->lo_cost_.begin(), this->hi_cost_.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(this->primal_variable_index_.end(), this->lo_cost_.end(), this->hi_cost_.end()));

        distribute_delta_func<REAL> func({thrust::raw_pointer_cast(delta_lo_.data()), thrust::raw_pointer_cast(delta_hi_.data())});

        thrust::for_each(first, last, func);
        this->flush_forward_states();
        this->flush_backward_states();

        thrust::fill(delta_lo_.begin(), delta_lo_.end(), 0.0f);
        thrust::fill(delta_hi_.begin(), delta_hi_.end(), 0.0f);
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
    struct tuple_sum
    {
        __host__ __device__
        thrust::tuple<REAL, REAL> operator()(const thrust::tuple<REAL, REAL>& t0, const thrust::tuple<REAL, REAL>& t1)
        {
            return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1), thrust::get<1>(t0) + thrust::get<1>(t1));
        }
    };

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::compute_delta()
    {
        auto first_val = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_permutation_iterator(thrust::make_transform_iterator(mm_diff_.begin(), pos_part<REAL>()), this->primal_variable_sorting_order_.begin()),
            thrust::make_permutation_iterator(thrust::make_transform_iterator(mm_diff_.begin(), abs_neg_part<REAL>()), this->primal_variable_sorting_order_.begin())));

        auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(delta_hi_.begin(), delta_lo_.begin()));

        thrust::equal_to<int> binary_pred;
        auto new_end = thrust::reduce_by_key(this->primal_variable_index_sorted_.begin(), this->primal_variable_index_sorted_.end() - this->nr_bdds_, first_val, 
                            thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_sum<REAL>());
        assert(thrust::distance(first_out_val, new_end.second) == delta_hi_.size());
        // thrust::reduce_by_key(thrust::make_permutation_iterator(this->primal_variable_index_.begin(), primal_variable_sorting_order_.begin()),
        //                     thrust::make_permutation_iterator(this->primal_variable_index_.end(), primal_variable_sorting_order_.end()), first_val, 
        //                     thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_sum<REAL>()); // Uses less memory but slower.

    }

    template<typename REAL>
    struct normalize_delta_func {
        __host__ __device__ void operator()(const thrust::tuple<REAL&, REAL&, int> t) const
        {
            const int norm = thrust::get<2>(t);
            REAL& hi_cost = thrust::get<0>(t);
            hi_cost /= norm;
            REAL& lo_cost = thrust::get<1>(t);
            lo_cost /= norm;
        }
    };

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::normalize_delta()
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(this->delta_hi_.begin(), this->delta_lo_.begin(), this->num_bdds_per_var_.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(this->delta_hi_.end(), this->delta_lo_.end(), this->num_bdds_per_var_.end()));
        thrust::for_each(first, last, normalize_delta_func<REAL>());
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::flush_mm()
    {   // Makes min marginals INF so that they can be populated again by in-place minimization
        thrust::fill(mm_lo_local_.begin(), mm_lo_local_.end(), CUDART_INF_F_HOST);
        thrust::fill(mm_diff_.begin(), mm_diff_.end(), CUDART_INF_F_HOST);
    }

    template class bdd_cuda_parallel_mma<float>;
    template class bdd_cuda_parallel_mma<double>;
}
