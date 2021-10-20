#include "bdd_cuda_parallel_mma.h"
#include "cuda_utils.h"
#include <chrono>

namespace LPMP {

    template<typename REAL>
    bdd_cuda_parallel_mma<REAL>::bdd_cuda_parallel_mma(const BDD::bdd_collection& bdd_col) : bdd_cuda_base<REAL>(bdd_col)
    {
        delta_lo_in_ = thrust::device_vector<REAL>(this->nr_variables(), 0.0);
        delta_hi_in_ = thrust::device_vector<REAL>(this->nr_variables(), 0.0);
        delta_lo_out_ = thrust::device_vector<REAL>(this->nr_variables());
        delta_hi_out_ = thrust::device_vector<REAL>(this->nr_variables());
        mm_lo_ = thrust::device_vector<REAL>(this->hi_cost_.size());
        mm_hi_ = thrust::device_vector<REAL>(this->hi_cost_.size());
        // Copy from arc costs because it contains infinity for arcs to bot sink
        hi_cost_out_ = thrust::device_vector<REAL>(this->hi_cost_);
        lo_cost_out_ = thrust::device_vector<REAL>(this->lo_cost_);
    }

    template<typename REAL>
    __global__ void min_marginals_from_directional_costs_cuda(const int cur_num_bdd_nodes, const int start_offset,
                                                            const int* const __restrict__ lo_bdd_node_index, 
                                                            const int* const __restrict__ hi_bdd_node_index, 
                                                            const int* const __restrict__ bdd_node_to_layer_map, 
                                                            const REAL* const __restrict__ lo_cost,
                                                            const REAL* const __restrict__ hi_cost,
                                                            const REAL* const __restrict__ cost_from_root,
                                                            const REAL* const __restrict__ cost_from_terminal,
                                                            REAL* __restrict__ mm_lo, REAL* __restrict__ mm_hi)
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

            atomicMin(&mm_lo[layer_idx], cur_c_from_root + lo_cost[layer_idx] + cost_from_terminal[next_lo_node]);
            atomicMin(&mm_hi[layer_idx], cur_c_from_root + hi_cost[layer_idx] + cost_from_terminal[next_hi_node]);
        }
    }

    // This function does not need lo_path_costs and hi_path_costs to compute min-marginals.
    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::min_marginals_from_directional_costs(const int hop_index)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        
        int threadCount = 256;
        int num_nodes_processed = 0;
        if (hop_index > 0) 
            num_nodes_processed = this->cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1];
        const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[hop_index] - num_nodes_processed;
        int blockCount = ceil(cur_num_bdd_nodes / (REAL) threadCount);
        min_marginals_from_directional_costs_cuda<<<blockCount, threadCount>>>(cur_num_bdd_nodes, num_nodes_processed,
                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                thrust::raw_pointer_cast(this->cost_from_root_.data()),
                                                thrust::raw_pointer_cast(this->cost_from_terminal_.data()),
                                                thrust::raw_pointer_cast(this->mm_lo_.data()),
                                                thrust::raw_pointer_cast(this->mm_hi_.data()));
        #ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
        #endif
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::iteration()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        thrust::fill(mm_lo_.begin(), mm_lo_.end(), CUDART_INF_F_HOST);
        thrust::fill(mm_hi_.begin(), mm_hi_.end(), CUDART_INF_F_HOST);
        // forward_iteration_layer_based(0.5);
        forward_iteration(0.5);
        thrust::fill(mm_lo_.begin(), mm_lo_.end(), CUDART_INF_F_HOST);
        thrust::fill(mm_hi_.begin(), mm_hi_.end(), CUDART_INF_F_HOST);
        backward_iteration(0.5);
    }

    template<typename REAL>
    __global__ void forward_step_with_solve(const int cur_num_bdd_nodes, const int start_offset, const REAL omega,
                                const int* const __restrict__ lo_bdd_node_index, 
                                const int* const __restrict__ hi_bdd_node_index, 
                                const int* const __restrict__ bdd_node_to_layer_map, 
                                const int* const __restrict__ primal_variable_index, 
                                const int* const __restrict__ num_bdds_per_var, 
                                const REAL* const __restrict__ delta_lo_in,
                                const REAL* const __restrict__ delta_hi_in,
                                const REAL* const __restrict__ mm_lo,
                                const REAL* const __restrict__ mm_hi,
                                const REAL* const __restrict__ lo_cost_in,
                                const REAL* const __restrict__ hi_cost_in,
                                REAL* __restrict__ lo_cost_out,
                                REAL* __restrict__ hi_cost_out,
                                REAL* __restrict__ delta_lo_out,
                                REAL* __restrict__ delta_hi_out,
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
            const REAL cur_mm_diff_hi_lo = mm_hi[layer_idx] - mm_lo[layer_idx];
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const REAL cur_hi_cost = hi_cost_in[layer_idx] + omega * min(-cur_mm_diff_hi_lo, 0.0f) + delta_hi_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx];
            const REAL cur_lo_cost = lo_cost_in[layer_idx] + omega * min(cur_mm_diff_hi_lo, 0.0f) + delta_lo_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx];

            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
            const REAL cur_c_from_root = cost_from_root[bdd_node_idx];

            // Update costs from root:
            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root + cur_lo_cost);
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + cur_hi_cost);

            // Select the leader thread for each BDD layer which will write out to delta_out
            // Leader thread would be the thread operating on first BDD node of each layer.
            bool is_leader = bdd_node_idx == 0 || layer_idx != bdd_node_to_layer_map[bdd_node_idx - 1];
            if  (is_leader)
            {
                lo_cost_out[layer_idx] = cur_lo_cost;
                hi_cost_out[layer_idx] = cur_hi_cost;
    
                if (cur_mm_diff_hi_lo > 0)
                    atomicAdd(&delta_hi_out[cur_primal_idx], omega * cur_mm_diff_hi_lo);
                else
                    atomicAdd(&delta_lo_out[cur_primal_idx], -omega * cur_mm_diff_hi_lo);
            }
        }
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::forward_iteration(const REAL omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        if(!this->backward_state_valid_)
            this->backward_run(false); //For the first iteration need to have costs from terminal. 
        
        // Set costs from root to INF and of root nodes to itself to 0:
        thrust::fill(this->cost_from_root_.begin(), this->cost_from_root_.end(), CUDART_INF_F_HOST);
        thrust::scatter(thrust::make_constant_iterator<REAL>(0.0), thrust::make_constant_iterator<REAL>(0.0) + this->root_indices_.size(), this->root_indices_.begin(), this->cost_from_root_.begin());

        // Set delta_out to zero:
        thrust::fill(delta_lo_out_.begin(), delta_lo_out_.end(), 0.0f);
        thrust::fill(delta_hi_out_.begin(), delta_hi_out_.end(), 0.0f);

        const int num_steps = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        int num_nodes_processed = 0;
        for (int s = 0; s < num_steps; s++)
        {
            // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
            min_marginals_from_directional_costs(s);

            const int threadCount = NUM_THREADS;
            const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            const int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);

            // 2. Subtract from hi_costs, update costs from root and add to delta_hi_out, delta_lo_out.
            forward_step_with_solve<<<blockCount, threadCount>>>(cur_num_bdd_nodes, num_nodes_processed, omega,
                                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                                thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                                thrust::raw_pointer_cast(this->num_bdds_per_var_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_in_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_in_.data()),
                                                                thrust::raw_pointer_cast(mm_lo_.data()),
                                                                thrust::raw_pointer_cast(mm_hi_.data()),
                                                                thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                                thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                                thrust::raw_pointer_cast(this->lo_cost_out_.data()),
                                                                thrust::raw_pointer_cast(this->hi_cost_out_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_out_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_out_.data()),
                                                                thrust::raw_pointer_cast(this->cost_from_root_.data()));
            num_nodes_processed += cur_num_bdd_nodes;
        }
        thrust::swap(this->lo_cost_, lo_cost_out_);
        thrust::swap(this->hi_cost_, hi_cost_out_);
        thrust::swap(delta_lo_in_, delta_lo_out_);
        thrust::swap(delta_hi_in_, delta_hi_out_);
        this->forward_state_valid_ = true;
        this->flush_backward_states();

        #ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
        #endif
    }

    template<typename REAL>
    __global__ void forward_step_with_solve_layer(const int cur_num_bdd_nodes, const int start_offset, const REAL omega,
                                                const int* const __restrict__ lo_bdd_node_index, 
                                                const int* const __restrict__ hi_bdd_node_index, 
                                                const int* const __restrict__ bdd_node_to_layer_map, 
                                                const int* const __restrict__ primal_variable_index, 
                                                const int* const __restrict__ num_bdds_per_var, 
                                                const REAL* const __restrict__ delta_lo_in,
                                                const REAL* const __restrict__ delta_hi_in,
                                                const REAL* const __restrict__ mm_lo,
                                                const REAL* const __restrict__ mm_hi,
                                                const int* const __restrict__ layer_offsets,
                                                REAL* __restrict__ lo_cost,
                                                REAL* __restrict__ hi_cost,
                                                REAL* __restrict__ delta_lo_out,
                                                REAL* __restrict__ delta_hi_out,
                                                REAL* __restrict__ cost_from_root)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int layer_idx = start_index + start_offset; layer_idx < cur_num_bdd_nodes + start_offset; layer_idx += num_threads) 
        {
            const int cur_primal_idx = primal_variable_index[layer_idx];
            if (cur_primal_idx == INT_MAX)
                continue; // terminal node.

            const REAL cur_mm_diff_hi_lo = mm_hi[layer_idx] - mm_lo[layer_idx];
            const REAL cur_hi_cost = hi_cost[layer_idx] + omega * min(-cur_mm_diff_hi_lo, 0.0f) + delta_hi_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx];
            const REAL cur_lo_cost = lo_cost[layer_idx] + omega * min(cur_mm_diff_hi_lo, 0.0f) + delta_lo_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx];
            lo_cost[layer_idx] = cur_lo_cost;
            hi_cost[layer_idx] = cur_hi_cost;

            const int start_bdd_node = layer_offsets[layer_idx];
            const int end_bdd_node = layer_offsets[layer_idx + 1];
            for (int bdd_node_idx = start_bdd_node; bdd_node_idx < end_bdd_node; bdd_node_idx++)
            {
                const int next_lo_node = lo_bdd_node_index[bdd_node_idx];
                const int next_hi_node = hi_bdd_node_index[bdd_node_idx];
                const REAL cur_c_from_root = cost_from_root[bdd_node_idx];

                // Update costs from root:
                cost_from_root[next_lo_node] = min(cost_from_root[next_lo_node], cur_c_from_root + cur_lo_cost);
                cost_from_root[next_hi_node] = min(cost_from_root[next_hi_node], cur_c_from_root + cur_hi_cost);
            }
            if (cur_mm_diff_hi_lo > 0)
                atomicAdd(&delta_hi_out[cur_primal_idx], omega * cur_mm_diff_hi_lo);
            else
                atomicAdd(&delta_lo_out[cur_primal_idx], -omega * cur_mm_diff_hi_lo);
        }
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::forward_iteration_layer_based(const REAL omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        assert(this->backward_state_valid_); //For the first iteration need to have costs from terminal. 
        
        // Set costs from root to INF and of root nodes to itself to 0:
        thrust::fill(this->cost_from_root_.begin(), this->cost_from_root_.end(), CUDART_INF_F_HOST);
        thrust::scatter(thrust::make_constant_iterator<REAL>(0.0), thrust::make_constant_iterator<REAL>(0.0) + this->root_indices_.size(), this->root_indices_.begin(), this->cost_from_root_.begin());

        // Set delta_out to zero:
        thrust::fill(delta_lo_out_.begin(), delta_lo_out_.end(), 0.0f);
        thrust::fill(delta_hi_out_.begin(), delta_hi_out_.end(), 0.0f);

        const int num_steps = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        int num_layers_processed = 0;
        for (int s = 0; s < num_steps; s++)
        {
            // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
            min_marginals_from_directional_costs(s);

            const int threadCount = NUM_THREADS;
            const int cur_num_layers = this->cum_nr_layers_per_hop_dist_[s] - num_layers_processed;
            const int blockCount = ceil(cur_num_layers / (float) threadCount);

            // 2. Subtract from hi_costs, update costs from root and add to delta_hi_out, delta_lo_out.
            forward_step_with_solve_layer<<<blockCount, threadCount>>>(cur_num_layers, num_layers_processed, omega,
                                                                    thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                                    thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                                    thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                                    thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                                    thrust::raw_pointer_cast(this->num_bdds_per_var_.data()),
                                                                    thrust::raw_pointer_cast(delta_lo_in_.data()),
                                                                    thrust::raw_pointer_cast(delta_hi_in_.data()),
                                                                    thrust::raw_pointer_cast(mm_lo_.data()),
                                                                    thrust::raw_pointer_cast(mm_hi_.data()),
                                                                    thrust::raw_pointer_cast(this->layer_offsets_.data()),
                                                                    thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                                    thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                                    thrust::raw_pointer_cast(delta_lo_out_.data()),
                                                                    thrust::raw_pointer_cast(delta_hi_out_.data()),
                                                                    thrust::raw_pointer_cast(this->cost_from_root_.data()));
            num_layers_processed += cur_num_layers;
        }
        thrust::swap(delta_lo_in_, delta_lo_out_);
        thrust::swap(delta_hi_in_, delta_hi_out_);
        this->forward_state_valid_ = true;
        this->flush_backward_states();

        #ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
        #endif
    }


    template<typename REAL>
    __global__ void backward_step_with_solve(const int cur_num_bdd_nodes, const int start_offset, const REAL omega,
                                            const int* const __restrict__ lo_bdd_node_index, 
                                            const int* const __restrict__ hi_bdd_node_index, 
                                            const int* const __restrict__ bdd_node_to_layer_map, 
                                            const int* const __restrict__ primal_variable_index, 
                                            const int* const __restrict__ num_bdds_per_var, 
                                            const REAL* const __restrict__ delta_lo_in,
                                            const REAL* const __restrict__ delta_hi_in,
                                            const REAL* const __restrict__ mm_lo,
                                            const REAL* const __restrict__ mm_hi,
                                            const REAL* const __restrict__ lo_cost_in,
                                            const REAL* const __restrict__ hi_cost_in,
                                            REAL* __restrict__ lo_cost_out,
                                            REAL* __restrict__ hi_cost_out,
                                            REAL* __restrict__ delta_lo_out,
                                            REAL* __restrict__ delta_hi_out,
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
            const REAL cur_mm_diff_hi_lo = mm_hi[layer_idx] - mm_lo[layer_idx];
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const REAL cur_hi_cost = hi_cost_in[layer_idx] + (omega * min(-cur_mm_diff_hi_lo, 0.0f)) + (delta_hi_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx]);
            const REAL cur_lo_cost = lo_cost_in[layer_idx] + (omega * min(cur_mm_diff_hi_lo, 0.0f)) + (delta_lo_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx]);

            const int next_hi_node = hi_bdd_node_index[bdd_node_idx];

            // Update costs from terminal:
            cost_from_terminal[bdd_node_idx] = min(cur_hi_cost + cost_from_terminal[next_hi_node], cur_lo_cost + cost_from_terminal[next_lo_node]);

            // Select the leader thread for each BDD layer which will write out to delta_out
            // Leader thread would be the thread operating on first BDD node of each layer.
            bool is_leader = bdd_node_idx == 0 || layer_idx != bdd_node_to_layer_map[bdd_node_idx - 1]; //TODO: Try warp shuffle.
            if  (is_leader)
            {
                lo_cost_out[layer_idx] = cur_lo_cost;
                hi_cost_out[layer_idx] = cur_hi_cost;
        
                if (cur_mm_diff_hi_lo > 0)
                    atomicAdd(&delta_hi_out[cur_primal_idx], omega * cur_mm_diff_hi_lo);
                else
                    atomicAdd(&delta_lo_out[cur_primal_idx], -omega * cur_mm_diff_hi_lo);
            }
        }
    }

    template<typename REAL>
    void bdd_cuda_parallel_mma<REAL>::backward_iteration(const REAL omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        assert(this->forward_state_valid_); 

        // Set delta_out to zero:
        thrust::fill(delta_lo_out_.begin(), delta_lo_out_.end(), 0.0f);
        thrust::fill(delta_hi_out_.begin(), delta_hi_out_.end(), 0.0f);

        for (int s = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 2; s >= 0; s--)
        {
            // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
            min_marginals_from_directional_costs(s);

            const int threadCount = NUM_THREADS;
            int start_offset = 0;
            if(s > 0)
                start_offset = this->cum_nr_bdd_nodes_per_hop_dist_[s - 1];

            const int cur_num_bdd_nodes = this->cum_nr_bdd_nodes_per_hop_dist_[s] - start_offset;
            const int blockCount = ceil(cur_num_bdd_nodes / (REAL) threadCount);

            // 2. Subtract from hi_costs, update costs from terminal and add to delta_hi_out, delta_lo_out.
            backward_step_with_solve<<<blockCount, threadCount>>>(cur_num_bdd_nodes, start_offset, omega,
                                                                thrust::raw_pointer_cast(this->lo_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->hi_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(this->bdd_node_to_layer_map_.data()),
                                                                thrust::raw_pointer_cast(this->primal_variable_index_.data()),
                                                                thrust::raw_pointer_cast(this->num_bdds_per_var_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_in_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_in_.data()),
                                                                thrust::raw_pointer_cast(mm_lo_.data()),
                                                                thrust::raw_pointer_cast(mm_hi_.data()),
                                                                thrust::raw_pointer_cast(this->lo_cost_.data()),
                                                                thrust::raw_pointer_cast(this->hi_cost_.data()),
                                                                thrust::raw_pointer_cast(lo_cost_out_.data()),
                                                                thrust::raw_pointer_cast(hi_cost_out_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_out_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_out_.data()),
                                                                thrust::raw_pointer_cast(this->cost_from_terminal_.data()));
        }
        thrust::swap(delta_lo_in_, delta_lo_out_);
        thrust::swap(delta_hi_in_, delta_hi_out_);
        thrust::swap(this->lo_cost_, lo_cost_out_);
        thrust::swap(this->hi_cost_, hi_cost_out_);
        this->flush_forward_states();
        this->backward_state_valid_ = true;

        #ifndef NDEBUG
            cudaDeviceSynchronize();  // Not necessary, only to compute exact timing of this function.
        #endif
    }

    template class bdd_cuda_parallel_mma<float>;
    template class bdd_cuda_parallel_mma<double>;
}
