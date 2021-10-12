#include "bdd_cuda_parallel_mma.h"
#include "cuda_utils.h"
#include <chrono>

namespace LPMP {

    bdd_cuda_parallel_mma::bdd_cuda_parallel_mma(const BDD::bdd_collection& bdd_col) : bdd_cuda_base(bdd_col)
    {
        delta_lo_in_ = thrust::device_vector<float>(nr_variables(), 0.0);
        delta_hi_in_ = thrust::device_vector<float>(nr_variables(), 0.0);
        delta_lo_out_ = thrust::device_vector<float>(nr_variables());
        delta_hi_out_ = thrust::device_vector<float>(nr_variables());
        mm_lo_ = thrust::device_vector<float>(hi_cost_.size());
        mm_hi_ = thrust::device_vector<float>(hi_cost_.size());
    }

    __global__ void min_marginals_from_directional_costs_cuda(const int cur_num_bdd_nodes, const int start_offset,
                                                            const int* const __restrict__ lo_bdd_node_index, 
                                                            const int* const __restrict__ hi_bdd_node_index, 
                                                            const int* const __restrict__ bdd_node_to_layer_map, 
                                                            const float* const __restrict__ lo_cost,
                                                            const float* const __restrict__ hi_cost,
                                                            const float* const __restrict__ cost_from_root,
                                                            const float* const __restrict__ cost_from_terminal,
                                                            float* __restrict__ mm_lo, float* __restrict__ mm_hi)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index + start_offset; bdd_idx < cur_num_bdd_nodes + start_offset; bdd_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_idx];
            if (next_lo_node < 0) // will matter when one row contains multiple BDDs, otherwise the terminal nodes are at the end anyway.
                continue; // nothing needs to be done for terminal node.

            const int next_hi_node = hi_bdd_node_index[bdd_idx];

            const float cur_c_from_root = cost_from_root[bdd_idx];
            const int layer_idx = bdd_node_to_layer_map[bdd_idx];

            // TODO: can possibly store the following 'indicator' variable in char or something?
            // or just set lo_cost to bot sink to infinity?
            const bool is_lo_bot_sink = lo_bdd_node_index[next_lo_node] == BOT_SINK_INDICATOR_CUDA;
            const bool is_hi_bot_sink = lo_bdd_node_index[next_hi_node] == BOT_SINK_INDICATOR_CUDA;

            if (!is_lo_bot_sink)
                atomicMin(&mm_lo[layer_idx], cur_c_from_root + lo_cost[layer_idx] + cost_from_terminal[next_lo_node]);
            if (!is_hi_bot_sink)
                atomicMin(&mm_hi[layer_idx], cur_c_from_root + hi_cost[layer_idx] + cost_from_terminal[next_hi_node]);
        }
    }

    // This function does not need lo_path_costs and hi_path_costs to compute min-marginals.
    void bdd_cuda_parallel_mma::min_marginals_from_directional_costs(const int hop_index)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        
        int threadCount = 256;
        int num_nodes_processed = 0;
        if (hop_index > 0) 
            num_nodes_processed = cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1];
        const int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[hop_index] - num_nodes_processed;
        int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);
        min_marginals_from_directional_costs_cuda<<<blockCount, threadCount>>>(cur_num_bdd_nodes, num_nodes_processed,
                                                thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                                thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                                thrust::raw_pointer_cast(lo_cost_.data()),
                                                thrust::raw_pointer_cast(hi_cost_.data()),
                                                thrust::raw_pointer_cast(cost_from_root_.data()),
                                                thrust::raw_pointer_cast(cost_from_terminal_.data()),
                                                thrust::raw_pointer_cast(mm_lo_.data()),
                                                thrust::raw_pointer_cast(mm_hi_.data()));
    }

    void bdd_cuda_parallel_mma::iteration()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        thrust::fill(mm_lo_.begin(), mm_lo_.end(), CUDART_INF_F);
        thrust::fill(mm_hi_.begin(), mm_hi_.end(), CUDART_INF_F);
        backward_iteration(0.5);
        // backward_run(false);
        forward_iteration(0.5);
    }

    __global__ void forward_step_with_solve(const int cur_num_bdd_nodes, const int start_offset, const float omega,
                                const int* const __restrict__ lo_bdd_node_index, 
                                const int* const __restrict__ hi_bdd_node_index, 
                                const int* const __restrict__ bdd_node_to_layer_map, 
                                const int* const __restrict__ primal_variable_index, 
                                const int* const __restrict__ num_bdds_per_var, 
                                const float* const __restrict__ delta_lo_in,
                                const float* const __restrict__ delta_hi_in,
                                const float* const __restrict__ mm_lo,
                                const float* const __restrict__ mm_hi,
                                float* __restrict__ lo_cost,
                                float* __restrict__ hi_cost,
                                float* __restrict__ delta_lo_out,
                                float* __restrict__ delta_hi_out,
                                float* __restrict__ cost_from_root)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index + start_offset; bdd_idx < cur_num_bdd_nodes + start_offset; bdd_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_idx];
            if (next_lo_node < 0) // will matter when one row contains multiple BDDs, otherwise the terminal nodes are at the end anyway.
                continue; // nothing needs to be done for terminal node.
            
            const int layer_idx = bdd_node_to_layer_map[bdd_idx];
            const float cur_mm_diff_hi_lo = mm_hi[layer_idx] - mm_lo[layer_idx];
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const float cur_hi_cost = hi_cost[layer_idx] + omega * min(-cur_mm_diff_hi_lo, 0.0f) + delta_hi_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx];
            const float cur_lo_cost = lo_cost[layer_idx] + omega * min(cur_mm_diff_hi_lo, 0.0f) + delta_lo_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx];

            lo_cost[layer_idx] = cur_lo_cost;
            hi_cost[layer_idx] = cur_hi_cost;

            const int next_hi_node = hi_bdd_node_index[bdd_idx];
            const float cur_c_from_root = cost_from_root[bdd_idx];

            // Update costs from root:
            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root + cur_lo_cost);
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + cur_hi_cost);

            // Select the leader thread for each BDD layer which will write out to delta_out
            // Leader thread would be the thread operating on first BDD node of each layer.
            bool is_leader = start_index == 0 || layer_idx != bdd_node_to_layer_map[bdd_idx - 1];
            if  (is_leader)
            {
                if (cur_mm_diff_hi_lo > 0)
                    atomicAdd(&delta_hi_out[cur_primal_idx], omega * cur_mm_diff_hi_lo);
                else
                    atomicAdd(&delta_lo_out[cur_primal_idx], -omega * cur_mm_diff_hi_lo);
            }
        }
    }

    void bdd_cuda_parallel_mma::forward_iteration(const float omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        assert(backward_state_valid_); //For the first iteration need to have costs from terminal. 
        
        // Set costs from root to INF and of root nodes to itself to 0:
        thrust::fill(cost_from_root_.begin(), cost_from_root_.end(), CUDART_INF_F);
        thrust::scatter(thrust::make_constant_iterator<float>(0.0), thrust::make_constant_iterator<float>(0.0) + root_indices_.size(), root_indices_.begin(), cost_from_root_.begin());

        // Set delta_out to zero:
        thrust::fill(delta_lo_out_.begin(), delta_lo_out_.end(), 0.0f);
        thrust::fill(delta_hi_out_.begin(), delta_hi_out_.end(), 0.0f);

        const int num_steps = cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        int num_nodes_processed = 0;
        for (int s = 0; s < num_steps; s++)
        {
            // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
            min_marginals_from_directional_costs(s);

            int threadCount = 256;
            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);

            // 2. Subtract from hi_costs, update costs from root and add to delta_hi_out, delta_lo_out.
            forward_step_with_solve<<<blockCount, threadCount>>>(cur_num_bdd_nodes, num_nodes_processed, omega,
                                                                thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                                                thrust::raw_pointer_cast(primal_variable_index_.data()),
                                                                thrust::raw_pointer_cast(num_bdds_per_var_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_in_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_in_.data()),
                                                                thrust::raw_pointer_cast(mm_lo_.data()),
                                                                thrust::raw_pointer_cast(mm_hi_.data()),
                                                                thrust::raw_pointer_cast(lo_cost_.data()),
                                                                thrust::raw_pointer_cast(hi_cost_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_out_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_out_.data()),
                                                                thrust::raw_pointer_cast(cost_from_root_.data()));
            num_nodes_processed += cur_num_bdd_nodes;
        }
        thrust::swap(delta_lo_in_, delta_lo_out_);
        thrust::swap(delta_hi_in_, delta_hi_out_);
        forward_state_valid_ = true;
        backward_state_valid_ = false;
    }

    __global__ void backward_step_with_solve(const int cur_num_bdd_nodes, const int start_offset, const float omega,
                                            const int* const __restrict__ lo_bdd_node_index, 
                                            const int* const __restrict__ hi_bdd_node_index, 
                                            const int* const __restrict__ bdd_node_to_layer_map, 
                                            const int* const __restrict__ primal_variable_index, 
                                            const int* const __restrict__ num_bdds_per_var, 
                                            const float* const __restrict__ delta_lo_in,
                                            const float* const __restrict__ delta_hi_in,
                                            const float* const __restrict__ mm_lo,
                                            const float* const __restrict__ mm_hi,
                                            float* __restrict__ lo_cost,
                                            float* __restrict__ hi_cost,
                                            float* __restrict__ delta_lo_out,
                                            float* __restrict__ delta_hi_out,
                                            float* __restrict__ cost_from_terminal)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index + start_offset; bdd_idx < cur_num_bdd_nodes + start_offset; bdd_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_idx];
            if (next_lo_node < 0)
                continue; // nothing needs to be done for terminal node.
            
            const int layer_idx = bdd_node_to_layer_map[bdd_idx];
            const float cur_mm_diff_hi_lo = mm_hi[layer_idx] - mm_lo[layer_idx];
            const int cur_primal_idx = primal_variable_index[layer_idx];

            const float cur_hi_cost = hi_cost[layer_idx] + omega * min(-cur_mm_diff_hi_lo, 0.0f) + delta_hi_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx];
            const float cur_lo_cost = lo_cost[layer_idx] + omega * min(cur_mm_diff_hi_lo, 0.0f) + delta_lo_in[cur_primal_idx] / num_bdds_per_var[cur_primal_idx];

            lo_cost[layer_idx] = cur_lo_cost;
            hi_cost[layer_idx] = cur_hi_cost;

            const int next_hi_node = hi_bdd_node_index[bdd_idx];

            // TODO: Skip following check by setting all arcs going to bot sink to infty.
            const bool is_lo_bot_sink = lo_bdd_node_index[next_lo_node] == BOT_SINK_INDICATOR_CUDA;
            const bool is_hi_bot_sink = lo_bdd_node_index[next_hi_node] == BOT_SINK_INDICATOR_CUDA;

            // Update costs from terminal:
            if(!is_lo_bot_sink && !is_hi_bot_sink)
                cost_from_terminal[bdd_idx] = min(cur_hi_cost + cost_from_terminal[next_hi_node], cur_lo_cost + cost_from_terminal[next_lo_node]);
            else if(!is_hi_bot_sink)
                cost_from_terminal[bdd_idx] = cur_hi_cost + cost_from_terminal[next_hi_node];
            else if(!is_lo_bot_sink)
                cost_from_terminal[bdd_idx] = cur_lo_cost + cost_from_terminal[next_lo_node];

            // Select the leader thread for each BDD layer which will write out to delta_out
            // Leader thread would be the thread operating on first BDD node of each layer.
            bool is_leader = start_index == 0 || layer_idx != bdd_node_to_layer_map[bdd_idx - 1]; //TODO: Try warp shuffle.
            if  (is_leader)
            {
                if (cur_mm_diff_hi_lo > 0)
                    atomicAdd(&delta_hi_out[cur_primal_idx], omega * cur_mm_diff_hi_lo);
                else
                    atomicAdd(&delta_lo_out[cur_primal_idx], -omega * cur_mm_diff_hi_lo);
            }
        }
    }

    void bdd_cuda_parallel_mma::backward_iteration(const float omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        assert(forward_state_valid_); 
        
        // Set costs of top sinks to 0:
        thrust::scatter(thrust::make_constant_iterator<float>(0.0), thrust::make_constant_iterator<float>(0.0) + top_sink_indices_.size(),
                        top_sink_indices_.begin(), cost_from_terminal_.begin());

        // Set delta_out to zero:
        thrust::fill(delta_lo_out_.begin(), delta_lo_out_.end(), 0.0f);
        thrust::fill(delta_hi_out_.begin(), delta_hi_out_.end(), 0.0f);

        for (int s = cum_nr_bdd_nodes_per_hop_dist_.size() - 2; s >= 0; s--)
        {
            // 1. Compute min-marginals using costs from root, costs from terminal and hi_costs, lo_costs for current hop
            min_marginals_from_directional_costs(s);

            int threadCount = 256;
            int start_offset = 0;
            if(s > 0)
                start_offset = cum_nr_bdd_nodes_per_hop_dist_[s - 1];

            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - start_offset;
            int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);

            // 2. Subtract from hi_costs, update costs from terminal and add to delta_hi_out, delta_lo_out.
            backward_step_with_solve<<<blockCount, threadCount>>>(cur_num_bdd_nodes, start_offset, omega,
                                                                thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                                                thrust::raw_pointer_cast(primal_variable_index_.data()),
                                                                thrust::raw_pointer_cast(num_bdds_per_var_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_in_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_in_.data()),
                                                                thrust::raw_pointer_cast(mm_lo_.data()),
                                                                thrust::raw_pointer_cast(mm_hi_.data()),
                                                                thrust::raw_pointer_cast(lo_cost_.data()),
                                                                thrust::raw_pointer_cast(hi_cost_.data()),
                                                                thrust::raw_pointer_cast(delta_lo_out_.data()),
                                                                thrust::raw_pointer_cast(delta_hi_out_.data()),
                                                                thrust::raw_pointer_cast(cost_from_terminal_.data()));
        }
        thrust::swap(delta_lo_in_, delta_lo_out_);
        thrust::swap(delta_hi_in_, delta_hi_out_);
        forward_state_valid_ = false;
        backward_state_valid_ = true;
    }
}
