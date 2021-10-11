#include "bdd_cuda_parallel_mma.h"
#include <chrono>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

namespace LPMP {

    struct is_terminal_func {
        __host__ __device__ bool operator()(const thrust::tuple<int, float>& t)
        {
            return thrust::get<0>(t) < 0;
        }
    };

    struct mm_diff_func {
        __host__ __device__ float operator()(const float& m1, const float& m0)
        {
            if(!isinf(m0) && !isinf(m1))
                return m1 - m0;
            assert(isinf(m0) && isinf(m1));
            return m0;
        }
    };

    void bdd_cuda_parallel_mma_sorting::iteration()
    {
        backward_iteration();
        forward_iteration();
    }

    __global__ void forward_step_with_solve(const int cur_num_bdd_nodes, const int start_offset, const float omega,
                                const int* const __restrict__ lo_bdd_node_index, 
                                const int* const __restrict__ hi_bdd_node_index, 
                                const int* const __restrict__ bdd_node_to_layer_map, 
                                const float* const __restrict__ lo_cost,
                                const float* const __restrict__ hi_cost,
                                const float* const __restrict__ delta_lo,
                                const float* const __restrict__ delta_hi,
                                const float* const __restrict__ cost_from_terminal,
                                float* __restrict__ cost_from_root)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index + start_offset; bdd_idx < cur_num_bdd_nodes + start_offset; bdd_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_idx];
            if (next_lo_node < 0) // will matter when one row contains multiple BDDs, otherwise the terminal nodes are at the end anyway.
                continue; // nothing needs to be done for terminal node.

            const int next_hi_node = hi_bdd_node_index[bdd_idx];
            assert(next_hi_node >= 0);

            const float cur_c_from_root = cost_from_root[bdd_idx];
            const int layer_idx = bdd_node_to_layer_map[bdd_idx];

            // Uncoalesced writes:
            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root + lo_cost[layer_idx]);
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + hi_cost[layer_idx]);
        }
    }

    void bdd_cuda_parallel_mma_sorting::forward_iteration(const float omega)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        assert(backward_state_valid_); //For the first iteration need to have costs from terminal. 
        
        // Set root nodes costs to INF:
        thrust::fill(cost_from_root_.begin(), cost_from_root_.end(), CUDART_INF_F);

        // Set costs of root nodes to 0:
        thrust::scatter(thrust::make_constant_iterator<float>(0.0), thrust::make_constant_iterator<float>(0.0) + root_indices_.size(),
                        root_indices_.begin(), cost_from_root_.begin());

        const int num_steps = cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        int num_nodes_processed = 0;
        for (int s = 0; s < num_steps; s++)
        {
            int threadCount = 256;
            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);
            std::cout<<"forward_iteration: "<<s<<", blockCount: "<<blockCount<<", cur_num_bdd_nodes: "<<cur_num_bdd_nodes<<"\n";
            forward_step_with_solve<<<blockCount, threadCount>>>(cur_num_bdd_nodes, num_nodes_processed, omega,
                                                                thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                                                thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                                                thrust::raw_pointer_cast(lo_cost_.data()),
                                                                thrust::raw_pointer_cast(hi_cost_.data()),
                                                                thrust::raw_pointer_cast(delta_lo.data()),
                                                                thrust::raw_pointer_cast(delta_hi.data()),
                                                                thrust::raw_pointer_cast(cost_from_terminal_.data())
                                                                thrust::raw_pointer_cast(cost_from_root_.data()));
            num_nodes_processed += cur_num_bdd_nodes;
        }
        forward_state_valid_ = true;
    }
    
    void bdd_cuda_parallel_mma_sorting::solve(const size_t max_iter, const double tolerance, const double time_limit)
    {
        const auto start_time = std::chrono::steady_clock::now();
        double lb_prev = this->lower_bound();
        double lb_post = lb_prev;
        std::cout << "initial lower bound = " << lb_prev;
        auto time = std::chrono::steady_clock::now();
        std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
        std::cout << "\n";

        backward_run(False);
        delta_lo = thrust::device_vector<float>(nr_variables(), 0.0);
        delta_hi = thrust::device_vector<float>(nr_variables(), 0.0);
        for(size_t iter=0; iter<max_iter; ++iter)
        {
            iteration();
            lb_prev = lb_post;
            lb_post = this->lower_bound();
            std::cout << "iteration " << iter << ", lower bound = " << lb_post;
            time = std::chrono::steady_clock::now();
            double time_spent = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
            std::cout << ", time = " << time_spent << " s";
            std::cout << "\n";
            if (time_spent > time_limit)
            {
                std::cout << "Time limit reached." << std::endl;
                break;
            }
            if (std::abs(lb_prev-lb_post) < std::abs(tolerance*lb_prev))
            {
                std::cout << "Relative progress less than tolerance (" << tolerance << ")\n";
                break;
            }
        }
        std::cout << "final lower bound = " << this->lower_bound() << "\n"; 
    }
}
