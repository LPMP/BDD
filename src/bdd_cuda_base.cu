#include "bdd_cuda_base.h"
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

namespace LPMP {
    // copied from: https://github.com/treecode/Bonsai/blob/8904dd3ebf395ccaaf0eacef38933002b49fc3ba/runtime/profiling/derived_atomic_functions.h#L186
    __device__ __forceinline__ float atomicMin(float *address, float val) //TODO: Check!
    {
        int ret = __float_as_int(*address);
        while(val < __int_as_float(ret))
        {
            int old = ret;
            if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
                break;
        }
        return __int_as_float(ret);
    }

    struct assign_new_indices_func {
        int* new_indices;
        __host__ __device__ void operator()(int& idx)
        {
            idx = new_indices[idx];
        }
    };

    bdd_cuda_base::bdd_cuda_base(BDD::bdd_collection& bdd_col)
    {
        std::vector<int> primal_variable_index; //TODO: Possibly store in compressed format?
        std::vector<int> lo_bdd_node_index;
        std::vector<int> hi_bdd_node_index;
        // Store hop distance from root node, so that all nodes with same hop distance can be processed in parallel:
        std::vector<int> bdd_hop_dist_root;
        std::vector<int> bdd_indices; // which bdd index does the bdd node belong to.
        std::vector<int> num_vars_per_bdd;

        std::unordered_map<size_t, int> primal_var_count;
        //TODO: Iterate over BDDs in sorted order w.r.t number of nodes.
        int storage_offset = 0;
        num_dual_variables_ = 0;
        for(size_t bdd_idx=0; bdd_idx < bdd_col.nr_bdds(); ++bdd_idx)
        {
            assert(bdd_col.is_qbdd(bdd_idx));
            assert(bdd_col.is_reordered(bdd_idx));
            int cur_hop_dist = 0;
            size_t prev_var = bdd_col(bdd_idx, 0).index;
            int cur_num_variables = 1; // root node
            for(size_t bdd_node_idx=0; bdd_node_idx < bdd_col.nr_bdd_nodes(bdd_idx); ++bdd_node_idx)
            {
                const auto cur_instr = bdd_col(bdd_idx, bdd_node_idx);
                const size_t var = cur_instr.index;
                if(prev_var != var)
                {
                    assert(prev_var < var);
                    prev_var = var;
                    cur_hop_dist++;
                    if(!cur_instr.is_terminal())
                        cur_num_variables++;
                }
                if(!cur_instr.is_terminal())
                {
                    assert(bdd_node_idx < bdd_col.nr_bdd_nodes(bdd_idx) - 2); // only last two nodes can be terminal nodes. 
                    primal_variable_index.push_back(var);
                    lo_bdd_node_index.push_back(cur_instr.lo + storage_offset);
                    hi_bdd_node_index.push_back(cur_instr.hi + storage_offset);
                    auto it = primal_var_count.find(var);
                    if(it != primal_var_count.end())
                        it->second++;
                    else
                        primal_var_count[var] = 1;
                }
                else if(cur_instr.is_topsink())
                {
                    primal_variable_index.push_back(-1);
                    lo_bdd_node_index.push_back(-1);
                    hi_bdd_node_index.push_back(-1);
                    assert(bdd_node_idx >= bdd_col.nr_bdd_nodes(bdd_idx) - 2);
                }
                else
                {
                    assert(cur_instr.is_botsink());
                    primal_variable_index.push_back(-2);
                    lo_bdd_node_index.push_back(-2);
                    hi_bdd_node_index.push_back(-2);
                    assert(bdd_node_idx >= bdd_col.nr_bdd_nodes(bdd_idx) - 2);
                }
                bdd_hop_dist_root.push_back(cur_hop_dist);
                bdd_indices.push_back(bdd_node_idx);
            }
            num_vars_per_bdd.push_back(cur_num_variables);
            storage_offset += bdd_col.nr_bdd_nodes(bdd_idx);
            num_dual_variables_ += cur_num_variables;
        }
        // copy to GPU
        primal_variable_index_ = thrust::device_vector<int>(primal_variable_index.begin(), primal_variable_index.end());
        bdd_index_ = thrust::device_vector<int>(bdd_indices.begin(), bdd_indices.end());
        lo_bdd_node_index_ = thrust::device_vector<int>(lo_bdd_node_index.begin(), lo_bdd_node_index.end());
        hi_bdd_node_index_ = thrust::device_vector<int>(hi_bdd_node_index.begin(), lo_bdd_node_index.end());
        thrust::device_vector<int> bdd_hop_dist(bdd_hop_dist_root.begin(), bdd_hop_dist_root.end());
        cost_from_root_ = thrust::device_vector<float>(lo_bdd_node_index.size(), std::numeric_limits<float>::max());
        cost_from_terminal_ = thrust::device_vector<float>(lo_bdd_node_index.size(), std::numeric_limits<float>::max());
        hi_cost_ = thrust::device_vector<float>(lo_bdd_node_index.size(), std::numeric_limits<float>::max());
        hi_path_cost_ = thrust::device_vector<float>(lo_bdd_node_index.size());
        lo_path_cost_ = thrust::device_vector<float>(lo_bdd_node_index.size());
        num_vars_per_bdd_ = thrust::device_vector<int>(num_vars_per_bdd.begin(), num_vars_per_bdd.end());

        // At this point all nodes of a BDD are contiguous in memory. Now we convert this so that nodes with same
        // hop distances become contiguous.
        
        // Determine ordering:
        thrust::device_vector<int> sorting_order(lo_bdd_node_index.size());
        thrust::sequence(lo_bdd_node_index.begin(), lo_bdd_node_index.end());
        thrust::sort_by_key(bdd_hop_dist.begin(), bdd_hop_dist.end(), sorting_order.begin());

        // Sort BDD nodes:
        thrust::gather(sorting_order.begin(), sorting_order.end(), primal_variable_index_.begin(), primal_variable_index_.begin());
        thrust::gather(sorting_order.begin(), sorting_order.end(), bdd_index_.begin(), bdd_index_.begin());
        
        // Since the ordering is changed so lo, hi indices also need to be updated:
        assign_new_indices_func func({thrust::raw_pointer_cast(sorting_order.data())});
        thrust::for_each(lo_bdd_node_index.begin(), lo_bdd_node_index.end(), func);
        thrust::for_each(hi_bdd_node_index.begin(), hi_bdd_node_index.end(), func);

        // Count number of BDD nodes per hop distance:
        cum_nr_bdd_nodes_per_hop_dist_ = thrust::device_vector<int>(lo_bdd_node_index.size());
        auto last_red = thrust::reduce_by_key(bdd_hop_dist.begin(), bdd_hop_dist.end(), thrust::make_constant_iterator<int>(1), 
                                                thrust::make_discard_iterator(), 
                                                cum_nr_bdd_nodes_per_hop_dist_.begin());
        cum_nr_bdd_nodes_per_hop_dist_.resize(thrust::distance(cum_nr_bdd_nodes_per_hop_dist_.begin(), last_red.second));
        assert(cum_nr_bdd_nodes_per_hop_dist_[0] == bdd_col.nr_bdds()); // root nodes are 0 distance away and each BDD has exactly one root node.
        assert(cum_nr_bdd_nodes_per_hop_dist_.back() == 2 * bdd_col.nr_bdds());

        // Convert to cumulative:
        thrust::inclusive_scan(cum_nr_bdd_nodes_per_hop_dist_.begin(), cum_nr_bdd_nodes_per_hop_dist_.end(), cum_nr_bdd_nodes_per_hop_dist_.begin());

        nr_vars_ = *thrust::max_element(primal_variable_index_.begin(), primal_variable_index_.end()) + 1;
        nr_bdds_ = bdd_col.nr_bdds();
        nr_bdd_nodes_ = lo_bdd_node_index.size();

        // Populate variable counts:
        assert(primal_var_count.size() == nr_vars_);

        std::vector<int> primal_variable_counts(nr_vars_);
        for (const auto& [var, count] : primal_var_count) {
            primal_variable_counts[var] = count;
        }
        primal_variable_counts_ = thrust::device_vector<int>(primal_variable_counts.begin(), primal_variable_counts.end());
    }

    struct set_var_cost_func {
        int var_index;
        float cost;
        __host__ __device__ void operator()(const thrust::tuple<int, float&> t) const
        {
            const int cur_var_index = thrust::get<0>(t);
            if(cur_var_index != var_index)
                return;
            float& hi_cost = thrust::get<1>(t);
            hi_cost = cost;
        }
    };

    void bdd_cuda_base::set_cost(const double c, const size_t var)
    {
        assert(var < nr_vars_);
        set_var_cost_func func({(int) var, (float) c / primal_variable_counts_[var]});

        auto first = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), hi_cost_.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.end(), hi_cost_.end()));

        thrust::for_each(first, last, func);
    }

    struct set_vars_costs_func {
        int* var_counts;
        float* primal_costs;
        __host__ __device__ void operator()(const thrust::tuple<int, float&> t) const
        {
            const int cur_var_index = thrust::get<0>(t);
            float& hi_cost = thrust::get<1>(t);
            hi_cost = primal_costs[cur_var_index] / var_counts[cur_var_index];
        }
    };

    template<typename COST_ITERATOR> 
    void bdd_cuda_base::set_costs(COST_ITERATOR begin, COST_ITERATOR end)
    {
        assert(std::distance(begin, end) == nr_variables());
        thrust::device_vector<float> primal_costs(begin, end);
        
        set_vars_costs_func func({primal_variable_counts_, primal_costs});
        auto first = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), hi_cost_.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.end(), hi_cost_.end()));

        thrust::for_each(first, last, func);
    }

    __global__ void forward_step(const int cur_num_bdd_nodes,
        const int* const __restrict__ lo_bdd_node_index, 
        const int* const __restrict__ hi_bdd_node_index, 
        const float* const __restrict__ hi_cost,
        float* __restrict__ cost_from_root, 
        const bool is_first_step)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index; bdd_idx < cur_num_bdd_nodes; bdd_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_idx];
            if (next_lo_node < 0) // will matter when one row contains multiple BDDs, otherwise the terminal nodes are at the end anyway.
                continue; // nothing needs to be done for terminal node.

            const int next_hi_node = hi_bdd_node_index[bdd_idx];
            assert(next_hi_node >= 0);

            float cur_c_from_root = 0.0;
            if (!is_first_step)
                cur_c_from_root = cost_from_root[bdd_idx];
            
            const float cur_hi_cost = hi_cost[bdd_idx];
            
            // Uncoalesced writes:
            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root); // TODO: Set cost_from_root to infinity before starting next iterations.
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + cur_hi_cost);
        }
    }

    void bdd_cuda_base::forward_run()
    {
        const int num_steps = cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        int num_nodes_processed = 0;
        for (int s = 0; s < num_steps; s++)
        {
            int threadCount = 256;
            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);
            forward_step<<<blockCount, threadCount>>>(cur_num_bdd_nodes, 
                thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                thrust::raw_pointer_cast(hi_cost_.data()),
                thrust::raw_pointer_cast(cost_from_root_.data()),
                s == 0);
            num_nodes_processed += cur_num_bdd_nodes;
        }
    }

    __global__ void backward_step(const int cur_num_bdd_nodes, const int start_offset,
        const int* const __restrict__ lo_bdd_node_index, 
        const int* const __restrict__ hi_bdd_node_index, 
        const float* const __restrict__ hi_cost,
        const float* __restrict__ cost_from_root, 
        float* __restrict__ cost_from_terminal,
        float* __restrict__ lo_path_cost, 
        float* __restrict__ hi_path_cost, 
        const bool is_first_step)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index + start_offset; bdd_idx < cur_num_bdd_nodes + start_offset; bdd_idx += num_threads) 
        {
            float next_lo_node_cost_terminal = 0.0;
            float next_hi_node_cost_terminal = 0.0;

            if(!is_first_step)
            {
                next_lo_node_cost_terminal = cost_from_terminal[lo_bdd_node_index[bdd_idx]];
                next_hi_node_cost_terminal = cost_from_terminal[hi_bdd_node_index[bdd_idx]];
            }
            const float cur_hi_cost_from_terminal = next_hi_node_cost_terminal + hi_cost[bdd_idx];
            cost_from_terminal[bdd_idx] = min(cur_hi_cost_from_terminal, next_lo_node_cost_terminal);

            const float cur_cost_from_root = cost_from_root[bdd_idx];
            hi_path_cost[bdd_idx] = cur_cost_from_root + cur_hi_cost_from_terminal;
            lo_path_cost[bdd_idx] = cur_cost_from_root + next_lo_node_cost_terminal;
        }
    }

    void bdd_cuda_base::backward_run()
    {
        const int num_steps = cum_nr_bdd_nodes_per_hop_dist_.size() - 1;

        for (int s = num_steps; s >= 0; s--)
        {
            int threadCount = 256;
            int start_offset = 0;
            if(s > 0)
                start_offset -= cum_nr_bdd_nodes_per_hop_dist_[s - 1];

            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - start_offset;
            int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);
            backward_step<<<blockCount, threadCount>>>(cur_num_bdd_nodes, start_offset,
                thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                thrust::raw_pointer_cast(hi_cost_.data()),
                thrust::raw_pointer_cast(cost_from_root_.data()),
                thrust::raw_pointer_cast(cost_from_terminal_.data()),
                thrust::raw_pointer_cast(lo_path_cost_.data()),
                thrust::raw_pointer_cast(hi_path_cost_.data()),
                s == num_steps);
        }
    }

    struct tuple_min
    {
        __host__ __device__
        thrust::tuple<float, float> operator()(const thrust::tuple<float, float>& t0, const thrust::tuple<float, float>& t1)
        {
            return thrust::make_tuple(min(thrust::get<0>(t0), thrust::get<0>(t1)), min(thrust::get<1>(t0), thrust::get<1>(t1)));
        }
    };

    // Compute min-marginals by knowing primal var index and also the bdd index of each bdd node.
    // TODO: Warp aggregation or not (?) https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
    std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>, thrust::device_vector<float>> 
        bdd_cuda_base::min_marginals_cuda()
    {
        forward_run();
        backward_run();

        thrust::device_vector<int> primal_variable_index_sorted = primal_variable_index_;
        thrust::device_vector<int> bdd_index_sorted = bdd_index_;
        thrust::device_vector<float> lo_path_cost_sorted = lo_path_cost_;
        thrust::device_vector<float> hi_path_cost_sorted = hi_path_cost_;

        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_index_sorted.begin(), primal_variable_index_sorted.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_index_sorted.end(), primal_variable_index_sorted.end()));

        auto first_val = thrust::make_zip_iterator(thrust::make_tuple(lo_path_cost_sorted.begin(), hi_path_cost_sorted.begin()));
        thrust::sort_by_key(first_key, last_key, first_val); //TODO: Necessary? reduce_by_key does not requires sorted only that all equal elements are consecutive.

        //TODO: Allocate less memory?
        thrust::device_vector<int> min_marginal_primal_index(nr_bdd_nodes_);
        thrust::device_vector<int> min_marginal_bdd_index(nr_bdd_nodes_);
        auto first_out_key = thrust::make_zip_iterator(thrust::make_tuple(min_marginal_bdd_index.begin(), min_marginal_primal_index.begin()));

        thrust::device_vector<float> min_marginals_lo(nr_bdd_nodes_);
        thrust::device_vector<float> min_marginals_hi(nr_bdd_nodes_); 
        auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(min_marginals_lo.begin(), min_marginals_hi.begin()));

        thrust::equal_to<thrust::tuple<int, int>> binary_pred;

        auto new_end = thrust::reduce_by_key(first_key, last_key, first_val, first_out_key, first_out_val, binary_pred, tuple_min());
        const int out_size = thrust::distance(first_out_key, new_end.first);
        assert(num_dual_variables_ == out_size);   //TODO: Check size and keep reusing old memory.

        min_marginals_lo.resize(out_size);
        min_marginals_hi.resize(out_size);
        min_marginal_primal_index.resize(out_size);
        min_marginal_bdd_index.resize(out_size);
        return {min_marginal_primal_index, min_marginal_bdd_index, min_marginals_lo, min_marginals_hi};
    }

    std::vector<std::vector<std::array<float, 2>>> bdd_cuda_base::min_marginals()
    {
        thrust::device_vector<int> mm_primal_index, mm_bdd_index;
        thrust::device_vector<float> mm_0, mm_1;

        std::tie(mm_primal_index, mm_bdd_index, mm_0, mm_1) = min_marginals_cuda();

        std::vector<int> num_vars_per_bdd(num_vars_per_bdd_.size());
        thrust::copy(num_vars_per_bdd_.begin(), num_vars_per_bdd_.end(), num_vars_per_bdd.begin());

        std::vector<int> h_mm_primal_index(mm_primal_index.size());
        thrust::copy(mm_primal_index.begin(), mm_primal_index.end(), h_mm_primal_index.begin());

        std::vector<int> h_mm_bdd_index(mm_primal_index.size());
        thrust::copy(mm_bdd_index.begin(), mm_bdd_index.end(), h_mm_bdd_index.begin());

        std::vector<float> h_mm_0(mm_primal_index.size());
        thrust::copy(mm_0.begin(), mm_0.end(), h_mm_0.begin());

        std::vector<float> h_mm_1(mm_primal_index.size());
        thrust::copy(mm_1.begin(), mm_1.end(), h_mm_1.begin());

        std::vector<std::vector<std::array<float,2>>> min_margs(nr_bdds());
        int idx_1d = 0;
        for(int bdd_idx=0; bdd_idx < nr_bdds(); ++bdd_idx)
        {
            for(int var = 0; var < num_vars_per_bdd[bdd_idx]; var++, idx_1d++)
            {
                std::array<float,2> mm = {h_mm_0[idx_1d], h_mm_1[idx_1d]};
                min_margs[bdd_idx].push_back(mm);
            }
        }
        return min_margs;
    }

    struct return_top_sink_costs
    {
        __host__ __device__ double operator()(const thrust::tuple<int, float>& t) const
        {
            const int primal_index = thrust::get<0>(t);
            if (primal_index != -1)
                return 0.0;
            return thrust::get<1>(t);
        }
    };

    double bdd_cuda_base::lower_bound()
    {
        // Gather all BDD nodes corresponding to top_sink (i.e. primal_variable_index == -1) and sum their costs_from_root
        auto first = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), cost_from_root_.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.end(), cost_from_root_.end()));

        return thrust::transform_reduce(first, last, return_top_sink_costs(), 0.0, thrust::plus<double>());
    }
}
