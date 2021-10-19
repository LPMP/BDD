#include "bdd_cuda_base.h"
#include "time_measure_util.h"
#include "cuda_utils.h"
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

namespace LPMP {

    struct assign_new_indices_func {
        const int* new_indices;
        __host__ __device__ void operator()(int& idx)
        {
            if(idx >= 0) // non-terminal nodes.
                idx = new_indices[idx];
        }
    };

    struct not_equal_to
    {
        const int* values;
        const int val_to_search;
        __host__ __device__
        bool operator()(const int i) const
        {
            return values[i] != val_to_search;
        }
    };

    bdd_cuda_base::bdd_cuda_base(const BDD::bdd_collection& bdd_col)
    {
        initialize(bdd_col);
        thrust::device_vector<int> bdd_hop_dist_root, bdd_depth;
        std::tie(bdd_hop_dist_root, bdd_depth) = populate_bdd_nodes(bdd_col);
        reorder_bdd_nodes(bdd_hop_dist_root, bdd_depth);
        set_special_nodes_indices(bdd_hop_dist_root);
        compress_bdd_nodes_to_layer(bdd_hop_dist_root);
        print_num_bdd_nodes_per_hop();
    }

    void bdd_cuda_base::initialize(const BDD::bdd_collection& bdd_col)
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        nr_vars_ = [&]() {
            size_t max_v=0;
            for(size_t bdd_nr=0; bdd_nr<bdd_col.nr_bdds(); ++bdd_nr)
                max_v = std::max(max_v, bdd_col.min_max_variables(bdd_nr)[1]);
            return max_v+1;
        }();
        nr_bdds_ = bdd_col.nr_bdds();
        std::vector<int> primal_variable_counts(nr_vars_, 0);
        std::vector<int> num_vars_per_bdd;
        for(size_t bdd_idx=0; bdd_idx < bdd_col.nr_bdds(); ++bdd_idx)
        {
            const std::vector<size_t> cur_bdd_variables = bdd_col.variables(bdd_idx);
            for (const auto& var : cur_bdd_variables)
                primal_variable_counts[var]++;
            num_vars_per_bdd.push_back(cur_bdd_variables.size());
            num_dual_variables_ += cur_bdd_variables.size();
            nr_bdd_nodes_ += bdd_col.nr_bdd_nodes(bdd_idx);
        }
        num_bdds_per_var_ = thrust::device_vector<int>(primal_variable_counts.begin(), primal_variable_counts.end());
        num_vars_per_bdd_ = thrust::device_vector<int>(num_vars_per_bdd.begin(), num_vars_per_bdd.end());
        // Initialize data per BDD node: 
        hi_cost_ = thrust::device_vector<float>(nr_bdd_nodes_, 0.0);
        lo_cost_ = thrust::device_vector<float>(nr_bdd_nodes_, 0.0);
        cost_from_root_ = thrust::device_vector<float>(nr_bdd_nodes_);
        cost_from_terminal_ = thrust::device_vector<float>(nr_bdd_nodes_);
        hi_path_cost_ = thrust::device_vector<float>(nr_bdd_nodes_);
        lo_path_cost_ = thrust::device_vector<float>(nr_bdd_nodes_);
    }

    std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> bdd_cuda_base::populate_bdd_nodes(const BDD::bdd_collection& bdd_col)
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        std::vector<int> primal_variable_index;
        std::vector<int> lo_bdd_node_index;
        std::vector<int> hi_bdd_node_index;
        std::vector<int> bdd_index;
        std::vector<int> bdd_depth;
        // Store hop distance from root node, so that all nodes with same hop distance can be processed in parallel:
        std::vector<int> bdd_hop_dist_root;

        for(size_t bdd_idx=0; bdd_idx < bdd_col.nr_bdds(); ++bdd_idx)
        {
            assert(bdd_col.is_qbdd(bdd_idx));
            assert(bdd_col.is_reordered(bdd_idx));
            int cur_hop_dist = 0;
            const size_t storage_offset = bdd_col.offset(bdd_idx);
            size_t prev_var = bdd_col(bdd_idx, storage_offset).index;
            for(size_t bdd_node_idx=0; bdd_node_idx < bdd_col.nr_bdd_nodes(bdd_idx); ++bdd_node_idx)
            {
                const auto cur_instr = bdd_col(bdd_idx, bdd_node_idx + storage_offset);
                const size_t var = cur_instr.index;
                if(prev_var != var)
                {
                    assert(prev_var < var || cur_instr.is_terminal());
                    prev_var = var;
                    if(!cur_instr.is_topsink())
                        cur_hop_dist++; // both terminal nodes can have same hop distance.
                }
                if(!cur_instr.is_terminal())
                {
                    assert(bdd_node_idx < bdd_col.nr_bdd_nodes(bdd_idx) - 2); // only last two nodes can be terminal nodes. 
                    primal_variable_index.push_back(var);
                    lo_bdd_node_index.push_back(cur_instr.lo);
                    hi_bdd_node_index.push_back(cur_instr.hi);
                }
                else
                {
                    primal_variable_index.push_back(INT_MAX);
                    const int terminal_indicator = cur_instr.is_topsink() ? TOP_SINK_INDICATOR_CUDA: BOT_SINK_INDICATOR_CUDA;
                    lo_bdd_node_index.push_back(terminal_indicator);
                    hi_bdd_node_index.push_back(terminal_indicator);
                    assert(bdd_node_idx >= bdd_col.nr_bdd_nodes(bdd_idx) - 2);
                }
                bdd_hop_dist_root.push_back(cur_hop_dist);
                bdd_index.push_back(bdd_idx);
            }
            bdd_depth.insert(bdd_depth.end(), bdd_col.nr_bdd_nodes(bdd_idx), cur_hop_dist);
        }

        // copy to GPU
        primal_variable_index_ = thrust::device_vector<int>(primal_variable_index.begin(), primal_variable_index.end());
        bdd_index_ = thrust::device_vector<int>(bdd_index.begin(), bdd_index.end());
        lo_bdd_node_index_ = thrust::device_vector<int>(lo_bdd_node_index.begin(), lo_bdd_node_index.end());
        hi_bdd_node_index_ = thrust::device_vector<int>(hi_bdd_node_index.begin(), hi_bdd_node_index.end());
        thrust::device_vector<int> bdd_hop_dist_dev(bdd_hop_dist_root.begin(), bdd_hop_dist_root.end());
        thrust::device_vector<int> bdd_depth_dev(bdd_depth.begin(), bdd_depth.end());
        return {bdd_hop_dist_dev, bdd_depth_dev};
    }

    void bdd_cuda_base::reorder_bdd_nodes(thrust::device_vector<int>& bdd_hop_dist_dev, thrust::device_vector<int>& bdd_depth_dev)
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        // Make nodes with same hop distance, BDD depth and bdd index contiguous in that order.
        thrust::device_vector<int> sorting_order(nr_bdd_nodes_);
        thrust::sequence(sorting_order.begin(), sorting_order.end());
        
        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.begin(), bdd_depth_dev.begin(), bdd_index_.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.end(), bdd_depth_dev.begin(), bdd_index_.end()));

        auto first_bdd_val = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), lo_bdd_node_index_.begin(), 
                                                                        hi_bdd_node_index_.begin(), sorting_order.begin()));
        thrust::sort_by_key(first_key, last_key, first_bdd_val);
        
        // Since the ordering is changed so lo, hi indices also need to be updated:
        thrust::device_vector<int> new_indices(sorting_order.size());
        thrust::scatter(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + sorting_order.size(), 
                        sorting_order.begin(), new_indices.begin());
        assign_new_indices_func func({thrust::raw_pointer_cast(new_indices.data())});
        thrust::for_each(lo_bdd_node_index_.begin(), lo_bdd_node_index_.end(), func);
        thrust::for_each(hi_bdd_node_index_.begin(), hi_bdd_node_index_.end(), func);

        // Count number of BDD nodes per hop distance. Need for launching CUDA kernel with appropiate offset and threads:
        thrust::device_vector<int> dev_cum_nr_bdd_nodes_per_hop_dist(nr_bdd_nodes_);
        auto last_red = thrust::reduce_by_key(bdd_hop_dist_dev.begin(), bdd_hop_dist_dev.end(), thrust::make_constant_iterator<int>(1), 
                                                thrust::make_discard_iterator(), 
                                                dev_cum_nr_bdd_nodes_per_hop_dist.begin());
        dev_cum_nr_bdd_nodes_per_hop_dist.resize(thrust::distance(dev_cum_nr_bdd_nodes_per_hop_dist.begin(), last_red.second));

        // Convert to cumulative:
        thrust::inclusive_scan(dev_cum_nr_bdd_nodes_per_hop_dist.begin(), dev_cum_nr_bdd_nodes_per_hop_dist.end(), dev_cum_nr_bdd_nodes_per_hop_dist.begin());

        cum_nr_bdd_nodes_per_hop_dist_ = std::vector<int>(dev_cum_nr_bdd_nodes_per_hop_dist.size());
        thrust::copy(dev_cum_nr_bdd_nodes_per_hop_dist.begin(), dev_cum_nr_bdd_nodes_per_hop_dist.end(), cum_nr_bdd_nodes_per_hop_dist_.begin());
    }

    void bdd_cuda_base::set_special_nodes_indices(const thrust::device_vector<int>& bdd_hop_dist_dev)
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        // Set indices of BDD nodes which are root, top, bot sinks.
        root_indices_ = thrust::device_vector<int>(nr_bdd_nodes_);
        thrust::sequence(root_indices_.begin(), root_indices_.end());
        auto last_root = thrust::remove_if(root_indices_.begin(), root_indices_.end(),
                                            not_equal_to({thrust::raw_pointer_cast(bdd_hop_dist_dev.data()), 0})); //TODO: This needs to be changed when multiple BDDs are in one row.
        root_indices_.resize(std::distance(root_indices_.begin(), last_root));
        assert(root_indices_.size() == nr_bdds_);

        bot_sink_indices_ = thrust::device_vector<int>(nr_bdd_nodes_);
        thrust::sequence(bot_sink_indices_.begin(), bot_sink_indices_.end());
        auto last_bot_sink = thrust::remove_if(bot_sink_indices_.begin(), bot_sink_indices_.end(),
                                            not_equal_to({thrust::raw_pointer_cast(lo_bdd_node_index_.data()), BOT_SINK_INDICATOR_CUDA}));
        bot_sink_indices_.resize(std::distance(bot_sink_indices_.begin(), last_bot_sink));
        assert(bot_sink_indices_.size() == nr_bdds_);

        top_sink_indices_ = thrust::device_vector<int>(nr_bdd_nodes_);
        thrust::sequence(top_sink_indices_.begin(), top_sink_indices_.end());
        auto last_top_sink = thrust::remove_if(top_sink_indices_.begin(), top_sink_indices_.end(),
                                            not_equal_to({thrust::raw_pointer_cast(lo_bdd_node_index_.data()), TOP_SINK_INDICATOR_CUDA}));
        top_sink_indices_.resize(std::distance(top_sink_indices_.begin(), last_top_sink));

        // Set costs of top sinks to itself to 0:
        thrust::scatter(thrust::make_constant_iterator<float>(0.0), thrust::make_constant_iterator<float>(0.0) + top_sink_indices_.size(),
                        top_sink_indices_.begin(), cost_from_terminal_.begin());

        // Set costs of bot sinks to top to infinity:
        thrust::scatter(thrust::make_constant_iterator<float>(CUDART_INF_F_HOST), thrust::make_constant_iterator<float>(CUDART_INF_F_HOST) + bot_sink_indices_.size(),
                        bot_sink_indices_.begin(), cost_from_terminal_.begin());

        assert(top_sink_indices_.size() == nr_bdds_);
    }

    // Removes redundant information in hi_costs, primal_index, bdd_index as it is duplicated across
    // multiple BDD nodes for each layer.
    void bdd_cuda_base::compress_bdd_nodes_to_layer(const thrust::device_vector<int>& bdd_hop_dist_dev)
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        thrust::device_vector<float> hi_cost_compressed(hi_cost_.size());
        thrust::device_vector<float> lo_cost_compressed(lo_cost_.size());
        thrust::device_vector<int> primal_index_compressed(primal_variable_index_.size()); 
        thrust::device_vector<int> bdd_index_compressed(bdd_index_.size());
        
        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.begin(), bdd_index_.begin(), primal_variable_index_.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.end(), bdd_index_.end(), primal_variable_index_.end()));

        auto first_out_key = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), bdd_index_compressed.begin(), primal_index_compressed.begin()));

        // Compute number of BDD nodes in each layer:
        bdd_layer_width_ = thrust::device_vector<int>(nr_bdd_nodes_);
        auto new_end = thrust::reduce_by_key(first_key, last_key, thrust::make_constant_iterator<int>(1), first_out_key, bdd_layer_width_.begin());
        const int out_size = thrust::distance(first_out_key, new_end.first);      

        // Assign bdd node to layer map:
        bdd_node_to_layer_map_ = thrust::device_vector<int>(out_size);
        thrust::sequence(bdd_node_to_layer_map_.begin(), bdd_node_to_layer_map_.end());
        bdd_node_to_layer_map_ = repeat_values(bdd_node_to_layer_map_, bdd_layer_width_);

        // Compress hi_costs_, lo_costs_ (although initially they are infinity, 0 resp.) and also populate how many BDD layers per hop dist.
        thrust::device_vector<int> bdd_hop_dist_compressed(out_size);
        auto first_cost_val = thrust::make_zip_iterator(thrust::make_tuple(hi_cost_.begin(), lo_cost_.begin(), bdd_hop_dist_dev.begin()));
        auto first_cost_val_compressed = thrust::make_zip_iterator(thrust::make_tuple(hi_cost_compressed.begin(), lo_cost_compressed.begin(), bdd_hop_dist_compressed.begin()));

        auto new_end_unique = thrust::unique_by_key_copy(first_key, last_key, first_cost_val, thrust::make_discard_iterator(), first_cost_val_compressed);
        assert(out_size == thrust::distance(first_cost_val_compressed, new_end_unique.second));

        hi_cost_compressed.resize(out_size);
        lo_cost_compressed.resize(out_size);
        primal_index_compressed.resize(out_size);
        bdd_index_compressed.resize(out_size);
        bdd_layer_width_.resize(out_size);

        thrust::swap(lo_cost_compressed, lo_cost_);
        thrust::swap(hi_cost_compressed, hi_cost_);
        thrust::swap(primal_index_compressed, primal_variable_index_);
        thrust::swap(bdd_index_compressed, bdd_index_);

        // For launching kernels where each thread operates on a BDD layer instead of a BDD node.
        layer_offsets_ = thrust::device_vector<int>(bdd_layer_width_.size() + 1);
        layer_offsets_[0] = 0;
        thrust::inclusive_scan(bdd_layer_width_.begin(), bdd_layer_width_.end(), layer_offsets_.begin() + 1);

        thrust::device_vector<int> dev_cum_nr_layers_per_hop_dist(cum_nr_bdd_nodes_per_hop_dist_.size());
        cum_nr_layers_per_hop_dist_ = std::vector<int>(dev_cum_nr_layers_per_hop_dist.size());

        thrust::reduce_by_key(bdd_hop_dist_compressed.begin(), bdd_hop_dist_compressed.end(), thrust::make_constant_iterator<int>(1), 
                            thrust::make_discard_iterator(), dev_cum_nr_layers_per_hop_dist.begin());

        thrust::inclusive_scan(dev_cum_nr_layers_per_hop_dist.begin(), dev_cum_nr_layers_per_hop_dist.end(), dev_cum_nr_layers_per_hop_dist.begin());
        thrust::copy(dev_cum_nr_layers_per_hop_dist.begin(), dev_cum_nr_layers_per_hop_dist.end(), cum_nr_layers_per_hop_dist_.begin());
    }

    void bdd_cuda_base::flush_forward_states()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        forward_state_valid_ = false;
        path_costs_valid_ = false;
    }

    void bdd_cuda_base::flush_backward_states()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        backward_state_valid_ = false;
        path_costs_valid_ = false;
    }

    void bdd_cuda_base::print_num_bdd_nodes_per_hop()
    {
        int prev = 0;
        for(int i = 0; i < cum_nr_bdd_nodes_per_hop_dist_.size(); i++)
        {
            std::cout<<"Hop: "<<i<<", # BDD nodes: "<<cum_nr_bdd_nodes_per_hop_dist_[i] - prev<<std::endl;
            prev = cum_nr_bdd_nodes_per_hop_dist_[i];
        }
    }

    struct set_var_cost_func {
        int var_index;
        float cost;
        __device__ void operator()(const thrust::tuple<int, float&> t) const
        {
            const int cur_var_index = thrust::get<0>(t);
            if(cur_var_index != var_index)
                return;
            float& arc_cost = thrust::get<1>(t);
            arc_cost += cost;
        }
    };

    void bdd_cuda_base::set_cost(const double c, const size_t var)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        assert(var < nr_vars_);
        set_var_cost_func func({(int) var, (float) c / num_bdds_per_var_[var]});

        auto first = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), hi_cost_.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.end(), hi_cost_.end()));

        thrust::for_each(first, last, func);
        flush_forward_states();
        flush_backward_states();
    }

    struct set_vars_costs_func {
        int* var_counts;
        float* primal_costs;
        __host__ __device__ void operator()(const thrust::tuple<int, float&> t) const
        {
            const int cur_var_index = thrust::get<0>(t);
            if (cur_var_index == INT_MAX)
                return; // terminal node.
            float& arc_cost = thrust::get<1>(t);
            const int count = var_counts[cur_var_index];
            assert(count > 0);
            arc_cost += primal_costs[cur_var_index] / count;
        }
    };

    template<typename COST_ITERATOR> 
    void bdd_cuda_base::update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        assert(std::distance(cost_lo_begin, cost_lo_end) == nr_variables() || std::distance(cost_lo_begin, cost_lo_end) == 0);
        assert(std::distance(cost_hi_begin, cost_hi_end) == nr_variables() || std::distance(cost_hi_begin, cost_hi_end) == 0);

        auto populate_costs = [&](auto cost_begin, auto cost_end, auto base_cost_begin, auto base_cost_end) {
            thrust::device_vector<float> primal_costs(cost_begin, cost_end);

            set_vars_costs_func func({thrust::raw_pointer_cast(num_bdds_per_var_.data()), 
                    thrust::raw_pointer_cast(primal_costs.data())});
            auto first = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), base_cost_begin));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.end(), base_cost_end));

            thrust::for_each(first, last, func);
        };

        if(std::distance(cost_lo_begin, cost_lo_end) > 0)
            populate_costs(cost_lo_begin, cost_lo_end, lo_cost_.begin(), lo_cost_.end());
        if(std::distance(cost_hi_begin, cost_hi_end) > 0)
            populate_costs(cost_hi_begin, cost_hi_end, hi_cost_.begin(), hi_cost_.end()); 

        flush_forward_states();
        flush_backward_states();
    }

    template void bdd_cuda_base::update_costs(double*, double*, double*, double*);
    template void bdd_cuda_base::update_costs(float*, float*, float*, float*);
    template void bdd_cuda_base::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_cuda_base::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_cuda_base::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_cuda_base::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);


    __global__ void forward_step(const int cur_num_bdd_nodes, const int start_offset,
                                const int* const __restrict__ lo_bdd_node_index, 
                                const int* const __restrict__ hi_bdd_node_index, 
                                const int* const __restrict__ bdd_node_to_layer_map, 
                                const float* const __restrict__ lo_cost,
                                const float* const __restrict__ hi_cost,
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

            const float cur_c_from_root = cost_from_root[bdd_idx];
            const int layer_idx = bdd_node_to_layer_map[bdd_idx];

            // Uncoalesced writes:
            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root + lo_cost[layer_idx]);
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + hi_cost[layer_idx]);
        }
    }

    void bdd_cuda_base::forward_run()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        if (forward_state_valid_)
            return;

        thrust::fill(cost_from_root_.begin(), cost_from_root_.end(), CUDART_INF_F_HOST);
        // Set costs of root nodes to 0:
        thrust::scatter(thrust::make_constant_iterator<float>(0.0), thrust::make_constant_iterator<float>(0.0) + root_indices_.size(),
                        root_indices_.begin(), cost_from_root_.begin());

        const int num_steps = cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        int num_nodes_processed = 0;
        for (int s = 0; s < num_steps; s++)
        {
            int threadCount = NUM_THREADS;
            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);
            forward_step<<<blockCount, threadCount>>>(cur_num_bdd_nodes, num_nodes_processed,
                                                    thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                                    thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                                    thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                                    thrust::raw_pointer_cast(lo_cost_.data()),
                                                    thrust::raw_pointer_cast(hi_cost_.data()),
                                                    thrust::raw_pointer_cast(cost_from_root_.data()));
            num_nodes_processed += cur_num_bdd_nodes;
        }
        forward_state_valid_ = true;
    }

    __global__ void backward_step_with_path_costs(const int cur_num_bdd_nodes, const int start_offset,
                                                const int* const __restrict__ lo_bdd_node_index, 
                                                const int* const __restrict__ hi_bdd_node_index, 
                                                const int* const __restrict__ bdd_node_to_layer_map, 
                                                const float* const __restrict__ lo_cost,
                                                const float* const __restrict__ hi_cost,
                                                const float* __restrict__ cost_from_root, 
                                                float* __restrict__ cost_from_terminal,
                                                float* __restrict__ lo_path_cost, 
                                                float* __restrict__ hi_path_cost)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index + start_offset; bdd_idx < cur_num_bdd_nodes + start_offset; bdd_idx += num_threads) 
        {
            const int lo_node = lo_bdd_node_index[bdd_idx];
            if (lo_node < 0)
                continue; // terminal node.
            const int hi_node = hi_bdd_node_index[bdd_idx];

            const int layer_idx = bdd_node_to_layer_map[bdd_idx];
            float cur_hi_cost_from_terminal = cost_from_terminal[hi_node] + hi_cost[layer_idx];
            float cur_lo_cost_from_terminal = cost_from_terminal[lo_node] + lo_cost[layer_idx];
            const float cur_cost_from_root = cost_from_root[bdd_idx];

            hi_path_cost[bdd_idx] = cur_cost_from_root + cur_hi_cost_from_terminal;
            lo_path_cost[bdd_idx] = cur_cost_from_root + cur_lo_cost_from_terminal;
            cost_from_terminal[bdd_idx] = min(cur_hi_cost_from_terminal, cur_lo_cost_from_terminal);
        }
    }

    __global__ void backward_step(const int cur_num_bdd_nodes, const int start_offset,
                                    const int* const __restrict__ lo_bdd_node_index, 
                                    const int* const __restrict__ hi_bdd_node_index, 
                                    const int* const __restrict__ bdd_node_to_layer_map, 
                                    const float* const __restrict__ lo_cost,
                                    const float* const __restrict__ hi_cost,
                                    float* __restrict__ cost_from_terminal)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index + start_offset; bdd_idx < cur_num_bdd_nodes + start_offset; bdd_idx += num_threads) 
        {
            const int lo_node = lo_bdd_node_index[bdd_idx];
            if (lo_node < 0)
                continue; // terminal node.
            const int hi_node = hi_bdd_node_index[bdd_idx];

            const int layer_idx = bdd_node_to_layer_map[bdd_idx];
            cost_from_terminal[bdd_idx] = min(cost_from_terminal[hi_node] + hi_cost[layer_idx], cost_from_terminal[lo_node] + lo_cost[layer_idx]);
        }
    }


    void bdd_cuda_base::backward_run(bool compute_path_costs)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        if ((backward_state_valid_ && path_costs_valid_) ||
            (!compute_path_costs && backward_state_valid_))
            return;

        for (int s = cum_nr_bdd_nodes_per_hop_dist_.size() - 2; s >= 0; s--)
        {
            int threadCount = 256;
            int start_offset = 0;
            if(s > 0)
                start_offset = cum_nr_bdd_nodes_per_hop_dist_[s - 1];

            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - start_offset;
            int blockCount = ceil(cur_num_bdd_nodes / (float) threadCount);
            if (compute_path_costs)
                backward_step_with_path_costs<<<blockCount, threadCount>>>(cur_num_bdd_nodes, start_offset,
                                                        thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                                        thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                                        thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                                        thrust::raw_pointer_cast(lo_cost_.data()),
                                                        thrust::raw_pointer_cast(hi_cost_.data()),
                                                        thrust::raw_pointer_cast(cost_from_root_.data()),
                                                        thrust::raw_pointer_cast(cost_from_terminal_.data()),
                                                        thrust::raw_pointer_cast(lo_path_cost_.data()),
                                                        thrust::raw_pointer_cast(hi_path_cost_.data()));
            else
                backward_step<<<blockCount, threadCount>>>(cur_num_bdd_nodes, start_offset,
                                                        thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                                        thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                                        thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                                        thrust::raw_pointer_cast(lo_cost_.data()),
                                                        thrust::raw_pointer_cast(hi_cost_.data()),
                                                        thrust::raw_pointer_cast(cost_from_terminal_.data()));

        }
        backward_state_valid_ = true;
        if (compute_path_costs)
            path_costs_valid_ = true;
    }

    struct tuple_min
    {
        __host__ __device__
        thrust::tuple<float, float> operator()(const thrust::tuple<float, float>& t0, const thrust::tuple<float, float>& t1)
        {
            return thrust::make_tuple(min(thrust::get<0>(t0), thrust::get<0>(t1)), min(thrust::get<1>(t0), thrust::get<1>(t1)));
        }
    };

    // Computes min-marginals by reduction.
    std::tuple<thrust::device_vector<float>, thrust::device_vector<float>> bdd_cuda_base::min_marginals_cuda()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        forward_run();
        backward_run();

        auto first_val = thrust::make_zip_iterator(thrust::make_tuple(lo_path_cost_.begin(), hi_path_cost_.begin()));

        thrust::device_vector<float> min_marginals_lo(hi_cost_.size());
        thrust::device_vector<float> min_marginals_hi(hi_cost_.size());
        auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(min_marginals_lo.begin(), min_marginals_hi.begin()));

        thrust::equal_to<int> binary_pred;

        auto new_end = thrust::reduce_by_key(bdd_node_to_layer_map_.begin(), bdd_node_to_layer_map_.end(), first_val, thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_min());
        const int out_size = thrust::distance(first_out_val, new_end.second);
        assert(out_size == hi_cost_.size());

        return {min_marginals_lo, min_marginals_hi};
    }

    two_dim_variable_array<std::array<double,2>> bdd_cuda_base::min_marginals()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        thrust::device_vector<float> mm_0, mm_1;

        std::tie(mm_0, mm_1) = min_marginals_cuda();

        // sort the min-marginals per bdd_index, primal_index:
        thrust::device_vector<int> bdd_index_sorted = bdd_index_;
        thrust::device_vector<int> primal_variable_index_sorted = primal_variable_index_;
        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_sorted.begin(), bdd_index_sorted.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_sorted.end(), bdd_index_sorted.end()));

        auto first_val = thrust::make_zip_iterator(thrust::make_tuple(mm_0.begin(), mm_1.begin()));

        thrust::sort_by_key(first_key, last_key, first_val);

        std::vector<int> num_bdds_per_var(num_bdds_per_var_.size());
        thrust::copy(num_bdds_per_var_.begin(), num_bdds_per_var_.end(), num_bdds_per_var.begin());

        std::vector<int> h_mm_primal_index(primal_variable_index_sorted.size());
        thrust::copy(primal_variable_index_sorted.begin(), primal_variable_index_sorted.end(), h_mm_primal_index.begin());

        std::vector<int> h_mm_bdd_index(bdd_index_sorted.size());
        thrust::copy(bdd_index_sorted.begin(), bdd_index_sorted.end(), h_mm_bdd_index.begin());

        std::vector<float> h_mm_0(mm_0.size());
        thrust::copy(mm_0.begin(), mm_0.end(), h_mm_0.begin());

        std::vector<float> h_mm_1(mm_1.size());
        thrust::copy(mm_1.begin(), mm_1.end(), h_mm_1.begin());

        std::vector<int> h_bdd_node_to_layer_map(bdd_node_to_layer_map_.size());
        thrust::copy(bdd_node_to_layer_map_.begin(), bdd_node_to_layer_map_.end(), h_bdd_node_to_layer_map.begin());

        two_dim_variable_array<std::array<double,2>> min_margs(num_bdds_per_var);

        for (int i = 0; i < nr_bdds_; ++i)
            assert(h_mm_primal_index[h_mm_primal_index.size() - 1 - i] == INT_MAX);
        int idx_1d = 0;
        for(int var = 0; var < nr_vars_; ++var)
        {
            assert(num_bdds_per_var[var] > 0);
            for(int bdd_idx = 0; bdd_idx < num_bdds_per_var[var]; ++bdd_idx, ++idx_1d)
            {
                assert(idx_1d < h_mm_primal_index.size() - nr_bdds_ && idx_1d < h_mm_0.size() - nr_bdds_ && idx_1d < h_mm_1.size() - nr_bdds_);
                assert(h_mm_primal_index[idx_1d] < INT_MAX); // Should ignore terminal nodes.
                std::array<double,2> mm = {h_mm_0[idx_1d], h_mm_1[idx_1d]};
                min_margs(var, bdd_idx) = mm;
            }
        }

        return min_margs;
    }

    void bdd_cuda_base::update_costs(const thrust::device_vector<float>& update_vec)
    {
        thrust::transform(hi_cost_.begin(), hi_cost_.end(), update_vec.begin(), hi_cost_.begin(), thrust::plus<float>());
        flush_forward_states();
        flush_backward_states();
    }

    double bdd_cuda_base::lower_bound()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        backward_run(false);
        // Sum costs_from_terminal of all root nodes. Since root nodes are always at the start (unless one row contains > 1 BDD then have to change TODO.)

        return thrust::reduce(cost_from_terminal_.begin(), cost_from_terminal_.begin() + nr_bdds_, 0.0);
    }
}
