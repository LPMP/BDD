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

    template<typename REAL>
    bdd_cuda_base<REAL>::bdd_cuda_base(const BDD::bdd_collection& bdd_col)
    {
        initialize(bdd_col);
        thrust::device_vector<int> bdd_hop_dist_root, bdd_depth;
        std::tie(bdd_hop_dist_root, bdd_depth) = populate_bdd_nodes(bdd_col);
        reorder_bdd_nodes(bdd_hop_dist_root, bdd_depth);
        compress_bdd_nodes_to_layer(bdd_hop_dist_root);
        reorder_within_bdd_layers();
        set_special_nodes_indices(bdd_hop_dist_root);
        set_special_nodes_costs();
        find_primal_variable_ordering();
        print_num_bdd_nodes_per_hop();
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::initialize(const BDD::bdd_collection& bdd_col)
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
        std::cout<<"Num vars: "<<nr_vars_<<", Num BDDs: "<<nr_bdds_<<", Num Nodes: "<<nr_bdd_nodes_ <<"\n";
        num_bdds_per_var_ = thrust::device_vector<int>(primal_variable_counts.begin(), primal_variable_counts.end());
        // Initialize data per BDD node: 
        hi_cost_ = thrust::device_vector<REAL>(nr_bdd_nodes_, 0.0);
        lo_cost_ = thrust::device_vector<REAL>(nr_bdd_nodes_, 0.0);
        cost_from_root_ = thrust::device_vector<REAL>(nr_bdd_nodes_);
        cost_from_terminal_ = thrust::device_vector<REAL>(nr_bdd_nodes_);
    }

    template<typename REAL>
    std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> bdd_cuda_base<REAL>::populate_bdd_nodes(const BDD::bdd_collection& bdd_col)
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
                    prev_var = var;
                    if(bdd_node_idx <= bdd_col.nr_bdd_nodes(bdd_idx) - 2)
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

    template<typename REAL>
    void bdd_cuda_base<REAL>::reorder_bdd_nodes(thrust::device_vector<int>& bdd_hop_dist_dev, thrust::device_vector<int>& bdd_depth_dev)
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        // Make nodes with same hop distance, BDD depth and bdd index contiguous in that order.
        thrust::device_vector<int> sorting_order(nr_bdd_nodes_);
        thrust::sequence(sorting_order.begin(), sorting_order.end());
        
        // auto first_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.begin(), bdd_depth_dev.begin(), bdd_index_.begin()));
        // auto last_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.end(), bdd_depth_dev.begin(), bdd_index_.end()));

        // auto first_bdd_val = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), lo_bdd_node_index_.begin(), 
        //                                                                 hi_bdd_node_index_.begin(), sorting_order.begin()));

        // Sort by primal indices within one BDD column, this is faster than the above scheme. (faster on MRF, CT, GM but slightly slower on QAPLib).
        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.begin(), primal_variable_index_.begin(), bdd_index_.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.end(), primal_variable_index_.end(), bdd_index_.end()));

        auto first_bdd_val = thrust::make_zip_iterator(thrust::make_tuple(bdd_depth_dev.begin(), lo_bdd_node_index_.begin(), 
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

    template<typename REAL>
    void bdd_cuda_base<REAL>::set_special_nodes_indices(const thrust::device_vector<int>& bdd_hop_dist_dev)
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
        assert(top_sink_indices_.size() == nr_bdds_);
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::set_special_nodes_costs()
    {
        // Set costs of top sinks to itself to 0:
        thrust::scatter(thrust::make_constant_iterator<REAL>(0.0), thrust::make_constant_iterator<REAL>(0.0) + top_sink_indices_.size(),
                        top_sink_indices_.begin(), cost_from_terminal_.begin());

        // Set costs of bot sinks to top to infinity:
        thrust::scatter(thrust::make_constant_iterator<REAL>(CUDART_INF_F_HOST), thrust::make_constant_iterator<REAL>(CUDART_INF_F_HOST) + bot_sink_indices_.size(),
                        bot_sink_indices_.begin(), cost_from_terminal_.begin());
    }

    struct valid_primal_index_func {
        __host__ __device__ int operator()(const int i) const
        {
            if(i < INT_MAX)
                return 1;
            return 0;
        }
    };

    // Removes redundant information in hi_costs, primal_index, bdd_index as it is duplicated across
    // multiple BDD nodes for each layer.
    template<typename REAL>
    void bdd_cuda_base<REAL>::compress_bdd_nodes_to_layer(const thrust::device_vector<int>& bdd_hop_dist_dev)
    {
        MEASURE_FUNCTION_EXECUTION_TIME
        thrust::device_vector<REAL> hi_cost_compressed(hi_cost_.size());
        thrust::device_vector<REAL> lo_cost_compressed(lo_cost_.size());
        thrust::device_vector<int> primal_index_compressed(primal_variable_index_.size()); 
        thrust::device_vector<int> bdd_index_compressed(bdd_index_.size());
        
        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.begin(), primal_variable_index_.begin(), bdd_index_.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_hop_dist_dev.end(), primal_variable_index_.end(), bdd_index_.end()));

        auto first_out_key = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_discard_iterator(), primal_index_compressed.begin(), bdd_index_compressed.begin()));

        // Compute number of BDD nodes in each layer:
        thrust::device_vector<int> bdd_layer_width(nr_bdd_nodes_);
        auto new_end = thrust::reduce_by_key(first_key, last_key, thrust::make_constant_iterator<int>(1), first_out_key, bdd_layer_width.begin());
        const int out_size = thrust::distance(first_out_key, new_end.first);      
        bdd_layer_width.resize(out_size);     

        // Assign bdd node to layer map:
        bdd_node_to_layer_map_ = thrust::device_vector<int>(out_size);
        thrust::sequence(bdd_node_to_layer_map_.begin(), bdd_node_to_layer_map_.end());
        bdd_node_to_layer_map_ = repeat_values(bdd_node_to_layer_map_, bdd_layer_width);

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

        thrust::swap(lo_cost_compressed, lo_cost_);
        thrust::swap(hi_cost_compressed, hi_cost_);
        thrust::swap(primal_index_compressed, primal_variable_index_);
        thrust::swap(bdd_index_compressed, bdd_index_);

        thrust::device_vector<int> dev_cum_nr_layers_per_hop_dist(cum_nr_bdd_nodes_per_hop_dist_.size());
        thrust::device_vector<int> dev_nr_variables_per_hop_dist(cum_nr_bdd_nodes_per_hop_dist_.size());

        auto first_val = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_constant_iterator<int>(1), 
                                                    thrust::make_transform_iterator(
                                                        primal_index_compressed.begin(), valid_primal_index_func()))); 
        auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(dev_cum_nr_layers_per_hop_dist.begin(), dev_nr_variables_per_hop_dist.begin())); 

        thrust::equal_to<int> binary_pred;
        thrust::reduce_by_key(bdd_hop_dist_compressed.begin(), bdd_hop_dist_compressed.end(), first_val, //thrust::make_constant_iterator<int>(1), 
                            thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_sum());

        thrust::inclusive_scan(dev_cum_nr_layers_per_hop_dist.begin(), dev_cum_nr_layers_per_hop_dist.end(), dev_cum_nr_layers_per_hop_dist.begin());

        cum_nr_layers_per_hop_dist_ = std::vector<int>(dev_cum_nr_layers_per_hop_dist.size());
        thrust::copy(dev_cum_nr_layers_per_hop_dist.begin(), dev_cum_nr_layers_per_hop_dist.end(), cum_nr_layers_per_hop_dist_.begin());

        nr_variables_per_hop_dist_ = std::vector<int>(dev_nr_variables_per_hop_dist.size());
        thrust::copy(dev_nr_variables_per_hop_dist.begin(), dev_nr_variables_per_hop_dist.end(), nr_variables_per_hop_dist_.begin());
    }

    struct set_bdd_node_priority_func {
        int* node_priority_lo;
        int* node_priority_hi;
        __device__ void operator()(const thrust::tuple<int, int, int>& t)
        {
            const int bdd_node_idx = thrust::get<0>(t);
            const int next_lo_node = thrust::get<1>(t);
            if(next_lo_node < 0)
                return;
            const int next_hi_node = thrust::get<2>(t);
            atomicMin(&node_priority_lo[next_lo_node], bdd_node_idx);
            atomicMin(&node_priority_hi[next_hi_node], bdd_node_idx);
        }
    };

    template<typename REAL>
    void bdd_cuda_base<REAL>::reorder_within_bdd_layers()
    {
        const int num_steps = this->cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        thrust::device_vector<int> sorting_order(nr_bdd_nodes_);
        thrust::sequence(sorting_order.begin(), sorting_order.end());
        thrust::device_vector<int> node_priority_lo(nr_bdd_nodes_, INT_MAX); // lower values mean that BDD node should occur early in the layer and viceversa.
        thrust::device_vector<int> node_priority_hi(nr_bdd_nodes_, INT_MAX);

        set_bdd_node_priority_func set_priority({thrust::raw_pointer_cast(node_priority_lo.data()), 
                                                thrust::raw_pointer_cast(node_priority_hi.data())});

        for (int hop_index = 0; hop_index < num_steps; hop_index++)
        {
            const int start_offset = hop_index > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1] : 0;
            const int end_offset = this->cum_nr_bdd_nodes_per_hop_dist_[hop_index];
            // Set priority of nodes in hop i + 1 by checking when are these nodes required in hop i. 
            auto first = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator<int>(0) + start_offset, 
                                                                    lo_bdd_node_index_.begin() + start_offset,
                                                                    hi_bdd_node_index_.begin() + start_offset)); 
            
            auto last = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator<int>(0) + end_offset, 
                                                                    lo_bdd_node_index_.begin() + end_offset,
                                                                    hi_bdd_node_index_.begin() + end_offset));
            thrust::for_each(first, last, set_priority);

            // Now sort nodes in hop i + 1 in the order of increasing priority w.r.t (bdd_node_to_layer_map_, node_priority_lo, node_priority_hi) i.e.
            // first w.r.t bdd_node_to_layer_map_ to keep the nodes within confines of a layer and then by
            // node_priority_lo and lastly by node_priority_hi

            const int next_end_offset = this->cum_nr_bdd_nodes_per_hop_dist_[hop_index + 1];
            auto first_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_node_to_layer_map_.begin() + end_offset,
                                                                    node_priority_lo.begin() + end_offset,
                                                                    node_priority_hi.begin() + end_offset));

            auto last_key = thrust::make_zip_iterator(thrust::make_tuple(bdd_node_to_layer_map_.begin() + next_end_offset,
                                                                    node_priority_lo.begin() + next_end_offset,
                                                                    node_priority_hi.begin() + next_end_offset));
            
            auto first_bdd_val = thrust::make_zip_iterator(thrust::make_tuple(lo_bdd_node_index_.begin() + end_offset, hi_bdd_node_index_.begin() + end_offset, 
                                                                            sorting_order.begin() + end_offset));

            thrust::sort_by_key(first_key, last_key, first_bdd_val);
        }

        // Since the ordering is changed so lo, hi indices also need to be updated. TODO : make function.
        thrust::device_vector<int> new_indices(sorting_order.size());
        thrust::scatter(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + sorting_order.size(), 
                        sorting_order.begin(), new_indices.begin());
        assign_new_indices_func func({thrust::raw_pointer_cast(new_indices.data())});
        thrust::for_each(lo_bdd_node_index_.begin(), lo_bdd_node_index_.end(), func);
        thrust::for_each(hi_bdd_node_index_.begin(), hi_bdd_node_index_.end(), func);
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::find_primal_variable_ordering()
    {
        // Populate primal variables sorting order to permute min-marginals such that reduction over adjacent values can be performed 
        // to compute values for each primal variable e.g. min-marginals sum for each primal variable. etc.
        primal_variable_sorting_order_ = thrust::device_vector<int>(primal_variable_index_.size());
        thrust::sequence(primal_variable_sorting_order_.begin(), primal_variable_sorting_order_.end());
        primal_variable_index_sorted_ = primal_variable_index_;
        thrust::device_vector<int> bdd_index_sorted = bdd_index_;
    
        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_sorted_.begin(), bdd_index_sorted.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_sorted_.end(), bdd_index_sorted.end()));
        thrust::sort_by_key(first_key, last_key, primal_variable_sorting_order_.begin());
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::flush_forward_states()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        forward_state_valid_ = false;
        path_costs_valid_ = false;
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::flush_backward_states()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        backward_state_valid_ = false;
        path_costs_valid_ = false;
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::print_num_bdd_nodes_per_hop()
    {
        int prev = 0;
        for(int i = 0; i < cum_nr_bdd_nodes_per_hop_dist_.size(); i++)
        {
            std::cout<<"Hop: "<<i<<", # BDD nodes: "<<cum_nr_bdd_nodes_per_hop_dist_[i] - prev<<std::endl;
            prev = cum_nr_bdd_nodes_per_hop_dist_[i];
        }
    }

    template<typename REAL>
    struct set_var_cost_func {
        int var_index;
        REAL cost;
        __device__ void operator()(const thrust::tuple<int, REAL&> t) const
        {
            const int cur_var_index = thrust::get<0>(t);
            if(cur_var_index != var_index)
                return;
            REAL& arc_cost = thrust::get<1>(t);
            arc_cost += cost;
        }
    };

    template<typename REAL>
    void bdd_cuda_base<REAL>::set_cost(const double c, const size_t var)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        assert(var < nr_vars_);
        set_var_cost_func<REAL> func({(int) var, (REAL) c / num_bdds_per_var_[var]});

        auto first = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), hi_cost_.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.end(), hi_cost_.end()));

        thrust::for_each(first, last, func);
        flush_forward_states();
        flush_backward_states();
    }

    template<typename REAL, typename REAL2>
    struct set_vars_costs_func {
        int* var_counts;
        const REAL2* primal_costs;
        __host__ __device__ void operator()(const thrust::tuple<int, REAL&> t) const
        {
            const int cur_var_index = thrust::get<0>(t);
            if (cur_var_index == INT_MAX)
                return; // terminal node.
            REAL& arc_cost = thrust::get<1>(t);
            const int count = var_counts[cur_var_index];
            assert(count > 0);
            arc_cost += primal_costs[cur_var_index] / count;
        }
    };

    template<typename REAL>
    template<typename COST_ITERATOR> 
    void bdd_cuda_base<REAL>::update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

        auto populate_costs = [&](auto cost_begin, auto cost_end, auto bdd_cost_begin, auto bdd_cost_end) {
            thrust::device_vector<REAL> primal_costs(cost_begin, cost_end);
            
            const int num_elements = primal_costs.size();
            set_vars_costs_func<REAL, REAL> func({thrust::raw_pointer_cast(num_bdds_per_var_.data()), thrust::raw_pointer_cast(primal_costs.data())});
            auto first = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), bdd_cost_begin));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin() + num_elements, bdd_cost_end));

            thrust::for_each(first, last, func);
        };

        if(std::distance(cost_lo_begin, cost_lo_end) > 0)
            populate_costs(cost_lo_begin, cost_lo_end, lo_cost_.begin(), lo_cost_.end());
        if(std::distance(cost_hi_begin, cost_hi_end) > 0)
            populate_costs(cost_hi_begin, cost_hi_end, hi_cost_.begin(), hi_cost_.end()); 

        flush_forward_states();
        flush_backward_states();
    }

    template void bdd_cuda_base<float>::update_costs(double*, double*, double*, double*);
    template void bdd_cuda_base<float>::update_costs(float*, float*, float*, float*);
    template void bdd_cuda_base<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_cuda_base<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_cuda_base<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_cuda_base<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template void bdd_cuda_base<double>::update_costs(double*, double*, double*, double*);
    template void bdd_cuda_base<double>::update_costs(float*, float*, float*, float*);
    template void bdd_cuda_base<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_cuda_base<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_cuda_base<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_cuda_base<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template<typename REAL>
    template<typename REAL_arg>
    void bdd_cuda_base<REAL>::update_costs(const thrust::device_vector<REAL_arg>& cost_delta_0, const thrust::device_vector<REAL_arg>& cost_delta_1)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        assert(cost_delta_0.size() == 0 || cost_delta_0.size() == nr_variables());
        assert(cost_delta_1.size() == 0 || cost_delta_1.size() == nr_variables());

        auto populate_costs = [&](const thrust::device_vector<REAL_arg>& cost_delta, auto base_cost_begin, auto base_cost_end) {
            set_vars_costs_func<REAL, REAL_arg> func({thrust::raw_pointer_cast(num_bdds_per_var_.data()), thrust::raw_pointer_cast(cost_delta.data())});

            auto first = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.begin(), base_cost_begin));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(primal_variable_index_.end(), base_cost_end));

            thrust::for_each(first, last, func);
        };

        if(cost_delta_0.size() > 0)
            populate_costs(cost_delta_0, lo_cost_.begin(), lo_cost_.end());
        if(cost_delta_1.size() > 0)
            populate_costs(cost_delta_1, hi_cost_.begin(), hi_cost_.end()); 

        flush_forward_states();
        flush_backward_states();
    }

    template<typename REAL>
    __global__ void forward_step(const int cur_num_bdd_nodes, const int start_offset,
                                const int* const __restrict__ lo_bdd_node_index, 
                                const int* const __restrict__ hi_bdd_node_index, 
                                const int* const __restrict__ bdd_node_to_layer_map, 
                                const REAL* const __restrict__ lo_cost,
                                const REAL* const __restrict__ hi_cost,
                                REAL* __restrict__ cost_from_root)
    {
        const int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        const int num_threads = blockDim.x * gridDim.x;
        for (int bdd_idx = start_index + start_offset; bdd_idx < cur_num_bdd_nodes + start_offset; bdd_idx += num_threads) 
        {
            const int next_lo_node = lo_bdd_node_index[bdd_idx];
            if (next_lo_node < 0) // will matter when one row contains multiple BDDs, otherwise the terminal nodes are at the end anyway.
                continue; // nothing needs to be done for terminal node.

            const int next_hi_node = hi_bdd_node_index[bdd_idx];

            const REAL cur_c_from_root = cost_from_root[bdd_idx];
            const int layer_idx = bdd_node_to_layer_map[bdd_idx];

            // Uncoalesced writes:
            atomicMin(&cost_from_root[next_lo_node], cur_c_from_root + lo_cost[layer_idx]);
            atomicMin(&cost_from_root[next_hi_node], cur_c_from_root + hi_cost[layer_idx]);
        }
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::forward_run()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        if (forward_state_valid_)
            return;

        flush_costs_from_root();

        const int num_steps = cum_nr_bdd_nodes_per_hop_dist_.size() - 1;
        int num_nodes_processed = 0;
        for (int s = 0; s < num_steps; s++)
        {
            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - num_nodes_processed;
            int blockCount = ceil(cur_num_bdd_nodes / (REAL) NUM_THREADS);
            forward_step<<<blockCount, NUM_THREADS>>>(cur_num_bdd_nodes, num_nodes_processed,
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

    template<typename REAL>
    __global__ void backward_step_with_path_costs(const int cur_num_bdd_nodes, const int start_offset,
                                                const int* const __restrict__ lo_bdd_node_index, 
                                                const int* const __restrict__ hi_bdd_node_index, 
                                                const int* const __restrict__ bdd_node_to_layer_map, 
                                                const REAL* const __restrict__ lo_cost,
                                                const REAL* const __restrict__ hi_cost,
                                                const REAL* __restrict__ cost_from_root, 
                                                REAL* __restrict__ cost_from_terminal,
                                                REAL* __restrict__ lo_path_cost, 
                                                REAL* __restrict__ hi_path_cost)
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
            REAL cur_hi_cost_from_terminal = cost_from_terminal[hi_node] + hi_cost[layer_idx];
            REAL cur_lo_cost_from_terminal = cost_from_terminal[lo_node] + lo_cost[layer_idx];
            const REAL cur_cost_from_root = cost_from_root[bdd_idx];

            hi_path_cost[bdd_idx] = cur_cost_from_root + cur_hi_cost_from_terminal;
            lo_path_cost[bdd_idx] = cur_cost_from_root + cur_lo_cost_from_terminal;
            cost_from_terminal[bdd_idx] = min(cur_hi_cost_from_terminal, cur_lo_cost_from_terminal);
        }
    }

    template<typename REAL>
    __global__ void backward_step(const int cur_num_bdd_nodes, const int start_offset,
                                    const int* const __restrict__ lo_bdd_node_index, 
                                    const int* const __restrict__ hi_bdd_node_index, 
                                    const int* const __restrict__ bdd_node_to_layer_map, 
                                    const REAL* const __restrict__ lo_cost,
                                    const REAL* const __restrict__ hi_cost,
                                    REAL* __restrict__ cost_from_terminal)
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


    template<typename REAL>
    std::tuple<thrust::device_vector<REAL>, thrust::device_vector<REAL>> bdd_cuda_base<REAL>::backward_run(bool compute_path_costs)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        thrust::device_vector<REAL> hi_path_cost, lo_path_cost; 
        if ((backward_state_valid_ && path_costs_valid_) ||
            (!compute_path_costs && backward_state_valid_))
            return {lo_path_cost, hi_path_cost};

        if (compute_path_costs)
        {
            hi_path_cost = thrust::device_vector<REAL>(nr_bdd_nodes_);
            lo_path_cost = thrust::device_vector<REAL>(nr_bdd_nodes_);
        }

        for (int s = cum_nr_bdd_nodes_per_hop_dist_.size() - 2; s >= 0; s--)
        {
            const int start_offset = s > 0 ? this->cum_nr_bdd_nodes_per_hop_dist_[s - 1]: 0;

            int cur_num_bdd_nodes = cum_nr_bdd_nodes_per_hop_dist_[s] - start_offset;
            int blockCount = ceil(cur_num_bdd_nodes / (REAL) NUM_THREADS);
            if (compute_path_costs)
                backward_step_with_path_costs<<<blockCount, NUM_THREADS>>>(cur_num_bdd_nodes, start_offset,
                                                        thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                                        thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                                        thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                                        thrust::raw_pointer_cast(lo_cost_.data()),
                                                        thrust::raw_pointer_cast(hi_cost_.data()),
                                                        thrust::raw_pointer_cast(cost_from_root_.data()),
                                                        thrust::raw_pointer_cast(cost_from_terminal_.data()),
                                                        thrust::raw_pointer_cast(lo_path_cost.data()),
                                                        thrust::raw_pointer_cast(hi_path_cost.data()));
            else
                backward_step<<<blockCount, NUM_THREADS>>>(cur_num_bdd_nodes, start_offset,
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
        return {lo_path_cost, hi_path_cost};
    }

    // Computes min-marginals by reduction.
    template<typename REAL>
    std::tuple<thrust::device_vector<int>, thrust::device_vector<REAL>, thrust::device_vector<REAL>> bdd_cuda_base<REAL>::min_marginals_cuda(bool get_sorted)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME

        forward_run();

        thrust::device_vector<REAL> min_marginals_lo(hi_cost_.size());
        thrust::device_vector<REAL> min_marginals_hi(hi_cost_.size());

        {
            thrust::device_vector<REAL> lo_path_cost, hi_path_cost; 
            std::tie(lo_path_cost, hi_path_cost) = backward_run();
            auto first_val = thrust::make_zip_iterator(thrust::make_tuple(lo_path_cost.begin(), hi_path_cost.begin()));
            auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(min_marginals_lo.begin(), min_marginals_hi.begin()));

            thrust::equal_to<int> binary_pred;
            auto new_end = thrust::reduce_by_key(bdd_node_to_layer_map_.begin(), bdd_node_to_layer_map_.end(), first_val, thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_min());
            const int out_size = thrust::distance(first_out_val, new_end.second);
            assert(out_size == hi_cost_.size());
        }

        if (get_sorted)
        {
            thrust::device_vector<REAL> min_marginals_lo_sorted(hi_cost_.size());
            thrust::device_vector<REAL> min_marginals_hi_sorted(hi_cost_.size());

            auto first_val = thrust::make_zip_iterator(thrust::make_tuple(min_marginals_lo.begin(), min_marginals_hi.begin()));
            auto first_val_sorted = thrust::make_zip_iterator(thrust::make_tuple(min_marginals_lo_sorted.begin(), min_marginals_hi_sorted.begin()));
            thrust::gather(primal_variable_sorting_order_.begin(), primal_variable_sorting_order_.end(), first_val, first_val_sorted);

            return {primal_variable_index_sorted_, min_marginals_lo_sorted, min_marginals_hi_sorted};
        }
        else
            return {primal_variable_index_, min_marginals_lo, min_marginals_hi};
    }

    template<typename REAL>
    two_dim_variable_array<std::array<double,2>> bdd_cuda_base<REAL>::min_marginals()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        thrust::device_vector<REAL> mm_0, mm_1;
        thrust::device_vector<int> primal_variable_index_sorted;

        std::tie(primal_variable_index_sorted, mm_0, mm_1) = min_marginals_cuda();

        std::vector<int> num_bdds_per_var(num_bdds_per_var_.size());
        thrust::copy(num_bdds_per_var_.begin(), num_bdds_per_var_.end(), num_bdds_per_var.begin());

        std::vector<REAL> h_mm_0(mm_0.size());
        thrust::copy(mm_0.begin(), mm_0.end(), h_mm_0.begin());

        std::vector<REAL> h_mm_1(mm_1.size());
        thrust::copy(mm_1.begin(), mm_1.end(), h_mm_1.begin());

        two_dim_variable_array<std::array<double,2>> min_margs(num_bdds_per_var);

        int idx_1d = 0;
        for(int var = 0; var < nr_vars_; ++var)
        {
            assert(num_bdds_per_var[var] > 0);
            for(int bdd_idx = 0; bdd_idx < num_bdds_per_var[var]; ++bdd_idx, ++idx_1d)
            {
                assert(idx_1d < h_mm_0.size() - nr_bdds_ && idx_1d < h_mm_1.size() - nr_bdds_);
                std::array<double,2> mm = {h_mm_0[idx_1d], h_mm_1[idx_1d]};
                min_margs(var, bdd_idx) = mm;
            }
        }

        return min_margs;
    }

    template<typename REAL>
    struct compute_bdd_sol_func {
        const int* bdd_node_to_layer_map;
        const int* lo_bdd_node_index;
        const int* hi_bdd_node_index;
        const REAL* lo_path_cost;
        const REAL* hi_path_cost;
        int* next_path_nodes;
        REAL* sol;
        __host__ __device__ void operator()(const int layer_index)
        {
            const int node_index = next_path_nodes[layer_index]; // which node to select in current bdd layer.
            const REAL cost_diff = hi_path_cost[node_index] - lo_path_cost[node_index]; // see if lo arc has lower solution cost or higher.
            if (cost_diff > 0) // high arc has more cost so assign 0.
            {
                sol[layer_index] = 0.0;
                const int next_bdd_node = lo_bdd_node_index[node_index];
                next_path_nodes[bdd_node_to_layer_map[next_bdd_node]] = next_bdd_node;
            }
            else
            {
                sol[layer_index] = 1.0;
                const int next_bdd_node = hi_bdd_node_index[node_index];
                next_path_nodes[bdd_node_to_layer_map[next_bdd_node]] = next_bdd_node;
            }
        }
    };

    
    template<typename REAL>
    thrust::device_vector<REAL> bdd_cuda_base<REAL>::bdds_solution_cuda()
    {
        forward_run();
        thrust::device_vector<REAL> lo_path_cost, hi_path_cost; 
        std::tie(lo_path_cost, hi_path_cost) = backward_run(true);

        thrust::device_vector<REAL> sol(primal_variable_index_.size());
        thrust::device_vector<int> next_path_nodes(primal_variable_index_.size(), -1);

        // Start from root nodes of all BDDs in parallel.
        thrust::sequence(next_path_nodes.begin(), next_path_nodes.begin() + cum_nr_layers_per_hop_dist_[0]);

        compute_bdd_sol_func<REAL> func({thrust::raw_pointer_cast(bdd_node_to_layer_map_.data()),
                                        thrust::raw_pointer_cast(lo_bdd_node_index_.data()),
                                        thrust::raw_pointer_cast(hi_bdd_node_index_.data()),
                                        thrust::raw_pointer_cast(lo_path_cost.data()),
                                        thrust::raw_pointer_cast(hi_path_cost.data()),
                                        thrust::raw_pointer_cast(next_path_nodes.data()),
                                        thrust::raw_pointer_cast(sol.data())});

        const int num_steps = cum_nr_layers_per_hop_dist_.size();
        int start_offset = 0;
        for (int s = 0; s < num_steps; s++)
        {
            const int end_offset = cum_nr_layers_per_hop_dist_[s];
            if (s < num_steps - 1)
            {
                thrust::for_each(thrust::make_counting_iterator<int>(start_offset), thrust::make_counting_iterator<int>(end_offset), func);
            }
            else 
            {   // all terminal nodes, thus copy 0's.
                thrust::fill(sol.begin() + start_offset, sol.end(), 0.0);
            }
            start_offset = end_offset;
        }
        return sol;
    }

    template<typename REAL>
    two_dim_variable_array<REAL> bdd_cuda_base<REAL>::bdds_solution()
    {
        thrust::device_vector<REAL> sol = bdds_solution_cuda();
        thrust::device_vector<REAL> sol_sorted(sol.size());
        thrust::gather(primal_variable_sorting_order_.begin(), primal_variable_sorting_order_.end(), 
                        sol.begin(), sol_sorted.begin());

        std::vector<int> num_bdds_per_var(num_bdds_per_var_.size());
        thrust::copy(num_bdds_per_var_.begin(), num_bdds_per_var_.end(), num_bdds_per_var.begin());

        std::vector<REAL> h_sol_sorted(sol_sorted.size());
        thrust::copy(sol_sorted.begin(), sol_sorted.end(), h_sol_sorted.begin());

        two_dim_variable_array<REAL> h_sol(num_bdds_per_var);

        int idx_1d = 0;
        for(int var = 0; var < nr_vars_; ++var)
        {
            assert(num_bdds_per_var[var] > 0);
            for(int bdd_idx = 0; bdd_idx < num_bdds_per_var[var]; ++bdd_idx, ++idx_1d)
            {
                assert(idx_1d < h_sol_sorted.size() - nr_bdds_);
                h_sol(var, bdd_idx) = h_sol_sorted[idx_1d];
            }
        }

        return h_sol;
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::update_costs(const thrust::device_vector<REAL>& update_vec)
    {
        thrust::transform(hi_cost_.begin(), hi_cost_.end(), update_vec.begin(), hi_cost_.begin(), thrust::plus<REAL>());
        flush_forward_states();
        flush_backward_states();
    }

    template<typename REAL>
    double bdd_cuda_base<REAL>::lower_bound()
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        backward_run(false);
        // Sum costs_from_terminal of all root nodes. Since root nodes are always at the start (unless one row contains > 1 BDD then have to change TODO.)

        return thrust::reduce(cost_from_terminal_.begin(), cost_from_terminal_.begin() + nr_bdds_, 0.0);
    }

    template<typename REAL>
    struct get_dual_costs_func {
        __host__ __device__ void operator()(const thrust::tuple<REAL&, REAL, REAL> t) const
        {
            REAL& current_dual_cost = thrust::get<0>(t);
            const REAL lo_arc_cost = thrust::get<1>(t);
            const REAL hi_arc_cost = thrust::get<2>(t);

            current_dual_cost = hi_arc_cost - lo_arc_cost;
        }
    };

    template<typename REAL>
    void bdd_cuda_base<REAL>::get_dual_costs(thrust::device_ptr<REAL> lambda_out_ptr) const
    {
        int output_start_offset = 0;
        for(int var_block_idx = 0; var_block_idx < nr_blocks(); var_block_idx++)
        {
            const int num_vars_in_block = nr_dual_variables(var_block_idx);
            const int in_start_offset = var_block_idx > 0 ? this->cum_nr_layers_per_hop_dist_[var_block_idx - 1]: 0;
            const int in_end_offset = in_start_offset + num_vars_in_block;
            const int output_end_offset = output_start_offset + num_vars_in_block;

            // Terminal nodes are at the end of each hop and they dont need to be copied.
            auto first = thrust::make_zip_iterator(thrust::make_tuple(lambda_out_ptr + output_start_offset, lo_cost_.begin() + in_start_offset, hi_cost_.begin() + in_start_offset));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(lambda_out_ptr + output_end_offset, lo_cost_.begin() + in_end_offset, hi_cost_.begin() + in_end_offset));

            thrust::for_each(first, last, get_dual_costs_func<REAL>());
            output_start_offset += num_vars_in_block;
        }
    }

    template<typename REAL>
    struct set_dual_costs_func {
        __host__ __device__ void operator()(const thrust::tuple<REAL, REAL&, REAL&> t) const
        {
            const REAL current_dual_cost = thrust::get<0>(t);
            if (current_dual_cost < 0)
            {
                REAL& lo_arc_cost = thrust::get<1>(t);
                lo_arc_cost = -current_dual_cost;
            }
            else
            {
                REAL& hi_arc_cost = thrust::get<2>(t);
                hi_arc_cost = current_dual_cost;
            }
        }
    };

    template<typename REAL>
    template<typename ITERATOR> 
    void bdd_cuda_base<REAL>::set_dual_costs(ITERATOR dual_costs_begin, ITERATOR dual_costs_end)
    {
        assert(thrust::distance(dual_costs_begin, dual_costs_end) == nr_dual_variables());

        int input_offset = 0;
        for(int var_block_idx = 0; var_block_idx < nr_blocks(); var_block_idx++)
        {
            const int num_vars_in_block = nr_dual_variables(var_block_idx);
            const int out_start_offset = var_block_idx > 0 ? this->cum_nr_layers_per_hop_dist_[var_block_idx - 1]: 0;
            const int out_end_offset = out_start_offset + num_vars_in_block;

            // Positive values goes to hi_costs, negative to lo_cost. Since terminal nodes are at the end of each hop, thus they remain unaffected.
            auto first = thrust::make_zip_iterator(thrust::make_tuple(dual_costs_begin + input_offset, lo_cost_.begin() + out_start_offset, hi_cost_.begin() + out_start_offset));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(dual_costs_begin + input_offset + num_vars_in_block, lo_cost_.begin() + out_end_offset, hi_cost_.begin() + out_end_offset));

            thrust::for_each(first, last, set_dual_costs_func<REAL>());
            input_offset += num_vars_in_block;
        }
        flush_forward_states();
        flush_backward_states();
    }

    template<typename REAL>
    std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> bdd_cuda_base<REAL>::var_constraint_indices() const
    {
        throw std::runtime_error("Not implemented.");
    }

    template<typename REAL>
    std::vector<REAL> bdd_cuda_base<REAL>::compute_primal_objective_vector()
    {
        thrust::device_vector<REAL> net_cost(hi_cost_.size());
        thrust::transform(hi_cost_.begin(), hi_cost_.end(), lo_cost_.begin(), net_cost.begin(), thrust::minus<REAL>());

        thrust::device_vector<REAL> primal_obj_vec(nr_vars_);
        auto new_end = thrust::reduce_by_key(this->primal_variable_index_sorted_.begin(), this->primal_variable_index_sorted_.end() - this->nr_bdds_, 
                            thrust::make_permutation_iterator(net_cost.begin(), this->primal_variable_sorting_order_.begin()),
                            thrust::make_discard_iterator(), primal_obj_vec.begin());
        assert(thrust::distance(primal_obj_vec.begin(), new_end.second) == nr_vars_);

        std::vector<REAL> h_primal_obj_vec(primal_obj_vec.size());
        thrust::copy(primal_obj_vec.begin(), primal_obj_vec.end(), h_primal_obj_vec.begin());
        return h_primal_obj_vec;
    }

    template<typename REAL>
    void bdd_cuda_base<REAL>::flush_costs_from_root()
    {
        thrust::fill(cost_from_root_.begin(), cost_from_root_.end(), CUDART_INF_F_HOST);
        // Set costs of root nodes to 0:
        thrust::scatter(thrust::make_constant_iterator<REAL>(0.0), thrust::make_constant_iterator<REAL>(0.0) + this->root_indices_.size(),
                        this->root_indices_.begin(), this->cost_from_root_.begin());
    }

    template void bdd_cuda_base<float>::update_costs(const thrust::device_vector<double>&, const thrust::device_vector<double>&);
    template void bdd_cuda_base<float>::update_costs(const thrust::device_vector<float>&, const thrust::device_vector<float>&);
    template void bdd_cuda_base<double>::update_costs(const thrust::device_vector<double>&, const thrust::device_vector<double>&);
    template void bdd_cuda_base<double>::update_costs(const thrust::device_vector<float>&, const thrust::device_vector<float>&);

    template <typename REAL>
    template <class Archive>
    void bdd_cuda_base<REAL>::save(Archive& archive) const
    {
        archive(
            primal_variable_index_,
            bdd_index_,
            hi_cost_,
            lo_cost_,
            lo_bdd_node_index_,
            hi_bdd_node_index_,
            bdd_node_to_layer_map_,
            num_bdds_per_var_,
            root_indices_,
            bot_sink_indices_,
            top_sink_indices_,
            primal_variable_sorting_order_,
            primal_variable_index_sorted_,
            cum_nr_bdd_nodes_per_hop_dist_,
            cum_nr_layers_per_hop_dist_,
            nr_variables_per_hop_dist_,
            nr_vars_, nr_bdds_, nr_bdd_nodes_, num_dual_variables_
        );
    }

    template <typename REAL>
    template <class Archive>
    void bdd_cuda_base<REAL>::load(Archive& archive)
    {
        // Copies to GPU automatically by using device_vector ctor.
        archive(
            primal_variable_index_,
            bdd_index_,
            hi_cost_,
            lo_cost_,
            lo_bdd_node_index_,
            hi_bdd_node_index_,
            bdd_node_to_layer_map_,
            num_bdds_per_var_,
            root_indices_,
            bot_sink_indices_,
            top_sink_indices_,
            primal_variable_sorting_order_,
            primal_variable_index_sorted_,
            cum_nr_bdd_nodes_per_hop_dist_,
            cum_nr_layers_per_hop_dist_,
            nr_variables_per_hop_dist_,
            nr_vars_, nr_bdds_, nr_bdd_nodes_, num_dual_variables_
        );
        cost_from_root_ = thrust::device_vector<REAL>(nr_bdd_nodes_);
        cost_from_terminal_ = thrust::device_vector<REAL>(nr_bdd_nodes_);
        set_special_nodes_costs();
    }

    template void bdd_cuda_base<float>::save(cereal::BinaryOutputArchive&) const;
    template void bdd_cuda_base<double>::save(cereal::BinaryOutputArchive&) const;

    template void bdd_cuda_base<float>::load(cereal::BinaryInputArchive&);
    template void bdd_cuda_base<double>::load(cereal::BinaryInputArchive&);

    template class bdd_cuda_base<float>;
    template class bdd_cuda_base<double>;

}
