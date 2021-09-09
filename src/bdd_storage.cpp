#include "bdd_storage.h"
#include "time_measure_util.h"

namespace LPMP {

    bdd_storage::bdd_storage(bdd_preprocessor& bdd_pre)
    {
        // TODO: parallelize by building up multiple bdd storages and then join them together
        MEASURE_FUNCTION_EXECUTION_TIME;

        // first add BDDs from bdd_sollection
        assert(bdd_pre.get_bdd_collection().nr_bdds() > 0);

        if(bdd_pre.get_bdd_collection().nr_bdds() > 0)
        {
            std::cout << "Add " << bdd_pre.get_bdd_collection().nr_bdds() << " bdds from bdd collection\n";
            for(size_t bdd_nr=0; bdd_nr<bdd_pre.get_bdd_collection().nr_bdds(); ++bdd_nr)
                add_bdd(bdd_pre.get_bdd_collection()[bdd_nr]);
        }


        /*
        // second, add BDDs from BDD manager
        const auto bdds = bdd_pre.get_bdds();
        std::cout << "Add " << bdds.size() << " bdds from bdd manager\n";
        const auto bdd_variable_indices = bdd_pre.get_bdd_indices();
        assert(bdds.size() == bdd_variable_indices.size());
        for(size_t bdd_index=0; bdd_index<bdds.size(); ++bdd_index)
            add_bdd(bdd_pre.get_bdd_manager(), bdds[bdd_index], bdd_variable_indices[bdd_index].begin(), bdd_variable_indices[bdd_index].end());
            */
    }

    bdd_storage::bdd_storage()
    {}

    void bdd_storage::add_bdd(BDD::bdd_collection_entry bdd)
    {
        const auto vars = bdd.variables();
        //std::unordered_map<size_t,size_t> rebase_to_iota;
        tsl::robin_map<size_t,size_t> rebase_to_iota;
        for(size_t i=0; i<vars.size(); ++i)
            rebase_to_iota.insert({vars[i], i});
        bdd.rebase(rebase_to_iota);

        auto get_node = [&](const size_t i) {
            const size_t j = bdd.nr_nodes() - 3 - i;
            return bdd[j]; 
        };
        auto get_next_node = [&](BDD::bdd_collection_node node) { return node.next_postorder(); };
        auto get_variable = [&](BDD::bdd_collection_node node) { return node.variable(); };

        add_bdd_impl<BDD::bdd_collection_node>(
                bdd.nr_nodes() - 2, // do not count in top- and botsink
                get_node,
                get_variable,
                get_next_node,
                vars.begin(), vars.end()
                );

        bdd.rebase(vars.begin(), vars.end());
    }

    inline void bdd_storage::check_node_valid(const bdd_node bdd) const
    {
        assert(bdd.low == bdd_node::terminal_0 || bdd.low == bdd_node::terminal_1 || bdd.low < bdd_nodes_.size());
        assert(bdd.high == bdd_node::terminal_0 || bdd.high == bdd_node::terminal_1 || bdd.high < bdd_nodes_.size());
        assert(bdd.variable < nr_variables());
        if(bdd.low != bdd_node::terminal_0 && bdd.low != bdd_node::terminal_1) {
            assert(bdd.variable < bdd_nodes_[bdd.low].variable);
        }
        if(bdd.high != bdd_node::terminal_0 && bdd.high != bdd_node::terminal_1) {
            assert(bdd.variable < bdd_nodes_[bdd.high].variable);
        }
        //assert(bdd.high != bdd.low); this can be so in our formulation, but not in ordinary BDDs
    }


    size_t bdd_storage::first_bdd_node(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        return bdd_nodes_[bdd_delimiters_[bdd_nr]].variable;
    }

    size_t bdd_storage::last_bdd_node(const size_t bdd_nr) const
    {
        assert(bdd_nr < nr_bdds());
        std::size_t max_node = 0;
        for(std::size_t i=bdd_delimiters_[bdd_nr]; i<bdd_delimiters_[bdd_nr+1]; ++i)
            max_node = std::max(max_node, bdd_nodes_[i].variable);
        return max_node;
    }

    std::vector<std::array<size_t,2>> bdd_storage::dependency_graph() const
    {
        tsl::robin_set<std::array<size_t,2>> edges;
        tsl::robin_set<size_t> cur_vars;
        std::vector<size_t> cur_vars_sorted;
        for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
        {
            cur_vars.clear();
            for(size_t i=bdd_delimiters_[bdd_nr]; i<bdd_delimiters_[bdd_nr+1]; ++i)
                cur_vars.insert(bdd_nodes_[i].variable);
            cur_vars_sorted.clear();
            for(const size_t v : cur_vars)
                cur_vars_sorted.push_back(v);
            std::sort(cur_vars_sorted.begin(), cur_vars_sorted.end());
            for(size_t i=0; i+1<cur_vars_sorted.size(); ++i)
                edges.insert({cur_vars_sorted[i], cur_vars_sorted[i+1]});
        }

        return std::vector<std::array<size_t,2>>(edges.begin(), edges.end());
    }

    ///////////////////////////
    // for BDD decomposition //
    ///////////////////////////

    bdd_storage::intervals bdd_storage::compute_intervals(const size_t nr_intervals_, const size_t min_nr_bdd_nodes)
    {
        // if number of intervals would lead to subproblems that have fewer than min_nr_bdd_nodes, decrease the number of intervals
        const size_t nr_intervals = [&]() -> size_t {
            if(bdd_nodes_.size() <= min_nr_bdd_nodes)
                return 1;
            else if(min_nr_bdd_nodes * nr_intervals_ >= bdd_nodes_.size())
                return bdd_nodes_.size() / min_nr_bdd_nodes;
            return std::min(nr_intervals_, nr_variables());
        }();
        assert(nr_intervals > 0 && nr_intervals <= nr_intervals_);
        std::cout << "nr intervals = " << nr_intervals << ", requested nr intervals = " << nr_intervals_ << ", min nr bdd nodes per interval = " << min_nr_bdd_nodes << "\n";

        if(nr_intervals == 1)
            return intervals{{}, {}}; 

        // partition into intervals with equal number of bdd nodes
        assert(nr_intervals > 1);
        std::vector<size_t> interval_boundaries;
        interval_boundaries.reserve(nr_intervals+1);
        interval_boundaries.push_back(0);

        std::vector<size_t> bdds_per_variable(nr_variables(), 0);
        for(const auto& bdd_node : bdd_nodes_)
        {
            assert(bdd_node.variable < nr_variables());
            bdds_per_variable[bdd_node.variable]++; 
        }
        size_t cumulative_sum = 0;
        for(size_t i=0; i<nr_variables(); ++i)
        {
            cumulative_sum += bdds_per_variable[i];
            if(cumulative_sum >= bdd_nodes_.size()*double(interval_boundaries.size())/double(nr_intervals))
                interval_boundaries.push_back(i+1); 
        }

        assert(interval_boundaries.size() == nr_intervals+1);
        assert(interval_boundaries.back() == nr_variables());

        std::vector<size_t> variable_interval;
        variable_interval.reserve(this->nr_variables());
        for(size_t interval = 0; interval+1<interval_boundaries.size(); ++interval)
            for(size_t var=interval_boundaries[interval]; var<interval_boundaries[interval+1]; ++var)
                variable_interval.push_back(interval);

        return intervals{variable_interval, interval_boundaries};
    }

    size_t bdd_storage::intervals::interval(const size_t variable) const
    {
        assert(interval_boundaries.size() > 2 || interval_boundaries.size() == 0);
        if(interval_boundaries.size() == 0)
            return 0;
        assert(variable < this->variable_interval.size());
        return variable_interval[variable];
    }

    size_t bdd_storage::intervals::nr_intervals() const
    {
        assert(interval_boundaries.size() > 2 || interval_boundaries.size() == 0);
        if(interval_boundaries.size() == 0)
            return 1;
        return interval_boundaries.size()-1;
    }

    // given bdd storage, split up bdds into nr_intervals sub-bdd_storages. Record where splitting of BDDs is done in second return structure.
    // TODO: do not use nr_bdd_nodes_per_interval in second part, i.e. filling in bdd nodes. bdd_nodes_.size() is enough
    std::tuple<std::vector<bdd_storage>, tsl::robin_set<bdd_storage::duplicate_variable, bdd_storage::duplicate_variable_hash>> bdd_storage::split_bdd_nodes(const intervals& intn)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        assert(intn.nr_intervals() > 1);

        std::vector<size_t> nr_bdd_nodes_per_interval(intn.nr_intervals(), 0);
        std::vector<size_t> nr_bdds_per_interval(intn.nr_intervals(), 1);
        //std::unordered_set<size_t> active_intervals;
        tsl::robin_set<size_t> active_intervals;

        for(size_t bdd_counter=0; bdd_counter<bdd_delimiters_.size()-1; ++bdd_counter)
        {
            const size_t last_bdd_interval = [&]() {
                size_t last_bdd_interval = 0;
                for(auto bdd_node_counter=bdd_delimiters_[bdd_counter]; bdd_node_counter<bdd_delimiters_[bdd_counter+1]; ++bdd_node_counter)
                {
                    const auto& bdd = bdd_nodes_[bdd_node_counter];
                    last_bdd_interval = std::max(intn.interval(bdd.variable), last_bdd_interval);
                }
                return last_bdd_interval;
            }();
            
            // there are the following cases:
            // (i) All bdd nodes are non-terminal and in the same interval
            // (ii) all bdd nodes are non-terminal but in different intervals
            // (iii) one arc is bottom and the other one is in the same interval
            // (iv) one arc is bottom and the other one is in the next interval
            // (v) At least one arc is top -> bdd node is in last interval
            
            // count in which intervals bdd has nodes
            active_intervals.clear();
            for(auto bdd_node_counter=bdd_delimiters_[bdd_counter]; bdd_node_counter<bdd_delimiters_[bdd_counter+1]; ++bdd_node_counter)
            {
                const auto& bdd = bdd_nodes_[bdd_node_counter];
                active_intervals.insert(intn.interval(bdd.variable));
            }
            for(const size_t i : active_intervals)
                ++nr_bdds_per_interval[i];

            // count number of bdd nodes per interval
            for(auto bdd_node_counter=bdd_delimiters_[bdd_counter]; bdd_node_counter<bdd_delimiters_[bdd_counter+1]; ++bdd_node_counter)
            {
                const auto& bdd = bdd_nodes_[bdd_node_counter];
                // first check if node is split node. If not, increase node count in correct interval. If split node, increment node count in both intervals straddled.
                // case (i)
                if(!bdd.low_is_terminal() && !bdd.high_is_terminal())
                {
                    const bdd_node& low = bdd_nodes_[bdd.low];
                    const bdd_node& high = bdd_nodes_[bdd.high];
                    assert(low.variable == high.variable);
                    if(intn.interval(bdd.variable) == intn.interval(low.variable)) // case (i)
                    {
                        ++nr_bdd_nodes_per_interval[intn.interval(bdd.variable)]; 
                    }
                    else // case (ii)
                    {
                        ++nr_bdd_nodes_per_interval[intn.interval(bdd.variable)];
                        ++nr_bdd_nodes_per_interval[intn.interval(bdd_nodes_[bdd.low].variable)]; // bdd nodes pointed  to by low and high will be counted in next interval again when visiting those 
                    }
                }
                else if(bdd.low_is_terminal() && bdd.high_is_terminal()) // case (v)
                {
                    assert(bdd.low == bdd_node::terminal_1 || bdd.high == bdd_node::terminal_1);
                    ++nr_bdd_nodes_per_interval[intn.interval(bdd.variable)]; 
                }
                else if(bdd.low == bdd_node::terminal_0 || bdd.high == bdd_node::terminal_0)
                {
                    if(bdd.low == bdd_node::terminal_0)
                    {
                        const size_t high_var = bdd_nodes_[bdd.high].variable;
                        if(intn.interval(bdd.variable) == intn.interval(high_var)) // case (iii)
                        {
                            ++nr_bdd_nodes_per_interval[intn.interval(bdd.variable)];
                        }
                        else // case (iv)
                        {
                            // TODO: not necessarily += 2, possibly += 1 if node is shared!
                            ++nr_bdd_nodes_per_interval[intn.interval(bdd.variable)];
                            ++nr_bdd_nodes_per_interval[intn.interval(high_var)];
                        }
                    }
                    else
                    {
                        assert(bdd.high == bdd_node::terminal_0);
                        const size_t low_var = bdd_nodes_[bdd.low].variable;
                        if(intn.interval(bdd.variable) == intn.interval(low_var)) // case (iii)
                        {
                            ++nr_bdd_nodes_per_interval[intn.interval(bdd.variable)];
                        }
                        else // case (iv)
                        {
                            // TODO: not necessarily += 2, possibly += 1 if node is shared!
                            ++nr_bdd_nodes_per_interval[intn.interval(bdd.variable)];
                            ++nr_bdd_nodes_per_interval[intn.interval(low_var)];
                        }
                    } 
                }
                else
                {
                    assert(false); // We should have covered all cases
                }
            }
        }

        // allocate structures for holding bdd nodes
        std::vector<bdd_storage> bdd_storages(intn.nr_intervals());
        for(size_t i=0; i<intn.nr_intervals(); ++i)
        {
            bdd_storages[i].bdd_nodes_.reserve(nr_bdd_nodes_per_interval[i]);
            bdd_storages[i].bdd_delimiters_.reserve(nr_bdds_per_interval[i]);
            bdd_storages[i].nr_variables_ = this->nr_variables_;
            assert(bdd_storages[i].bdd_delimiters_.size() == 1 && bdd_storages[i].bdd_delimiters_[0] == 0);
        }


        //two_dim_variable_array<bdd_node> split_bdd_nodes(nr_bdd_nodes_per_interval.begin(), nr_bdd_nodes_per_interval.end());
        std::fill(nr_bdd_nodes_per_interval.begin(), nr_bdd_nodes_per_interval.end(), 0);

        //two_dim_variable_array<size_t> split_bdd_delimiters(nr_bdds_per_interval.begin(), nr_bdds_per_interval.end());
        std::fill(nr_bdds_per_interval.begin(), nr_bdds_per_interval.end(), 1);
        //for(size_t i=0; i<split_bdd_delimiters.size(); ++i)
        //    split_bdd_delimiters(i,0) = 0;

        // fill split bdd nodes, record duplicated bdd variables
        tsl::robin_set<duplicate_variable, duplicate_variable_hash> duplicated_variables;
        tsl::robin_map<std::array<size_t,2>,size_t> split_bdd_node_indices; // bdd index in bdd_nodes_, interval
        for(size_t bdd_counter=0; bdd_counter<bdd_delimiters_.size()-1; ++bdd_counter)
        {
            split_bdd_node_indices.clear();
            for(auto bdd_node_counter=bdd_delimiters_[bdd_counter]; bdd_node_counter<bdd_delimiters_[bdd_counter+1]; ++bdd_node_counter)
            { 
                const bdd_node& bdd = bdd_nodes_[bdd_node_counter];
                // case (i) & (ii)
                if(!bdd.low_is_terminal() && !bdd.high_is_terminal())
                {
                    const size_t i = intn.interval(bdd.variable);
                    const bdd_node& low = bdd_nodes_[bdd.low];
                    const bdd_node& high = bdd_nodes_[bdd.high];
                    assert(low.variable == high.variable);

                    if(intn.interval(bdd.variable) == intn.interval(low.variable)) // case (i)
                    {
                        assert(split_bdd_node_indices.count({bdd.low, i}) > 0);
                        const size_t low_idx = split_bdd_node_indices.find({bdd.low, i})->second;
                        assert(split_bdd_node_indices.count({bdd.high, i}) > 0);
                        const size_t high_idx = split_bdd_node_indices.find({bdd.high, i})->second;

                        bdd_storages[i].bdd_nodes_.push_back({low_idx, high_idx, bdd.variable});
                        //split_bdd_nodes(i, nr_bdd_nodes_per_interval[i]) = {bdd.variable, low_idx, high_idx};
                        split_bdd_node_indices.insert(std::make_pair(std::array<size_t,2>{bdd_node_counter, i}, nr_bdd_nodes_per_interval[i]));
                        ++nr_bdd_nodes_per_interval[i];
                        assert(bdd_storages[i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[i]);
                    }
                    else // case (ii)
                    {
                        // in interval i, low and high arcs should point to topsink
                        bdd_storages[i].bdd_nodes_.push_back({bdd_node::terminal_1, bdd_node::terminal_1, bdd.variable});
                        //split_bdd_nodes(i, nr_bdd_nodes_per_interval[i]) = {bdd.variable, bdd_node::terminal_1, bdd_node::terminal_1};
                        split_bdd_node_indices.insert(std::make_pair(std::array<size_t,2>{bdd_node_counter, i}, nr_bdd_nodes_per_interval[i]));
                        ++nr_bdd_nodes_per_interval[i];
                        assert(bdd_storages[i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[i]);

                        // in next interval
                        const size_t next_i = intn.interval(bdd_nodes_[bdd.low].variable);
                        assert(i < next_i);
                        const size_t next_lo_idx = split_bdd_node_indices.find({bdd.low, next_i})->second;
                        const size_t next_hi_idx = split_bdd_node_indices.find({bdd.high, next_i})->second;
                        bdd_storages[next_i].bdd_nodes_.push_back({next_lo_idx, next_hi_idx, bdd.variable});
                        //split_bdd_nodes(next_i, nr_bdd_nodes_per_interval[next_i]) = {bdd.variable, next_lo_idx, next_hi_idx};
                        split_bdd_node_indices.insert(std::make_pair(std::array<size_t,2>{bdd_node_counter, next_i}, nr_bdd_nodes_per_interval[next_i]));
                        ++nr_bdd_nodes_per_interval[next_i]; 
                        assert(bdd_storages[next_i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[next_i]);

                        duplicated_variables.insert({i, nr_bdds_per_interval[i]-1, next_i, nr_bdds_per_interval[next_i]-1});
                    }
                }
                else if(bdd.low_is_terminal() && bdd.high_is_terminal()) // case (v)
                {
                    assert(bdd.low == bdd_node::terminal_1 || bdd.high == bdd_node::terminal_1);
                    const size_t i = intn.interval(bdd.variable);
                    bdd_storages[i].bdd_nodes_.push_back({bdd.low, bdd.high, bdd.variable});
                    //split_bdd_nodes(i, nr_bdd_nodes_per_interval[i]) = {bdd.variable, bdd.low, bdd.high};
                    split_bdd_node_indices.insert(std::make_pair(std::array<size_t,2>{bdd_node_counter, i}, nr_bdd_nodes_per_interval[i])); 
                    ++nr_bdd_nodes_per_interval[i]; 
                    assert(bdd_storages[i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[i]);
                }
                else if(bdd.low == bdd_node::terminal_0 || bdd.high == bdd_node::terminal_0)
                {
                    const size_t i = intn.interval(bdd.variable);
                    if(bdd.low == bdd_node::terminal_0)
                    {
                        const size_t high_var = bdd_nodes_[bdd.high].variable;
                        if(i == intn.interval(high_var)) // case (iii)
                        {
                            assert(split_bdd_node_indices.count({bdd.high,i}) > 0);
                            const size_t high_idx = split_bdd_node_indices.find({bdd.high,i})->second;
                            bdd_storages[i].bdd_nodes_.push_back({bdd_node::terminal_0, high_idx, bdd.variable});
                            //split_bdd_nodes(i, nr_bdd_nodes_per_interval[i]) = {bdd.variable, bdd_node::terminal_0, high_idx};
                            split_bdd_node_indices.insert(std::make_pair(std::array<size_t,2>{bdd_node_counter, i}, nr_bdd_nodes_per_interval[i])); 
                            ++nr_bdd_nodes_per_interval[i]; 
                            assert(bdd_storages[i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[i]);
                        }
                        else // case (iv)
                        {
                            bdd_storages[i].bdd_nodes_.push_back({bdd_node::terminal_0, bdd_node::terminal_1, bdd.variable});
                            //split_bdd_nodes(i, nr_bdd_nodes_per_interval[i]) = {bdd.variable, bdd_node::terminal_0, bdd_node::terminal_1};
                            split_bdd_node_indices.insert(std::make_pair(std::array<size_t,2>{bdd_node_counter, i}, nr_bdd_nodes_per_interval[i])); 
                            ++nr_bdd_nodes_per_interval[i]; 
                            assert(bdd_storages[i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[i]);

                            const size_t next_i = intn.interval(high_var);
                            assert(split_bdd_node_indices.count({bdd.high, next_i}) > 0);
                            const size_t next_high_idx = split_bdd_node_indices.find({bdd.high, next_i})->second;
                            bdd_storages[next_i].bdd_nodes_.push_back({bdd_node::terminal_0, next_high_idx, bdd.variable});
                            //split_bdd_nodes(next_i, nr_bdd_nodes_per_interval[next_i]) = {bdd.variable, bdd_node::terminal_0, next_high_idx};
                            ++nr_bdd_nodes_per_interval[next_i]; 
                            assert(bdd_storages[next_i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[next_i]);

                            duplicated_variables.insert({i, nr_bdds_per_interval[i]-1, next_i, nr_bdds_per_interval[next_i]-1});
                        }
                    }
                    else
                    {
                        assert(bdd.high == bdd_node::terminal_0);
                        const size_t low_var = bdd_nodes_[bdd.low].variable;
                        if(intn.interval(bdd.variable) == intn.interval(low_var)) // case (iii)
                        {
                            assert(split_bdd_node_indices.count({bdd.low,i}) > 0);
                            const size_t low_idx = split_bdd_node_indices.find({bdd.low,i})->second;
                            bdd_storages[i].bdd_nodes_.push_back({low_idx, bdd_node::terminal_0, bdd.variable});
                            //split_bdd_nodes(i, nr_bdd_nodes_per_interval[i]) = {bdd.variable, low_idx, bdd_node::terminal_0};
                            split_bdd_node_indices.insert(std::make_pair(std::array<size_t,2>{bdd_node_counter, i}, nr_bdd_nodes_per_interval[i])); 
                            ++nr_bdd_nodes_per_interval[i]; 
                            assert(bdd_storages[i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[i]);
                        }
                        else // case (iv)
                        {
                            bdd_storages[i].bdd_nodes_.push_back({bdd_node::terminal_1, bdd_node::terminal_0, bdd.variable});
                            //split_bdd_nodes(i, nr_bdd_nodes_per_interval[i]) = {bdd.variable, bdd_node::terminal_1, bdd_node::terminal_0};
                            split_bdd_node_indices.insert(std::make_pair(std::array<size_t,2>{bdd_node_counter, i}, nr_bdd_nodes_per_interval[i])); 
                            ++nr_bdd_nodes_per_interval[i]; 
                            assert(bdd_storages[i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[i]);

                            const size_t next_i = intn.interval(low_var);
                            assert(split_bdd_node_indices.count({bdd.low, next_i}) > 0);
                            const size_t next_low_idx = split_bdd_node_indices.find({bdd.low, next_i})->second;
                            bdd_storages[next_i].bdd_nodes_.push_back({next_low_idx, bdd_node::terminal_0, bdd.variable});
                            //split_bdd_nodes(next_i, nr_bdd_nodes_per_interval[next_i]) = {bdd.variable, next_low_idx, bdd_node::terminal_0};
                            ++nr_bdd_nodes_per_interval[next_i]; 
                            assert(bdd_storages[next_i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[next_i]);

                            duplicated_variables.insert({i, nr_bdds_per_interval[i]-1, next_i, nr_bdds_per_interval[next_i]-1});
                        }
                    } 
                }
                else
                {
                    assert(false);
                }
            }
            // go over each affected interval and set new bdd delimiter values
            active_intervals.clear();
            for(auto bdd_node_counter=bdd_delimiters_[bdd_counter]; bdd_node_counter<bdd_delimiters_[bdd_counter+1]; ++bdd_node_counter)
            {
                const auto& bdd = bdd_nodes_[bdd_node_counter];
                active_intervals.insert(intn.interval(bdd.variable));
            }
            for(const size_t i : active_intervals)
            {
                assert(bdd_storages[i].bdd_nodes_.size() == nr_bdd_nodes_per_interval[i]);
                bdd_storages[i].bdd_delimiters_.push_back(bdd_storages[i].bdd_nodes_.size());
                assert(bdd_storages[i].bdd_delimiters_.size() >= 2 && bdd_storages[i].bdd_delimiters_.back() > bdd_storages[i].bdd_delimiters_[bdd_storages[i].bdd_delimiters_.size()-2]);
                //split_bdd_delimiters(i, nr_bdds_per_interval[i]) = nr_bdd_nodes_per_interval[i];
                ++nr_bdds_per_interval[i];
            }
        }

        return {bdd_storages, duplicated_variables};
    }

    std::vector<typename bdd_storage::bdd_endpoints_> bdd_storage::bdd_endpoints() const
    {
        // iterate through bdd storage bdds and get first and last variable of bdd. Then record bdd indices at corresponding variables
        std::vector<bdd_endpoints_> endpoints;
        endpoints.reserve(nr_bdds());
        std::vector<size_t> bdd_index_counter(nr_variables(), 0);

        tsl::robin_set<size_t> bdd_variables;
        for(size_t bdd_index=0; bdd_index<nr_bdds(); ++bdd_index)
        {
            size_t first_bdd_var = std::numeric_limits<size_t>::max();
            size_t last_bdd_var = 0;
            bdd_variables.clear();
            for(size_t i=bdd_delimiters()[bdd_index]; i<bdd_delimiters()[bdd_index+1]; ++i)
            {
                const size_t bdd_var = bdd_nodes()[i].variable;
                if(bdd_var < nr_variables()) // otherwise top and bottom sink
                {
                    first_bdd_var = std::min(bdd_var, first_bdd_var);
                    last_bdd_var = std::max(bdd_var, last_bdd_var); 
                    bdd_variables.insert(bdd_var);
                }
            }

            assert(first_bdd_var <= last_bdd_var);
            assert(last_bdd_var < this->nr_variables());
            assert(bdd_variables.count(first_bdd_var) > 0);
            assert(bdd_variables.count(last_bdd_var) > 0);

            endpoints.push_back({first_bdd_var, bdd_index_counter[first_bdd_var], last_bdd_var, bdd_index_counter[last_bdd_var]});
            for(size_t v : bdd_variables)
                ++bdd_index_counter[v];
        }

        return endpoints;
    }

    two_dim_variable_array<size_t> bdd_storage::compute_variable_groups() const
    {
        const auto dep_graph_arcs = dependency_graph();
        std::vector<size_t> nr_outgoing_arcs(nr_variables(), 0);
        std::vector<size_t> nr_incoming_arcs(nr_variables(), 0);
        for(const auto [i,j] : dep_graph_arcs)
        {
            assert(i < j);
            ++nr_outgoing_arcs[i];
            ++nr_incoming_arcs[j];
        }
        two_dim_variable_array<size_t> dep_graph_adj(nr_outgoing_arcs.begin(), nr_outgoing_arcs.end());
        std::fill(nr_outgoing_arcs.begin(), nr_outgoing_arcs.end(), 0);
        for(const auto [i,j] : dep_graph_arcs)
            dep_graph_adj(i,nr_outgoing_arcs[i]++) = j;

        two_dim_variable_array<size_t> variable_groups;
        std::vector<size_t> current_nodes;

        // first group consists of all variables with in-degree zero
        for(size_t i=0; i<nr_incoming_arcs.size(); ++i)
            if(nr_incoming_arcs[i] == 0)
                current_nodes.push_back(i);
        std::cout << "nr initial nodes in first variable group = " << current_nodes.size() << "\n";

        std::vector<size_t> next_nodes;
        while(current_nodes.size() > 0)
        {
            next_nodes.clear();
            variable_groups.push_back(current_nodes.begin(), current_nodes.end());
            // decrease in-degree of every node that has incoming arc from one of current nodes. If in-degree reaches zero, schedule nodes to be added to next variable group; 
            for(const size_t i : current_nodes)
            {
                for(const size_t j : dep_graph_adj[i])
                {
                    assert(nr_incoming_arcs[j] > 0);
                    --nr_incoming_arcs[j];
                    if(nr_incoming_arcs[j] == 0)
                        next_nodes.push_back(j); 
                }
            }
            std::swap(current_nodes, next_nodes);
        }

        for(const size_t d : nr_incoming_arcs)
            assert(d == 0);

        return variable_groups;
    } 

}
