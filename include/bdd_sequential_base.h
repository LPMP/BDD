#pragma once

#include <vector>
#include <array>
#include <Eigen/SparseCore>
#include "bdd_storage.h"
#include "two_dimensional_variable_array.hxx"

namespace LPMP {

    // store BDDs one after the other.
    // allows for efficient computation of min marginals
    template<typename BDD_BRANCH_NODE>
        class bdd_sequential_base {
            public:
            bdd_sequential_base(const bdd_storage& stor) { init(stor); }

            void init(const bdd_storage& stor);

            size_t nr_bdds() const;
            size_t nr_bdds(const size_t var) const;
            size_t nr_variables() const;
            size_t nr_variables(const size_t bdd_nr) const;
            size_t variable(const size_t bdd_nr, const size_t bdd_index) const;
            size_t nr_bdd_variables() const;

            double lower_bound();
            std::vector<float> lower_bound_per_bdd();

            void forward_run();
            void backward_run();
            two_dim_variable_array<std::array<float,2>> min_marginals();
            using min_marginal_type = Eigen::Matrix<typename BDD_BRANCH_NODE::value_type, Eigen::Dynamic, 2>;
            std::tuple<min_marginal_type, std::vector<char>> min_marginals_stacked();

            template<typename COST_ITERATOR>
                void set_costs(COST_ITERATOR begin, COST_ITERATOR end);
            void update_costs(const two_dim_variable_array<std::array<float,2>>& delta);
            void update_costs(const min_marginal_type& delta);

            std::vector<float> get_costs();
            using vector_type = Eigen::Matrix<typename BDD_BRANCH_NODE::value_type, Eigen::Dynamic, 1>;
            void update_costs(const vector_type& delta);

            // make a step that is guaranteed to be non-decreasing in the lower bound.
            void diffusion_step(const two_dim_variable_array<std::array<float,2>>& min_margs, const float damping_step = 1.0);

            // Both operations below are inverses of each other
            // Given elements in order bdd_nr/bdd_index, transpose to variable/bdd_index with same variable.
            template<typename T>
                two_dim_variable_array<T> transpose_to_var_order(const two_dim_variable_array<T>& m) const;
            // Given elements in order var/bdd_index with same variable, transpose to bdd_nr/bdd_index.
            template<typename T>
                two_dim_variable_array<T> transpose_to_bdd_order(const two_dim_variable_array<T>& m) const;

            Eigen::SparseMatrix<float> Lagrange_constraint_matrix() const;

            private:
            enum class message_passing_state {
                after_forward_pass,
                after_backward_pass,
                none 
            } message_passing_state_ = message_passing_state::none;

            enum class lower_bound_state {
                valid,
                invalid 
            } lower_bound_state_ = lower_bound_state::invalid; 
            double lower_bound_ = -std::numeric_limits<double>::infinity();

            double compute_lower_bound();
            double compute_lower_bound_after_forward_pass();
            double compute_lower_bound_after_backward_pass();

            std::vector<float> lower_bound_per_bdd_after_forward_pass();
            std::vector<float> lower_bound_per_bdd_after_backward_pass();

            std::array<size_t,2> bdd_range(const size_t bdd_nr) const;
                std::array<size_t,2> bdd_index_range(const size_t bdd_nr, const size_t bdd_idx) const;

                std::vector<BDD_BRANCH_NODE> bdd_branch_nodes_;

                // holds ranges of bdd branch instructions of specific bdd with specific variable
                struct bdd_variable {
                    size_t offset;
                    size_t variable; 
                };
                two_dim_variable_array<bdd_variable> bdd_variables_;
                std::vector<size_t> nr_bdds_per_variable_;
        };

    ////////////////////
    // implementation //
    ////////////////////

    template<typename BDD_BRANCH_NODE>
        size_t bdd_sequential_base<BDD_BRANCH_NODE>::nr_bdds() const
        {
            assert(bdd_variables_.size() > 0);
            return bdd_variables_.size() - 1;
        }

    template<typename BDD_BRANCH_NODE>
        size_t bdd_sequential_base<BDD_BRANCH_NODE>::nr_variables() const
        {
            return nr_bdds_per_variable_.size(); 
        }

    template<typename BDD_BRANCH_NODE>
        size_t bdd_sequential_base<BDD_BRANCH_NODE>::nr_bdds(const size_t variable) const
        {
            assert(variable < nr_variables());
            return nr_bdds_per_variable_[variable];
        }

    template<typename BDD_BRANCH_NODE>
        size_t bdd_sequential_base<BDD_BRANCH_NODE>::nr_variables(const size_t bdd_nr) const
        {
            assert(bdd_nr < nr_bdds());
            assert(bdd_variables_.size(bdd_nr) > 0);
            return bdd_variables_.size(bdd_nr) - 1; 
        }

    template<typename BDD_BRANCH_NODE>
            size_t bdd_sequential_base<BDD_BRANCH_NODE>::variable(const size_t bdd_nr, const size_t bdd_index) const
            {
                assert(bdd_nr < nr_bdds());
                assert(bdd_index < nr_variables(bdd_nr));
                return bdd_variables_(bdd_nr, bdd_index).variable; 
            }

    template<typename BDD_BRANCH_NODE>
        size_t bdd_sequential_base<BDD_BRANCH_NODE>::nr_bdd_variables() const
        {
            return std::accumulate(nr_bdds_per_variable_.begin(), nr_bdds_per_variable_.end(), 0); 
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_sequential_base<BDD_BRANCH_NODE>::init(const bdd_storage& stor)
        {
            bdd_branch_nodes_.clear();
            bdd_branch_nodes_.reserve(stor.bdd_nodes().size());
            bdd_variables_.clear();
            nr_bdds_per_variable_.clear();
            nr_bdds_per_variable_.resize(stor.nr_variables(), 0);

            for(size_t bdd_nr=0; bdd_nr<stor.nr_bdds(); ++bdd_nr)
            {
                const size_t first_bdd_node = stor.bdd_delimiters()[bdd_nr];
                const size_t last_bdd_node = stor.bdd_delimiters()[bdd_nr+1];

                std::unordered_map<size_t, size_t> bdd_node_counter_per_var;
                for(size_t bdd_storage_node_idx = first_bdd_node; bdd_storage_node_idx<last_bdd_node; ++bdd_storage_node_idx) 
                {
                    const auto& bdd_storage_node = stor.bdd_nodes()[bdd_storage_node_idx];
                    if(bdd_node_counter_per_var.count(bdd_storage_node.variable) == 0)
                        bdd_node_counter_per_var.insert({bdd_storage_node.variable, 0});
                    bdd_node_counter_per_var.find(bdd_storage_node.variable)->second++;
                }

                struct bdd_node_counter { size_t variable; size_t nr_bdd_nodes; };
                std::vector<bdd_node_counter> bdd_variable_nr_sorted;
                for(const auto [var, nr_nodes] : bdd_node_counter_per_var)
                {
                    assert(nr_nodes > 0);
                    bdd_variable_nr_sorted.push_back({var, nr_nodes});
                }
                std::sort(bdd_variable_nr_sorted.begin(), bdd_variable_nr_sorted.end(), [](const bdd_node_counter& a, const bdd_node_counter& b) { return a.variable < b.variable; });

                std::vector<bdd_variable> cur_bdd_variables;
                size_t cur_offset = bdd_branch_nodes_.size();
                for(const auto [var, nr_bdd_nodes] : bdd_variable_nr_sorted)
                {
                    cur_bdd_variables.push_back({cur_offset, var});
                    cur_offset += nr_bdd_nodes;
                    ++nr_bdds_per_variable_[var];
                }
                bdd_branch_nodes_.resize(cur_offset);

                cur_bdd_variables.push_back({cur_offset, std::numeric_limits<size_t>::max()}); // For extra delimiter at the end
                bdd_variables_.push_back(cur_bdd_variables.begin(), cur_bdd_variables.end());
                cur_bdd_variables.resize(cur_bdd_variables.size()-1);

                // fill in bdd branch nodes
                for(const auto [offset, var] : cur_bdd_variables)
                    bdd_node_counter_per_var.find(var)->second = offset;

                std::unordered_map<size_t, size_t> bdd_storage_index_to_branch_node_index;
                auto bdd_branch_node_index_from_bdd_storage_index = [&](const size_t cur_idx, const size_t storage_idx) -> size_t {
                    if(storage_idx == bdd_storage::bdd_node::terminal_0)
                        return BDD_BRANCH_NODE::terminal_0_offset;
                    else if(storage_idx == bdd_storage::bdd_node::terminal_1)
                        return BDD_BRANCH_NODE::terminal_1_offset;
                    assert(bdd_storage_index_to_branch_node_index.count(storage_idx) > 0);
                    const size_t pointed_to_idx = bdd_storage_index_to_branch_node_index.find(storage_idx)->second; 
                    assert(cur_idx < pointed_to_idx);
                    return pointed_to_idx - cur_idx;
                };

                auto new_branch_node_index = [&](size_t idx) -> size_t {
                    assert(bdd_storage_index_to_branch_node_index.count(idx) == 0);
                    const bdd_storage::bdd_node bdd_storage_node = stor.bdd_nodes()[idx];
                    const size_t variable = bdd_storage_node.variable;
                    const size_t bdd_branch_node_index = bdd_node_counter_per_var.find(variable)->second;
                    bdd_node_counter_per_var.find(variable)->second++;
                    assert(bdd_node_counter_per_var.find(variable)->second > 0);
                    bdd_storage_index_to_branch_node_index.insert(std::make_pair(idx, bdd_branch_node_index));
                    return bdd_branch_node_index;
                };

                for(size_t bdd_storage_node_idx = first_bdd_node; bdd_storage_node_idx<last_bdd_node; ++bdd_storage_node_idx) 
                {
                    const auto& bdd_storage_node = stor.bdd_nodes()[bdd_storage_node_idx];
                    const size_t branch_node_index = new_branch_node_index(bdd_storage_node_idx);
                    auto& bdd_branch_node = bdd_branch_nodes_[branch_node_index];
                    assert(bdd_branch_node.offset_low == 0 && bdd_branch_node.offset_high == 0);
                    assert(bdd_branch_node.low_cost == 0 && bdd_branch_node.high_cost == 0);

                    bdd_branch_node.offset_low = bdd_branch_node_index_from_bdd_storage_index(branch_node_index, bdd_storage_node.low); 
                    if(bdd_branch_node.offset_low == BDD_BRANCH_NODE::terminal_0_offset)
                        bdd_branch_node.low_cost = std::numeric_limits<decltype(bdd_branch_node.low_cost)>::infinity();

                    bdd_branch_node.offset_high = bdd_branch_node_index_from_bdd_storage_index(branch_node_index, bdd_storage_node.high);
                    if(bdd_branch_node.offset_high == BDD_BRANCH_NODE::terminal_0_offset)
                        bdd_branch_node.high_cost = std::numeric_limits<decltype(bdd_branch_node.high_cost)>::infinity();
                }
            }

            // add last entry for offset
            std::vector<bdd_variable> tmp_bdd_variables;
            tmp_bdd_variables.push_back({bdd_branch_nodes_.size(), std::numeric_limits<size_t>::max()});
            bdd_variables_.push_back(tmp_bdd_variables.begin(), tmp_bdd_variables.end());
        }

    template<typename BDD_BRANCH_NODE>
        double bdd_sequential_base<BDD_BRANCH_NODE>::lower_bound()
        {
            if(lower_bound_state_ == lower_bound_state::invalid)
                compute_lower_bound();
            assert(lower_bound_state_ == lower_bound_state::valid);
            return lower_bound_; 
        }

    template<typename BDD_BRANCH_NODE>
        double bdd_sequential_base<BDD_BRANCH_NODE>::compute_lower_bound()
        {
            if(message_passing_state_ == message_passing_state::after_backward_pass)
            {
                lower_bound_ = compute_lower_bound_after_backward_pass();
            }
            else if(message_passing_state_ == message_passing_state::after_forward_pass)
            {
                lower_bound_ = compute_lower_bound_after_forward_pass();
            }
            else if(message_passing_state_ == message_passing_state::none)
            {
                backward_run();
                lower_bound_ = compute_lower_bound_after_backward_pass();
            }

            lower_bound_state_ = lower_bound_state::valid; 
            return lower_bound_;
        }

    template<typename BDD_BRANCH_NODE>
        double bdd_sequential_base<BDD_BRANCH_NODE>::compute_lower_bound_after_backward_pass()
        {
            assert(message_passing_state_ == message_passing_state::after_backward_pass);
            double lb = 0.0;

            // TODO: works only for non-split BDDs
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                const auto [first,last] = bdd_index_range(bdd_nr, 0);
                assert(first+1 == last);
                lb += bdd_branch_nodes_[first].m;
            }

            return lb;
        }

    template<typename BDD_BRANCH_NODE>
        double bdd_sequential_base<BDD_BRANCH_NODE>::compute_lower_bound_after_forward_pass()
        {
            assert(message_passing_state_ == message_passing_state::after_forward_pass);
            double lb = 0.0;

            // TODO: works only for non-split BDDs
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                const auto [first,last] = bdd_index_range(bdd_nr, nr_variables(bdd_nr)-1);
                float bdd_lb = std::numeric_limits<float>::infinity();
                for(size_t idx=first; idx<last; ++idx)
                {
                    const auto mm = bdd_branch_nodes_[idx].min_marginals();
                    bdd_lb = std::min({bdd_lb, mm[0], mm[1]});
                }
                lb += bdd_lb;
            }

            return lb;
        }

    template<typename BDD_BRANCH_NODE>
        std::vector<float> bdd_sequential_base<BDD_BRANCH_NODE>::lower_bound_per_bdd()
        {
            if(message_passing_state_ == message_passing_state::after_backward_pass)
            {
                return lower_bound_per_bdd_after_backward_pass();
            }
            else if(message_passing_state_ == message_passing_state::after_forward_pass)
            {
                return lower_bound_per_bdd_after_forward_pass();
            }
            else
            {
                assert(message_passing_state_ == message_passing_state::none);
                backward_run();
                return lower_bound_per_bdd_after_backward_pass();
            }
        }

    // TODO: possibly implement template functino that takes lambda and can compute lower bound and lower bound per bdd

    template<typename BDD_BRANCH_NODE>
        std::vector<float> bdd_sequential_base<BDD_BRANCH_NODE>::lower_bound_per_bdd_after_forward_pass()
        {
            assert(message_passing_state_ == message_passing_state::after_forward_pass);
            std::vector<float> lbs;
            lbs.reserve(nr_bdds());
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                const auto [first,last] = bdd_index_range(bdd_nr, nr_variables(bdd_nr)-1);
                float bdd_lb = std::numeric_limits<float>::infinity();
                for(size_t idx=first; idx<last; ++idx)
                {
                    const auto mm = bdd_branch_nodes_[idx].min_marginals();
                    bdd_lb = std::min({bdd_lb, mm[0], mm[1]});
                }
                lbs.push_back(bdd_lb);
            }

            return lbs;
        }

    template<typename BDD_BRANCH_NODE>
        std::vector<float> bdd_sequential_base<BDD_BRANCH_NODE>::lower_bound_per_bdd_after_backward_pass()
        {
            assert(message_passing_state_ == message_passing_state::after_backward_pass);
            std::vector<float> lbs;
            lbs.reserve(nr_bdds());

            // TODO: works only for non-split BDDs
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                const auto [first,last] = bdd_index_range(bdd_nr, 0);
                assert(first+1 == last);
                lbs.push_back(bdd_branch_nodes_[first].m);
            }

            return lbs;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_sequential_base<BDD_BRANCH_NODE>::forward_run()
        {
            if(message_passing_state_ == message_passing_state::after_forward_pass)
                return;
            message_passing_state_ = message_passing_state::none;
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                const auto [first_bdd_node, last_bdd_node] = bdd_range(bdd_nr);
                // TODO: This only works for non-split BDDs with exactly one root node
                assert(first_bdd_node + 1 == last_bdd_node);
                bdd_branch_nodes_[first_bdd_node].m = 0.0;
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                    bdd_branch_nodes_[i].prepare_forward_step(); 
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                    bdd_branch_nodes_[i].forward_step(); 
            } 
            message_passing_state_ = message_passing_state::after_forward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_sequential_base<BDD_BRANCH_NODE>::backward_run()
        {
            if(message_passing_state_ == message_passing_state::after_backward_pass)
                return;
            message_passing_state_ = message_passing_state::none;
//#pragma omp parallel for schedule(guided,128)
            for(std::ptrdiff_t bdd_nr=nr_bdds()-1; bdd_nr>=0; --bdd_nr)
            {
                const auto [first_bdd_node, last_bdd_node] = bdd_range(bdd_nr);
                for(std::ptrdiff_t i=last_bdd_node-1; i>=std::ptrdiff_t(first_bdd_node); --i)
                    bdd_branch_nodes_[i].backward_step(); 
            }

            message_passing_state_ = message_passing_state::after_backward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        std::array<size_t,2> bdd_sequential_base<BDD_BRANCH_NODE>::bdd_index_range(const size_t bdd_nr, const size_t bdd_idx) const
        {
            assert(bdd_nr < nr_bdds());
            assert(bdd_idx < nr_variables(bdd_nr));
            const size_t first_bdd_node = bdd_variables_(bdd_nr, bdd_idx).offset;
            const size_t last_bdd_node = bdd_variables_(bdd_nr, bdd_idx+1).offset;
            return {first_bdd_node, last_bdd_node};
        }

    template<typename BDD_BRANCH_NODE>
        std::array<size_t,2> bdd_sequential_base<BDD_BRANCH_NODE>::bdd_range(const size_t bdd_nr) const
        {
            assert(bdd_nr < nr_bdds());
            const size_t first = bdd_variables_(bdd_nr, 0).offset;
            const size_t last = bdd_variables_(bdd_nr+1, 0).offset;
            return {first, last}; 
        }

    template<typename BDD_BRANCH_NODE>
        two_dim_variable_array<std::array<float,2>> bdd_sequential_base<BDD_BRANCH_NODE>::min_marginals()
        {
            backward_run();
            std::vector<size_t> nr_bdd_variables;
            nr_bdd_variables.reserve(nr_bdds());
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
                nr_bdd_variables.push_back(nr_variables(bdd_nr));
            two_dim_variable_array<std::array<float,2>> min_margs(nr_bdd_variables);

//#pragma omp parallel for schedule(guided,128)
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                // intialize
                const auto [first,last] = bdd_index_range(bdd_nr, 0);
                assert(first + 1 == last);
                bdd_branch_nodes_[first].m = 0.0;

                for(size_t idx=0; idx<nr_variables(bdd_nr); ++idx)
                {
                    std::array<float,2> mm = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
                    const auto [first,last] = bdd_index_range(bdd_nr, idx);
                    for(size_t i=first; i<last; ++i)
                    {
                        const std::array<float,2> cur_mm = bdd_branch_nodes_[i].min_marginals();
                        mm[0] = std::min(mm[0], cur_mm[0]);
                        mm[1] = std::min(mm[1], cur_mm[1]); 
                    }

                    min_margs(bdd_nr, idx) = mm;

                    for(size_t i=first; i<last; ++i)
                        bdd_branch_nodes_[i].prepare_forward_step(); 
                    for(size_t i=first; i<last; ++i)
                        bdd_branch_nodes_[i].forward_step();
                }
            }

            message_passing_state_ = message_passing_state::after_forward_pass;

            return min_margs;
        }
    
    template<typename BDD_BRANCH_NODE>
        std::tuple<typename bdd_sequential_base<BDD_BRANCH_NODE>::min_marginal_type, std::vector<char>> bdd_sequential_base<BDD_BRANCH_NODE>::min_marginals_stacked()
        {
            backward_run();
            min_marginal_type min_margs(nr_bdd_variables(), 2);
            std::vector<char> solutions;
            solutions.reserve(nr_bdd_variables());

//#pragma omp parallel for schedule(guided,128)
            size_t c = 0;
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                // intialize
                const auto [first,last] = bdd_index_range(bdd_nr, 0);
                assert(first + 1 == last);
                const float bdd_lb = bdd_branch_nodes_[first].m; 
                bdd_branch_nodes_[first].m = 0.0;

                size_t next_node = first;
                for(size_t idx=0; idx<nr_variables(bdd_nr); ++idx, ++c)
                {
                    std::array<float,2> mm = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};

                    const auto [first,last] = bdd_index_range(bdd_nr, idx);
                    for(size_t i=first; i<last; ++i)
                    {
                        const std::array<float,2> cur_mm = bdd_branch_nodes_[i].min_marginals();
                        mm[0] = std::min(mm[0], cur_mm[0]);
                        mm[1] = std::min(mm[1], cur_mm[1]); 

                        // see if active path points to current node;
                        if(next_node == i)
                        {
                            if(cur_mm[0] < cur_mm[1])
                            {
                                assert(std::abs(bdd_lb - mm[0]) <= 1e-6);
                                solutions.push_back(0); 
                                if(bdd_branch_nodes_[i].offset_low == BDD_BRANCH_NODE::terminal_0_offset)
                                {
                                    assert(false); // this cannot happen
                                }
                                else if(bdd_branch_nodes_[i].offset_low == BDD_BRANCH_NODE::terminal_1_offset)
                                {
                                    // we have arrived at the last variable of the bdd
                                    assert(idx+1 == nr_variables(bdd_nr));
                                }
                                else
                                {
                                    next_node = std::distance(&bdd_branch_nodes_[0], bdd_branch_nodes_[i].address(bdd_branch_nodes_[i].offset_low));
                                }
                            }
                            else
                            {
                                assert(std::abs(bdd_lb - mm[1]) <= 1e-6);
                                solutions.push_back(1); 
                                if(bdd_branch_nodes_[i].offset_high == BDD_BRANCH_NODE::terminal_0_offset)
                                {
                                    assert(false); // this cannot happen
                                }
                                else if(bdd_branch_nodes_[i].offset_high == BDD_BRANCH_NODE::terminal_1_offset)
                                {
                                    // we have arrived at the last variable of the bdd
                                    assert(idx+1 == nr_variables(bdd_nr));
                                }
                                else
                                {
                                    next_node = std::distance(&bdd_branch_nodes_[0], bdd_branch_nodes_[i].address(bdd_branch_nodes_[i].offset_high));
                                }
                            }
                        }
                    }

                    min_margs(int(c),0) = mm[0];
                    min_margs(int(c),1) = mm[1];

                    for(size_t i=first; i<last; ++i)
                        bdd_branch_nodes_[i].prepare_forward_step(); 
                    for(size_t i=first; i<last; ++i)
                        bdd_branch_nodes_[i].forward_step();
                }
            }

            message_passing_state_ = message_passing_state::after_forward_pass;
            std::cout << "solutions size " << solutions.size() << "\n";

            return {min_margs, solutions};
        }

    template<typename BDD_BRANCH_NODE>
        std::vector<float> bdd_sequential_base<BDD_BRANCH_NODE>::get_costs()
        {
            std::vector<float> costs;
            costs.reserve(nr_bdd_variables());

            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                for(size_t idx=0; idx<nr_variables(bdd_nr); ++idx)
                {
                    const auto [first,last] = bdd_index_range(bdd_nr, idx);
                    for(size_t i=first; i<last; ++i)
                    {
                        const auto& bdd = bdd_branch_nodes_[i];
                        if(bdd.offset_low != BDD_BRANCH_NODE::terminal_0_offset)
                            assert(bdd.low_cost == 0.0);
                        if(bdd.offset_high != BDD_BRANCH_NODE::terminal_0_offset)
                        {
                            costs.push_back(bdd.high_cost);
                            break;
                        } 
                    }

                }
            }

            assert(costs.size() == nr_bdd_variables());

            return costs;
        }

    template<typename BDD_BRANCH_NODE>
        template<typename COST_ITERATOR> 
        void bdd_sequential_base<BDD_BRANCH_NODE>::set_costs(COST_ITERATOR begin, COST_ITERATOR end)
        {
            assert(std::distance(begin, end) == nr_variables());
//#pragma omp parallel for schedule(guided,128)
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                for(size_t bdd_idx=0; bdd_idx<nr_variables(bdd_nr); ++bdd_idx)
                {
                    const auto [first_node, last_node] = bdd_index_range(bdd_nr, bdd_idx);
                    const size_t var = variable(bdd_nr, bdd_idx);
                    const auto cost = *(begin+var)/nr_bdds(var);;
                    assert(std::isfinite(cost));
                    for(size_t i=first_node; i<last_node; ++i)
                    {
                        if(bdd_branch_nodes_[i].offset_low == BDD_BRANCH_NODE::terminal_0_offset)
                            assert(bdd_branch_nodes_[i].low_cost == std::numeric_limits<decltype(bdd_branch_nodes_[i].low_cost)>::infinity());
                        else
                            assert(bdd_branch_nodes_[i].low_cost == 0.0);

                        if(bdd_branch_nodes_[i].offset_high == BDD_BRANCH_NODE::terminal_0_offset)
                            assert(bdd_branch_nodes_[i].high_cost == std::numeric_limits<decltype(bdd_branch_nodes_[i].high_cost)>::infinity());
                        else
                            assert(bdd_branch_nodes_[i].high_cost == 0.0);

                        if(bdd_branch_nodes_[i].offset_high != BDD_BRANCH_NODE::terminal_0_offset)
                            bdd_branch_nodes_[i].high_cost = cost;
                    }
                }
            }
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_sequential_base<BDD_BRANCH_NODE>::update_costs(const two_dim_variable_array<std::array<float,2>>& delta)
        {
            message_passing_state_ = message_passing_state::none;
            assert(delta.size() == nr_bdds());
            const auto delta_t = transpose_to_bdd_order(delta);
#pragma omp parallel for schedule(guided,128)
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                for(size_t bdd_idx=0; bdd_idx<nr_variables(bdd_nr); ++bdd_idx)
                {
                    const auto [first_node, last_node] = bdd_index_range(bdd_nr, bdd_idx);
                    for(size_t i=first_node; i<last_node; ++i)
                    {
                        bdd_branch_nodes_[i].low_cost += delta_t(bdd_nr, bdd_idx)[0]; 
                        bdd_branch_nodes_[i].high_cost += delta_t(bdd_nr, bdd_idx)[1]; 
                    }
                }
            } 
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_sequential_base<BDD_BRANCH_NODE>::update_costs(const min_marginal_type& delta)
        {
            message_passing_state_ = message_passing_state::none;
            assert(delta.rows() == nr_bdd_variables());
            assert(delta.cols() == 2);
//#pragma omp parallel for schedule(guided,128)
            size_t c = 0;
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                for(size_t bdd_idx=0; bdd_idx<nr_variables(bdd_nr); ++bdd_idx, ++c)
                {
                    const auto [first_node, last_node] = bdd_index_range(bdd_nr, bdd_idx);
                    for(size_t i=first_node; i<last_node; ++i)
                    {
                        bdd_branch_nodes_[i].low_cost += delta(c, 0);
                        bdd_branch_nodes_[i].high_cost += delta(c, 1);
                    }
                }
            }
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_sequential_base<BDD_BRANCH_NODE>::update_costs(const vector_type& delta)
        {
            message_passing_state_ = message_passing_state::none;
            assert(delta.rows() == nr_bdd_variables());
            assert(delta.cols() == 1);
//#pragma omp parallel for schedule(guided,128)
            size_t c = 0;
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                for(size_t bdd_idx=0; bdd_idx<nr_variables(bdd_nr); ++bdd_idx, ++c)
                {
                    const auto [first_node, last_node] = bdd_index_range(bdd_nr, bdd_idx);
                    for(size_t i=first_node; i<last_node; ++i)
                    {
                        bdd_branch_nodes_[i].high_cost += delta(c, 0);
                    }
                }
            }
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_sequential_base<BDD_BRANCH_NODE>::diffusion_step(const two_dim_variable_array<std::array<float,2>>& min_margs, const float damping_step)
        {
            throw std::runtime_error("not correct yet");
            message_passing_state_ = message_passing_state::none;
            assert(min_margs.size() == nr_bdds());
            assert(damping_step >= 0.0 && damping_step <= 1.0);
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                assert(min_margs.size(bdd_nr) == nr_bdd_variables(bdd_nr));
                for(size_t bdd_idx=0; bdd_idx<nr_bdd_variables(bdd_nr); ++bdd_idx)
                {
                    const size_t var = variable(bdd_nr, bdd_idx);
                    float denom = 1.0 / (nr_bdds(var)-1);
                    const auto [first_bdd_node, last_bdd_node] = bdd_index_range(bdd_nr, bdd_idx);
                    for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                    {
                        bdd_branch_nodes_[i].low_cost -= min_margs(bdd_nr, bdd_idx)[0];
                        bdd_branch_nodes_[i].high_cost -= min_margs(bdd_nr, bdd_idx)[1];
                    } 
                } 
            } 
        }

    template<typename BDD_BRANCH_NODE>
        template<typename T>
        two_dim_variable_array<T> bdd_sequential_base<BDD_BRANCH_NODE>::transpose_to_var_order(const two_dim_variable_array<T>& m) const
        {
            assert(m.size() == nr_bdds());
            std::vector<size_t> counter(nr_variables(), 0);

            two_dim_variable_array<T> transposed(nr_bdds_per_variable_);
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                assert(m.size(bdd_nr) == nr_variables(bdd_nr));
                for(size_t bdd_idx=0; bdd_idx<nr_variables(bdd_nr); ++bdd_idx)
                {
                    const size_t var = variable(bdd_nr, bdd_idx);
                    transposed(var, counter[var]++) = m(bdd_nr, bdd_idx); 
                }
            }

            return transposed;
        }

    template<typename BDD_BRANCH_NODE>
    template<typename T>
        two_dim_variable_array<T> bdd_sequential_base<BDD_BRANCH_NODE>::transpose_to_bdd_order(const two_dim_variable_array<T>& m) const
        {
            assert(m.size() == nr_variables());
            std::vector<size_t> counter;
            counter.reserve(nr_bdds());
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
                counter.push_back(nr_variables(bdd_nr));

            two_dim_variable_array<T> transposed(counter);
            counter.clear();
            counter.resize(nr_variables(),0);

            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                for(size_t bdd_idx=0; bdd_idx<nr_variables(bdd_nr); ++bdd_idx)
                {
                    const size_t var = variable(bdd_nr, bdd_idx);
                    transposed(bdd_nr, bdd_idx) = m(var, counter[var]++);
                }
            }

            return transposed; 
        }

    template<typename BDD_BRANCH_NODE>
        Eigen::SparseMatrix<float> bdd_sequential_base<BDD_BRANCH_NODE>::Lagrange_constraint_matrix() const
        {
            using T = Eigen::Triplet<float>;
            std::vector<T> coefficients;
            size_t c = 0;
            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                for(size_t bdd_idx=0; bdd_idx<nr_variables(bdd_nr); ++bdd_idx, ++c)
                {
                    const size_t var = variable(bdd_nr, bdd_idx);
                    coefficients.push_back(T(var,c,1)); 
                }
            }

            Eigen::SparseMatrix<float> A(nr_variables(), nr_bdd_variables());
            A.setFromTriplets(coefficients.begin(), coefficients.end());
            return A; 
        }
}
