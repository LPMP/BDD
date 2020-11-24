#pragma once

#include <tsl/robin_map.h>
#include <vector>
#include <array>
#include <cstddef>
#include "bdd_storage.h"
#include "two_dimensional_variable_array.hxx"
#include "time_measure_util.h"
#include <iostream>

namespace LPMP {

    template<typename T, size_t N, std::size_t... I>
        constexpr auto create_iota_array_impl(std::index_sequence<I...>) {
            return std::array<T, N>{ {I...} };
        }

    template<typename T, size_t N>
        constexpr auto create_iota_array() {
            return create_iota_array_impl<T, N>(std::make_index_sequence<N>{});
        } 

    template<typename T, size_t N, std::size_t... I>
        constexpr auto create_constant_array_impl(const T& value, std::index_sequence<I...>) {
            return std::array<T, N>{{(static_cast<void>(I), value)...}};
        }

    template<typename T, size_t N>
        constexpr auto create_constant_array(const T& value) {
            return create_constant_array_impl<T, N>(value, std::make_index_sequence<N>{});
        } 
 
    template<size_t N>
    class bdd_branch_node_vec {
        public:
            // offsets are added to the address of the current bdd_branch_node_vec. The compute address points to the bdd_branch_node_vec
            std::array<uint32_t,N> offset_low = create_iota_array<uint32_t,N>();
            std::array<uint32_t,N> offset_high = create_iota_array<uint32_t,N>();
            std::array<float,N> m = create_constant_array<float,N>(std::numeric_limits<float>::infinity());
            std::array<float,N> low_cost = {0.0};
            std::array<float,N> high_cost = {0.0};

            constexpr static uint32_t inactive_bdd_index = std::numeric_limits<uint32_t>::max();
            std::array<uint32_t,N> bdd_index = create_constant_array<uint32_t,N>(inactive_bdd_index);

            void prepare_forward_step();
            void forward_step();
            void backward_step();

            void min_marginal(float* reduced_min_marginals0, float* reduced_min_marginals1);
            void set_marginal(float* min_marginals0, float* min_marginals1, const std::array<float,2> avg_marginals);

            constexpr static uint32_t terminal_0_offset = std::numeric_limits<uint32_t>::max();
            constexpr static uint32_t terminal_1_offset = std::numeric_limits<uint32_t>::max()-1;

            std::tuple<bdd_branch_node_vec*, size_t> address(uint32_t offset);
            uint32_t synthesize_address(bdd_branch_node_vec<N>* node, const uint32_t intra_node_offset);

            constexpr static uint32_t log_N = std::log2(N);

            ~bdd_branch_node_vec()
            {
                static_assert(N == 1 || N == 2 || N == 4 || N == 8, "vector width not compatible");
                static_assert(sizeof(float) == 4, "float must be quadword");
                static_assert(sizeof(uint32_t) == 4, "uint32_t must be quadword");
                static_assert(std::pow(2,log_N) == N, "N must be a power of 2");
            }
    };

    template<size_t N>
    std::tuple<bdd_branch_node_vec<N>*, size_t> bdd_branch_node_vec<N>::address(uint32_t offset)
    {
        assert(offset != terminal_0_offset && offset != terminal_1_offset);
        const uint32_t i = offset % N;
        const uint32_t node_offset = offset >> log_N;
        bdd_branch_node_vec<N>* addr = this + node_offset;
        return {addr, i}; 
    }

    template<size_t N>
        uint32_t bdd_branch_node_vec<N>::synthesize_address(bdd_branch_node_vec<N>* node, const uint32_t intra_node_offset)
        {
            assert(intra_node_offset < N);
            assert(this < node);
            assert(std::distance(this, node) < std::numeric_limits<uint32_t>::max()/N);
            assert(std::distance(this, node) > 0);
            const uint32_t node_offset = std::distance(this, node);
            const uint32_t offset = (node_offset << log_N) + intra_node_offset;
            assert(std::get<0>(address(offset)) == node);
            assert(std::get<1>(address(offset)) == intra_node_offset);
            return offset;
        }

    template<size_t N>
        void bdd_branch_node_vec<N>::backward_step()
        {
            for(int i=0; i<N; ++i)
            {
                if(offset_low[i] == terminal_0_offset)
                {
                    assert(low_cost[i] == std::numeric_limits<float>::infinity());
                    m[i] = std::numeric_limits<float>::infinity();
                }
                else if(offset_low[i] == terminal_1_offset)
                {
                    assert(std::isfinite(low_cost[i]));
                    m[i] = low_cost[i];
                }
                else
                {
                    const auto [low_branch_node, o] = address(offset_low[i]);
                    m[i] = low_branch_node->m[o] + low_cost[i];
                }
            }
            for(int i=0; i<N; ++i)
            {
                if(offset_high[i] == terminal_0_offset)
                {
                    assert(high_cost[i] == std::numeric_limits<float>::infinity());
                }
                else if(offset_high[i] == terminal_1_offset)
                {
                    assert(std::isfinite(high_cost[i]));
                    m[i] = std::min(m[i], high_cost[i]);
                }
                else
                {
                    const auto [high_branch_node, o] = address(offset_high[i]);
                    m[i] = std::min(m[i], high_branch_node->m[o] + high_cost[i]);
                }
            }
            for(int i=0; i<N; ++i)
            {
                if(bdd_index[i] != inactive_bdd_index)
                {
                    assert(std::isfinite(m[i]));
                }
            }
        }

    template<size_t N>
        void bdd_branch_node_vec<N>::prepare_forward_step()
        {
            for(int i=0; i<N; ++i)
            {
                if(offset_low[i] != terminal_0_offset && offset_low[i] != terminal_1_offset)
                {
                    const auto [low_branch_node, o] = address(offset_low[i]);
                    low_branch_node->m[o] = std::numeric_limits<float>::infinity(); 
                }
            }

            for(int i=0; i<N; ++i)
            {
                if(offset_high[i] != terminal_0_offset && offset_high[i] != terminal_1_offset)
                {
                    const auto [high_branch_node, o] = address(offset_high[i]);
                    high_branch_node->m[o] = std::numeric_limits<float>::infinity();
                }
            }
        }

    template<size_t N>
        void bdd_branch_node_vec<N>::forward_step()
        {
            for(int i=0; i<N; ++i)
            {
                if(offset_low[i] != terminal_0_offset && offset_low[i] != terminal_1_offset)
                {
                    const auto [low_branch_node, o] = address(offset_low[i]);
                    low_branch_node->m[o] = std::min(low_branch_node->m[o], m[i] + low_cost[i]);
                }
            }

            for(int i=0; i<N; ++i)
            {
                if(offset_high[i] != terminal_0_offset && offset_high[i] != terminal_1_offset)
                {
                    const auto [high_branch_node, o] = address(offset_high[i]);
                    high_branch_node->m[o] = std::min(high_branch_node->m[o], m[i] + high_cost[i]);
                }
            } 
        }

    template<size_t N>
        void bdd_branch_node_vec<N>::min_marginal(float* reduced_min_marginals0, float* reduced_min_marginals1)
        {
            for(int i=0; i<N; ++i)
            {
                if(bdd_index[i] == inactive_bdd_index)
                    continue;
                if(offset_low[i] == terminal_0_offset)
                {
                    //min_marg0[i] = std::numeric_limits<float>::infinity();
                }
                else if(offset_low[i] == terminal_1_offset)
                {
                    //min_marg0[i] = m[i] + low_cost[i];
                    reduced_min_marginals0[bdd_index[i]] = std::min(m[i] + low_cost[i], reduced_min_marginals0[bdd_index[i]]);
                }
                else
                {
                    const auto [low_branch_node, o] = address(offset_low[i]);
                    //min_marg0[i] = m[i] + low_cost[i] + low_branch_node->m[o];
                    reduced_min_marginals0[bdd_index[i]] = std::min(m[i] + low_cost[i] + low_branch_node->m[o], reduced_min_marginals0[bdd_index[i]]);
                }
            }

            for(int i=0; i<N; ++i)
            {
                if(bdd_index[i] == inactive_bdd_index)
                    continue;
                if(offset_high[i] == terminal_0_offset)
                {
                    //min_marg1[i] = std::numeric_limits<float>::infinity();
                }
                else if(offset_high[i] == terminal_1_offset)
                {
                    //min_marg1[i] = m[i] + high_cost[i];
                    reduced_min_marginals1[bdd_index[i]] = std::min(m[i] + high_cost[i], reduced_min_marginals1[bdd_index[i]]);
                }
                else
                {
                    const auto [high_branch_node, o] = address(offset_high[i]);
                    //min_marg1[i] = m[i] + high_cost[i] + high_branch_node->m[o];
                    reduced_min_marginals1[bdd_index[i]] = std::min(m[i] + high_cost[i] + high_branch_node->m[o], reduced_min_marginals1[bdd_index[i]]);
                }
            }
        }

    template<size_t N>
        void bdd_branch_node_vec<N>::set_marginal(float* reduced_min_marginals0, float* reduced_min_marginals1, const std::array<float,2> avg_marginals)
        {
            assert(std::isfinite(avg_marginals[0]));
            assert(std::isfinite(avg_marginals[1]));
            for(int i=0; i<N; ++i)
            {
                if(bdd_index[i] != inactive_bdd_index)
                {
                    assert(std::isfinite(reduced_min_marginals0[bdd_index[i]]));
                    low_cost[i] += -reduced_min_marginals0[bdd_index[i]] + avg_marginals[0];
                }
            }
            for(int i=0; i<N; ++i)
            {
                if(bdd_index[i] != inactive_bdd_index)
                {
                    assert(std::isfinite(reduced_min_marginals1[bdd_index[i]]));
                    high_cost[i] += -reduced_min_marginals1[bdd_index[i]] + avg_marginals[1]; 
                }
            } 
        }

    // bdds are stored in variable groups. Each variable group is a set of variables that can be processed in parallel.
    template<size_t N>
    class bdd_mma_base_vec {
        public:
            bdd_mma_base_vec(const bdd_storage& bdd_storage_) { init(bdd_storage_); }
            two_dim_variable_array<size_t> compute_variable_groups(const bdd_storage& bdd_storage_) const;
            void init(const bdd_storage& bdd_storage_);
            size_t nr_variables() const;
            size_t nr_variable_groups() const;
            size_t nr_bdd_vectors(const size_t var_group) const;
            size_t nr_bdds(const size_t var_group) const { assert(var_group < nr_variable_groups()); return nr_bdds_[var_group]; }

            void forward_step(const size_t var_group);
            void min_marginal_averaging_forward();
            void min_marginal_averaging_step_forward(const size_t var_group);

            void backward_step(const size_t var_group);
            void min_marginal_averaging_backward();
            void min_marginal_averaging_step_backward(const size_t var_group);

            float average_marginals(float* marginals, const size_t nr_marginals);

            void iteration();
            void backward_run();
            void compute_lower_bound(); 
            void solve(const size_t max_iter); 
            double lower_bound() const { return lower_bound_; }
            void set_cost(const double c, const size_t var);

        private:
            std::vector<bdd_branch_node_vec<N>> bdd_branch_nodes_;
            std::vector<size_t> bdd_branch_node_offsets_; // offsets into where bdd branch nodes belonging to a variable start 
            std::vector<size_t> bdd_branch_node_group_offsets_; // offsets into where bdd branch nodes belonging to a variable group start 
            std::vector<size_t> nr_bdds_;
            struct first_bdd_index { size_t bdd_node_index; uint32_t slot; };
            std::vector<first_bdd_index> first_bdd_node_indices_; // used for computing lower bound
            double lower_bound_ = -std::numeric_limits<double>::infinity();
    };

    template<size_t N>
        size_t bdd_mma_base_vec<N>::nr_variables() const
        {
            return bdd_branch_node_offsets_.size()-1; 
        }

    template<size_t N>
        size_t bdd_mma_base_vec<N>::nr_variable_groups() const
        {
            return bdd_branch_node_group_offsets_.size()-1; 
        }

    template<size_t N>
        size_t bdd_mma_base_vec<N>::nr_bdd_vectors(const size_t var) const
        {
            assert(var < nr_variables());
            return bdd_branch_node_offsets_[var+1] - bdd_branch_node_offsets_[var];
        }

    template<size_t N>
        two_dim_variable_array<size_t> bdd_mma_base_vec<N>::compute_variable_groups(const bdd_storage& bdd_storage_) const
        {
            const auto dep_graph_arcs = bdd_storage_.dependency_graph();
            std::vector<size_t> nr_outgoing_arcs(bdd_storage_.nr_variables(), 0);
            std::vector<size_t> nr_incoming_arcs(bdd_storage_.nr_variables(), 0);
            for(const auto [i,j] : dep_graph_arcs)
            {
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

    template<size_t N>
        void bdd_mma_base_vec<N>::init(const bdd_storage& bdd_storage_)
        {
            //const auto variable_groups = compute_variable_groups(bdd_storage_);

            std::cout << "N = " << N << ", log(N) = " << bdd_branch_node_vec<N>::log_N << "\n";
            // count bdd branch nodes per variable
            std::vector<size_t> bdd_branch_nodes_per_var(bdd_storage_.nr_variables(), 0);
            for(const auto& stored_bdd_node : bdd_storage_.bdd_nodes())
            {
                ++bdd_branch_nodes_per_var[stored_bdd_node.variable]; 
            }
            // round up to nearest multiple of N
            for(size_t& x : bdd_branch_nodes_per_var)
                if(x % N != 0)
                    x += N - (x%N);

            bdd_branch_node_offsets_.reserve(bdd_storage_.nr_variables()+1);
            bdd_branch_node_offsets_.push_back(0);
            for(const size_t i : bdd_branch_nodes_per_var)
            {
                const size_t nr_bdd_vecs = i/N;
                bdd_branch_node_offsets_.push_back(bdd_branch_node_offsets_.back() + nr_bdd_vecs);
            }

            bdd_branch_nodes_.resize(bdd_branch_node_offsets_.back());
            std::vector<size_t> bdd_branch_nodes_counter(bdd_storage_.nr_variables(), 0);

            struct bdd_index_to_offset {
                bdd_branch_node_vec<N>* ptr;
                size_t intra_vec_index; 
            };
            std::vector<bdd_index_to_offset> stored_bdd_index_to_bdd_offset(bdd_storage_.bdd_nodes().size());
            //tsl::robin_map<size_t, bdd_index_to_offset> stored_bdd_index_to_bdd_offset;

            for(size_t bdd_index=0; bdd_index<bdd_storage_.nr_bdds(); ++bdd_index)
            {
                //stored_bdd_index_to_bdd_offset.clear();
                const size_t first_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index]; 
                const size_t last_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index+1];
                for(size_t stored_bdd_node_index=first_stored_bdd_node; stored_bdd_node_index<last_stored_bdd_node; ++stored_bdd_node_index)
                {
                    const auto& stored_bdd = bdd_storage_.bdd_nodes()[stored_bdd_node_index];
                    const size_t v = stored_bdd.variable;
                    const size_t bdd_branch_index = bdd_branch_node_offsets_[v] + bdd_branch_nodes_counter[v] / N;

                    const size_t bdd_branch_slot = bdd_branch_nodes_counter[v] % N;
                    ++bdd_branch_nodes_counter[v];

                    if(v == bdd_storage_.bdd_nodes()[last_stored_bdd_node-1].variable)
                        first_bdd_node_indices_.push_back({bdd_branch_index, bdd_branch_slot});

                    if(stored_bdd.low == bdd_storage::bdd_node::terminal_0)
                    {
                        bdd_branch_nodes_[bdd_branch_index].offset_low[bdd_branch_slot] = bdd_branch_node_vec<N>::terminal_0_offset;;;
                        bdd_branch_nodes_[bdd_branch_index].low_cost[bdd_branch_slot] = std::numeric_limits<float>::infinity();
                    }
                    else if(stored_bdd.low == bdd_storage::bdd_node::terminal_1)
                    {
                        assert(bdd_branch_nodes_[bdd_branch_index].low_cost[bdd_branch_slot] == 0.0);
                        bdd_branch_nodes_[bdd_branch_index].offset_low[bdd_branch_slot] = bdd_branch_node_vec<N>::terminal_1_offset;;
                    }
                    else
                    {
                        assert(bdd_branch_nodes_[bdd_branch_index].low_cost[bdd_branch_slot] == 0.0);
                        //assert(stored_bdd_index_to_bdd_offset.count(stored_bdd.low) > 0);
                        //const auto [low_ptr, low_intra_vec_index] = stored_bdd_index_to_bdd_offset.find(stored_bdd.low)->second;
                        const auto [low_ptr, low_intra_vec_index] = stored_bdd_index_to_bdd_offset[stored_bdd.low];
                        bdd_branch_nodes_[bdd_branch_index].offset_low[bdd_branch_slot] = bdd_branch_nodes_[bdd_branch_index].synthesize_address(low_ptr, low_intra_vec_index);
                    }

                    if(stored_bdd.high == bdd_storage::bdd_node::terminal_0)
                    {
                        bdd_branch_nodes_[bdd_branch_index].offset_high[bdd_branch_slot] = bdd_branch_node_vec<N>::terminal_0_offset;;;
                        bdd_branch_nodes_[bdd_branch_index].high_cost[bdd_branch_slot] = std::numeric_limits<float>::infinity();
                    }
                    else if(stored_bdd.high == bdd_storage::bdd_node::terminal_1)
                    {
                        assert(bdd_branch_nodes_[bdd_branch_index].high_cost[bdd_branch_slot] == 0.0);
                        bdd_branch_nodes_[bdd_branch_index].offset_high[bdd_branch_slot] = bdd_branch_node_vec<N>::terminal_1_offset;;
                    }
                    else
                    {
                        assert(bdd_branch_nodes_[bdd_branch_index].high_cost[bdd_branch_slot] == 0.0);
                        //assert(stored_bdd_index_to_bdd_offset.count(stored_bdd.high) > 0);
                        //const auto [high_ptr, high_intra_vec_index] = stored_bdd_index_to_bdd_offset.find(stored_bdd.high)->second;
                        const auto [high_ptr, high_intra_vec_index] = stored_bdd_index_to_bdd_offset[stored_bdd.high];
                        bdd_branch_nodes_[bdd_branch_index].offset_high[bdd_branch_slot] = bdd_branch_nodes_[bdd_branch_index].synthesize_address(high_ptr, high_intra_vec_index);
                    }

                    assert(bdd_index <= std::numeric_limits<uint32_t>::max()); // TODO: write alternative mechanism for this case
                    bdd_branch_nodes_[bdd_branch_index].bdd_index[bdd_branch_slot] = bdd_index;

                    //stored_bdd_index_to_bdd_offset.insert({stored_bdd_node_index, bdd_index_to_offset{&bdd_branch_nodes_[bdd_branch_index], bdd_branch_slot}});
                    stored_bdd_index_to_bdd_offset[stored_bdd_node_index] = bdd_index_to_offset{&bdd_branch_nodes_[bdd_branch_index], bdd_branch_slot};
                }
            }

            assert(first_bdd_node_indices_.size() == bdd_storage_.nr_bdds()); // only holds for non-split BDD storage

            nr_bdds_.clear();
            nr_bdds_.reserve(nr_variables());
            // TODO: reduce bdd indices in bdd_branch_node_vec to contiguous indices
            // TODO: replace by vector?
            tsl::robin_map<uint32_t, uint32_t> bdd_index_redux;
            for(size_t i=0; i<bdd_storage_.nr_variables(); ++i)
            {
                bdd_index_redux.clear();
                for(size_t vec_idx=bdd_branch_node_offsets_[i]; vec_idx<bdd_branch_node_offsets_[i+1]; ++vec_idx)
                {
                    auto& bdd_vec = bdd_branch_nodes_[vec_idx]; 
                    for(size_t slot=0; slot<N; ++slot)
                    {
                        const uint32_t bdd_index = bdd_vec.bdd_index[slot];
                        if(bdd_index != bdd_branch_node_vec<N>::inactive_bdd_index && bdd_index_redux.count(bdd_index) == 0)
                            bdd_index_redux.insert({bdd_index, bdd_index_redux.size()}); 
                    }
                }
                for(size_t vec_idx=bdd_branch_node_offsets_[i]; vec_idx<bdd_branch_node_offsets_[i+1]; ++vec_idx)
                {
                    auto& bdd_vec = bdd_branch_nodes_[vec_idx]; 
                    for(size_t slot=0; slot<N; ++slot)
                        if(bdd_vec.bdd_index[slot] != bdd_branch_node_vec<N>::inactive_bdd_index)
                            bdd_vec.bdd_index[slot] = bdd_index_redux.find(bdd_vec.bdd_index[slot])->second;
                }
                nr_bdds_.push_back(bdd_index_redux.size());
            }
        }

    template<size_t N>
        void bdd_mma_base_vec<N>::forward_step(const size_t var)
        {
            assert(var < nr_variables());

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].prepare_forward_step();

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].forward_step();
        }

    // TODO: count backwards in loop?
    template<size_t N>
        void bdd_mma_base_vec<N>::backward_step(const size_t var)
        {
            assert(var < nr_variables());

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].backward_step();
        }

    template<size_t N>
        float bdd_mma_base_vec<N>::average_marginals(float* marginals, const size_t nr_marginals)
        {
            float avg_margs = 0.0;
            for(size_t i=0; i<nr_marginals; ++i)
            {
                assert(std::isfinite(marginals[i]));
                avg_margs += marginals[i];
            }
            avg_margs /= float(nr_marginals);
            assert(std::isfinite(avg_margs));
            return avg_margs;
        }

    template<size_t N>
        void bdd_mma_base_vec<N>::min_marginal_averaging_step_forward(const size_t var)
        {
            // TODO: pad to four so that SIMD instructions can be applied?
            float min_marginals0[nr_bdds(var)];
            std::fill(min_marginals0, min_marginals0 + nr_bdds(var), std::numeric_limits<float>::infinity());
            float min_marginals1[nr_bdds(var)];
            std::fill(min_marginals1, min_marginals1 + nr_bdds(var), std::numeric_limits<float>::infinity());

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].min_marginal(min_marginals0, min_marginals1);

            std::array<float,2> avg_marginals = {average_marginals(min_marginals0, nr_bdds(var)), average_marginals(min_marginals1, nr_bdds(var))};

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].set_marginal(min_marginals0, min_marginals1, avg_marginals);

            forward_step(var);
        }

    template<size_t N>
        void bdd_mma_base_vec<N>::min_marginal_averaging_step_backward(const size_t var)
        {
            // TODO: pad to four so that SIMD instructions can be applied?
            float min_marginals0[nr_bdds(var)];
            std::fill(min_marginals0, min_marginals0 + nr_bdds(var), std::numeric_limits<float>::infinity());
            float min_marginals1[nr_bdds(var)];
            std::fill(min_marginals1, min_marginals1 + nr_bdds(var), std::numeric_limits<float>::infinity());

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].min_marginal(min_marginals0, min_marginals1);

            std::array<float,2> avg_marginals = {average_marginals(min_marginals0, nr_bdds(var)), average_marginals(min_marginals1, nr_bdds(var))};

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].set_marginal(min_marginals0, min_marginals1, avg_marginals);

            backward_step(var);
        }

    template<size_t N>
        void bdd_mma_base_vec<N>::min_marginal_averaging_forward()
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            for(size_t i=0; i<nr_variables(); ++i)
                min_marginal_averaging_step_forward(i);
        }

    template<size_t N>
        void bdd_mma_base_vec<N>::min_marginal_averaging_backward()
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            for(std::ptrdiff_t i=nr_variables()-1; i>=0; --i)
                min_marginal_averaging_step_backward(i);
        }

    template<size_t N>
        void bdd_mma_base_vec<N>::solve(const size_t max_iter)
        {
            std::cout << "initial lower bound = " << lower_bound() << "\n";
            for(size_t iter=0; iter<max_iter; ++iter)
            {
                iteration();
                std::cout << "iteration " << iter << ", lower bound = " << lower_bound() << "\n";
            } 
        }

    template<size_t N>
        void bdd_mma_base_vec<N>::iteration()
        {
            min_marginal_averaging_forward();
            min_marginal_averaging_backward();
            compute_lower_bound();
        }

    template<size_t N>
        void bdd_mma_base_vec<N>::backward_run()
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            for(std::ptrdiff_t i=bdd_branch_nodes_.size()-1; i>=0; --i)
                bdd_branch_nodes_[i].backward_step();
            compute_lower_bound();
        }


    template<size_t N>
        void bdd_mma_base_vec<N>::compute_lower_bound()
        {
            double lb = 0.0;
            for(const auto [bdd_node_index, bdd_node_slot] : first_bdd_node_indices_)
            {
                assert(std::isfinite(bdd_branch_nodes_[bdd_node_index].m[bdd_node_slot])); 
                lb += bdd_branch_nodes_[bdd_node_index].m[bdd_node_slot]; // only works if BDDs have one root node (not necessarily so for split BDDs.
            }

            lower_bound_ = lb;
        } 

    template<size_t N>
        void bdd_mma_base_vec<N>::set_cost(const double c, const size_t var)
        {
            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                for(size_t slot=0; slot<N; ++slot)
                    bdd_branch_nodes_[i].high_cost[slot] += c / float(nr_bdds(var));
        }
}
