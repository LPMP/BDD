#pragma once

// TODO: rename file and classes
// TODO: allow float and double via templates
// TODO: templatize to allow for different bdd branch nodes

#include <tsl/robin_map.h>
#include <vector>
#include <array>
#include <cstddef>
#include "bdd_storage.h"
#include "two_dimensional_variable_array.hxx"
#include "time_measure_util.h"
#include "kahan_summation.hxx"
#include "bdd_filtration.hxx"
#include "bdd_manager/bdd.h"
#include <iostream>

namespace LPMP {

    // TODO: add const to appropriate functions
    class bdd_branch_node_vec {
        public:
            // offsets are added to the address of the current bdd_branch_node_vec. The compute address points to the bdd_branch_node_vec
            uint32_t offset_low = 0;
            uint32_t offset_high = 0;
            float m = std::numeric_limits<float>::infinity();
            float low_cost = 0.0;
            float high_cost = 0.0;

            constexpr static uint32_t inactive_bdd_index = std::numeric_limits<uint32_t>::max();
            uint32_t bdd_index = inactive_bdd_index;

            void prepare_forward_step();
            void forward_step();
            void backward_step();

            std::array<float,2> min_marginals();
            void min_marginal(std::array<float,2>* reduced_min_marginals);
            void set_marginal(std::array<float,2>* min_marginals, const std::array<float,2> avg_marginals);

            constexpr static uint32_t terminal_0_offset = std::numeric_limits<uint32_t>::max();
            constexpr static uint32_t terminal_1_offset = std::numeric_limits<uint32_t>::max()-1;

            bdd_branch_node_vec* address(uint32_t offset);
            uint32_t synthesize_address(bdd_branch_node_vec* node);

            ~bdd_branch_node_vec()
            {
                static_assert(sizeof(float) == 4, "float must be quadword");
                static_assert(sizeof(uint32_t) == 4, "uint32_t must be quadword");
            }
    };

    inline bdd_branch_node_vec* bdd_branch_node_vec::address(uint32_t offset)
    {
        assert(offset != terminal_0_offset && offset != terminal_1_offset);
        return this + offset;
    }

    inline uint32_t bdd_branch_node_vec::synthesize_address(bdd_branch_node_vec* node)
    {
        assert(this < node);
        assert(std::distance(this, node) < std::numeric_limits<uint32_t>::max());
        assert(std::distance(this, node) > 0);
        return std::distance(this, node);
    }

    inline void bdd_branch_node_vec::backward_step()
    {
        if(offset_low == terminal_0_offset)
            assert(low_cost == std::numeric_limits<float>::infinity());

        if(offset_low == terminal_0_offset || offset_low == terminal_1_offset)
        {
            m = low_cost;
        }
        else
        {
            const auto low_branch_node = address(offset_low);
            m = low_branch_node->m + low_cost;
        }

        if(offset_high == terminal_0_offset)
            assert(high_cost == std::numeric_limits<float>::infinity());

        if(offset_high == terminal_0_offset || offset_high == terminal_1_offset)
        {
            m = std::min(m, high_cost);
        }
        else
        {
            const auto high_branch_node = address(offset_high);
            m = std::min(m, high_branch_node->m + high_cost);
        }

        assert(std::isfinite(m));
    }

    inline void bdd_branch_node_vec::prepare_forward_step()
    {
        if(offset_low != terminal_0_offset && offset_low != terminal_1_offset)
        {
            const auto low_branch_node = address(offset_low);
            low_branch_node->m = std::numeric_limits<float>::infinity(); 
        }

        if(offset_high != terminal_0_offset && offset_high != terminal_1_offset)
        {
            const auto high_branch_node = address(offset_high);
            high_branch_node->m = std::numeric_limits<float>::infinity();
        }
    }

    inline void bdd_branch_node_vec::forward_step()
    {
        if(offset_low != terminal_0_offset && offset_low != terminal_1_offset)
        {
            const auto low_branch_node = address(offset_low);
            low_branch_node->m = std::min(low_branch_node->m, m + low_cost);
        }

        if(offset_high != terminal_0_offset && offset_high != terminal_1_offset)
        {
            const auto high_branch_node = address(offset_high);
            high_branch_node->m = std::min(high_branch_node->m, m + high_cost);
        }
    }

    inline std::array<float,2> bdd_branch_node_vec::min_marginals()
    {
        std::array<float,2> mm;
        if(offset_low == terminal_0_offset || offset_low == terminal_1_offset)
        {
            mm[0] = m + low_cost;
        }
        else
        {
            const auto low_branch_node = address(offset_low);
            mm[0] = m + low_cost + low_branch_node->m;
        }

        if(offset_high == terminal_0_offset || offset_high == terminal_1_offset)
        {
            mm[1] = m + high_cost;
        }
        else
        {
            const auto high_branch_node = address(offset_high);
            mm[1] = m + high_cost + high_branch_node->m;
        }

        assert(std::isfinite(std::min(mm[0],mm[1])));
        return mm;
    }

    inline void bdd_branch_node_vec::min_marginal(std::array<float,2>* reduced_min_marginals)
    {
        // TODO: use above min marginal
        const auto mm = min_marginals();
        reduced_min_marginals[bdd_index][0] = std::min(mm[0], reduced_min_marginals[bdd_index][0]);
        reduced_min_marginals[bdd_index][1] = std::min(mm[1], reduced_min_marginals[bdd_index][1]);
        return;
        if(offset_low == terminal_0_offset || offset_low == terminal_1_offset)
        {
            reduced_min_marginals[bdd_index][0] = std::min(m + low_cost, reduced_min_marginals[bdd_index][0]);
        }
        else
        {
            const auto low_branch_node = address(offset_low);
            reduced_min_marginals[bdd_index][0] = std::min(m + low_cost + low_branch_node->m, reduced_min_marginals[bdd_index][0]);
        }

        if(offset_high == terminal_0_offset || offset_high == terminal_1_offset)
        {
            reduced_min_marginals[bdd_index][1] = std::min(m + high_cost, reduced_min_marginals[bdd_index][1]);
        }
        else
        {
            const auto high_branch_node = address(offset_high);
            reduced_min_marginals[bdd_index][1] = std::min(m + high_cost + high_branch_node->m, reduced_min_marginals[bdd_index][1]);
        }
    }

    inline void bdd_branch_node_vec::set_marginal(std::array<float,2>* reduced_min_marginals, const std::array<float,2> avg_marginals)
    {
        assert(std::isfinite(avg_marginals[0]));
        assert(std::isfinite(avg_marginals[1]));
        assert(std::isfinite(reduced_min_marginals[bdd_index][0]));
        low_cost += -reduced_min_marginals[bdd_index][0] + avg_marginals[0];
        assert(std::isfinite(reduced_min_marginals[bdd_index][1]));
        high_cost += -reduced_min_marginals[bdd_index][1] + avg_marginals[1]; 
    }

    // bdds are stored in variable groups. Each variable group is a set of variables that can be processed in parallel.
    class bdd_mma_base_vec {
        public:
            bdd_mma_base_vec() {}
            bdd_mma_base_vec(const bdd_storage& bdd_storage_) { add_bdds(bdd_storage_); } // { init(bdd_storage_); }
            void init(const bdd_storage& bdd_storage_);
            size_t nr_variables() const;
            size_t nr_variable_groups() const;
            size_t nr_bdd_vectors(const size_t var_group) const;
            size_t nr_bdds(const size_t var) const { assert(var < nr_variables()); return nr_bdds_[var]; }
            size_t nr_bdd_nodes() const { return bdd_branch_nodes_.size(); }
            size_t nr_bdd_nodes(const size_t v) const { assert(v < nr_variables()); return bdd_branch_node_offsets_[v+1] - bdd_branch_node_offsets_[v]; }
            size_t nr_bdds() const { return first_bdd_node_indices_.size(); }

            void forward_step(const size_t var_group);
            void min_marginal_averaging_forward();
            void min_marginal_averaging_step_forward(const size_t var_group);

            void backward_step(const size_t var_group);
            void min_marginal_averaging_backward();
            void min_marginal_averaging_step_backward(const size_t var_group);

            std::array<float,2> average_marginals(std::array<float,2>* marginals, const size_t nr_marginals);

            void iteration();
            void backward_run();
            void forward_run();

            void compute_lower_bound(); 
            void compute_lower_bound_after_forward_pass(); 
            void compute_lower_bound_after_backward_pass(); 

            std::vector<double> total_min_marginals();
            void solve(const size_t max_iter, const double tolerance, const double time_limit); 
            double lower_bound() const { return lower_bound_; }
            void set_cost(const double c, const size_t var);

            // get variable costs from bdd
            std::vector<float> get_costs(const size_t bdd_nr);
            // add costs from cost iterator to costs of bdd. Assume that variables given are subset of variables of bdd
            template<typename COST_ITERATOR, typename VARIABLE_ITERATOR>
                void update_costs(const size_t bdd_nr,
                        COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
                        VARIABLE_ITERATOR variable_begin, VARIABLE_ITERATOR variable_end);

            template<typename ITERATOR>
                void update_arc_costs(const size_t first_node, ITERATOR begin, ITERATOR end);
            void transfer_cost(const size_t from_bdd_nr, const size_t to_bdd_nr);
            void get_arc_marginals(const size_t first_node, const size_t last_node, std::vector<double>& arc_marginals);

            std::array<size_t,2> bdd_branch_node_offset(const size_t var, const size_t bdd_index) const;
            size_t variable(const size_t bdd_offset) const;

        protected:
            std::vector<bdd_branch_node_vec> bdd_branch_nodes_;
            std::vector<size_t> bdd_branch_node_offsets_; // offsets into where bdd branch nodes belonging to a variable start 
            std::vector<size_t> bdd_branch_node_group_offsets_; // offsets into where bdd branch nodes belonging to a variable group start 
            std::vector<size_t> nr_bdds_; // nr bdds per variable
            two_dim_variable_array<size_t> first_bdd_node_indices_;  // used for computing lower bound
            two_dim_variable_array<size_t> last_bdd_node_indices_;  // used for computing lower bound
            double lower_bound_ = -std::numeric_limits<double>::infinity();

            enum class message_passing_state {
                after_forward_pass,
                after_backward_pass,
                none 
            } message_passing_state_ = message_passing_state::none;

        private:
            std::vector<size_t> compute_bdd_branch_instruction_variables() const;
            void tighten_bdd(const float eps);
            two_dim_variable_array<size_t> tighten_bdd_groups(const std::vector<char>& tighten_variables);
            mutable std::vector<size_t> bdd_branch_instruction_variables_;

        public: // TODO: change to private again
            template<typename LAMBDA>
                void visit_nodes(const size_t bdd_nr, LAMBDA&& f);

            // for BDD tightening
        public:
            std::vector<float> min_marginal_differences(const float eps);
            // min marginals for each variable and each bdd
            two_dim_variable_array<std::array<float,2>> min_marginals();
            // export BDDs that cover the given variables
            // TODO: unify with init?
            std::vector<size_t> add_bdds(const bdd_storage& stor);
            template<typename BDD_NR_ITERATOR>
                std::vector<size_t> add_bdds(BDD::bdd_collection& bdd_col, BDD_NR_ITERATOR bdd_nrs_begin, BDD_NR_ITERATOR bdd_nrs_end);

            std::vector<size_t> variables(const size_t bdd_idx);
            std::vector<bdd_branch_instruction<float>> export_bdd(const size_t bdd_idx);
            size_t export_bdd(BDD::bdd_collection& bdd_col, const size_t bdd_idx);
            std::tuple<BDD::node_ref, std::vector<size_t>> export_bdd(BDD::bdd_mgr& bdd_mgr, const size_t bdd_idx);

    };


    inline size_t bdd_mma_base_vec::nr_variables() const
    {
        return nr_bdds_.size(); 
    }

    inline size_t bdd_mma_base_vec::nr_variable_groups() const
    {
        throw std::runtime_error("not usable");
        return bdd_branch_node_group_offsets_.size()-1; 
    } 

    inline size_t bdd_mma_base_vec::nr_bdd_vectors(const size_t var) const
    {
        assert(var < nr_variables());
        return bdd_branch_node_offsets_[var+1] - bdd_branch_node_offsets_[var];
    } 

    inline void bdd_mma_base_vec::init(const bdd_storage& bdd_storage_)
    {
        bdd_branch_nodes_.clear();
        bdd_branch_node_offsets_.clear();
        nr_bdds_.clear();
        std::vector<size_t> cur_first_bdd_node_indices;
        std::vector<size_t> cur_last_bdd_node_indices;
        first_bdd_node_indices_.clear();
        last_bdd_node_indices_.clear();
        lower_bound_ = -std::numeric_limits<double>::infinity();

        /*
           const auto variable_groups = compute_variable_groups(bdd_storage_);
           std::cout << "# variable groups: = " << variable_groups.size() << "\n";
           std::cout << "# variable : = " << bdd_storage_.nr_variables() << "\n";
           for(size_t i=0; i<variable_groups.size(); ++i)
           {
           for(const size_t v : variable_groups[i])
           std::cout << v << ", ";
           std::cout << "\n";
           }
           std::cout << "\n";
           */

        // count bdd branch nodes per variable
        std::vector<size_t> bdd_branch_nodes_per_var(bdd_storage_.nr_variables(), 0);
        for(const auto& stored_bdd_node : bdd_storage_.bdd_nodes())
            ++bdd_branch_nodes_per_var[stored_bdd_node.variable]; 

        bdd_branch_node_offsets_.reserve(bdd_storage_.nr_variables()+1);
        bdd_branch_node_offsets_.push_back(0);
        for(const size_t i : bdd_branch_nodes_per_var)
            bdd_branch_node_offsets_.push_back(bdd_branch_node_offsets_.back() + i);

        bdd_branch_nodes_.resize(bdd_branch_node_offsets_.back());
        std::vector<size_t> bdd_branch_nodes_counter(bdd_storage_.nr_variables(), 0);

        std::vector<bdd_branch_node_vec*> stored_bdd_index_to_bdd_offset(bdd_storage_.bdd_nodes().size(), nullptr);

        for(size_t bdd_index=0; bdd_index<bdd_storage_.nr_bdds(); ++bdd_index)
        {
            cur_first_bdd_node_indices.clear();
            cur_last_bdd_node_indices.clear();
            const size_t first_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index]; 
            const size_t last_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index+1];
            const size_t first_var = bdd_storage_.bdd_nodes()[last_stored_bdd_node-1].variable;
            const size_t last_var = bdd_storage_.bdd_nodes()[first_stored_bdd_node].variable;
            for(size_t i=first_stored_bdd_node; i<last_stored_bdd_node; ++i)
            {
                assert(first_var <= bdd_storage_.bdd_nodes()[i].variable);
                assert(last_var >= bdd_storage_.bdd_nodes()[i].variable);
            }

            for(size_t stored_bdd_node_index=first_stored_bdd_node; stored_bdd_node_index<last_stored_bdd_node; ++stored_bdd_node_index)
            {
                const auto& stored_bdd = bdd_storage_.bdd_nodes()[stored_bdd_node_index];
                const size_t v = stored_bdd.variable;
                const size_t bdd_branch_index = bdd_branch_node_offsets_[v] + bdd_branch_nodes_counter[v];
                ++bdd_branch_nodes_counter[v];

                if(v == first_var)
                    cur_first_bdd_node_indices.push_back(bdd_branch_index);
                if(v == last_var)
                    cur_last_bdd_node_indices.push_back(bdd_branch_index);

                if(stored_bdd.low == bdd_storage::bdd_node::terminal_0)
                {
                    bdd_branch_nodes_[bdd_branch_index].offset_low = bdd_branch_node_vec::terminal_0_offset;
                    bdd_branch_nodes_[bdd_branch_index].low_cost = std::numeric_limits<float>::infinity();
                }
                else if(stored_bdd.low == bdd_storage::bdd_node::terminal_1)
                {
                    assert(bdd_branch_nodes_[bdd_branch_index].low_cost == 0.0);
                    bdd_branch_nodes_[bdd_branch_index].offset_low = bdd_branch_node_vec::terminal_1_offset;
                }
                else
                {
                    assert(bdd_branch_nodes_[bdd_branch_index].low_cost == 0.0);
                    assert(stored_bdd_index_to_bdd_offset[stored_bdd.low] != nullptr);
                    bdd_branch_node_vec* low_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.low];
                    bdd_branch_nodes_[bdd_branch_index].offset_low = bdd_branch_nodes_[bdd_branch_index].synthesize_address(low_ptr);
                }

                if(stored_bdd.high == bdd_storage::bdd_node::terminal_0)
                {
                    bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_node_vec::terminal_0_offset;
                    bdd_branch_nodes_[bdd_branch_index].high_cost = std::numeric_limits<float>::infinity();
                }
                else if(stored_bdd.high == bdd_storage::bdd_node::terminal_1)
                {
                    assert(bdd_branch_nodes_[bdd_branch_index].high_cost == 0.0);
                    bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_node_vec::terminal_1_offset;
                }
                else
                {
                    assert(bdd_branch_nodes_[bdd_branch_index].high_cost == 0.0);
                    assert(stored_bdd_index_to_bdd_offset[stored_bdd.high] != nullptr);
                    bdd_branch_node_vec* high_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.high];
                    bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_nodes_[bdd_branch_index].synthesize_address(high_ptr);
                }

                assert(bdd_index <= std::numeric_limits<uint32_t>::max()); // TODO: write alternative mechanism for this case
                bdd_branch_nodes_[bdd_branch_index].bdd_index = bdd_index;
                assert(stored_bdd_index_to_bdd_offset[stored_bdd_node_index] == nullptr);

                stored_bdd_index_to_bdd_offset[stored_bdd_node_index] = &bdd_branch_nodes_[bdd_branch_index];
            }

            assert(cur_first_bdd_node_indices.size() > 0);
            first_bdd_node_indices_.push_back(cur_first_bdd_node_indices.begin(), cur_first_bdd_node_indices.end()); 
            assert(cur_last_bdd_node_indices.size() > 0);
            last_bdd_node_indices_.push_back(cur_last_bdd_node_indices.begin(), cur_last_bdd_node_indices.end()); 
        }

        assert(first_bdd_node_indices_.size() == bdd_storage_.nr_bdds());
        assert(last_bdd_node_indices_.size() == bdd_storage_.nr_bdds());

        nr_bdds_.clear();
        nr_bdds_.reserve(nr_variables());
        // TODO: replace by vector?
        tsl::robin_map<uint32_t, uint32_t> bdd_index_redux;
        for(size_t i=0; i<bdd_storage_.nr_variables(); ++i)
        {
            bdd_index_redux.clear();
            for(size_t vec_idx=bdd_branch_node_offsets_[i]; vec_idx<bdd_branch_node_offsets_[i+1]; ++vec_idx)
            {
                auto& bdd_vec = bdd_branch_nodes_[vec_idx]; 
                const uint32_t bdd_index = bdd_vec.bdd_index;
                assert(bdd_index != bdd_branch_node_vec::inactive_bdd_index);
                if(bdd_index_redux.count(bdd_index) == 0)
                    bdd_index_redux.insert({bdd_index, bdd_index_redux.size()}); 
            }
            for(size_t vec_idx=bdd_branch_node_offsets_[i]; vec_idx<bdd_branch_node_offsets_[i+1]; ++vec_idx)
            {
                auto& bdd_vec = bdd_branch_nodes_[vec_idx]; 
                bdd_vec.bdd_index = bdd_index_redux.find(bdd_vec.bdd_index)->second;
            }
            nr_bdds_.push_back(bdd_index_redux.size());
        }

        std::cout << "nr bdd nodes = " << bdd_branch_nodes_.size() << "\n";
    }

    inline void bdd_mma_base_vec::forward_step(const size_t var)
    {
        assert(var < nr_variables());

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].prepare_forward_step();

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].forward_step();
    }


    inline void bdd_mma_base_vec::backward_step(const size_t var)
    {
        assert(var < nr_variables());

        // TODO: count backwards in loop?
        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].backward_step();
    } 

    inline std::array<float,2> bdd_mma_base_vec::average_marginals(std::array<float,2>* marginals, const size_t nr_marginals)
    {
        std::array<float,2> avg_margs = {0.0,0.0};
        for(size_t i=0; i<nr_marginals; ++i)
        {
            assert(std::isfinite(marginals[i][0]));
            assert(std::isfinite(marginals[i][1]));
            avg_margs[0] += marginals[i][0];
            avg_margs[1] += marginals[i][1];
        }
        avg_margs[0] /= float(nr_marginals);
        avg_margs[1] /= float(nr_marginals);
        assert(std::isfinite(avg_margs[0]));
        assert(std::isfinite(avg_margs[1]));
        return avg_margs;
    } 

    inline void bdd_mma_base_vec::min_marginal_averaging_step_forward(const size_t var)
    {
        // TODO: pad to four so that SIMD instructions can be applied?
        const size_t _nr_bdds = nr_bdds(var);
        if(_nr_bdds == 0)
            return;

        std::array<float,2> min_marginals[_nr_bdds];
        std::fill(min_marginals, min_marginals + _nr_bdds, std::array<float,2>{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()});

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].min_marginal(min_marginals);

        std::array<float,2> avg_marginals = average_marginals(min_marginals, _nr_bdds);

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].set_marginal(min_marginals, avg_marginals);

        forward_step(var);
    }

    inline void bdd_mma_base_vec::min_marginal_averaging_step_backward(const size_t var)
    {
        // TODO: pad to four so that SIMD instructions can be applied?
        const size_t _nr_bdds = nr_bdds(var);
        if(_nr_bdds == 0)
            return;
        std::array<float,2> min_marginals[_nr_bdds];
        std::fill(min_marginals, min_marginals + _nr_bdds, std::array<float,2>{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()});

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].min_marginal(min_marginals);

        std::array<float,2> avg_marginals = average_marginals(min_marginals, _nr_bdds);

        //std::cout << "backward step for var " << var << ", offset = " << bdd_branch_node_offsets_[var] << ", #nodes = " << bdd_branch_nodes_.size() << "\n";
        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
        {
            bdd_branch_nodes_[i].set_marginal(min_marginals, avg_marginals);
            bdd_branch_nodes_[i].backward_step();
        }
    }

    inline void bdd_mma_base_vec::min_marginal_averaging_forward()
    {
        if(message_passing_state_ != message_passing_state::after_backward_pass)
            backward_run();
        message_passing_state_ = message_passing_state::none;
        //MEASURE_FUNCTION_EXECUTION_TIME;
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        for(size_t i=0; i<nr_variables(); ++i)
            min_marginal_averaging_step_forward(i);
        message_passing_state_ = message_passing_state::after_forward_pass;
    }

    inline void bdd_mma_base_vec::min_marginal_averaging_backward()
    {
        if(message_passing_state_ != message_passing_state::after_forward_pass)
            forward_run();
        message_passing_state_ = message_passing_state::none;
        //MEASURE_FUNCTION_EXECUTION_TIME;
        for(std::ptrdiff_t i=nr_variables()-1; i>=0; --i)
            min_marginal_averaging_step_backward(i);
        message_passing_state_ = message_passing_state::after_backward_pass;
    }

    inline void bdd_mma_base_vec::solve(const size_t max_iter, const double tolerance, const double time_limit)
    {
        const auto start_time = std::chrono::steady_clock::now();
        double lb_prev = this->lower_bound();
        double lb_post = lb_prev;
        std::cout << "initial lower bound = " << lb_prev;
        auto time = std::chrono::steady_clock::now();
        std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
        std::cout << "\n";
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
        const auto mmd = min_marginal_differences(0.001);
        std::vector<char> tighten_variables(nr_variables(), false);
        for(size_t i=0; i<mmd.size(); ++i)
            if(std::abs(mmd[i]) < 1e-3)
                tighten_variables[i] = true;
        tighten_bdd_groups(tighten_variables);
    }

    inline void bdd_mma_base_vec::iteration()
    {
        min_marginal_averaging_forward();
        compute_lower_bound();
        min_marginal_averaging_backward();
        compute_lower_bound();
    }

    inline void bdd_mma_base_vec::backward_run()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        // TODO: if we already have done a backward_run, we do not need to do it again. Check state!
        message_passing_state_ = message_passing_state::none;
        for(std::ptrdiff_t i=bdd_branch_nodes_.size()-1; i>=0; --i)
            bdd_branch_nodes_[i].backward_step();
        message_passing_state_ = message_passing_state::after_backward_pass;
        compute_lower_bound();
    }

    inline void bdd_mma_base_vec::forward_run()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        // TODO: if we already have done a forward_run, we do not need to do it again. Check state!
        message_passing_state_ = message_passing_state::none;
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        for(size_t i=0; i<nr_variables(); ++i)
            forward_step(i);
        message_passing_state_ = message_passing_state::after_forward_pass;
        compute_lower_bound();
    }

    inline void bdd_mma_base_vec::compute_lower_bound()
    {
        if(message_passing_state_ == message_passing_state::after_forward_pass)
            compute_lower_bound_after_forward_pass();
        else if(message_passing_state_ == message_passing_state::after_backward_pass)
            compute_lower_bound_after_backward_pass();
        else
            throw std::runtime_error("Cannot compute valid lower bound");
    }

    inline void bdd_mma_base_vec::compute_lower_bound_after_backward_pass()
    {
        tkahan<double> lb;
        //double lb = 0.0;
        for(size_t i=0; i<first_bdd_node_indices_.size(); ++i)
        {
            float bdd_lb = std::numeric_limits<float>::infinity();
            for(size_t j=0; j<first_bdd_node_indices_.size(i); ++j)
                bdd_lb = std::min(bdd_branch_nodes_[first_bdd_node_indices_(i,j)].m, bdd_lb);
            lb += bdd_lb;
        }

        assert(lb.value() >= lower_bound_ - 1e-8);
        lower_bound_ = lb.value();
    } 

    inline void bdd_mma_base_vec::compute_lower_bound_after_forward_pass()
    {
        tkahan<double> lb;
        //double lb = 0.0;
        for(size_t i=0; i<last_bdd_node_indices_.size(); ++i)
        {
            float bdd_lb = std::numeric_limits<float>::infinity();
            for(size_t j=0; j<last_bdd_node_indices_.size(i); ++j)
            {
                auto& bdd_node = bdd_branch_nodes_[last_bdd_node_indices_(i,j)];
                assert(bdd_node.offset_low == bdd_branch_node_vec::terminal_0_offset || bdd_node.offset_low == bdd_branch_node_vec::terminal_1_offset);
                assert(bdd_node.offset_high == bdd_branch_node_vec::terminal_0_offset || bdd_node.offset_high == bdd_branch_node_vec::terminal_1_offset);
                //bdd_lb = std::min({bdd_node.m + bdd_node.low_cost, bdd_node.m + bdd_node.high_cost, bdd_lb});
                const auto mm = bdd_node.min_marginals();
                bdd_lb = std::min({mm[0], mm[1], bdd_lb});
            }
            lb += bdd_lb;
        }

        assert(lb.value() >= lower_bound_ - 1e-8);
        lower_bound_ = lb.value();
    }

    inline void bdd_mma_base_vec::set_cost(const double c, const size_t var)
    {
        assert(nr_bdds(var) > 0);
        assert(std::isfinite(c));

        lower_bound_ = -std::numeric_limits<double>::infinity();
        message_passing_state_ = message_passing_state::none;

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].high_cost += c / float(nr_bdds(var));
    }

    template<typename ITERATOR>
        void bdd_mma_base_vec::update_arc_costs(const size_t first_node, ITERATOR begin, ITERATOR end)
        {
            assert(std::distance(begin,end) % 2 == 0);
            assert(first_node + std::distance(begin,end)/2 <= bdd_branch_nodes_.size());
            size_t l=first_node;
            for(auto it=begin; it!=end; ++l)
            {
                assert(bdd_branch_nodes_[first_node].bdd_index == bdd_branch_nodes_[l].bdd_index);
                if(bdd_branch_nodes_[l].offset_low != bdd_branch_node_vec::terminal_0_offset)
                    bdd_branch_nodes_[l].low_cost += *it; 
                ++it;
                if(bdd_branch_nodes_[l].offset_high != bdd_branch_node_vec::terminal_0_offset)
                    bdd_branch_nodes_[l].high_cost += *it; 
                ++it;
            }
        }

    template<typename LAMBDA>
        void bdd_mma_base_vec::visit_nodes(const size_t bdd_nr, LAMBDA&& f)
        {
            assert(bdd_nr < nr_bdds());
            std::deque<size_t> dq;
            tsl::robin_set<size_t> visited;
            assert(first_bdd_node_indices_.size(bdd_nr) == 1);
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_nr); ++j)
                dq.push_back(first_bdd_node_indices_(bdd_nr, j));
            while(!dq.empty())
            {
                const size_t i = dq.front();
                dq.pop_front();
                if(visited.count(i) > 0)
                    continue;
                visited.insert(i);

                f(i);

                auto calculate_offset = [&](bdd_branch_node_vec& bdd, const size_t offset) {
                    assert(offset != bdd_branch_node_vec::terminal_0_offset && offset != bdd_branch_node_vec::terminal_1_offset);
                    assert(&bdd >= &bdd_branch_nodes_[0]);
                    assert(std::distance(&bdd_branch_nodes_[0], &bdd) < bdd_branch_nodes_.size());
                    const size_t translated_offset = std::distance(&bdd_branch_nodes_[0], bdd.address(offset)); 
                    assert(translated_offset < bdd_branch_nodes_.size());
                    return translated_offset;
                };

                auto& bdd = bdd_branch_nodes_[i];
                if(bdd.offset_low != bdd_branch_node_vec::terminal_0_offset && bdd.offset_low != bdd_branch_node_vec::terminal_1_offset)
                    dq.push_back(calculate_offset(bdd, bdd.offset_low));
                if(bdd.offset_high != bdd_branch_node_vec::terminal_0_offset && bdd.offset_high != bdd_branch_node_vec::terminal_1_offset)
                    dq.push_back(calculate_offset(bdd, bdd.offset_high));
            } 
        }

    inline std::vector<float> bdd_mma_base_vec::get_costs(const size_t bdd_nr)
    {
        std::vector<float> costs;

        size_t cur_var = std::numeric_limits<size_t>::max();
        float hi_cost = std::numeric_limits<float>::infinity();
        float lo_cost = std::numeric_limits<float>::infinity();
        visit_nodes(bdd_nr, [&](const size_t i) {
                bdd_branch_node_vec& bdd = bdd_branch_nodes_[i];
                if(cur_var != this->variable(i) && cur_var != std::numeric_limits<size_t>::max())
                {
                costs.push_back(hi_cost - lo_cost); 
                hi_cost = std::numeric_limits<float>::infinity();
                lo_cost = std::numeric_limits<float>::infinity();
                }
                hi_cost = std::min(hi_cost, bdd.high_cost);
                lo_cost = std::min(lo_cost, bdd.low_cost);
                cur_var = this->variable(i); 
                });
        costs.push_back(hi_cost - lo_cost);

        for(const float x : costs)
            assert(std::isfinite(x));
        assert(costs.size() == variables(bdd_nr).size());

        return costs; 
    }

    template<typename COST_ITERATOR, typename VARIABLE_ITERATOR>
        void bdd_mma_base_vec::update_costs(const size_t bdd_nr,
                COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
                VARIABLE_ITERATOR variable_begin, VARIABLE_ITERATOR variable_end)
        {
            assert(std::distance(cost_begin, cost_end) == std::distance(variable_begin, variable_end));
            assert(std::is_sorted(variable_begin, variable_end));

            lower_bound_ = -std::numeric_limits<double>::infinity();
            message_passing_state_ = message_passing_state::none;

            auto cost_it = cost_begin;
            auto var_it = variable_begin;

            visit_nodes(bdd_nr, [&](const size_t i) {
                    bdd_branch_node_vec& bdd = bdd_branch_nodes_[i];
                    if(variable(i) < *var_it)
                    return;
                    if(variable(i) > *var_it && var_it < variable_end)
                    {
                    ++var_it;
                    ++cost_it; 
                    }
                    assert(std::isfinite(*cost_it));
                    if(*var_it == variable(i))
                    bdd_branch_nodes_[i].high_cost += *cost_it;
                    });
        }

    inline void bdd_mma_base_vec::get_arc_marginals(const size_t first_node, const size_t last_node, std::vector<double>& arc_marginals)
    {
        arc_marginals.clear();
        assert(first_node < bdd_branch_nodes_.size());
        for(size_t l=first_node; l<last_node; ++l)
        {
            assert(bdd_branch_nodes_[first_node].bdd_index == bdd_branch_nodes_[l].bdd_index);
            const auto m = bdd_branch_nodes_[l].min_marginals();
            arc_marginals.push_back(m[0]);
            arc_marginals.push_back(m[1]);
        } 
    }

    inline std::array<size_t,2> bdd_mma_base_vec::bdd_branch_node_offset(const size_t var, const size_t bdd_index) const
    {
        assert(var < nr_variables());
        std::array<size_t,2> offsets;
        size_t i = bdd_branch_node_offsets_[var];
        for(; i<bdd_branch_node_offsets_[var+1]; ++i)
            if(bdd_branch_nodes_[i].bdd_index == bdd_index)
                break;
        offsets[0] = i;

        assert(i < bdd_branch_node_offsets_[var+1]);
        for(; i<bdd_branch_node_offsets_[var+1]; ++i)
            if(bdd_branch_nodes_[i].bdd_index != bdd_index)
                break;
        offsets[1] = i; 

        assert(offsets[0] < bdd_branch_node_offsets_[var+1]);
        assert(offsets[0] < offsets[1]);

        return offsets;
    }

    inline size_t bdd_mma_base_vec::variable(const size_t bdd_offset) const
    {
        assert(bdd_offset < bdd_branch_nodes_.size());
        if(bdd_branch_instruction_variables_.size() != bdd_branch_nodes_.size())
            bdd_branch_instruction_variables_ = compute_bdd_branch_instruction_variables();
        assert(bdd_branch_instruction_variables_.size() == bdd_branch_nodes_.size());
        return bdd_branch_instruction_variables_[bdd_offset]; 
    }

    inline std::vector<double> bdd_mma_base_vec::total_min_marginals()
    {
        std::cout << "compute total min marginals in mma_vec\n";
        if(message_passing_state_ != message_passing_state::after_backward_pass)
            this->backward_run();
        message_passing_state_ = message_passing_state::none;
        // prepare forward run
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        std::vector<double> total_min_marginals_vec;
        total_min_marginals_vec.reserve(nr_variables());

        std::cout << "after forward run\n";

        for(size_t var=0; var<this->nr_variables(); ++var)
        {
            const size_t _nr_bdds = nr_bdds(var);
            std::array<float,2> min_marginals[_nr_bdds];
            std::fill(min_marginals, min_marginals + _nr_bdds, std::array<float,2>{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()});

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].min_marginal(min_marginals);

            float total_min_marg = 0.0;
            for(size_t i=0; i<_nr_bdds; ++i)
            {
                assert(std::isfinite(min_marginals[i][0]));
                assert(std::isfinite(min_marginals[i][1]));
                total_min_marg += (min_marginals[i][1] - min_marginals[i][0]);
            }
            this->forward_step(var);

            total_min_marginals_vec.push_back(total_min_marg); 
        }
        std::cout << "return mms\n";

        message_passing_state_ = message_passing_state::after_forward_pass;
        return total_min_marginals_vec;
    }

    inline std::vector<size_t> bdd_mma_base_vec::compute_bdd_branch_instruction_variables() const
    {
        std::vector<size_t> vars;
        vars.reserve(bdd_branch_nodes_.size());
        for(size_t var=0; var<nr_variables(); ++var)
            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                vars.push_back(var);
        return vars;
    }

    inline void bdd_mma_base_vec::tighten_bdd(const float epsilon)
    {
        backward_run();
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        assert(epsilon >= 0.0);
        const std::vector<size_t> vars = compute_bdd_branch_instruction_variables();
        std::vector<char> visited(bdd_branch_nodes_.size(), false);
        std::vector<size_t> bdd_offset_map(bdd_branch_nodes_.size(), std::numeric_limits<size_t>::max()); // maps offsets from bdd_branch_nodes_ onto offsets used for storing a single bdd in a vector

        BDD::bdd_mgr bdd_mgr;
        BDD::bdd_collection bdd_collection;

        // Currently only works for single root BDDs
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            assert(first_bdd_node_indices_.size(bdd_index) == 1);

        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
        {
            // first compute optimal value of current BDD
            const float bdd_lb = [&]() {
                float lb = std::numeric_limits<float>::infinity();
                for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                {
                    const auto mm = bdd_branch_nodes_[first_bdd_node_indices_(bdd_index, j)].min_marginals(); 
                    lb = std::min({mm[0], mm[1], lb});
                }
                return lb; 
            }();
            std::cout << "bdd lb = " << bdd_lb << "\n";
            size_t prev_var = vars[first_bdd_node_indices_(bdd_index, 0)];

            // TODO: move out to top
            std::vector<bdd_node> cur_bdd;
            tsl::robin_map<size_t,size_t> var_to_index_map; // from original variables to consecutive index
            var_to_index_map.insert({prev_var, 0});
            std::vector<size_t> index_to_var_map = {prev_var};  // from consecutive indices to original variables
            std::deque<size_t> dq;
            std::vector<size_t> last_var_bdd_nodes;
            size_t nr_redirected_arcs = 0;
            assert(first_bdd_node_indices_.size(bdd_index) == 1);
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                dq.push_back(first_bdd_node_indices_(bdd_index, j));
            while(!dq.empty())
            {
                const size_t i = dq.front();
                dq.pop_front();
                if(visited[i] == true)
                    continue;
                const size_t var = vars[i];
                if(prev_var != var)
                {
                    // tmp: check if lower bound correct.
                    float check_lb = std::numeric_limits<float>::infinity();
                    for(const size_t i : last_var_bdd_nodes)
                    {
                        const auto mm = bdd_branch_nodes_[i].min_marginals();
                        check_lb = std::min({mm[0], mm[1], check_lb});
                    }
                    assert(std::abs(check_lb - bdd_lb) <= 1e-3);
                    // TODO: remove above again!
                    for(const size_t i : last_var_bdd_nodes)
                        bdd_branch_nodes_[i].prepare_forward_step();
                    for(const size_t i : last_var_bdd_nodes)
                        bdd_branch_nodes_[i].forward_step();
                    last_var_bdd_nodes.clear();
                    var_to_index_map.insert({var, var_to_index_map.size()});
                    index_to_var_map.push_back(var);
                    prev_var = var;
                }
                last_var_bdd_nodes.push_back(i);
                visited[i] = true;
                auto& bdd = bdd_branch_nodes_[i];
                const auto mm = bdd.min_marginals();
                bdd_node n;
                n.var = var;

                auto calculate_offset = [&](const float min_marg, const size_t offset) {
                    assert(min_marg >= bdd_lb - 1e-3);
                    // check if low resp. high arc min-marginal is within epsilon of lower bound
                    if(min_marg <= bdd_lb + epsilon || offset == bdd_branch_node_vec::terminal_0_offset) // leave arc as is
                    {
                        if(offset == bdd_branch_node_vec::terminal_0_offset)
                            return bdd_node::terminal_0;
                        else if(offset == bdd_branch_node_vec::terminal_1_offset)
                            return bdd_node::terminal_1;
                        else
                        {
                            const size_t translated_offset = std::distance(&bdd_branch_nodes_[0], bdd.address(offset)); 
                            dq.push_back(translated_offset);
                            return translated_offset;
                        }
                    }
                    else // reroute arc to true terminal if it is not a pointer to false terminal
                    {
                        if(offset == bdd_branch_node_vec::terminal_0_offset)
                        {
                            assert(false);
                            return bdd_node::terminal_0;
                        }
                        else
                        {
                            ++nr_redirected_arcs;
                            return bdd_node::terminal_1; 
                        }
                    }
                };

                n.low = calculate_offset(mm[0], bdd.offset_low);
                n.high = calculate_offset(mm[1], bdd.offset_high);
                cur_bdd.push_back(n);
                bdd_offset_map[i] = cur_bdd.size()-1; 
            }

            // map bdd offsets
            for(size_t i=0; i<cur_bdd.size(); ++i)
            {
                auto& n = cur_bdd[i];
                assert(var_to_index_map.count(n.var) > 0);
                n.var = var_to_index_map.find(n.var)->second;
                if(n.low != bdd_node::terminal_0 && n.low != bdd_node::terminal_1)
                {
                    assert(bdd_offset_map[n.low] != std::numeric_limits<size_t>::max());
                    assert(bdd_offset_map[n.low] > i);
                    n.low = bdd_offset_map[n.low];
                }
                if(n.high != bdd_node::terminal_0 && n.high != bdd_node::terminal_1)
                {
                    assert(bdd_offset_map[n.high] != std::numeric_limits<size_t>::max());
                    assert(bdd_offset_map[n.high] > i);
                    n.high = bdd_offset_map[n.high];
                } 
            }

            // add bdd to the bdd base. This reduces the BDD
            // TODO: this only works with single root BDDs
            assert(first_bdd_node_indices_.size(bdd_index) == 1);
            std::vector<BDD::node_ref> node_refs(cur_bdd.size());
            for(std::ptrdiff_t i=cur_bdd.size()-1; i>=0; --i)
            {
                const auto& n = cur_bdd[i];
                auto get_node_ref = [&](const size_t offset) -> BDD::node_ref {
                    if(offset == bdd_node::terminal_0)
                        return bdd_mgr.botsink();
                    if(offset == bdd_node::terminal_1)
                        return bdd_mgr.topsink();
                    assert(offset < node_refs.size() && offset > i);
                    return node_refs[offset];
                };
                BDD::node_ref low = get_node_ref(n.low);
                BDD::node_ref high = get_node_ref(n.high);
                BDD::node_ref bdd_var = bdd_mgr.projection(n.var);
                node_refs[i] = bdd_mgr.ite_rec(bdd_var, low, high);
            }

            std::cout << "Original bdd of size = " << cur_bdd.size() << ", # redirected arcs = " << nr_redirected_arcs << ", after reduction size = " << node_refs[0].nr_nodes() << "\n";

            if(node_refs[0].is_terminal())
            {
                assert(node_refs[0].is_topsink());
                continue;
            }
            // check number of solutions. If there is exactly one solution, forget about BDD (the relevant information should be present in other BDDs as well
            if(node_refs[0].exactly_one_solution())
            {
                std::cout << "reduced BDD has only one solution, discard\n";
                assert(false); // This never seems to happen!
                continue; 
            }

            const size_t bdd_nr = bdd_collection.add_bdd(node_refs[0]);
            bdd_collection.rebase(bdd_nr, index_to_var_map.begin(), index_to_var_map.end());
        }

        std::vector<size_t> bdd_indices(bdd_collection.nr_bdds());
        std::iota(bdd_indices.begin(), bdd_indices.end(), 0);
        std::cout << "nr bdds in tightening = " << bdd_indices.size() << "\n";
        const size_t new_bdd_nr = bdd_collection.bdd_and(bdd_indices.begin(), bdd_indices.end());
        std::cout << "new bdd size = " << bdd_collection.nr_bdd_nodes(new_bdd_nr) << "\n";
    }

    inline std::vector<float> bdd_mma_base_vec::min_marginal_differences(const float eps)
    {
        // go over all variables and see where min-marginal difference
        this->backward_run();
        // prepare forward run
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        std::vector<float> min_marg_diffs;
        min_marg_diffs.reserve(nr_variables());

        for(size_t var=0; var<this->nr_variables(); ++var)
        {
            const size_t _nr_bdds = nr_bdds(var);
            std::array<float,2> min_marginals[_nr_bdds];
            std::fill(min_marginals, min_marginals + _nr_bdds, std::array<float,2>{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()});
            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].min_marginal(min_marginals);

            float min_diff = 0.0; //std::numeric_limits<float>::infinity(); 
            bool negative = true;
            bool positive = true;
            for(size_t i=0; i<_nr_bdds; ++i)
            {
                min_diff += std::abs(min_marginals[i][1] - min_marginals[i][0]);
                assert(std::isfinite(min_marginals[i][0]));
                assert(std::isfinite(min_marginals[i][1]));
                if(min_marginals[i][1] - min_marginals[i][0] >= eps)
                {
                    negative = false; 
                }
                else if(min_marginals[i][1] - min_marginals[i][0] <= -eps)
                {
                    positive = false;
                }
                else
                {
                    negative = false;
                    positive = false; 
                }
            }
            assert(_nr_bdds == 0 || (!(positive == true && negative == true)));
            if(negative)
                min_marg_diffs.push_back(-min_diff);
            else if(positive)
                min_marg_diffs.push_back(min_diff);
            else
                min_marg_diffs.push_back(0.0);

            this->forward_step(var);
        }

        const size_t nr_positive_min_marg_differences = std::count_if(min_marg_diffs.begin(), min_marg_diffs.end(), [&](const float x) { return x > eps; });
        const size_t nr_negative_min_marg_differences = std::count_if(min_marg_diffs.begin(), min_marg_diffs.end(), [&](const float x) { return x < -eps; });
        const size_t nr_zero_min_marg_differences = min_marg_diffs.size() - nr_positive_min_marg_differences - nr_negative_min_marg_differences;
        std::cout << "%zero min margs = " << 100.0 * double(nr_zero_min_marg_differences) / double(min_marg_diffs.size()) << "\n";
        std::cout << "#zero min margs = " << nr_zero_min_marg_differences << "\n";
        std::cout << "%positive min margs = " << 100.0 * double(nr_positive_min_marg_differences) / double(min_marg_diffs.size()) << "\n";
        std::cout << "#positive min margs = " << nr_positive_min_marg_differences << "\n";
        std::cout << "%negative min margs = " << 100.0 * double(nr_negative_min_marg_differences) / double(min_marg_diffs.size()) << "\n";
        std::cout << "#negative min margs = " << nr_negative_min_marg_differences << "\n";

        //for(float x : min_marg_diffs)
        //    std::cout << x << " ";
        //std::cout << "\n";
        
        return min_marg_diffs;
    }

    inline two_dim_variable_array<std::array<float,2>> bdd_mma_base_vec::min_marginals()
    {
        std::cout << "compute all min marginals in mma_vec\n";
        two_dim_variable_array<std::array<float,2>> mm;

        // TODO: seems not to work. Somewhere state is not set correctly!
        if(message_passing_state_ != message_passing_state::after_backward_pass)
            backward_run();
        message_passing_state_ = message_passing_state::none;

        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        for(size_t var=0; var<nr_variables(); ++var)
        {
            const size_t _nr_bdds = nr_bdds(var);
            std::array<float,2> min_marginals[_nr_bdds];
            std::fill(min_marginals, min_marginals + _nr_bdds, std::array<float,2>{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()});

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].min_marginal(min_marginals);

            mm.push_back(min_marginals, min_marginals + _nr_bdds);
            this->forward_step(var);
        }

        message_passing_state_ = message_passing_state::after_forward_pass;
        return mm;
    }

    inline two_dim_variable_array<size_t> bdd_mma_base_vec::tighten_bdd_groups(const std::vector<char>& tighten_variables)
    {
        // (i) collect all BDDs that have support on at least one of the variables to participate in tightening.
        std::vector<size_t> tighten_bdds;
        std::vector<size_t> bdd_node_vars = compute_bdd_branch_instruction_variables();
        std::vector<char> visited(bdd_branch_nodes_.size(), false);
        for(size_t bdd_idx=0; bdd_idx<first_bdd_node_indices_.size(); ++bdd_idx)
        {
            assert(first_bdd_node_indices_.size(bdd_idx) == 1);
            std::deque<size_t> dq;
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_idx); ++j)
                dq.push_back(first_bdd_node_indices_(bdd_idx, j));
            while(!dq.empty())
            {
                const size_t i = dq.front();
                dq.pop_front();
                if(visited[i] == true)
                    continue;
                visited[i] = true;
                const size_t var = bdd_node_vars[i];
                if(tighten_variables[var])
                    tighten_bdds.push_back(bdd_idx);
                auto& bdd_node = bdd_branch_nodes_[i];
                if(bdd_node.offset_low != bdd_branch_node_vec::terminal_0_offset && bdd_node.offset_low != bdd_branch_node_vec::terminal_1_offset)
                {
                    const size_t low_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_low)); 
                    dq.push_back(low_offset);
                }
                if(bdd_node.offset_high != bdd_branch_node_vec::terminal_0_offset && bdd_node.offset_high != bdd_branch_node_vec::terminal_1_offset)
                {
                    const size_t high_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_high)); 
                    dq.push_back(high_offset);
                }
            }
        }

        // (ii) partition BDDs into groups defined by disconnected components in the variable/BDD adjacency graph.
        // first build up var bdd adjacency matrix
        std::vector<std::array<size_t,2>> var_bdd_adjacencies;
        std::vector<size_t> var_bdd_adjacencies_size(tighten_bdds.size() + nr_variables(), 0);
        std::fill(visited.begin(), visited.end(), false);
        for(size_t c=0; c<tighten_bdds.size(); ++c)
        {
            const size_t bdd_idx = tighten_bdds[c];
            assert(first_bdd_node_indices_.size(bdd_idx) == 1);
            std::deque<size_t> dq;
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_idx); ++j)
                dq.push_back(first_bdd_node_indices_(bdd_idx, j));
            while(!dq.empty())
            {
                const size_t i = dq.front();
                dq.pop_front();
                if(visited[i] == true)
                    continue;
                visited[i] = true;
                const size_t var = variable(i);
                var_bdd_adjacencies.push_back({c, tighten_bdds.size() + var});
                ++var_bdd_adjacencies_size[c];
                ++var_bdd_adjacencies_size[tighten_bdds.size() + var];
                auto& bdd_node = bdd_branch_nodes_[i];
                if(bdd_node.offset_low != bdd_branch_node_vec::terminal_0_offset && bdd_node.offset_low != bdd_branch_node_vec::terminal_1_offset)
                {
                    const size_t low_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_low)); 
                    dq.push_back(low_offset);
                }
                if(bdd_node.offset_high != bdd_branch_node_vec::terminal_0_offset && bdd_node.offset_high != bdd_branch_node_vec::terminal_1_offset)
                {
                    const size_t high_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_high)); 
                    dq.push_back(high_offset); 
                }
            } 
        }

        // build var bdd adjacency graph
        two_dim_variable_array<size_t> g(var_bdd_adjacencies_size);
        std::fill(var_bdd_adjacencies_size.begin(), var_bdd_adjacencies_size.end(), 0);
        for(const auto [x,y] : var_bdd_adjacencies)
        {
            g(x, var_bdd_adjacencies_size[x]++) = y;
            g(y, var_bdd_adjacencies_size[y]++) = x; 
        }

        // determine connected components
        visited.resize(nr_variables() + tighten_bdds.size());
        std::fill(visited.begin(), visited.end(), false);
        two_dim_variable_array<size_t> ccs;
        for(size_t i=0; i<visited.size(); ++i)
        {
            if(visited[i])
                continue;
            std::stack<size_t> s;
            std::vector<size_t> cc_indices;
            s.push(i);
            while(!s.empty())
            {
                const size_t i = s.top();
                s.pop();
                if(visited[i])
                    continue;
                visited[i] = true;
                if(i < tighten_bdds.size())
                    cc_indices.push_back(i);
                for(const size_t j : g[i])
                    s.push(j); 
            }
            if(cc_indices.size() > 0)
                ccs.push_back(cc_indices.begin(), cc_indices.end());
        }

        std::cout << "#bdd groups = " << ccs.size() << "\n";
        size_t nr_bdds = 0;
        size_t max_group_size = 0;
        for(size_t c=0; c<ccs.size(); ++c)
        {
            nr_bdds += ccs.size(c);
            max_group_size = std::max(max_group_size, ccs.size(c)); 
        }
        std::cout << "#bdds = " << nr_bdds << "\n";
        std::cout << "max bdd group size = " << max_group_size << "\n";

        return ccs;
    }

    inline std::vector<size_t> bdd_mma_base_vec::variables(const size_t bdd_idx)
    {
        assert(bdd_idx < nr_bdds());
        std::deque<size_t> dq;
        tsl::robin_set<size_t> visited;
        tsl::robin_set<size_t> vars;
        for(size_t j=0; j<first_bdd_node_indices_.size(bdd_idx); ++j)
            dq.push_back(first_bdd_node_indices_(bdd_idx, j));
        while(!dq.empty())
        {
            const size_t i = dq.front();
            dq.pop_front();
            if(visited.count(i) > 0)
                continue;
            visited.insert(i);
            const size_t var = variable(i);
            vars.insert(var);
            auto& bdd_node = bdd_branch_nodes_[i];
            if(bdd_node.offset_low != bdd_branch_node_vec::terminal_0_offset && bdd_node.offset_low != bdd_branch_node_vec::terminal_1_offset)
            {
                const size_t low_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_low)); 
                dq.push_back(low_offset);
            }
            if(bdd_node.offset_high != bdd_branch_node_vec::terminal_0_offset && bdd_node.offset_high != bdd_branch_node_vec::terminal_1_offset)
            {
                const size_t high_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_high)); 
                dq.push_back(high_offset);
            }
        } 
        std::vector<size_t> var_vec(vars.begin(), vars.end());
        std::sort(var_vec.begin(), var_vec.end());
        return var_vec;
    }

    inline std::vector<bdd_branch_instruction<float>> bdd_mma_base_vec::export_bdd(const size_t bdd_idx)
    {
        assert(bdd_idx < nr_bdds());
        std::vector<bdd_branch_instruction<float>> bdds;
        std::vector<std::array<size_t,2>> bdd_offsets;
        std::deque<size_t> dq;
        // TODO: replace visited by using bdd_offset_map
        tsl::robin_set<size_t> visited;
        tsl::robin_map<size_t,size_t> bdd_offset_map;
        for(size_t j=0; j<first_bdd_node_indices_.size(bdd_idx); ++j)
            dq.push_back(first_bdd_node_indices_(bdd_idx, j));
        while(!dq.empty())
        {
            const size_t i = dq.front();
            dq.pop_front();
            if(visited.count(i) > 0)
                continue;
            const size_t var = variable(i);
            visited.insert(i);
            auto& instr = bdd_branch_nodes_[i];

            auto calculate_offset = [&](auto& bdd, const size_t offset) {
                // check if low resp. high arc min-marginal is within epsilon of lower bound
                if(offset == bdd_branch_node_vec::terminal_0_offset)
                    return bdd_node::terminal_0;
                else if(offset == bdd_branch_node_vec::terminal_1_offset)
                    return bdd_node::terminal_1;
                else
                {
                    const size_t translated_offset = std::distance(&bdd_branch_nodes_[0], bdd.address(offset)); 
                    dq.push_back(translated_offset);
                    return translated_offset;
                }
            };

            bdd_branch_instruction<float> bdd;
            bdd.low_cost = instr.low_cost;
            bdd.high_cost = instr.high_cost;
            bdd.offset_low = instr.offset_low;
            bdd.offset_high = instr.offset_high;
            bdds.push_back(bdd);
            bdd_offsets.push_back({calculate_offset(instr, instr.offset_low), calculate_offset(instr, instr.offset_high)});
            bdd_offset_map[i] = bdds.size()-1; 
        }

        // map bdd offsets
        for(size_t i=0; i<bdds.size(); ++i)
        {
            auto& instr = bdds[i];
            if(instr.offset_low != bdd_node::terminal_0 && instr.offset_low != bdd_node::terminal_1)
            {
                assert(bdd_offset_map[instr.offset_low] != std::numeric_limits<size_t>::max());
                assert(bdd_offset_map[instr.offset_low] > i);
                instr.offset_low = bdd_offset_map[instr.offset_low];
            }
            if(instr.offset_high != bdd_node::terminal_0 && instr.offset_high != bdd_node::terminal_1)
            {
                assert(bdd_offset_map[instr.offset_high] != std::numeric_limits<size_t>::max());
                assert(bdd_offset_map[instr.offset_high] > i);
                instr.offset_high = bdd_offset_map[instr.offset_high];
            } 
        }

        return bdds;
    }

    inline size_t bdd_mma_base_vec::export_bdd(BDD::bdd_collection& bdd_col, const size_t bdd_idx)
    {
        const size_t new_bdd_nr = bdd_col.new_bdd();
        std::unordered_map<size_t, BDD::bdd_collection_node> bdd_col_nodes; // position in bdd_branch_nodes to node in bdd collection

        auto calculate_offset = [&](bdd_branch_node_vec& bdd, const size_t offset) {
            // check if low resp. high arc min-marginal is within epsilon of lower bound
            if(offset == bdd_branch_node_vec::terminal_0_offset)
                return bdd_node::terminal_0;
            else if(offset == bdd_branch_node_vec::terminal_1_offset)
                return bdd_node::terminal_1;
            else
            {
                const size_t translated_offset = std::distance(&bdd_branch_nodes_[0], bdd.address(offset)); 
                assert(translated_offset > 0 && translated_offset < bdd_branch_nodes_.size());
                return translated_offset;
            }
        };

        std::deque<size_t> dq;
        for(size_t j=0; j<first_bdd_node_indices_.size(bdd_idx); ++j)
            dq.push_back(first_bdd_node_indices_(bdd_idx, j));
        tsl::robin_set<size_t> visited;
        while(!dq.empty())
        {
            const size_t i = dq.front();
            dq.pop_front();
            if(visited.count(i) > 0)
                continue;
            const size_t var = variable(i);
            visited.insert(i);
            auto& instr = bdd_branch_nodes_[i];

            const size_t low_i = calculate_offset(instr, instr.offset_low);
            if(low_i != bdd_node::terminal_0 && low_i != bdd_node::terminal_1) 
                dq.push_back(low_i);

            const size_t high_i = calculate_offset(instr, instr.offset_high);
            if(high_i != bdd_node::terminal_0 && high_i != bdd_node::terminal_1) 
                dq.push_back(high_i);

            BDD::bdd_collection_node node = bdd_col.add_bdd_node(var);
            bdd_col_nodes.insert({i, node});
        }

        // set correct offsets for arcs in previously added bdd nodes
        for(auto& [i, node] : bdd_col_nodes)
        {
            const size_t lo_offset = calculate_offset(bdd_branch_nodes_[i], bdd_branch_nodes_[i].offset_low);
            if(lo_offset == bdd_node::terminal_0)
                node.set_lo_to_0_terminal();
            else if(lo_offset == bdd_node::terminal_1)
                node.set_lo_to_1_terminal();
            else
            {
                assert(bdd_col_nodes.count(lo_offset) > 0);
                node.set_lo_arc(bdd_col_nodes.find(lo_offset)->second);
            }

            const size_t hi_offset = calculate_offset(bdd_branch_nodes_[i], bdd_branch_nodes_[i].offset_high);
            if(hi_offset == bdd_node::terminal_0)
                node.set_hi_to_0_terminal();
            else if(hi_offset == bdd_node::terminal_1)
                node.set_hi_to_1_terminal();
            else
            {
                assert(bdd_col_nodes.count(hi_offset) > 0);
                node.set_hi_arc(bdd_col_nodes.find(hi_offset)->second);
            }
        }

        bdd_col.close_bdd(); 

        return new_bdd_nr;
    }

    inline std::tuple<BDD::node_ref, std::vector<size_t>> bdd_mma_base_vec::export_bdd(BDD::bdd_mgr& bdd_mgr, const size_t bdd_idx)
    {
        assert(bdd_idx < nr_bdds());

        // TODO: this should be its own function
        auto calculate_offset = [&](bdd_branch_node_vec& bdd, const size_t offset) {
            assert(offset != bdd_branch_node_vec::terminal_0_offset && offset != bdd_branch_node_vec::terminal_1_offset);
            const size_t translated_offset = std::distance(&bdd_branch_nodes_[0], bdd.address(offset)); 
            assert(translated_offset > 0 && translated_offset < bdd_branch_nodes_.size());
            return translated_offset;
        };

        std::deque<size_t> dq;
        assert(first_bdd_node_indices_.size(bdd_idx) == 1);
        for(size_t j=0; j<first_bdd_node_indices_.size(bdd_idx); ++j)
            dq.push_back(first_bdd_node_indices_(bdd_idx, j));
        tsl::robin_set<size_t> visited;
        std::vector<size_t> variable_vec;
        std::vector<size_t> bdd_nodes;
        while(!dq.empty())
        {
            const size_t i = dq.front();
            dq.pop_front();
            if(visited.count(i) > 0)
                continue;
            const size_t var = variable(i);
            visited.insert(i);
            bdd_nodes.push_back(i);
            auto& instr = bdd_branch_nodes_[i];
            variable_vec.push_back(variable(i));

            if(instr.offset_low != bdd_branch_node_vec::terminal_0_offset && instr.offset_low != bdd_branch_node_vec::terminal_1_offset)
                dq.push_back(calculate_offset(instr, instr.offset_low));

            if(instr.offset_high != bdd_branch_node_vec::terminal_0_offset && instr.offset_high != bdd_branch_node_vec::terminal_1_offset)
                dq.push_back(calculate_offset(instr, instr.offset_high));
        }

        std::sort(variable_vec.begin(), variable_vec.end());
        std::unordered_map<size_t,size_t> variable_map;
        for(size_t i=0; i<variable_vec.size(); ++i)
            variable_map.insert({variable_vec[i], i});

        std::reverse(bdd_nodes.begin(), bdd_nodes.end());
        std::unordered_map<size_t, BDD::node_ref> bdd_map;

        for(const size_t i : bdd_nodes)
        {
            bdd_branch_node_vec& instr = bdd_branch_nodes_[i];
            const size_t var = variable(i);
            auto get_node_ref = [&](bdd_branch_node_vec& bdd, const size_t offset) {
                if(offset == bdd_branch_node_vec::terminal_0_offset)
                    return bdd_mgr.botsink();
                if(offset == bdd_branch_node_vec::terminal_1_offset)
                    return bdd_mgr.topsink();
                else
                {
                    const size_t j = calculate_offset(bdd, offset);
                    assert(bdd_map.count(j) > 0);
                    return bdd_map.find(j)->second; 
                } 
            };
            BDD::node_ref low_bdd = get_node_ref(instr, instr.offset_low);
            BDD::node_ref high_bdd = get_node_ref(instr, instr.offset_high);
            assert(variable_map.count(var) > 0);

            const size_t index = variable_map.find(var)->second;
            bdd_map.insert({i, bdd_mgr.ite_rec(bdd_mgr.projection(index), high_bdd, low_bdd)}); 
        }

        assert(bdd_map.count(first_bdd_node_indices_(bdd_idx,0)) > 0);
        return {bdd_map.find(first_bdd_node_indices_(bdd_idx,0))->second, variable_vec};
    }

    // TODO: replace by add_bdds with bdd collection
    inline std::vector<size_t> bdd_mma_base_vec::add_bdds(const bdd_storage& bdd_storage_)
    {
        message_passing_state_ = message_passing_state::none;

        std::vector<bdd_branch_node_vec> new_bdd_branch_nodes_;
        std::vector<size_t> new_bdd_branch_node_offsets_;
        std::vector<size_t> new_nr_bdds_;
        two_dim_variable_array<size_t> new_first_bdd_node_indices_;
        two_dim_variable_array<size_t> new_last_bdd_node_indices_;

        std::vector<size_t> cur_first_bdd_node_indices;
        std::vector<size_t> cur_last_bdd_node_indices;

        const size_t new_nr_variables = std::max(nr_variables(), bdd_storage_.nr_variables());
        // count bdd branch nodes per variable
        std::vector<size_t> new_bdd_branch_nodes_per_var(new_nr_variables, 0);
        for(const auto& stored_bdd_node : bdd_storage_.bdd_nodes())
            ++new_bdd_branch_nodes_per_var[stored_bdd_node.variable]; 
        for(size_t v=0; v<nr_variables(); ++v)
            new_bdd_branch_nodes_per_var[v] += nr_bdd_nodes(v);

        new_bdd_branch_node_offsets_.reserve(new_nr_variables+1);
        new_bdd_branch_node_offsets_.push_back(0);
        for(const size_t i : new_bdd_branch_nodes_per_var)
            new_bdd_branch_node_offsets_.push_back(new_bdd_branch_node_offsets_.back() + i);

        new_bdd_branch_nodes_.resize(new_bdd_branch_node_offsets_.back());
        std::vector<size_t> new_bdd_branch_nodes_counter(new_nr_variables, 0); 

        // fill in previous bdd nodes
        for(size_t bdd_idx=0; bdd_idx<nr_bdds(); ++bdd_idx)
        {
            // TODO: put in front of loop and clear before use
            std::deque<size_t> dq;
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_idx); ++j)
                dq.push_back(first_bdd_node_indices_(bdd_idx, j));
            // TODO: more efficient: get variables from bdd storage
            const size_t first_var = variable(first_bdd_node_indices_(bdd_idx, 0));
            const size_t last_var = variable(last_bdd_node_indices_(bdd_idx, 0));
            std::vector<size_t> new_cur_first_bdd_node_indices;
            std::vector<size_t> new_cur_last_bdd_node_indices;
            // TODO: same as above.
            tsl::robin_map<size_t,size_t> index_map;
            while(!dq.empty())
            {
                const size_t old_i = dq.front();
                auto& old_bdd = bdd_branch_nodes_[old_i];
                dq.pop_front();
                if(index_map.count(old_i) > 0)
                    continue;
                const size_t var = variable(old_i);
                const size_t new_i = new_bdd_branch_node_offsets_[var] + new_bdd_branch_nodes_counter[var];
                ++new_bdd_branch_nodes_counter[var];
                if(var == first_var)
                    new_cur_first_bdd_node_indices.push_back(new_i);
                if(var == last_var)
                    new_cur_last_bdd_node_indices.push_back(new_i);
                //assert(old_i <= new_i); // not necessarily true if we reorder bdd branch nodes belonging to same variable and bdd
                index_map.insert({old_i, new_i});

                if(old_bdd.offset_low != bdd_branch_node_vec::terminal_0_offset && old_bdd.offset_low != bdd_branch_node_vec::terminal_1_offset) 
                {
                    const size_t old_low_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_low)); 
                    dq.push_back(old_low_i);
                }
                if(old_bdd.offset_high != bdd_branch_node_vec::terminal_0_offset && old_bdd.offset_high != bdd_branch_node_vec::terminal_1_offset) 
                {
                    const size_t old_high_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_high)); 
                    dq.push_back(old_high_i);
                }
            }

            new_first_bdd_node_indices_.push_back(new_cur_first_bdd_node_indices.begin(), new_cur_first_bdd_node_indices.end());
            new_last_bdd_node_indices_.push_back(new_cur_last_bdd_node_indices.begin(), new_cur_last_bdd_node_indices.end());

            // set address offsets right
            for(const auto [old_i, new_i] : index_map)
            {
                bdd_branch_node_vec& old_bdd = bdd_branch_nodes_[old_i];
                bdd_branch_node_vec& new_bdd = new_bdd_branch_nodes_[new_i];
                new_bdd = old_bdd;
                std::cout << "old i " << old_i << ", old i " << old_i << "\n";

                if(old_bdd.offset_low != bdd_branch_node_vec::terminal_0_offset && old_bdd.offset_low != bdd_branch_node_vec::terminal_1_offset) 
                {
                    const size_t old_low_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_low)); 
                    assert(index_map.count(old_low_i) > 0);
                    const size_t new_low_i = index_map.find(old_low_i)->second;
                    std::cout << "new low i " << new_low_i << ", old low i " << old_low_i << "\n";
                    new_bdd.offset_low = new_bdd.synthesize_address(&new_bdd_branch_nodes_[new_low_i]); 
                }

                if(old_bdd.offset_high != bdd_branch_node_vec::terminal_0_offset && old_bdd.offset_high != bdd_branch_node_vec::terminal_1_offset) 
                {
                    const size_t old_high_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_high)); 
                    assert(index_map.count(old_high_i) > 0);
                    const size_t new_high_i = index_map.find(old_high_i)->second;
                    new_bdd.offset_high = new_bdd.synthesize_address(&new_bdd_branch_nodes_[new_high_i]); 
                }
            } 
        }

        std::vector<bdd_branch_node_vec*> stored_bdd_index_to_bdd_offset(bdd_branch_nodes_.size() + bdd_storage_.bdd_nodes().size(), nullptr);
        std::vector<size_t> new_bdd_nrs;
        // fill in bdds from bdd_storage_
        for(size_t bdd_index=0; bdd_index<bdd_storage_.nr_bdds(); ++bdd_index)
        {
            new_bdd_nrs.push_back(bdd_index + nr_bdds());
            cur_first_bdd_node_indices.clear();
            cur_last_bdd_node_indices.clear();
            const size_t first_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index]; 
            const size_t last_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index+1];
            const size_t first_var = bdd_storage_.bdd_nodes()[last_stored_bdd_node-1].variable;
            const size_t last_var = bdd_storage_.bdd_nodes()[first_stored_bdd_node].variable;
            for(size_t i=first_stored_bdd_node; i<last_stored_bdd_node; ++i)
            {
                assert(first_var <= bdd_storage_.bdd_nodes()[i].variable);
                assert(last_var >= bdd_storage_.bdd_nodes()[i].variable);
            }

            for(size_t stored_bdd_node_index=first_stored_bdd_node; stored_bdd_node_index<last_stored_bdd_node; ++stored_bdd_node_index)
            {
                const auto& stored_bdd = bdd_storage_.bdd_nodes()[stored_bdd_node_index];
                const size_t v = stored_bdd.variable;
                const size_t bdd_branch_index = new_bdd_branch_node_offsets_[v] + new_bdd_branch_nodes_counter[v];
                assert(new_bdd_branch_nodes_[bdd_branch_index].offset_low == 0 && new_bdd_branch_nodes_[bdd_branch_index].offset_high == 0 && new_bdd_branch_nodes_[bdd_branch_index].bdd_index == bdd_branch_node_vec::inactive_bdd_index);
                ++new_bdd_branch_nodes_counter[v];

                if(v == first_var)
                    cur_first_bdd_node_indices.push_back(bdd_branch_index);
                if(v == last_var)
                    cur_last_bdd_node_indices.push_back(bdd_branch_index);

                if(stored_bdd.low == bdd_storage::bdd_node::terminal_0)
                {
                    new_bdd_branch_nodes_[bdd_branch_index].offset_low = bdd_branch_node_vec::terminal_0_offset;
                    new_bdd_branch_nodes_[bdd_branch_index].low_cost = std::numeric_limits<float>::infinity();
                }
                else if(stored_bdd.low == bdd_storage::bdd_node::terminal_1)
                {
                    assert(new_bdd_branch_nodes_[bdd_branch_index].low_cost == 0.0);
                    new_bdd_branch_nodes_[bdd_branch_index].offset_low = bdd_branch_node_vec::terminal_1_offset;
                }
                else
                {
                    assert(new_bdd_branch_nodes_[bdd_branch_index].low_cost == 0.0);
                    assert(stored_bdd_index_to_bdd_offset[stored_bdd.low] != nullptr);
                    bdd_branch_node_vec* low_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.low];
                    new_bdd_branch_nodes_[bdd_branch_index].offset_low = new_bdd_branch_nodes_[bdd_branch_index].synthesize_address(low_ptr);
                }

                if(stored_bdd.high == bdd_storage::bdd_node::terminal_0)
                {
                    new_bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_node_vec::terminal_0_offset;
                    new_bdd_branch_nodes_[bdd_branch_index].high_cost = std::numeric_limits<float>::infinity();
                }
                else if(stored_bdd.high == bdd_storage::bdd_node::terminal_1)
                {
                    assert(new_bdd_branch_nodes_[bdd_branch_index].high_cost == 0.0);
                    new_bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_node_vec::terminal_1_offset;
                }
                else
                {
                    assert(new_bdd_branch_nodes_[bdd_branch_index].high_cost == 0.0);
                    assert(stored_bdd_index_to_bdd_offset[stored_bdd.high] != nullptr);
                    bdd_branch_node_vec* high_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.high];
                    new_bdd_branch_nodes_[bdd_branch_index].offset_high = new_bdd_branch_nodes_[bdd_branch_index].synthesize_address(high_ptr);
                }

                if(bdd_index + nr_bdds() >= std::numeric_limits<uint32_t>::max())
                    throw std::runtime_error("bdd indices exceed 2^32"); // TODO: write alternative mechanism for this case
                new_bdd_branch_nodes_[bdd_branch_index].bdd_index = nr_bdds() + bdd_index;
                assert(stored_bdd_index_to_bdd_offset[stored_bdd_node_index] == nullptr);

                stored_bdd_index_to_bdd_offset[stored_bdd_node_index] = &new_bdd_branch_nodes_[bdd_branch_index];
            }

            assert(cur_first_bdd_node_indices.size() > 0);
            new_first_bdd_node_indices_.push_back(cur_first_bdd_node_indices.begin(), cur_first_bdd_node_indices.end()); 
            assert(cur_last_bdd_node_indices.size() > 0);
            new_last_bdd_node_indices_.push_back(cur_last_bdd_node_indices.begin(), cur_last_bdd_node_indices.end()); 
        }

        new_nr_bdds_.clear();
        new_nr_bdds_.reserve(new_nr_variables);
        tsl::robin_map<uint32_t, uint32_t> bdd_index_redux;
        for(size_t i=0; i<new_nr_variables; ++i)
        {
            bdd_index_redux.clear();
            for(size_t vec_idx=new_bdd_branch_node_offsets_[i]; vec_idx<new_bdd_branch_node_offsets_[i+1]; ++vec_idx)
            {
                auto& bdd_vec = new_bdd_branch_nodes_[vec_idx]; 
                const uint32_t bdd_index = bdd_vec.bdd_index;
                assert(bdd_index != bdd_branch_node_vec::inactive_bdd_index);
                if(bdd_index_redux.count(bdd_index) == 0)
                    bdd_index_redux.insert({bdd_index, bdd_index_redux.size()}); 
            }
            for(size_t vec_idx=new_bdd_branch_node_offsets_[i]; vec_idx<new_bdd_branch_node_offsets_[i+1]; ++vec_idx)
            {
                auto& bdd_vec = new_bdd_branch_nodes_[vec_idx];
                bdd_vec.bdd_index = bdd_index_redux.find(bdd_vec.bdd_index)->second;
            }
            new_nr_bdds_.push_back(bdd_index_redux.size());
        }


        bdd_branch_instruction_variables_.clear(); // to force recomputation for variable of bdd node
        // swap new and old data structures
        std::swap(new_bdd_branch_nodes_, bdd_branch_nodes_);
        std::swap(new_bdd_branch_node_offsets_, bdd_branch_node_offsets_);
        std::swap(new_nr_bdds_, nr_bdds_);
        std::swap(new_first_bdd_node_indices_, first_bdd_node_indices_);
        std::swap(new_last_bdd_node_indices_, last_bdd_node_indices_);
        message_passing_state_ = message_passing_state::none;

        return new_bdd_nrs;
    }

    template<typename BDD_NR_ITERATOR>
        std::vector<size_t> bdd_mma_base_vec::add_bdds(BDD::bdd_collection& bdd_col, BDD_NR_ITERATOR bdd_nrs_begin, BDD_NR_ITERATOR bdd_nrs_end)
        {
            message_passing_state_ = message_passing_state::none;

            std::vector<bdd_branch_node_vec> new_bdd_branch_nodes_;
            std::vector<size_t> new_bdd_branch_node_offsets_;
            std::vector<size_t> new_nr_bdds_;
            two_dim_variable_array<size_t> new_first_bdd_node_indices_;
            two_dim_variable_array<size_t> new_last_bdd_node_indices_;

            std::vector<size_t> cur_first_bdd_node_indices;
            std::vector<size_t> cur_last_bdd_node_indices;

            const size_t bdd_col_max_var = [&]() {
                size_t max_var = 0;
                for(auto bdd_nr_it=bdd_nrs_begin; bdd_nr_it!=bdd_nrs_end; ++bdd_nr_it)
                    max_var = std::max(max_var, bdd_col.min_max_variables(*bdd_nr_it)[1]);
                return max_var; 
            }();
            const size_t new_nr_variables = std::max(nr_variables(), bdd_col_max_var);
            // count bdd branch nodes per variable
            std::vector<size_t> new_bdd_branch_nodes_per_var(new_nr_variables, 0);
            for(auto bdd_nr_it=bdd_nrs_begin; bdd_nr_it!=bdd_nrs_end; ++bdd_nr_it)
                for(auto bdd_it=bdd_col.begin(*bdd_nr_it); bdd_it!=bdd_col.end(*bdd_nr_it); ++bdd_it)
                    if(!bdd_it->is_terminal())
                        ++new_bdd_branch_nodes_per_var[bdd_it->index];
            for(size_t v=0; v<nr_variables(); ++v)
                new_bdd_branch_nodes_per_var[v] += nr_bdd_nodes(v);

            new_bdd_branch_node_offsets_.reserve(new_nr_variables+1);
            new_bdd_branch_node_offsets_.push_back(0);
            for(const size_t i : new_bdd_branch_nodes_per_var)
                new_bdd_branch_node_offsets_.push_back(new_bdd_branch_node_offsets_.back() + i);

            new_bdd_branch_nodes_.resize(new_bdd_branch_node_offsets_.back());
            std::vector<size_t> new_bdd_branch_nodes_counter(new_nr_variables, 0); 

            // fill in previous bdd nodes
            for(size_t bdd_idx=0; bdd_idx<nr_bdds(); ++bdd_idx)
            {
                // TODO: put in front of loop and clear before use
                std::deque<size_t> dq;
                for(size_t j=0; j<first_bdd_node_indices_.size(bdd_idx); ++j)
                    dq.push_back(first_bdd_node_indices_(bdd_idx, j));
                // TODO: more efficient: get variables from bdd storage
                const size_t first_var = variable(first_bdd_node_indices_(bdd_idx, 0));
                const size_t last_var = variable(last_bdd_node_indices_(bdd_idx, 0));
                std::vector<size_t> new_cur_first_bdd_node_indices;
                std::vector<size_t> new_cur_last_bdd_node_indices;
                // TODO: same as above.
                tsl::robin_map<size_t,size_t> index_map;
                while(!dq.empty())
                {
                    const size_t old_i = dq.front();
                    auto& old_bdd = bdd_branch_nodes_[old_i];
                    dq.pop_front();
                    if(index_map.count(old_i) > 0)
                        continue;
                    const size_t var = variable(old_i);
                    const size_t new_i = new_bdd_branch_node_offsets_[var] + new_bdd_branch_nodes_counter[var];
                    ++new_bdd_branch_nodes_counter[var];
                    if(var == first_var)
                        new_cur_first_bdd_node_indices.push_back(new_i);
                    if(var == last_var)
                        new_cur_last_bdd_node_indices.push_back(new_i);
                    //assert(old_i <= new_i); // not necessarily true if we reorder bdd branch nodes belonging to same variable and bdd
                    index_map.insert({old_i, new_i});

                    if(old_bdd.offset_low != bdd_branch_node_vec::terminal_0_offset && old_bdd.offset_low != bdd_branch_node_vec::terminal_1_offset) 
                    {
                        const size_t old_low_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_low)); 
                        dq.push_back(old_low_i);
                    }
                    if(old_bdd.offset_high != bdd_branch_node_vec::terminal_0_offset && old_bdd.offset_high != bdd_branch_node_vec::terminal_1_offset) 
                    {
                        const size_t old_high_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_high)); 
                        dq.push_back(old_high_i);
                    }
                }

                new_first_bdd_node_indices_.push_back(new_cur_first_bdd_node_indices.begin(), new_cur_first_bdd_node_indices.end());
                new_last_bdd_node_indices_.push_back(new_cur_last_bdd_node_indices.begin(), new_cur_last_bdd_node_indices.end());

                // set address offsets right
                for(const auto [old_i, new_i] : index_map)
                {
                    bdd_branch_node_vec& old_bdd = bdd_branch_nodes_[old_i];
                    bdd_branch_node_vec& new_bdd = new_bdd_branch_nodes_[new_i];
                    new_bdd = old_bdd;

                    if(old_bdd.offset_low != bdd_branch_node_vec::terminal_0_offset && old_bdd.offset_low != bdd_branch_node_vec::terminal_1_offset) 
                    {
                        const size_t old_low_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_low)); 
                        assert(index_map.count(old_low_i) > 0);
                        const size_t new_low_i = index_map.find(old_low_i)->second;
                        new_bdd.offset_low = new_bdd.synthesize_address(&new_bdd_branch_nodes_[new_low_i]); 
                    }

                    if(old_bdd.offset_high != bdd_branch_node_vec::terminal_0_offset && old_bdd.offset_high != bdd_branch_node_vec::terminal_1_offset) 
                    {
                        const size_t old_high_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_high)); 
                        assert(index_map.count(old_high_i) > 0);
                        const size_t new_high_i = index_map.find(old_high_i)->second;
                        new_bdd.offset_high = new_bdd.synthesize_address(&new_bdd_branch_nodes_[new_high_i]); 
                    }
                } 
            }

            //const size_t nr_bdd_col_nodes = [&]() {
            //    size_t nr_nodes = 0;
            //    for(auto bdd_nr_it=bdd_nrs_begin; bdd_nr_it!=bdd_nrs_end; ++bdd_nr_it)
            //        nr_nodes += bdd_col.nr_bdd_nodes(*bdd_nr_it);
            //    return nr_nodes; 
            //}();
            std::unordered_map<size_t,bdd_branch_node_vec*> stored_bdd_index_to_bdd_offset;//(bdd_branch_nodes_.size() + nr_bdd_col_nodes, nullptr);
            std::vector<size_t> new_bdd_nrs;
            // fill in bdds from bdd_collection
            for(auto bdd_nr_it=bdd_nrs_begin; bdd_nr_it!=bdd_nrs_end; ++bdd_nr_it)
            {
                const size_t bdd_nr = *bdd_nr_it;
                assert(bdd_col.is_reordered(bdd_nr));
                assert(bdd_col.is_qbdd(bdd_nr));
                new_bdd_nrs.push_back(std::distance(bdd_nrs_begin,bdd_nr_it) + nr_bdds());
                cur_first_bdd_node_indices.clear();
                cur_last_bdd_node_indices.clear();
                const auto [first_var, last_var] = bdd_col.min_max_variables(bdd_nr);

                for(auto bdd_it=bdd_col.rbegin(*bdd_nr_it); bdd_it!=bdd_col.rend(*bdd_nr_it); ++bdd_it)
                {
                    const auto stored_bdd = *bdd_it;
                    assert(!stored_bdd.is_terminal());
                    const size_t v = stored_bdd.index;
                    const size_t bdd_branch_index = new_bdd_branch_node_offsets_[v] + new_bdd_branch_nodes_counter[v];
                    assert(new_bdd_branch_nodes_[bdd_branch_index].offset_low == 0 && new_bdd_branch_nodes_[bdd_branch_index].offset_high == 0 && new_bdd_branch_nodes_[bdd_branch_index].bdd_index == bdd_branch_node_vec::inactive_bdd_index);
                    ++new_bdd_branch_nodes_counter[v];

                    if(v == first_var)
                        cur_first_bdd_node_indices.push_back(bdd_branch_index);
                    if(v == last_var)
                        cur_last_bdd_node_indices.push_back(bdd_branch_index);

                    if(bdd_col(bdd_nr,stored_bdd.lo).is_botsink())
                    {
                        new_bdd_branch_nodes_[bdd_branch_index].offset_low = bdd_branch_node_vec::terminal_0_offset;
                        new_bdd_branch_nodes_[bdd_branch_index].low_cost = std::numeric_limits<float>::infinity();
                    }
                    else if(bdd_col(bdd_nr,stored_bdd.lo).is_topsink())
                    {
                        assert(new_bdd_branch_nodes_[bdd_branch_index].low_cost == 0.0);
                        new_bdd_branch_nodes_[bdd_branch_index].offset_low = bdd_branch_node_vec::terminal_1_offset;
                    }
                    else
                    {
                        assert(new_bdd_branch_nodes_[bdd_branch_index].low_cost == 0.0);
                        assert(stored_bdd_index_to_bdd_offset.count(stored_bdd.lo) > 0);
                        bdd_branch_node_vec* low_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.lo];
                        new_bdd_branch_nodes_[bdd_branch_index].offset_low = new_bdd_branch_nodes_[bdd_branch_index].synthesize_address(low_ptr);
                    }

                    if(bdd_col(bdd_nr,stored_bdd.hi).is_botsink())
                    {
                        new_bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_node_vec::terminal_0_offset;
                        new_bdd_branch_nodes_[bdd_branch_index].high_cost = std::numeric_limits<float>::infinity();
                    }
                    else if(bdd_col(bdd_nr,stored_bdd.hi).is_topsink())
                    {
                        assert(new_bdd_branch_nodes_[bdd_branch_index].high_cost == 0.0);
                        new_bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_node_vec::terminal_1_offset;
                    }
                    else
                    {
                        assert(new_bdd_branch_nodes_[bdd_branch_index].high_cost == 0.0);
                        assert(stored_bdd_index_to_bdd_offset.count(stored_bdd.hi) > 0);// != nullptr);
                        bdd_branch_node_vec* high_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.hi];
                        new_bdd_branch_nodes_[bdd_branch_index].offset_high = new_bdd_branch_nodes_[bdd_branch_index].synthesize_address(high_ptr);
                    }

                    const size_t bdd_counter = std::distance(bdd_nrs_begin, bdd_nr_it);
                    if(bdd_counter + nr_bdds() >= std::numeric_limits<uint32_t>::max())
                        throw std::runtime_error("bdd indices exceed 2^32"); // TODO: write alternative mechanism for this case
                    new_bdd_branch_nodes_[bdd_branch_index].bdd_index = nr_bdds() + bdd_counter;
                    assert(stored_bdd_index_to_bdd_offset.count(bdd_col.offset(*bdd_it)) == 0);
                    stored_bdd_index_to_bdd_offset[bdd_col.offset(*bdd_it)] = &new_bdd_branch_nodes_[bdd_branch_index];
                }

                assert(cur_first_bdd_node_indices.size() > 0);
                new_first_bdd_node_indices_.push_back(cur_first_bdd_node_indices.begin(), cur_first_bdd_node_indices.end()); 
                assert(cur_last_bdd_node_indices.size() > 0);
                new_last_bdd_node_indices_.push_back(cur_last_bdd_node_indices.begin(), cur_last_bdd_node_indices.end()); 
            }

            new_nr_bdds_.clear();
            new_nr_bdds_.reserve(new_nr_variables);
            tsl::robin_map<uint32_t, uint32_t> bdd_index_redux;
            for(size_t i=0; i<new_nr_variables; ++i)
            {
                bdd_index_redux.clear();
                for(size_t vec_idx=new_bdd_branch_node_offsets_[i]; vec_idx<new_bdd_branch_node_offsets_[i+1]; ++vec_idx)
                {
                    auto& bdd_vec = new_bdd_branch_nodes_[vec_idx]; 
                    const uint32_t bdd_index = bdd_vec.bdd_index;
                    assert(bdd_index != bdd_branch_node_vec::inactive_bdd_index);
                    if(bdd_index_redux.count(bdd_index) == 0)
                        bdd_index_redux.insert({bdd_index, bdd_index_redux.size()}); 
                }
                for(size_t vec_idx=new_bdd_branch_node_offsets_[i]; vec_idx<new_bdd_branch_node_offsets_[i+1]; ++vec_idx)
                {
                    auto& bdd_vec = new_bdd_branch_nodes_[vec_idx];
                    bdd_vec.bdd_index = bdd_index_redux.find(bdd_vec.bdd_index)->second;
                }
                new_nr_bdds_.push_back(bdd_index_redux.size());
            }


            bdd_branch_instruction_variables_.clear(); // to force recomputation for variable of bdd node
            // swap new and old data structures
            std::swap(new_bdd_branch_nodes_, bdd_branch_nodes_);
            std::swap(new_bdd_branch_node_offsets_, bdd_branch_node_offsets_);
            std::swap(new_nr_bdds_, nr_bdds_);
            std::swap(new_first_bdd_node_indices_, first_bdd_node_indices_);
            std::swap(new_last_bdd_node_indices_, last_bdd_node_indices_);
            message_passing_state_ = message_passing_state::none;

            return new_bdd_nrs;

        }

}
