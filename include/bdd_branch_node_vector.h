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
#include "bdd.h"
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
            bdd_mma_base_vec(const bdd_storage& bdd_storage_) { init(bdd_storage_); }
            void init(const bdd_storage& bdd_storage_);
            size_t nr_variables() const;
            size_t nr_variable_groups() const;
            size_t nr_bdd_vectors(const size_t var_group) const;
            size_t nr_bdds(const size_t var) const { assert(var < nr_variables()); return nr_bdds_[var]; }
            size_t nr_bdd_nodes() const { return bdd_branch_nodes_.size(); }

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
            void solve(const size_t max_iter, const double tolerance); 
            double lower_bound() const { return lower_bound_; }
            void set_cost(const double c, const size_t var);
            template<typename ITERATOR>
                void update_arc_costs(const size_t first_node, ITERATOR begin, ITERATOR end);
            void get_arc_marginals(const size_t first_node, const size_t last_node, std::vector<double>& arc_marginals);

            std::array<size_t,2> bdd_branch_node_offset(const size_t var, const size_t bdd_index) const;

        protected:
            std::vector<bdd_branch_node_vec> bdd_branch_nodes_;
            std::vector<size_t> bdd_branch_node_offsets_; // offsets into where bdd branch nodes belonging to a variable start 
            std::vector<size_t> bdd_branch_node_group_offsets_; // offsets into where bdd branch nodes belonging to a variable group start 
            std::vector<size_t> nr_bdds_;
            two_dim_variable_array<size_t> first_bdd_node_indices_;  // used for computing lower bound
            two_dim_variable_array<size_t> last_bdd_node_indices_;  // used for computing lower bound
            double lower_bound_ = -std::numeric_limits<double>::infinity();

            enum class message_passing_state {
                after_forward_pass,
                after_backward_pass,
                none 
            } message_passing_state_ = message_passing_state::none;

        private:
            std::vector<size_t> bdd_node_variables() const;
            void tighten_bdd(const float eps);
            std::vector<float> min_marginal_differences(const float eps);
            two_dim_variable_array<size_t> tighten_bdd_groups(const std::vector<char>& tighten_variables);
    };


    inline size_t bdd_mma_base_vec::nr_variables() const
    {
        return bdd_branch_node_offsets_.size()-1; 
    }

    inline size_t bdd_mma_base_vec::nr_variable_groups() const
    {
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
                    bdd_branch_nodes_[bdd_branch_index].offset_low = bdd_branch_node_vec::terminal_1_offset;;
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
                    bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_node_vec::terminal_1_offset;;
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

    inline void bdd_mma_base_vec::solve(const size_t max_iter, const double tolerance)
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
            std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
            std::cout << "\n";
            if (std::abs(lb_prev-lb_post) < std::abs(0.001*tolerance*lb_prev))
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

    inline std::vector<double> bdd_mma_base_vec::total_min_marginals()
    {
        this->backward_run();
        // prepare forward run
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        std::vector<double> total_min_marginals;
        total_min_marginals.reserve(nr_variables());

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

            total_min_marginals.push_back(total_min_marg); 
        }
        return total_min_marginals;
    }

    inline std::vector<size_t> bdd_mma_base_vec::bdd_node_variables() const
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
        const std::vector<size_t> vars = bdd_node_variables();
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
            std::unordered_map<size_t,size_t> var_to_index_map; // from original variables to consecutive index
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

    inline two_dim_variable_array<size_t> bdd_mma_base_vec::tighten_bdd_groups(const std::vector<char>& tighten_variables)
    {
        // (i) collect all BDDs that have support on at least one of the variables to participate in tightening.
        std::vector<size_t> tighten_bdds;
        std::vector<size_t> bdd_node_vars = bdd_node_variables();
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
                const size_t var = bdd_node_vars[i];
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

}
