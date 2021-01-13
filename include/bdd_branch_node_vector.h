#pragma once

// TODO: rename file and classes

#include <tsl/robin_map.h>
#include <vector>
#include <array>
#include <cstddef>
#include "bdd_storage.h"
#include "two_dimensional_variable_array.hxx"
#include "time_measure_util.h"
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

            void forward_step(const size_t var_group);
            void min_marginal_averaging_forward();
            void min_marginal_averaging_step_forward(const size_t var_group);

            void backward_step(const size_t var_group);
            void min_marginal_averaging_backward();
            void min_marginal_averaging_step_backward(const size_t var_group);

            std::array<float,2> average_marginals(std::array<float,2>* marginals, const size_t nr_marginals);

            void iteration();
            void backward_run();
            void compute_lower_bound(); 
            std::vector<double> total_min_marginals();
            void solve(const size_t max_iter); 
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
            double lower_bound_ = -std::numeric_limits<double>::infinity();
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
        first_bdd_node_indices_.clear();
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

        std::vector<bdd_branch_node_vec*> stored_bdd_index_to_bdd_offset(bdd_storage_.bdd_nodes().size());

        for(size_t bdd_index=0; bdd_index<bdd_storage_.nr_bdds(); ++bdd_index)
        {
            cur_first_bdd_node_indices.clear();
            const size_t first_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index]; 
            const size_t last_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index+1];
            for(size_t stored_bdd_node_index=first_stored_bdd_node; stored_bdd_node_index<last_stored_bdd_node; ++stored_bdd_node_index)
            {
                const auto& stored_bdd = bdd_storage_.bdd_nodes()[stored_bdd_node_index];
                const size_t v = stored_bdd.variable;
                const size_t bdd_branch_index = bdd_branch_node_offsets_[v] + bdd_branch_nodes_counter[v];
                ++bdd_branch_nodes_counter[v];

                if(v == bdd_storage_.bdd_nodes()[last_stored_bdd_node-1].variable)
                    cur_first_bdd_node_indices.push_back(bdd_branch_index);

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
                    //assert(stored_bdd_index_to_bdd_offset.count(stored_bdd.high) > 0);
                    //const auto [high_ptr, high_intra_vec_index] = stored_bdd_index_to_bdd_offset.find(stored_bdd.high)->second;
                    bdd_branch_node_vec* high_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.high];
                    bdd_branch_nodes_[bdd_branch_index].offset_high = bdd_branch_nodes_[bdd_branch_index].synthesize_address(high_ptr);
                }

                assert(bdd_index <= std::numeric_limits<uint32_t>::max()); // TODO: write alternative mechanism for this case
                bdd_branch_nodes_[bdd_branch_index].bdd_index = bdd_index;

                stored_bdd_index_to_bdd_offset[stored_bdd_node_index] = &bdd_branch_nodes_[bdd_branch_index];
            }

            first_bdd_node_indices_.push_back(cur_first_bdd_node_indices.begin(), cur_first_bdd_node_indices.end()); 
        }

        assert(first_bdd_node_indices_.size() == bdd_storage_.nr_bdds());

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
        MEASURE_FUNCTION_EXECUTION_TIME;
        for(size_t i=0; i<nr_variables(); ++i)
            min_marginal_averaging_step_forward(i);
    }

    inline void bdd_mma_base_vec::min_marginal_averaging_backward()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        for(std::ptrdiff_t i=nr_variables()-1; i>=0; --i)
            min_marginal_averaging_step_backward(i);
    }

    inline void bdd_mma_base_vec::solve(const size_t max_iter)
    {
        std::cout << "initial lower bound = " << lower_bound() << "\n";
        for(size_t iter=0; iter<max_iter; ++iter)
        {
            iteration();
            std::cout << "iteration " << iter << ", lower bound = " << lower_bound() << "\n";
        } 
    } 

    inline void bdd_mma_base_vec::iteration()
    {
        min_marginal_averaging_forward();
        min_marginal_averaging_backward();
        compute_lower_bound();
    }

    inline void bdd_mma_base_vec::backward_run()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        for(std::ptrdiff_t i=bdd_branch_nodes_.size()-1; i>=0; --i)
            bdd_branch_nodes_[i].backward_step();
        compute_lower_bound();
    }

    inline void bdd_mma_base_vec::compute_lower_bound()
    {
        double lb = 0.0;
        for(size_t i=0; i<first_bdd_node_indices_.size(); ++i)
        {
            float bdd_lb = std::numeric_limits<float>::infinity();
            for(size_t j=0; j<first_bdd_node_indices_.size(i); ++j)
                bdd_lb = std::min(bdd_branch_nodes_[first_bdd_node_indices_(i,j)].m, bdd_lb);
            lb += bdd_lb;
        }

        assert(lb >= lower_bound_ - 1e-8);
        lower_bound_ = lb;
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
        std::vector<double> total_min_marginals;
        return total_min_marginals;
        // TODO implement
    } 
}
