#pragma once

#include <vectorclass.h>
#include <tsl/robin_map.h>
#include <iostream>
#include <vector>
#include <array>
#include "bdd_storage.h"
#include "two_dimensional_variable_array.hxx"
#include "time_measure_util.h"

namespace LPMP {

    class bdd_branch_node_8f {
        public:
            constexpr static int N = 8;
            // offsets in sizeof(float) steps are added to the address of the current bdd_branch_node_8f and point to the m-entries
            Vec8f m = 0.0;//std::numeric_limits<float>::infinity();
            Vec8i offset_low = Vec8i(0,1,2,3,4,5,6,7);
            Vec8i offset_high = Vec8i(0,1,2,3,4,5,6,7);
            Vec8f low_cost = 0.0;
            Vec8f high_cost = 0.0;

            constexpr static int inactive_bdd_index = std::numeric_limits<int>::max();
            Vec8i bdd_index = inactive_bdd_index;

            void prepare_forward_step();
            void forward_step();
            void backward_step();

            void min_marginal(float* reduced_min_marginals0, float* reduced_min_marginals1);
            void set_marginal(float* min_marginals0, float* min_marginals1, const std::array<float,2> avg_marginals);

            ~bdd_branch_node_8f()
            {
                static_assert(sizeof(float) == 4, "float must be quadword");
                static_assert(sizeof(Vec8i) == 8*4, "int must be quadword");
            }

            constexpr static int ls = 1073741824;// == std::pow(2,31); // lookup size
    };

    inline void bdd_branch_node_8f::backward_step()
    {
        static Vec8f zero(0.0f);
        static Vec8i iota(0,1,2,3,4,5,6,7);

        Vec8f low_m = lookup<ls>(offset_low, reinterpret_cast<float*>(this));
        //for(int i=0; i<N; ++i)
        //    assert(low_m[i] == *(reinterpret_cast<float*>(this) + offset_low[i]));
        low_m = select(offset_low == iota, zero, low_m);
        Vec8f m0 = low_cost + low_m;

        Vec8f high_m = lookup<ls>(offset_high, reinterpret_cast<float*>(this));
        //for(int i=0; i<N; ++i)
        //    assert(high_m[i] == *(reinterpret_cast<float*>(this) + offset_high[i]));
        high_m = select(offset_high == iota, zero, high_m);
        Vec8f m1 = high_cost + high_m;
        m = min(m0, m1);
    }

    inline void bdd_branch_node_8f::prepare_forward_step()
    {
        static const Vec8i iota(0,1,2,3,4,5,6,7);
        static Vec8f inf(std::numeric_limits<float>::infinity());

        Vec8f low_m = lookup<ls>(offset_low, reinterpret_cast<float*>(this));
        Vec8f low_prep = select(offset_low == iota, low_m, inf);
        scatter(offset_low, ls, low_prep, reinterpret_cast<float*>(this));

        Vec8f high_m = lookup<ls>(offset_high, reinterpret_cast<float*>(this));
        Vec8f high_prep = select(offset_high == iota, high_m, inf);
        scatter(offset_high, ls, high_prep, reinterpret_cast<float*>(this));
    }

    inline void bdd_branch_node_8f::forward_step()
    {
        static const Vec8i iota(0,1,2,3,4,5,6,7);

        // TODO: assert that all offsets are distinct.
        Vec8f m0 = m + low_cost;
        Vec8f low_m = lookup<ls>(offset_low, reinterpret_cast<float*>(this));
        low_m = select(offset_low == iota, low_m, min(m0,low_m));
        scatter(offset_low, ls, low_m, reinterpret_cast<float*>(this));

        Vec8f m1 = m + high_cost;
        Vec8f high_m = lookup<ls>(offset_high, reinterpret_cast<float*>(this));
        high_m = select(offset_high == iota, high_m, min(m1,high_m));//min(m1, high_m);
        scatter(offset_high, ls, high_m, reinterpret_cast<float*>(this)); 
    }

    inline void bdd_branch_node_8f::min_marginal(float* reduced_min_marginals0, float* reduced_min_marginals1)
    {
        static Vec8f zero(0.0f);
        static Vec8i iota(0,1,2,3,4,5,6,7);

        Vec8f low_m = lookup<ls>(offset_low, reinterpret_cast<float*>(this));
        low_m = select(offset_low == iota, zero, low_m);
        Vec8f low_min_marg = m + low_cost + low_m;

        // TODO: is SIMD possible somehow?
        for(int i=0; i<N; ++i)
            reduced_min_marginals0[bdd_index[i]] = std::min(low_min_marg[i], reduced_min_marginals0[bdd_index[i]]);

        Vec8f high_m = lookup<ls>(offset_high, reinterpret_cast<float*>(this));
        high_m = select(offset_high == iota, zero, high_m);
        Vec8f high_min_marg = m + high_cost + high_m;

        for(int i=0; i<N; ++i)
            reduced_min_marginals1[bdd_index[i]] = std::min(high_min_marg[i], reduced_min_marginals1[bdd_index[i]]);

        // only true for bdd bipartite test
        for(int i=0; i<N; ++i)
            if(bdd_index[i] <= 1)
            {
                //std::cout << bdd_index[i] << ", ";
                //assert(std::isfinite(std::min(high_min_marg[i], low_min_marg[i])));
            }
        //std::cout << "\n";
    }

    inline void bdd_branch_node_8f::set_marginal(float* reduced_min_marginals0, float* reduced_min_marginals1, const std::array<float,2> avg_marginals)
    {
        low_cost += -lookup<ls>(bdd_index, reduced_min_marginals0) + avg_marginals[0];
        high_cost += -lookup<ls>(bdd_index, reduced_min_marginals1) + avg_marginals[1];
    }

    class bdd_mma_base_8f {
        public:
            static constexpr size_t N = 8;
            bdd_mma_base_8f(const bdd_storage& bdd_storage_) { init(bdd_storage_); }
            void init(const bdd_storage& bdd_storage_);
            size_t nr_variables() const;
            size_t nr_bdds(const size_t var) const { assert(var < nr_variables()); return nr_bdds_[var]; }

            void forward_step(const size_t var);
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
            std::vector<bdd_branch_node_8f> bdd_branch_nodes_;
            std::vector<size_t> bdd_branch_node_offsets_; // offsets into where bdd branch nodes belonging to a variable start 
            std::vector<size_t> bdd_branch_node_group_offsets_; // offsets into where bdd branch nodes belonging to a variable group start 
            std::vector<size_t> nr_bdds_;

            std::vector<Vec8f> terminal_vals;
            std::vector<Vec8ib> terminal_masks;

            struct first_bdd_index { size_t bdd_node_index; size_t slot; };
            std::vector<first_bdd_index> first_bdd_node_indices_; // used for computing lower bound
            double lower_bound_ = -std::numeric_limits<double>::infinity();
    };

    inline size_t bdd_mma_base_8f::nr_variables() const
    {
        return bdd_branch_node_offsets_.size()-1; 
    }

    inline void bdd_mma_base_8f::init(const bdd_storage& bdd_storage_)
    {
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
            size_t bdd_index;
            size_t slot; 
        };
        std::vector<bdd_index_to_offset> stored_bdd_index_to_bdd_offset(bdd_storage_.bdd_nodes().size());

        for(size_t bdd_index=0; bdd_index<bdd_storage_.nr_bdds(); ++bdd_index)
        {
            const size_t first_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index]; 
            const size_t last_stored_bdd_node = bdd_storage_.bdd_delimiters()[bdd_index+1];
            for(size_t stored_bdd_node_index=first_stored_bdd_node; stored_bdd_node_index<last_stored_bdd_node; ++stored_bdd_node_index)
            {
                const auto& stored_bdd = bdd_storage_.bdd_nodes()[stored_bdd_node_index];
                const size_t v = stored_bdd.variable;
                const size_t bdd_branch_index = bdd_branch_node_offsets_[v] + (bdd_branch_nodes_counter[v] / N);

                const size_t bdd_branch_slot = bdd_branch_nodes_counter[v] % N;
                ++bdd_branch_nodes_counter[v];

                if(v == bdd_storage_.bdd_nodes()[last_stored_bdd_node-1].variable)
                    first_bdd_node_indices_.push_back({bdd_branch_index, bdd_branch_slot});

                if(stored_bdd.low == bdd_storage::bdd_node::terminal_0)
                {
                    bdd_branch_nodes_[bdd_branch_index].offset_low.insert(bdd_branch_slot, bdd_branch_slot);
                    bdd_branch_nodes_[bdd_branch_index].low_cost.insert(bdd_branch_slot, std::numeric_limits<float>::infinity());
                }
                else if(stored_bdd.low == bdd_storage::bdd_node::terminal_1)
                {
                    assert(bdd_branch_nodes_[bdd_branch_index].low_cost[bdd_branch_slot] == 0.0);
                    bdd_branch_nodes_[bdd_branch_index].offset_low.insert(bdd_branch_slot, bdd_branch_slot);
                }
                else
                {
                    assert(bdd_branch_nodes_[bdd_branch_index].low_cost[bdd_branch_slot] == 0.0);
                    const auto [bdd_index, slot] = stored_bdd_index_to_bdd_offset[stored_bdd.low];
                    assert(bdd_index > bdd_branch_index);
                    assert(slot < N);
                    const int offset = std::distance(reinterpret_cast<float*>(&bdd_branch_nodes_[bdd_branch_index]), reinterpret_cast<float*>(&bdd_branch_nodes_[bdd_index].m) + slot);
                    assert(offset > bdd_branch_slot);
                    bdd_branch_nodes_[bdd_branch_index].offset_low.insert(bdd_branch_slot, offset);
                }

                if(stored_bdd.high == bdd_storage::bdd_node::terminal_0)
                {
                    bdd_branch_nodes_[bdd_branch_index].offset_high.insert(bdd_branch_slot, bdd_branch_slot);
                    bdd_branch_nodes_[bdd_branch_index].high_cost.insert(bdd_branch_slot, std::numeric_limits<float>::infinity());
                }
                else if(stored_bdd.high == bdd_storage::bdd_node::terminal_1)
                {
                    assert(bdd_branch_nodes_[bdd_branch_index].high_cost[bdd_branch_slot] == 0.0);
                    bdd_branch_nodes_[bdd_branch_index].offset_high.insert(bdd_branch_slot, bdd_branch_slot);
                }
                else
                {
                    assert(bdd_branch_nodes_[bdd_branch_index].high_cost[bdd_branch_slot] == 0.0);
                    const auto [bdd_index, slot] = stored_bdd_index_to_bdd_offset[stored_bdd.high];
                    assert(bdd_index > bdd_branch_index); 
                    assert(slot < N);
                    const int offset = std::distance(reinterpret_cast<float*>(&bdd_branch_nodes_[bdd_branch_index]), reinterpret_cast<float*>(&bdd_branch_nodes_[bdd_index]) + slot);
                    assert(offset > bdd_branch_slot);
                    bdd_branch_nodes_[bdd_branch_index].offset_high.insert(bdd_branch_slot, offset);
                }

                assert(bdd_index <= std::numeric_limits<int>::max()); // TODO: write alternative mechanism for this case
                bdd_branch_nodes_[bdd_branch_index].bdd_index.insert(bdd_branch_slot, bdd_index);

                stored_bdd_index_to_bdd_offset[stored_bdd_node_index] = bdd_index_to_offset{bdd_branch_index, bdd_branch_slot};
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
                    //std::cout << bdd_index << ", ";
                    if(bdd_index != bdd_branch_node_8f::inactive_bdd_index && bdd_index_redux.count(bdd_index) == 0)
                        bdd_index_redux.insert({bdd_index, bdd_index_redux.size()}); 
                }
                //std::cout << "\n";
            }
            for(size_t vec_idx=bdd_branch_node_offsets_[i]; vec_idx<bdd_branch_node_offsets_[i+1]; ++vec_idx)
            {
                auto& bdd_vec = bdd_branch_nodes_[vec_idx]; 
                for(size_t slot=0; slot<N; ++slot)
                {
                    if(bdd_vec.bdd_index[slot] != bdd_branch_node_8f::inactive_bdd_index)
                        bdd_vec.bdd_index.insert(slot, bdd_index_redux.find(bdd_vec.bdd_index[slot])->second);
                    else
                        bdd_vec.bdd_index.insert(slot, bdd_index_redux.size());
                    //std::cout << bdd_vec.bdd_index[slot] << ", ";
                }
                //std::cout << "\n";
            }
            nr_bdds_.push_back(bdd_index_redux.size());
        }
    }

    inline void bdd_mma_base_8f::forward_step(const size_t var)
    {
        assert(var < nr_variables());
        //std::cout << "forward step for var " << var << "\n";

        // TODO: only for bdd bipartite matching test
        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            for(int slot=0; slot<N; ++slot)
                if(bdd_branch_nodes_[i].bdd_index[slot] < 2)
                    assert(std::isfinite(bdd_branch_nodes_[i].m[slot])); 

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].prepare_forward_step();

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].forward_step();
    }

    // TODO: count backwards in loop?
    inline void bdd_mma_base_8f::backward_step(const size_t var)
    {
        assert(var < nr_variables());

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].backward_step();
    }

    inline float bdd_mma_base_8f::average_marginals(float* marginals, const size_t nr_marginals)
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

    inline void bdd_mma_base_8f::min_marginal_averaging_step_forward(const size_t var)
    {
        // TODO: pad to eight so that SIMD instructions can be applied
        // +1 below for inactive bdd index
        float min_marginals0[nr_bdds(var)+1];
        std::fill(min_marginals0, min_marginals0 + nr_bdds(var), std::numeric_limits<float>::infinity());
        float min_marginals1[nr_bdds(var)+1];
        std::fill(min_marginals1, min_marginals1 + nr_bdds(var), std::numeric_limits<float>::infinity());

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].min_marginal(min_marginals0, min_marginals1);

        std::array<float,2> avg_marginals = {average_marginals(min_marginals0, nr_bdds(var)), average_marginals(min_marginals1, nr_bdds(var))};

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].set_marginal(min_marginals0, min_marginals1, avg_marginals);

        forward_step(var);
    }

    inline void bdd_mma_base_8f::min_marginal_averaging_step_backward(const size_t var)
    {
        // TODO: pad to four so that SIMD instructions can be applied?
        const size_t _nr_bdds = nr_bdds(var);
        float min_marginals0[_nr_bdds+1];
        std::fill(min_marginals0, min_marginals0 + _nr_bdds, std::numeric_limits<float>::infinity());
        float min_marginals1[_nr_bdds+1];
        std::fill(min_marginals1, min_marginals1 + _nr_bdds, std::numeric_limits<float>::infinity());

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].min_marginal(min_marginals0, min_marginals1);

        std::array<float,2> avg_marginals = {average_marginals(min_marginals0, _nr_bdds), average_marginals(min_marginals1, _nr_bdds)};

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].set_marginal(min_marginals0, min_marginals1, avg_marginals);

        backward_step(var);
    }

    inline void bdd_mma_base_8f::min_marginal_averaging_forward()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        for(size_t i=0; i<nr_variables(); ++i)
            min_marginal_averaging_step_forward(i);
    }

    inline void bdd_mma_base_8f::min_marginal_averaging_backward()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        for(std::ptrdiff_t i=nr_variables()-1; i>=0; --i)
            min_marginal_averaging_step_backward(i);
    }

    inline void bdd_mma_base_8f::solve(const size_t max_iter)
    {
        std::cout << "initial lower bound = " << lower_bound() << "\n";
        for(size_t iter=0; iter<max_iter; ++iter)
        {
            iteration();
            std::cout << "iteration " << iter << ", lower bound = " << lower_bound() << "\n";
        } 
    }

    inline void bdd_mma_base_8f::iteration()
    {
        min_marginal_averaging_forward();
        min_marginal_averaging_backward();
        compute_lower_bound();
    }

    inline void bdd_mma_base_8f::backward_run()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        for(std::ptrdiff_t i=bdd_branch_nodes_.size()-1; i>=0; --i)
            bdd_branch_nodes_[i].backward_step();
        compute_lower_bound();
    }

    inline void bdd_mma_base_8f::compute_lower_bound()
    {
        double lb = 0.0;
        for(const auto [bdd_node_index, bdd_node_slot] : first_bdd_node_indices_)
        {
            assert(std::isfinite(bdd_branch_nodes_[bdd_node_index].m[bdd_node_slot])); 
            lb += bdd_branch_nodes_[bdd_node_index].m[bdd_node_slot]; // only works if BDDs have one root node (not necessarily so for split BDDs.
        }

        lower_bound_ = lb;
    } 

    inline void bdd_mma_base_8f::set_cost(const double c, const size_t var)
    {
        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].high_cost += c / float(nr_bdds(var));
    }
}
