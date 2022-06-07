#pragma once

// TODO: rename file and classes

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include <vector>
#include <array>
#include <cstddef>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <unordered_set>
#include <queue>
#include <stack>
#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include "time_measure_util.h"
#include "bdd_filtration.hxx"
#include "bdd_manager/bdd.h"
#include "min_marginal_utils.h"
#include <iostream>
#include <stack>

namespace LPMP {

    // bdds are stored in variable groups. Each variable group is a set of variables that can be processed in parallel.
    template<typename BDD_BRANCH_NODE>
        class bdd_mma_base {
            public:
                using value_type = typename BDD_BRANCH_NODE::value_type;
                bdd_mma_base() {}
                bdd_mma_base(BDD::bdd_collection& bdd_col) { add_bdds(bdd_col); } 
                size_t nr_variables() const;
                size_t nr_variable_groups() const;
                size_t nr_bdd_vectors(const size_t var_group) const;
                size_t nr_bdds(const size_t var) const { assert(var < nr_variables()); return nr_bdds_[var]; }
                size_t nr_bdd_nodes() const { return bdd_branch_nodes_.size(); }
                size_t nr_bdd_nodes(const size_t v) const { assert(v < nr_variables()); return bdd_branch_node_offsets_[v+1] - bdd_branch_node_offsets_[v]; }
                size_t nr_bdds() const { return first_bdd_node_indices_.size(); }
                const std::vector<size_t> nr_bdds_vector() const { return nr_bdds_; }

                void forward_step(const size_t var_group);
                void min_marginal_averaging_forward();
                void min_marginal_averaging_step_forward(const size_t var);

                void backward_step(const size_t var_group);
                void min_marginal_averaging_backward();
                void min_marginal_averaging_step_backward(const size_t var);

                std::array<value_type,2> average_marginals(std::array<value_type,2>* marginals, const size_t nr_marginals);

                void iteration();
                void backward_run();
                void forward_run();

                void compute_lower_bound_after_forward_pass(); 
                void compute_lower_bound_after_backward_pass(); 

                two_dim_variable_array<std::array<double,2>> min_marginals();
                void solve(const size_t max_iter, const double tolerance, const double time_limit); 
                double lower_bound();
                void update_cost(const double lo_cost, const double hi_cost, const size_t var);
                void fix_variable(const size_t var, const bool value);

                // get variable costs from bdd
                std::vector<value_type> get_costs(const size_t bdd_nr);
                // add costs from cost iterator to costs of bdd. Assume that variables given are subset of variables of bdd
                template<typename COST_ITERATOR, typename VARIABLE_ITERATOR>
                    void update_bdd_costs(const size_t bdd_nr,
                            COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
                            VARIABLE_ITERATOR variable_begin, VARIABLE_ITERATOR variable_end);
                //template<typename COST_ITERATOR>
                //    void update_costs(COST_ITERATOR cost_begin, COST_ITERATOR cost_end);
                template<typename REAL>
                    void update_costs(const two_dim_variable_array<std::array<REAL,2>>& delta);

                void add_to_constant(const double c) { constant_ += c; }
                double constant() const { return constant_; }

                template<typename ITERATOR>
                    void update_arc_costs(const size_t first_node, ITERATOR begin, ITERATOR end);
                void transfer_cost(const size_t from_bdd_nr, const size_t to_bdd_nr);
                void get_arc_marginals(const size_t first_node, const size_t last_node, std::vector<double>& arc_marginals);

                std::array<size_t,2> bdd_branch_node_offset(const size_t var, const size_t bdd_index) const;
                size_t variable(const size_t bdd_offset) const;

                // record for each bdd and each of its variables whether the solution is feasible for it
                template<typename ITERATOR>
                    two_dim_variable_array<char> bdd_feasibility(ITERATOR sol_begin, ITERATOR sol_end) const;

                void export_graphviz(const char* filename);
                void export_graphviz(const std::string& filename);
                template<typename STREAM>
                    void export_graphviz(STREAM& s, const size_t bdd_nr);

            protected:
                std::vector<BDD_BRANCH_NODE> bdd_branch_nodes_;
                std::vector<size_t> bdd_branch_node_offsets_; // offsets into where bdd branch nodes belonging to a variable start 
                // TODO: remove all group stuff
                std::vector<size_t> bdd_branch_node_group_offsets_; // offsets into where bdd branch nodes belonging to a variable group start 
                std::vector<size_t> nr_bdds_; // nr bdds per variable
                // TODO: use std::vector<std::array<size_t,2>> for range of first resp. last bdd nodes of each BDD. Also sort them for faster access time
                two_dim_variable_array<size_t> first_bdd_node_indices_;  // used for computing lower bound
                two_dim_variable_array<size_t> last_bdd_node_indices_;  // used for computing lower bound
                double lower_bound_ = -std::numeric_limits<double>::infinity();
                enum class lower_bound_state {
                    valid,
                    invalid
                } lower_bound_state_ = lower_bound_state::invalid;

                enum class message_passing_state {
                    after_forward_pass,
                    after_backward_pass,
                    none 
                } message_passing_state_ = message_passing_state::none;

                double constant_ = 0.0;

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
                //std::vector<float> min_marginal_differences(const float eps);
                // min marginals for each variable and each bdd
                //two_dim_variable_array<std::array<float,2>> min_marginals();
                // export BDDs that cover the given variables
                // TODO: unify with init?
                void add_bdds(BDD::bdd_collection& bdd_col);
                template<typename BDD_NR_ITERATOR>
                    std::vector<size_t> add_bdds(BDD::bdd_collection& bdd_col, BDD_NR_ITERATOR bdd_nrs_begin, BDD_NR_ITERATOR bdd_nrs_end);

                std::vector<size_t> variables(const size_t bdd_idx);
                std::vector<bdd_branch_instruction<float,uint32_t>> export_bdd(const size_t bdd_idx);
                size_t export_bdd(BDD::bdd_collection& bdd_col, const size_t bdd_idx);
                std::tuple<BDD::node_ref, std::vector<size_t>> export_bdd(BDD::bdd_mgr& bdd_mgr, const size_t bdd_idx);
        };

    template<typename BDD_BRANCH_NODE>
    size_t bdd_mma_base<BDD_BRANCH_NODE>::nr_variables() const
    {
        return nr_bdds_.size(); 
    }

    template<typename BDD_BRANCH_NODE>
    size_t bdd_mma_base<BDD_BRANCH_NODE>::nr_variable_groups() const
    {
        throw std::runtime_error("not usable");
        return bdd_branch_node_group_offsets_.size()-1; 
    } 

    template<typename BDD_BRANCH_NODE>
    size_t bdd_mma_base<BDD_BRANCH_NODE>::nr_bdd_vectors(const size_t var) const
    {
        assert(var < nr_variables());
        return bdd_branch_node_offsets_[var+1] - bdd_branch_node_offsets_[var];
    } 

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::forward_step(const size_t var)
    {
        assert(var < nr_variables());

        assert(var+1 < bdd_branch_node_offsets_.size());
        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
        {
            assert(i < bdd_branch_nodes_.size());
            bdd_branch_nodes_[i].prepare_forward_step();
        }

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].forward_step();
    }


    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::backward_step(const size_t var)
    {
        assert(var < nr_variables());

        // TODO: count backwards in loop?
        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].backward_step();
    } 

    template<typename BDD_BRANCH_NODE>
    std::array<typename BDD_BRANCH_NODE::value_type,2> bdd_mma_base<BDD_BRANCH_NODE>::average_marginals(std::array<typename BDD_BRANCH_NODE::value_type,2>* marginals, const size_t nr_marginals)
    {
        std::array<value_type,2> avg_margs = {0.0,0.0};
        for(size_t i=0; i<nr_marginals; ++i)
        {
            //assert(std::isfinite(marginals[i][0])); // need not hold true after fix_variable
            //assert(std::isfinite(marginals[i][1]));
            avg_margs[0] += marginals[i][0];
            avg_margs[1] += marginals[i][1];
        }
        avg_margs[0] /= value_type(nr_marginals);
        avg_margs[1] /= value_type(nr_marginals);
        //assert(std::isfinite(avg_margs[0]));
        //assert(std::isfinite(avg_margs[1]));
        return avg_margs;
    } 

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::min_marginal_averaging_step_forward(const size_t var)
    {
        assert(var < nr_variables());
        const size_t _nr_bdds = nr_bdds(var);
        if(_nr_bdds == 0)
            return;

        std::array<value_type,2> min_marginals[_nr_bdds];
        std::fill(min_marginals, min_marginals + _nr_bdds, std::array<value_type,2>{std::numeric_limits<value_type>::infinity(), std::numeric_limits<value_type>::infinity()});

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].min_marginal(min_marginals);

        std::array<value_type,2> avg_marginals = average_marginals(min_marginals, _nr_bdds);

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].set_marginal(min_marginals, avg_marginals);

        forward_step(var);
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::min_marginal_averaging_step_backward(const size_t var)
    {
        // TODO: pad to four so that SIMD instructions can be applied?
        const size_t _nr_bdds = nr_bdds(var);
        if(_nr_bdds == 0)
            return;
        std::array<value_type,2> min_marginals[_nr_bdds];
        std::fill(min_marginals, min_marginals + _nr_bdds, std::array<value_type,2>{std::numeric_limits<value_type>::infinity(), std::numeric_limits<value_type>::infinity()});

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            bdd_branch_nodes_[i].min_marginal(min_marginals);

        std::array<value_type,2> avg_marginals = average_marginals(min_marginals, _nr_bdds);

        //std::cout << "backward step for var " << var << ", offset = " << bdd_branch_node_offsets_[var] << ", #nodes = " << bdd_branch_nodes_.size() << "\n";
        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
        {
            bdd_branch_nodes_[i].set_marginal(min_marginals, avg_marginals);
            bdd_branch_nodes_[i].backward_step();
        }
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::min_marginal_averaging_forward()
    {
        if(message_passing_state_ != message_passing_state::after_backward_pass)
            backward_run();
        message_passing_state_ = message_passing_state::none;
        lower_bound_state_ = lower_bound_state::invalid;
        //MEASURE_FUNCTION_EXECUTION_TIME;
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        for(size_t i=0; i<nr_variables(); ++i)
            min_marginal_averaging_step_forward(i);
        message_passing_state_ = message_passing_state::after_forward_pass;
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::min_marginal_averaging_backward()
    {
        if(message_passing_state_ != message_passing_state::after_forward_pass)
            forward_run();
        message_passing_state_ = message_passing_state::none;
        lower_bound_state_ = lower_bound_state::invalid;
        //MEASURE_FUNCTION_EXECUTION_TIME;
        for(std::ptrdiff_t i=nr_variables()-1; i>=0; --i)
            min_marginal_averaging_step_backward(i);
        message_passing_state_ = message_passing_state::after_backward_pass;
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::iteration()
    {
        min_marginal_averaging_forward();
        min_marginal_averaging_backward();
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::backward_run()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        if(message_passing_state_ == message_passing_state::after_backward_pass)
            return;
        // TODO: if we already have done a backward_run, we do not need to do it again. Check state!
        message_passing_state_ = message_passing_state::none;
        for(std::ptrdiff_t i=bdd_branch_nodes_.size()-1; i>=0; --i)
            bdd_branch_nodes_[i].backward_step();
        message_passing_state_ = message_passing_state::after_backward_pass;
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::forward_run()
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        if(message_passing_state_ == message_passing_state::after_forward_pass)
            return;
        // TODO: if we already have done a forward_run, we do not need to do it again. Check state!
        message_passing_state_ = message_passing_state::none;
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        for(size_t i=0; i<nr_variables(); ++i)
            forward_step(i);
        message_passing_state_ = message_passing_state::after_forward_pass;
    }

    template<typename BDD_BRANCH_NODE>
    double bdd_mma_base<BDD_BRANCH_NODE>::lower_bound()
    {
        if(lower_bound_state_ == lower_bound_state::valid)
            return lower_bound_ + constant_;
        if(message_passing_state_ == message_passing_state::after_forward_pass)
            compute_lower_bound_after_forward_pass();
        else if(message_passing_state_ == message_passing_state::after_backward_pass)
            compute_lower_bound_after_backward_pass();
        else
        {
            backward_run();
            compute_lower_bound_after_backward_pass();
        }
        lower_bound_state_ = lower_bound_state::valid;
        return lower_bound_ + constant_;
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::compute_lower_bound_after_backward_pass()
    {
        assert(message_passing_state_ == message_passing_state::after_backward_pass);
        double lb = 0.0;
        for(size_t i=0; i<first_bdd_node_indices_.size(); ++i)
        {
            value_type bdd_lb = std::numeric_limits<value_type>::infinity();
            for(size_t j=0; j<first_bdd_node_indices_.size(i); ++j)
                bdd_lb = std::min(bdd_branch_nodes_[first_bdd_node_indices_(i,j)].m, bdd_lb);
            lb += bdd_lb;
        }

        //assert(lb.value() >= lower_bound_ - 1e-6);
        //lower_bound_ = lb.value();
        lower_bound_ = lb;
        lower_bound_state_ = lower_bound_state::valid;
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::compute_lower_bound_after_forward_pass()
    {
        assert(message_passing_state_ == message_passing_state::after_forward_pass);
        double lb = 0.0;
        for(size_t i=0; i<last_bdd_node_indices_.size(); ++i)
        {
            value_type bdd_lb = std::numeric_limits<value_type>::infinity();
            for(size_t j=0; j<last_bdd_node_indices_.size(i); ++j)
            {
                const auto& bdd_node = bdd_branch_nodes_[last_bdd_node_indices_(i,j)];
                assert(bdd_node.offset_low == BDD_BRANCH_NODE::terminal_0_offset || bdd_node.offset_low == BDD_BRANCH_NODE::terminal_1_offset);
                assert(bdd_node.offset_high == BDD_BRANCH_NODE::terminal_0_offset || bdd_node.offset_high == BDD_BRANCH_NODE::terminal_1_offset);
                //bdd_lb = std::min({bdd_node.m + bdd_node.low_cost, bdd_node.m + bdd_node.high_cost, bdd_lb});
                const auto mm = bdd_node.min_marginals();
                bdd_lb = std::min({mm[0], mm[1], bdd_lb});
            }
            lb += bdd_lb;
        }

        //assert(lb.value() >= lower_bound_ - 1e-6);
        //lower_bound_ = lb.value();
        lower_bound_ = lb;
        lower_bound_state_ = lower_bound_state::valid;
    }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base<BDD_BRANCH_NODE>::update_cost(const double lo_cost, const double hi_cost, const size_t var)
        {
            assert(nr_bdds(var) > 0);
            assert(std::isfinite(std::min(lo_cost, hi_cost)));

            lower_bound_state_ = lower_bound_state::invalid;
            message_passing_state_ = message_passing_state::none;

            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
            {
                bdd_branch_nodes_[i].low_cost += lo_cost / value_type(nr_bdds(var));
                bdd_branch_nodes_[i].high_cost += hi_cost / value_type(nr_bdds(var));
            }
        }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::fix_variable(const size_t var, const bool value)
    {
        assert(nr_bdds(var) > 0);

        lower_bound_state_ = lower_bound_state::invalid;
        message_passing_state_ = message_passing_state::none;

        for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
        {
            if(value == 0)
                bdd_branch_nodes_[i].high_cost = std::numeric_limits<value_type>::infinity();
            else
                bdd_branch_nodes_[i].low_cost = std::numeric_limits<value_type>::infinity();
        }
    }

    template<typename BDD_BRANCH_NODE>
    template<typename ITERATOR>
        void bdd_mma_base<BDD_BRANCH_NODE>::update_arc_costs(const size_t first_node, ITERATOR begin, ITERATOR end)
        {
            assert(std::distance(begin,end) % 2 == 0);
            assert(first_node + std::distance(begin,end)/2 <= bdd_branch_nodes_.size());
            size_t l=first_node;
            for(auto it=begin; it!=end; ++l)
            {
                assert(bdd_branch_nodes_[first_node].bdd_index == bdd_branch_nodes_[l].bdd_index);
                if(bdd_branch_nodes_[l].offset_low != BDD_BRANCH_NODE::terminal_0_offset)
                    bdd_branch_nodes_[l].low_cost += *it; 
                ++it;
                if(bdd_branch_nodes_[l].offset_high != BDD_BRANCH_NODE::terminal_0_offset)
                    bdd_branch_nodes_[l].high_cost += *it; 
                ++it;
            }
        }

    template<typename BDD_BRANCH_NODE>
    template<typename LAMBDA>
        void bdd_mma_base<BDD_BRANCH_NODE>::visit_nodes(const size_t bdd_nr, LAMBDA&& f)
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

                auto calculate_offset = [&](BDD_BRANCH_NODE& bdd, const size_t offset) {
                    assert(offset != BDD_BRANCH_NODE::terminal_0_offset && offset != BDD_BRANCH_NODE::terminal_1_offset);
                    assert(&bdd >= &bdd_branch_nodes_[0]);
                    assert(std::distance(&bdd_branch_nodes_[0], &bdd) < bdd_branch_nodes_.size());
                    const size_t translated_offset = std::distance(&bdd_branch_nodes_[0], bdd.address(offset)); 
                    assert(translated_offset < bdd_branch_nodes_.size());
                    return translated_offset;
                };

                auto& bdd = bdd_branch_nodes_[i];
                if(bdd.offset_low != BDD_BRANCH_NODE::terminal_0_offset && bdd.offset_low != BDD_BRANCH_NODE::terminal_1_offset)
                    dq.push_back(calculate_offset(bdd, bdd.offset_low));
                if(bdd.offset_high != BDD_BRANCH_NODE::terminal_0_offset && bdd.offset_high != BDD_BRANCH_NODE::terminal_1_offset)
                    dq.push_back(calculate_offset(bdd, bdd.offset_high));
            } 
        }

    template<typename BDD_BRANCH_NODE>
    std::vector<typename BDD_BRANCH_NODE::value_type> bdd_mma_base<BDD_BRANCH_NODE>::get_costs(const size_t bdd_nr)
    {
        std::vector<value_type> costs;

        size_t cur_var = std::numeric_limits<size_t>::max();
        value_type hi_cost = std::numeric_limits<value_type>::infinity();
        value_type lo_cost = std::numeric_limits<value_type>::infinity();
        visit_nodes(bdd_nr, [&](const size_t i) {
                const BDD_BRANCH_NODE& bdd = bdd_branch_nodes_[i];
                if(cur_var != this->variable(i) && cur_var != std::numeric_limits<size_t>::max())
                {
                costs.push_back(hi_cost - lo_cost); 
                hi_cost = std::numeric_limits<value_type>::infinity();
                lo_cost = std::numeric_limits<value_type>::infinity();
                }
                hi_cost = std::min(hi_cost, bdd.high_cost);
                lo_cost = std::min(lo_cost, bdd.low_cost);
                cur_var = this->variable(i); 
                });
        costs.push_back(hi_cost - lo_cost);

        for(const value_type x : costs)
            assert(std::isfinite(x));
        assert(costs.size() == variables(bdd_nr).size());

        return costs; 
    }

    template<typename BDD_BRANCH_NODE>
    template<typename COST_ITERATOR, typename VARIABLE_ITERATOR>
        void bdd_mma_base<BDD_BRANCH_NODE>::update_bdd_costs(const size_t bdd_nr,
                COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
                VARIABLE_ITERATOR variable_begin, VARIABLE_ITERATOR variable_end)
        {
            assert(std::distance(cost_begin, cost_end) == std::distance(variable_begin, variable_end));
            assert(std::is_sorted(variable_begin, variable_end));

            lower_bound_ = -std::numeric_limits<double>::infinity();
            lower_bound_state_ = lower_bound_state::invalid;
            message_passing_state_ = message_passing_state::none;

            auto cost_it = cost_begin;
            auto var_it = variable_begin;

            visit_nodes(bdd_nr, [&](const size_t i) {
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

    /*
    template<typename BDD_BRANCH_NODE>
    template<typename COST_ITERATOR>
        void bdd_mma_base<BDD_BRANCH_NODE>::update_costs(COST_ITERATOR cost_begin, COST_ITERATOR cost_end)
        {
            assert(std::distance(cost_begin, cost_end) == nr_variables());
            for(size_t var=0; var<nr_variables(); ++var)
                for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                    bdd_branch_nodes_[i].high_cost += *(cost_begin+i) / value_type(nr_bdds(var));
        }
        */


    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::get_arc_marginals(const size_t first_node, const size_t last_node, std::vector<double>& arc_marginals)
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

    template<typename BDD_BRANCH_NODE>
    std::array<size_t,2> bdd_mma_base<BDD_BRANCH_NODE>::bdd_branch_node_offset(const size_t var, const size_t bdd_index) const
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

    template<typename BDD_BRANCH_NODE>
    size_t bdd_mma_base<BDD_BRANCH_NODE>::variable(const size_t bdd_offset) const
    {
        assert(bdd_offset < bdd_branch_nodes_.size());
        if(bdd_branch_instruction_variables_.size() != bdd_branch_nodes_.size())
            bdd_branch_instruction_variables_ = compute_bdd_branch_instruction_variables();
        assert(bdd_branch_instruction_variables_.size() == bdd_branch_nodes_.size());
        return bdd_branch_instruction_variables_[bdd_offset]; 
    }

    template<typename BDD_BRANCH_NODE>
    two_dim_variable_array<std::array<double,2>> bdd_mma_base<BDD_BRANCH_NODE>::min_marginals()
    {
        std::cout << "[bdd mma base] compute min marginals\n";
        if(message_passing_state_ != message_passing_state::after_backward_pass)
            this->backward_run();
        message_passing_state_ = message_passing_state::none;
        // prepare forward run
        for(size_t bdd_index=0; bdd_index<first_bdd_node_indices_.size(); ++bdd_index)
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_index); ++j)
                bdd_branch_nodes_[first_bdd_node_indices_(bdd_index,j)].m = 0.0;

        two_dim_variable_array<std::array<double,2>> mms;

        for(size_t var=0; var<this->nr_variables(); ++var)
        {
            const size_t _nr_bdds = nr_bdds(var);
            std::array<value_type,2> min_marginals[_nr_bdds];
            std::fill(min_marginals, min_marginals + _nr_bdds, std::array<value_type,2>{std::numeric_limits<value_type>::infinity(), std::numeric_limits<value_type>::infinity()});
            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                bdd_branch_nodes_[i].min_marginal(min_marginals);

            std::array<double,2> min_marginals_double[_nr_bdds];
            for(size_t i=0; i<_nr_bdds; ++i)
            {
                //assert(std::isfinite(min_marginals[i][0]));
                //assert(std::isfinite(min_marginals[i][1]));
                min_marginals_double[i][0] = min_marginals[i][0];
                min_marginals_double[i][1] = min_marginals[i][1];
            }
            this->forward_step(var);

            mms.push_back(min_marginals_double, min_marginals_double + _nr_bdds); 
        }

        message_passing_state_ = message_passing_state::after_forward_pass;
        return mms;
    }

    template<typename BDD_BRANCH_NODE>
        template<typename REAL>
        void bdd_mma_base<BDD_BRANCH_NODE>::update_costs(const two_dim_variable_array<std::array<REAL,2>>& delta)
        {
            lower_bound_state_ = lower_bound_state::invalid;
            message_passing_state_ = message_passing_state::none;

            assert(delta.size() == nr_variables());

            for(size_t var=0; var<delta.size(); ++var)
            {
                assert(delta.size(var) == nr_bdds(var));
                for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                {
                    auto& bdd = bdd_branch_nodes_[i];
                    const size_t idx = bdd.bdd_index;
                    bdd.low_cost += delta(var,idx)[0];
                    bdd.high_cost += delta(var,idx)[1];
                }
            }
        }

    template<typename BDD_BRANCH_NODE>
    std::vector<size_t> bdd_mma_base<BDD_BRANCH_NODE>::compute_bdd_branch_instruction_variables() const
    {
        std::vector<size_t> vars;
        vars.reserve(bdd_branch_nodes_.size());
        for(size_t var=0; var<nr_variables(); ++var)
            for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                vars.push_back(var);
        return vars;
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::tighten_bdd(const float epsilon)
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
            const value_type bdd_lb = [&]() {
                value_type lb = std::numeric_limits<value_type>::infinity();
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
                    value_type check_lb = std::numeric_limits<value_type>::infinity();
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

                auto calculate_offset = [&](const value_type min_marg, const size_t offset) {
                    assert(min_marg >= bdd_lb - 1e-3);
                    // check if low resp. high arc min-marginal is within epsilon of lower bound
                    if(min_marg <= bdd_lb + epsilon || offset == BDD_BRANCH_NODE::terminal_0_offset) // leave arc as is
                    {
                        if(offset == BDD_BRANCH_NODE::terminal_0_offset)
                            return bdd_node::terminal_0;
                        else if(offset == BDD_BRANCH_NODE::terminal_1_offset)
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
                        if(offset == BDD_BRANCH_NODE::terminal_0_offset)
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

    template<typename BDD_BRANCH_NODE>
    two_dim_variable_array<size_t> bdd_mma_base<BDD_BRANCH_NODE>::tighten_bdd_groups(const std::vector<char>& tighten_variables)
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
                if(bdd_node.offset_low != BDD_BRANCH_NODE::terminal_0_offset && bdd_node.offset_low != BDD_BRANCH_NODE::terminal_1_offset)
                {
                    const size_t low_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_low)); 
                    dq.push_back(low_offset);
                }
                if(bdd_node.offset_high != BDD_BRANCH_NODE::terminal_0_offset && bdd_node.offset_high != BDD_BRANCH_NODE::terminal_1_offset)
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
                if(bdd_node.offset_low != BDD_BRANCH_NODE::terminal_0_offset && bdd_node.offset_low != BDD_BRANCH_NODE::terminal_1_offset)
                {
                    const size_t low_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_low)); 
                    dq.push_back(low_offset);
                }
                if(bdd_node.offset_high != BDD_BRANCH_NODE::terminal_0_offset && bdd_node.offset_high != BDD_BRANCH_NODE::terminal_1_offset)
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

    template<typename BDD_BRANCH_NODE>
    std::vector<size_t> bdd_mma_base<BDD_BRANCH_NODE>::variables(const size_t bdd_idx)
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
            if(bdd_node.offset_low != BDD_BRANCH_NODE::terminal_0_offset && bdd_node.offset_low != BDD_BRANCH_NODE::terminal_1_offset)
            {
                const size_t low_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_low)); 
                dq.push_back(low_offset);
            }
            if(bdd_node.offset_high != BDD_BRANCH_NODE::terminal_0_offset && bdd_node.offset_high != BDD_BRANCH_NODE::terminal_1_offset)
            {
                const size_t high_offset = std::distance(&bdd_branch_nodes_[0], bdd_node.address(bdd_node.offset_high)); 
                dq.push_back(high_offset);
            }
        } 
        std::vector<size_t> var_vec(vars.begin(), vars.end());
        std::sort(var_vec.begin(), var_vec.end());
        return var_vec;
    }

    template<typename BDD_BRANCH_NODE>
    std::vector<bdd_branch_instruction<float,uint32_t>> bdd_mma_base<BDD_BRANCH_NODE>::export_bdd(const size_t bdd_idx)
    {
        assert(bdd_idx < nr_bdds());
        std::vector<bdd_branch_instruction<float,uint32_t>> bdds;
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
                if(offset == BDD_BRANCH_NODE::terminal_0_offset)
                    return bdd_node::terminal_0;
                else if(offset == BDD_BRANCH_NODE::terminal_1_offset)
                    return bdd_node::terminal_1;
                else
                {
                    const size_t translated_offset = std::distance(&bdd_branch_nodes_[0], bdd.address(offset)); 
                    dq.push_back(translated_offset);
                    return translated_offset;
                }
            };

            bdd_branch_instruction<float,uint32_t> bdd;
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

    template<typename BDD_BRANCH_NODE>
    size_t bdd_mma_base<BDD_BRANCH_NODE>::export_bdd(BDD::bdd_collection& bdd_col, const size_t bdd_idx)
    {
        const size_t new_bdd_nr = bdd_col.new_bdd();
        std::unordered_map<size_t, BDD::bdd_collection_node> bdd_col_nodes; // position in bdd_branch_nodes to node in bdd collection

        auto calculate_offset = [&](BDD_BRANCH_NODE& bdd, const size_t offset) {
            // check if low resp. high arc min-marginal is within epsilon of lower bound
            if(offset == BDD_BRANCH_NODE::terminal_0_offset)
                return bdd_node::terminal_0;
            else if(offset == BDD_BRANCH_NODE::terminal_1_offset)
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

    template<typename BDD_BRANCH_NODE>
    std::tuple<BDD::node_ref, std::vector<size_t>> bdd_mma_base<BDD_BRANCH_NODE>::export_bdd(BDD::bdd_mgr& bdd_mgr, const size_t bdd_idx)
    {
        assert(bdd_idx < nr_bdds());

        // TODO: this should be its own function
        auto calculate_offset = [&](BDD_BRANCH_NODE& bdd, const size_t offset) {
            assert(offset != BDD_BRANCH_NODE::terminal_0_offset && offset != BDD_BRANCH_NODE::terminal_1_offset);
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

            if(instr.offset_low != BDD_BRANCH_NODE::terminal_0_offset && instr.offset_low != BDD_BRANCH_NODE::terminal_1_offset)
                dq.push_back(calculate_offset(instr, instr.offset_low));

            if(instr.offset_high != BDD_BRANCH_NODE::terminal_0_offset && instr.offset_high != BDD_BRANCH_NODE::terminal_1_offset)
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
            BDD_BRANCH_NODE& instr = bdd_branch_nodes_[i];
            const size_t var = variable(i);
            auto get_node_ref = [&](BDD_BRANCH_NODE& bdd, const size_t offset) {
                if(offset == BDD_BRANCH_NODE::terminal_0_offset)
                    return bdd_mgr.botsink();
                if(offset == BDD_BRANCH_NODE::terminal_1_offset)
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

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::add_bdds(BDD::bdd_collection& bdd_col)
    {
        std::vector<size_t> bdd_nrs(bdd_col.nr_bdds());
        std::iota(bdd_nrs.begin(), bdd_nrs.end(), 0);
        add_bdds(bdd_col, bdd_nrs.begin(), bdd_nrs.end());
    }

    template<typename BDD_BRANCH_NODE>
        template<typename BDD_NR_ITERATOR>
        std::vector<size_t> bdd_mma_base<BDD_BRANCH_NODE>::add_bdds(BDD::bdd_collection& bdd_col, BDD_NR_ITERATOR bdd_nrs_begin, BDD_NR_ITERATOR bdd_nrs_end)
        {
            for(auto bdd_it=bdd_nrs_begin; bdd_it!=bdd_nrs_end; ++bdd_it)
                assert(bdd_col.variables_sorted(*bdd_it));

            message_passing_state_ = message_passing_state::none;

            std::vector<BDD_BRANCH_NODE> new_bdd_branch_nodes_;
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
            const size_t new_nr_variables = std::max(nr_variables(), bdd_col_max_var+1);
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
            for(const auto& b : new_bdd_branch_nodes_)
                assert(b.offset_low == 0 && b.offset_high == 0 && b.bdd_index == BDD_BRANCH_NODE::inactive_bdd_index);
            std::vector<size_t> new_bdd_branch_nodes_counter(new_nr_variables, 0); 

            // fill in previous bdd nodes
            for(size_t bdd_idx=0; bdd_idx<nr_bdds(); ++bdd_idx)
            {
                assert(bdd_col.is_qbdd(bdd_idx));
                assert(bdd_col.is_reordered(bdd_idx));
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
                    assert(new_bdd_branch_nodes_[new_i].offset_low == 0 && new_bdd_branch_nodes_[new_i].offset_high == 0 && new_bdd_branch_nodes_[new_i].bdd_index == BDD_BRANCH_NODE::inactive_bdd_index);
                    ++new_bdd_branch_nodes_counter[var];
                    if(var == first_var)
                        new_cur_first_bdd_node_indices.push_back(new_i);
                    if(var == last_var)
                        new_cur_last_bdd_node_indices.push_back(new_i);
                    //assert(old_i <= new_i); // not necessarily true if we reorder bdd branch nodes belonging to same variable and bdd
                    index_map.insert({old_i, new_i});

                    if(old_bdd.offset_low != BDD_BRANCH_NODE::terminal_0_offset && old_bdd.offset_low != BDD_BRANCH_NODE::terminal_1_offset) 
                    {
                        const size_t old_low_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_low)); 
                        dq.push_back(old_low_i);
                    }
                    if(old_bdd.offset_high != BDD_BRANCH_NODE::terminal_0_offset && old_bdd.offset_high != BDD_BRANCH_NODE::terminal_1_offset) 
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
                    BDD_BRANCH_NODE& old_bdd = bdd_branch_nodes_[old_i];
                    BDD_BRANCH_NODE& new_bdd = new_bdd_branch_nodes_[new_i];
                    new_bdd = old_bdd;

                    if(old_bdd.offset_low != BDD_BRANCH_NODE::terminal_0_offset && old_bdd.offset_low != BDD_BRANCH_NODE::terminal_1_offset) 
                    {
                        const size_t old_low_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_low)); 
                        assert(index_map.count(old_low_i) > 0);
                        const size_t new_low_i = index_map.find(old_low_i)->second;
                        new_bdd.offset_low = new_bdd.synthesize_address(&new_bdd_branch_nodes_[new_low_i]); 
                    }

                    if(old_bdd.offset_high != BDD_BRANCH_NODE::terminal_0_offset && old_bdd.offset_high != BDD_BRANCH_NODE::terminal_1_offset) 
                    {
                        const size_t old_high_i = std::distance(&bdd_branch_nodes_[0], old_bdd.address(old_bdd.offset_high)); 
                        assert(index_map.count(old_high_i) > 0);
                        const size_t new_high_i = index_map.find(old_high_i)->second;
                        new_bdd.offset_high = new_bdd.synthesize_address(&new_bdd_branch_nodes_[new_high_i]); 
                    }
                } 
            }

            // fill in bdds from bdd_collection
            std::unordered_map<size_t,BDD_BRANCH_NODE*> stored_bdd_index_to_bdd_offset;
            std::vector<size_t> new_bdd_nrs;
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
                    ++new_bdd_branch_nodes_counter[v];
                    assert(bdd_branch_index < new_bdd_branch_nodes_.size());
                    assert(new_bdd_branch_nodes_[bdd_branch_index].offset_low == 0 && new_bdd_branch_nodes_[bdd_branch_index].offset_high == 0 && new_bdd_branch_nodes_[bdd_branch_index].bdd_index == BDD_BRANCH_NODE::inactive_bdd_index);
                    assert(v == new_nr_variables-1 || bdd_branch_index < new_bdd_branch_node_offsets_[v+1]);
                    assert(new_bdd_branch_nodes_[bdd_branch_index].offset_low == 0 && new_bdd_branch_nodes_[bdd_branch_index].offset_high == 0 && new_bdd_branch_nodes_[bdd_branch_index].bdd_index == BDD_BRANCH_NODE::inactive_bdd_index);

                    if(v == first_var)
                        cur_first_bdd_node_indices.push_back(bdd_branch_index);
                    if(v == last_var)
                        cur_last_bdd_node_indices.push_back(bdd_branch_index);

                    if(bdd_col(bdd_nr,stored_bdd.lo).is_botsink())
                    {
                        new_bdd_branch_nodes_[bdd_branch_index].offset_low = BDD_BRANCH_NODE::terminal_0_offset;
                        new_bdd_branch_nodes_[bdd_branch_index].low_cost = std::numeric_limits<value_type>::infinity();
                    }
                    else if(bdd_col(bdd_nr,stored_bdd.lo).is_topsink())
                    {
                        assert(new_bdd_branch_nodes_[bdd_branch_index].low_cost == 0.0);
                        new_bdd_branch_nodes_[bdd_branch_index].offset_low = BDD_BRANCH_NODE::terminal_1_offset;
                    }
                    else
                    {
                        assert(new_bdd_branch_nodes_[bdd_branch_index].low_cost == 0.0);
                        assert(stored_bdd_index_to_bdd_offset.count(stored_bdd.lo) > 0);
                        BDD_BRANCH_NODE* low_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.lo];
                        new_bdd_branch_nodes_[bdd_branch_index].offset_low = new_bdd_branch_nodes_[bdd_branch_index].synthesize_address(low_ptr);
                    }

                    if(bdd_col(bdd_nr,stored_bdd.hi).is_botsink())
                    {
                        new_bdd_branch_nodes_[bdd_branch_index].offset_high = BDD_BRANCH_NODE::terminal_0_offset;
                        new_bdd_branch_nodes_[bdd_branch_index].high_cost = std::numeric_limits<value_type>::infinity();
                    }
                    else if(bdd_col(bdd_nr,stored_bdd.hi).is_topsink())
                    {
                        assert(new_bdd_branch_nodes_[bdd_branch_index].high_cost == 0.0);
                        new_bdd_branch_nodes_[bdd_branch_index].offset_high = BDD_BRANCH_NODE::terminal_1_offset;
                    }
                    else
                    {
                        assert(new_bdd_branch_nodes_[bdd_branch_index].high_cost == 0.0);
                        assert(stored_bdd_index_to_bdd_offset.count(stored_bdd.hi) > 0);// != nullptr);
                        BDD_BRANCH_NODE* high_ptr = stored_bdd_index_to_bdd_offset[stored_bdd.hi];
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
            // TODO: use bdd_index type of BDD_BRANCH_NODE
            tsl::robin_map<uint32_t, uint32_t> bdd_index_redux;
            for(size_t i=0; i<new_nr_variables; ++i)
            {
                bdd_index_redux.clear();
                for(size_t vec_idx=new_bdd_branch_node_offsets_[i]; vec_idx<new_bdd_branch_node_offsets_[i+1]; ++vec_idx)
                {
                    auto& bdd_vec = new_bdd_branch_nodes_[vec_idx]; 
                    const uint32_t bdd_index = bdd_vec.bdd_index;
                    assert(bdd_index != BDD_BRANCH_NODE::inactive_bdd_index);
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

            for(size_t i=0; i<bdd_branch_nodes_.size(); ++i)
                assert(bdd_branch_nodes_[i].node_initialized());

            const double lb = lower_bound();
            std::cout << "lb = " << lb << "\n";
            return new_bdd_nrs;
        }

    template<typename BDD_BRANCH_NODE>
        template<typename ITERATOR>
        two_dim_variable_array<char> bdd_mma_base<BDD_BRANCH_NODE>::bdd_feasibility(ITERATOR sol_begin, ITERATOR sol_end) const
        {
            assert(std::distance(sol_begin, sol_end) == nr_variables());

            std::vector<size_t> bdd_feas_size;
            bdd_feas_size.reserve(nr_variables());
            for(size_t i=0; i<nr_variables(); ++i)
                bdd_feas_size.push_back(nr_bdds(i));
            two_dim_variable_array<char> bdd_feas(bdd_feas_size);
            for(size_t i=0; i<bdd_feas.size(); ++i)
                for(size_t j=0; j<bdd_feas.size(i); ++j)
                    bdd_feas(i,j) = true;


            std::unordered_set<size_t> forward_instr_ptrs; // TODO: possibly remove for vector?
            struct i_var { size_t i, var; }; // bdd_branch_nodes_ offset and variable
            std::unordered_map<size_t, i_var> backward_instr_ptrs; // for backtracking and setting corresponding elements to false

            for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
            {
                assert(first_bdd_node_indices_.size(bdd_nr) == 1);
                const size_t i = first_bdd_node_indices_(bdd_nr,0);
                forward_instr_ptrs.insert(i);
            }

            // todo: there might be additional (var, bdd_index) tuples corresponding to current bdd that must be set to false!
            auto backtrack = [&](const size_t i, const size_t var) -> void {
                auto backtrack_impl = [&](const size_t i, const size_t var, auto& backtrack_ref) -> void {
                    assert(i < bdd_branch_nodes_.size());
                    assert(var < nr_variables());

                    const auto& bdd = bdd_branch_nodes_[i];
                    const size_t bdd_index = bdd.bdd_index;

                    if(bdd_feas(var, bdd_index) == false)
                        return;
                    bdd_feas(var, bdd_index) = false;

                    auto prev_it = backward_instr_ptrs.find(i);
                    if(prev_it != backward_instr_ptrs.end())
                    {
                        const auto [prev_i, prev_var] = prev_it->second;
                        backtrack_ref(prev_i, prev_var, backtrack_ref);
                    }
                };
                return backtrack_impl(i, var, backtrack_impl);
            };

            // insert into doubly linked list
            auto insert_list = [&](const size_t i, const size_t var, const size_t offset) {
                assert(i < bdd_branch_nodes_.size());
                assert(var < nr_variables());
                const auto& bdd = bdd_branch_nodes_[i];
                assert(offset == bdd.offset_low || offset == bdd.offset_high);
                const size_t total_offset = std::distance(&bdd_branch_nodes_[0], bdd.address(offset));

                forward_instr_ptrs.insert(total_offset);
                backward_instr_ptrs.insert({total_offset, i_var{i,var}});
            };

            // see whether previous BDD node
            auto set_from_previous = [&](const size_t i, const size_t var) {
                assert(i < bdd_branch_nodes_.size());
                assert(var < nr_variables());

                // get pointer to previous
                auto prev_it = backward_instr_ptrs.find(i);
                if(prev_it != backward_instr_ptrs.end())
                {
                    // if it is set to false, set current entry to false as well
                    const auto [prev_i, prev_var] = prev_it->second;
                    const auto& bdd_prev = bdd_branch_nodes_[prev_i];
                    if(bdd_feas(prev_var, bdd_prev.bdd_index) == false)
                    {
                        const auto& bdd = bdd_branch_nodes_[i];
                        bdd_feas(var, bdd.bdd_index) = false;
                    }
                }
            };

            for(size_t var=0; var<nr_variables(); ++var)
            {
                assert(sol_begin[var] == 0 || sol_begin[var] == 1);
                const bool x = sol_begin[var];

                for(size_t i=bdd_branch_node_offsets_[var]; i<bdd_branch_node_offsets_[var+1]; ++i)
                {
                    const auto& bdd = bdd_branch_nodes_[i];

                    set_from_previous(i, var);

                    if(forward_instr_ptrs.count(i) > 0)
                    {
                        if(x == false)
                        {
                            if(bdd.offset_low == bdd.terminal_0_offset)
                            {
                                backtrack(i, var);
                                if(bdd.offset_high != bdd.terminal_1_offset)
                                    insert_list(i, var, bdd.offset_high);
                            }
                            else if(bdd.offset_low != bdd.terminal_1_offset)
                            {
                                insert_list(i, var, bdd.offset_low);
                            }
                        }
                        else
                        {
                            if(bdd.offset_high == bdd.terminal_0_offset)
                            {
                                backtrack(i, var);
                                if(bdd.offset_low != bdd.terminal_1_offset)
                                    insert_list(i, var, bdd.offset_low);
                            }
                            else if(bdd.offset_high != bdd.terminal_1_offset)
                            {
                                insert_list(i, var, bdd.offset_high);
                            }
                        }
                    }
                }
            }

            return bdd_feas;
        }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::export_graphviz(const char* filename)
    {
        const std::string f(filename);    
        export_graphviz(f);
    }

    template<typename BDD_BRANCH_NODE>
    void bdd_mma_base<BDD_BRANCH_NODE>::export_graphviz(const std::string& filename)
    {
        const std::string base_filename = std::filesystem::path(filename).replace_extension("").c_str();
        for(size_t bdd_nr=0; bdd_nr<nr_bdds(); ++bdd_nr)
        {
            const std::string dot_file = base_filename + "_" + std::to_string(bdd_nr) + ".dot";
            std::fstream f;
            f.open(dot_file, std::fstream::out | std::ofstream::trunc);
            export_graphviz(f, bdd_nr);
            f.close();
            const std::string png_file = base_filename + "_" + std::to_string(bdd_nr) + ".png";
            const std::string convert_command = "dot -Tpng " + dot_file + " > " + png_file;
            std::system(convert_command.c_str());
        }
    }

    template<typename BDD_BRANCH_NODE>
    template<typename STREAM>
        void bdd_mma_base<BDD_BRANCH_NODE>::export_graphviz(STREAM& s, const size_t bdd_nr)
        {
            std::unordered_set<BDD_BRANCH_NODE*> visited;
            std::queue<BDD_BRANCH_NODE*> q;

            s << "digraph BDD\n";
            s << "{\n";
            for(size_t j=0; j<first_bdd_node_indices_.size(bdd_nr); ++j)
            {
                auto& bdd = bdd_branch_nodes_[first_bdd_node_indices_(bdd_nr,j)];
                q.push(&bdd);
            }

            while(!q.empty())
            {
                BDD_BRANCH_NODE* bdd = q.front();
                q.pop();
                if(visited.count(bdd) > 0)
                    continue;
                visited.insert(bdd);

                if(bdd->offset_low != BDD_BRANCH_NODE::terminal_0_offset && bdd->offset_low != BDD_BRANCH_NODE::terminal_1_offset)
                {
                    q.push(bdd->address(bdd->offset_low));
                    s << "\"" << bdd << "\" -> \"" << bdd->address(bdd->offset_low) << "\" [style=\"dashed\"];\n";;
                }
                else
                    s << "\"" << bdd << "\" -> " << " bot [style=\"dashed\"];\n";;

                if(bdd->offset_high != BDD_BRANCH_NODE::terminal_0_offset && bdd->offset_high != BDD_BRANCH_NODE::terminal_1_offset)
                {
                    q.push(bdd->address(bdd->offset_high));
                    s << "\"" << bdd << "\" -> \"" << bdd->address(bdd->offset_high) << "\";\n";
                }
                else
                    s << "\"" << bdd << "\" -> " << " top;\n";;
            }
            s << "}\n";

        }

}
