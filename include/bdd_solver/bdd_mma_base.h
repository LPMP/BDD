#pragma once

// TODO: remove unnecessary headers
#include <vector>
#include <array>
#include <cstddef>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <queue>
#include <stack>
#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include "bdd_solver/bdd_branch_instruction.h"
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
                bdd_mma_base(const BDD::bdd_collection& bdd_col);
                bdd_mma_base(const BDD::bdd_collection& bdd_col, const std::vector<double>& costs_hi);
                //template<typename COST_ITERATOR>
                //bdd_mma_base(const BDD::bdd_collection& bdd_col, COST_ITERATOR costs_begin, COST_ITERATOR costs_end) { add_bdds(bdd_col); update_costs(costs_begin, costs_begin, costs_begin, costs_end); } 
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
                void update_costs(const std::vector<value_type>& costs_lo, const std::vector<value_type>& costs_hi);
                //template<typename REAL>
                //    void update_costs(const two_dim_variable_array<std::array<REAL,2>>& delta);

                void add_to_constant(const double c) { constant_ += c; }
                double constant() const { return constant_; }

                template<typename ITERATOR>
                    void update_arc_costs(const size_t first_node, ITERATOR begin, ITERATOR end);
                void transfer_cost(const size_t from_bdd_nr, const size_t to_bdd_nr);
                void get_arc_marginals(const size_t first_node, const size_t last_node, std::vector<double>& arc_marginals);

                std::array<size_t,2> bdd_branch_node_offset(const size_t var, const size_t bdd_index) const;
                size_t variable(const size_t bdd_offset) const;

                // record for each bdd and each of its variables whether the solution is feasible for it
                two_dim_variable_array<char> bdd_feasibility(const std::vector<char>& sol) const;
                template<typename ITERATOR>
                    two_dim_variable_array<char> bdd_feasibility(ITERATOR sol_begin, ITERATOR sol_end) const;

                void export_graphviz(const char* filename);
                void export_graphviz(const std::string& filename);
                template<typename STREAM>
                    void export_graphviz(STREAM& s, const size_t bdd_nr);

            protected:
                // add costs from cost iterator to costs of bdd. Assume that variables given are subset of variables of bdd
                template<typename COST_ITERATOR, typename VARIABLE_ITERATOR>
                    void update_bdd_costs(const size_t bdd_nr,
                            COST_ITERATOR cost_begin, COST_ITERATOR cost_end,
                            VARIABLE_ITERATOR variable_begin, VARIABLE_ITERATOR variable_end);
                //template<typename COST_ITERATOR>
                //    void update_costs(COST_ITERATOR cost_begin, COST_ITERATOR cost_end);
                template <typename COST_ITERATOR>
                void update_costs(COST_ITERATOR lo_cost_begin, COST_ITERATOR lo_cost_end, COST_ITERATOR hi_cost_begin, COST_ITERATOR hi_cost_end);

                struct bdd_node {
                    constexpr static size_t terminal_0 = std::numeric_limits<size_t>::max() - 1;
                    constexpr static size_t terminal_1 = std::numeric_limits<size_t>::max();
                    size_t var;
                    size_t low;
                    size_t high;
                };

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
                void add_bdds(const BDD::bdd_collection& bdd_col);
                template<typename BDD_NR_ITERATOR>
                    std::vector<size_t> add_bdds(const BDD::bdd_collection& bdd_col, BDD_NR_ITERATOR bdd_nrs_begin, BDD_NR_ITERATOR bdd_nrs_end);

                std::vector<size_t> variables(const size_t bdd_idx);
                std::vector<bdd_branch_instruction<float,uint32_t>> export_bdd(const size_t bdd_idx);
                size_t export_bdd(BDD::bdd_collection& bdd_col, const size_t bdd_idx);
                std::tuple<BDD::node_ref, std::vector<size_t>> export_bdd(BDD::bdd_mgr& bdd_mgr, const size_t bdd_idx);
        };
}