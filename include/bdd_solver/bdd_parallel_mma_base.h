#pragma once

// TODO: remove unnecessary header files
#include <vector>
#include <array>
#include <Eigen/SparseCore>
#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <unordered_set>
#include "bdd_logging.h"
#include "time_measure_util.h"
#include "atomic_ref.hpp"

namespace LPMP {

    // store BDDs one after the other.
    // allows for efficient computation of min marginals and parallel mma
    template<typename BDD_BRANCH_NODE>
        class bdd_parallel_mma_base {
            public:
            using value_type = typename BDD_BRANCH_NODE::value_type;
            bdd_parallel_mma_base() {}
            bdd_parallel_mma_base(const BDD::bdd_collection& bdd_col);
            bdd_parallel_mma_base(const BDD::bdd_collection& bdd_col, const std::vector<double>& costs_hi);
            //template<typename COST_ITERATOR>
            //bdd_parallel_mma_base(const BDD::bdd_collection& bdd_col, COST_ITERATOR costs_begin, COST_ITERATOR costs_end) { add_bdds(bdd_col); update_costs(costs_begin, costs_begin, costs_begin, costs_end); }

            void add_bdds(const BDD::bdd_collection& bdd_col);

            size_t nr_bdds() const;
            size_t nr_bdds(const size_t var) const;
            size_t nr_variables() const;
            size_t nr_variables(const size_t bdd_nr) const;
            size_t variable(const size_t bdd_nr, const size_t bdd_index) const;
            size_t nr_bdd_variables() const;

            double lower_bound();
            using vector_type = Eigen::Matrix<typename BDD_BRANCH_NODE::value_type, Eigen::Dynamic, 1>;
            vector_type lower_bound_per_bdd();

            void update_costs(const std::vector<value_type>& costs_lo, const std::vector<value_type>& costs_hi);

            void forward_run();
            void backward_run();
            void backward_run(const size_t bdd_nr);
            //two_dim_variable_array<std::array<value_type,2>> min_marginals();
            two_dim_variable_array<std::array<double,2>> min_marginals();
            using min_marginal_type = Eigen::Matrix<typename BDD_BRANCH_NODE::value_type, Eigen::Dynamic, 2>;
            std::tuple<min_marginal_type, std::vector<char>> min_marginals_stacked();
            std::tuple<std::vector<value_type>, std::vector<value_type>> min_marginals_vec(); // returns primal variables, lo mms, hi mms

            
            // TODO: remove these! //
            //void update_costs(const two_dim_variable_array<std::array<value_type,2>>& delta);
            //void update_costs(const min_marginal_type& delta);
            //vector_type get_costs();
            //void update_costs(const vector_type& delta);
            void add_to_constant(const double c);
            /////////////////////////

            void fix_variables(const std::vector<size_t> zero_variables, const std::vector<size_t> one_variables);
            void fix_variable(const size_t var, const bool value);

            // compute incremental min marginals and perform min-marginal averaging subsequently
            void iteration();
            void forward_mm(const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega, 
                    std::vector<std::array<value_type,2>>& delta_out, std::vector<std::array<value_type,2>>& delta_in);
            value_type backward_mm(const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega, std::vector<std::array<value_type,2>>& delta_out, std::vector<std::array<value_type,2>>& delta_in);
            void forward_mm(const value_type omega, std::vector<std::array<value_type,2>>& delta);
            double backward_mm(const value_type omega, std::vector<std::array<value_type,2>>& delta);
            void distribute_delta();
            Eigen::SparseMatrix<value_type> Lagrange_constraint_matrix() const;

            void export_graphviz(const char* filename);
            void export_graphviz(const std::string& filename);

            // LBFGS operations
            std::vector<char> bdds_solution_vec();
            std::vector<value_type> net_solver_costs();
            void make_dual_feasible(std::vector<value_type>& duals) const;
            bool dual_feasible(const std::vector<value_type>& duals) const;
            void gradient_step(const std::vector<value_type>& duals, const double step_size);
            size_t nr_layers() const;

        protected:
            template <typename COST_ITERATOR>
            void update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end);
            template<typename ITERATOR>
                void fix_variables(ITERATOR zero_fixations_begin, ITERATOR zero_fixations_end, ITERATOR one_fixations_begin, ITERATOR one_fixations_end);

            template<typename STREAM>
                void export_graphviz(STREAM& s, const size_t bdd_nr);

            // Both operations below are inverses of each other
            // Given elements in order bdd_nr/bdd_index, transpose to variable/bdd_index with same variable.
            template<typename T>
                two_dim_variable_array<T> transpose_to_var_order(const two_dim_variable_array<T>& m) const;
            // Given elements in order var/bdd_index with same variable, transpose to bdd_nr/bdd_index.
            template<typename T>
                two_dim_variable_array<T> transpose_to_bdd_order(const two_dim_variable_array<T>& m) const;


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
            double constant_ = 0.0;

            double compute_lower_bound();
            double compute_lower_bound_after_forward_pass();
            double compute_lower_bound_after_backward_pass();

            vector_type lower_bound_per_bdd_after_forward_pass();
            vector_type lower_bound_per_bdd_after_backward_pass();

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

            // for parallel mma
            std::vector<std::array<value_type,2>> delta_out_;
            std::vector<std::array<value_type,2>> delta_in_;
        };

    template<typename REAL>
    void atomic_add(REAL& f, const REAL d) 
    {
        if(d == 0)
            return;

        // TODO: use std::atomic_ref when available in C++20
        Foo::atomic_ref<REAL> f_ref{f};
        f_ref += d;
    }

    template<typename REAL>
    void atomic_store(REAL& f, const REAL d) 
    {
        Foo::atomic_ref<REAL> f_ref{f};
        f_ref.store(d);
    }
}
