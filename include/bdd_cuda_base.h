#pragma once

#include <array>
#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#define TOP_SINK_INDICATOR_CUDA -1
#define BOT_SINK_INDICATOR_CUDA -2
#define NUM_THREADS 256

namespace LPMP {

    template<typename REAL>
    class bdd_cuda_base {
        public:
            using value_type = REAL;
            bdd_cuda_base(const BDD::bdd_collection& bdd_col);

            void flush_forward_states();
            void flush_backward_states();

            double lower_bound();

            template<typename COST_ITERATOR>
                void update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end);
            void update_costs(const thrust::device_vector<REAL>& cost_delta_0, const thrust::device_vector<REAL>& cost_delta_1);
            void set_cost(const double c, const size_t var);
            
            two_dim_variable_array<std::array<double,2>> min_marginals();
            std::tuple<thrust::device_vector<REAL>, thrust::device_vector<REAL>> min_marginals_cuda();

            size_t nr_variables() const { return nr_vars_; }
            size_t nr_bdds() const { return nr_bdds_; }

            void forward_run();
            void backward_run(bool compute_path_costs = true);

            const thrust::device_vector<int>& primal_variable_index() const { return primal_variable_index_; }
        protected:
            void update_costs(const thrust::device_vector<REAL>& update_vec);
            void flush_costs_from_root();

            // Following arrays have one entry per layer of BDD in each BDD:
            thrust::device_vector<int> primal_variable_index_;
            thrust::device_vector<int> bdd_index_;
            thrust::device_vector<REAL> hi_cost_, lo_cost_;

            thrust::device_vector<int> layer_offsets_; // Similar to CSR representation where row is layer index, and column is bdd node.
            std::vector<int> cum_nr_layers_per_hop_dist_; // Similar to cum_nr_bdd_nodes_per_hop_dist_ but for BDD layer instead of BDD node.

            // Following arrays are allocated for each bdd node:
            thrust::device_vector<int> lo_bdd_node_index_; // = 0
            thrust::device_vector<int> hi_bdd_node_index_; // = 1
            thrust::device_vector<REAL> cost_from_root_;
            thrust::device_vector<REAL> cost_from_terminal_;
            thrust::device_vector<int> bdd_node_to_layer_map_;

            thrust::device_vector<REAL> hi_path_cost_; // Cost of shortest path through BDD node where hi arc is used.
            thrust::device_vector<REAL> lo_path_cost_; // Cost of shortest path through BDD node where lo arc is used. 

            // Other information:
            size_t nr_vars_, nr_bdds_;
            size_t nr_bdd_nodes_ = 0;
            size_t num_dual_variables_ = 0;
            std::vector<int> cum_nr_bdd_nodes_per_hop_dist_; // How many BDD nodes (cumulative) are present with a given hop distance away from root node.
            thrust::device_vector<int> num_bdds_per_var_; // In how many BDDs does a primal variable appear.
            thrust::device_vector<int> num_vars_per_bdd_;
            thrust::device_vector<int> bdd_layer_width_; // Counts number of repetitions of a primal variable in a BDD. 
            thrust::device_vector<int> root_indices_, bot_sink_indices_, top_sink_indices_;
            bool forward_state_valid_ = false; // true means cost from root valid.
            bool backward_state_valid_ = false; // true means cost from terminal are valid.

        private:
            bool path_costs_valid_ = false; // here valid means lo, hi path paths are valid.
            void initialize(const BDD::bdd_collection& bdd_col);
            std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> populate_bdd_nodes(const BDD::bdd_collection& bdd_col);
            void reorder_bdd_nodes(thrust::device_vector<int>& bdd_hop_dist_root, thrust::device_vector<int>& bdd_depth);
            void populate_counts(const BDD::bdd_collection& bdd_col);
            void set_special_nodes_indices(const thrust::device_vector<int>& bdd_hop_dist);
            void compress_bdd_nodes_to_layer(const thrust::device_vector<int>& bdd_hop_dist);
            void print_num_bdd_nodes_per_hop();

    };

}
