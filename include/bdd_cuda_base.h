#pragma once

#include <array>
#include "bdd_collection/bdd_collection.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#define TOP_SINK_INDICATOR_CUDA -1
#define BOT_SINK_INDICATOR_CUDA -2

namespace LPMP {

    class bdd_cuda_base {
        public:
            bdd_cuda_base(const BDD::bdd_collection& bdd_col);

            void flush_forward_states();
            void flush_backward_states();

            double lower_bound();

            template<typename COST_ITERATOR>
                void set_costs(COST_ITERATOR begin, COST_ITERATOR end);
            void set_cost(const double c, const size_t var);
            
            std::vector<std::vector<std::array<float,2>>> min_marginals();
            std::tuple<thrust::device_vector<float>, thrust::device_vector<float>> min_marginals_cuda();

            size_t nr_variables() const { return nr_vars_; }
            size_t nr_bdds() const { return nr_bdds_; }

            void forward_run(); //TODO: Should these functions be called from outside?
            void backward_run();

        protected:

            void update_costs(const thrust::device_vector<float>& update_vec);
            
            // Following arrays have one entry per layer of BDD in each BDD:
            thrust::device_vector<int> primal_variable_index_;
            thrust::device_vector<int> bdd_index_;
            thrust::device_vector<float> hi_cost_;

            // Following arrays are allocated for each bdd node:
            thrust::device_vector<int> lo_bdd_node_index_; // = 0
            thrust::device_vector<int> hi_bdd_node_index_; // = 1
            thrust::device_vector<float> cost_from_root_;
            thrust::device_vector<float> cost_from_terminal_;
            thrust::device_vector<int> bdd_node_to_layer_map_;

            thrust::device_vector<float> hi_path_cost_; // Cost of shortest path through BDD node where hi arc is used.
            thrust::device_vector<float> lo_path_cost_; // Cost of shortest path through BDD node where lo arc is used. 

            // Other information:
            size_t nr_vars_, nr_bdds_;
            size_t nr_bdd_nodes_ = 0;
            size_t num_dual_variables_ = 0;
            thrust::device_vector<int> cum_nr_bdd_nodes_per_hop_dist_; // How many BDD nodes (cumulative) are present with a given hop distance away from root node.
            thrust::device_vector<int> num_bdds_per_var_; // In how many BDDs does a primal variable appear.
            thrust::device_vector<int> num_vars_per_bdd_;
            thrust::device_vector<int> bdd_layer_width_; // Counts number of repetitions of a primal variable in a BDD. 
            thrust::device_vector<int> root_indices_, bot_sink_indices_, top_sink_indices_;

        private:
            void initialize(const BDD::bdd_collection& bdd_col);
            std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> populate_bdd_nodes(const BDD::bdd_collection& bdd_col);
            void reorder_bdd_nodes(thrust::device_vector<int>& bdd_hop_dist_root, thrust::device_vector<int>& bdd_depth);
            void populate_counts(const BDD::bdd_collection& bdd_col);
            void set_special_nodes_indices(const thrust::device_vector<int>& bdd_hop_dist);
            void compress_bdd_nodes_to_layer();

            bool forward_state_valid_ = false;
            bool backward_state_valid_ = false;

    };

}
