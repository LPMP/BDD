#pragma once

#include <array>
#include "bdd_collection/bdd_collection.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "two_dimensional_variable_array.hxx"

inline float __int_as_float_host(int a)
{
    union {int a; float b;} u;
    u.a = a;
    return u.b;
}

#define CUDART_INF_F __int_as_float_host(0x7f800000)

namespace LPMP {

    class bdd_cuda_base {
        public:
            bdd_cuda_base(BDD::bdd_collection& bdd_col);

            void initialize_costs();

            double lower_bound();

            template<typename COST_ITERATOR>
                void set_costs(COST_ITERATOR begin, COST_ITERATOR end);
            void set_cost(const double c, const size_t var);
            
            two_dim_variable_array<std::array<float,2>> min_marginals();
            std::tuple<thrust::device_vector<int>, thrust::device_vector<int>, thrust::device_vector<float>, thrust::device_vector<float>> min_marginals_cuda();

            size_t nr_variables() const { return nr_vars_; }
            size_t nr_bdds() const { return nr_bdds_; }

            void forward_run(); //TODO: Should these functions be called from outside?
            void backward_run();

        protected:
            // Following arrays are allocated for each bdd node:
            thrust::device_vector<int> primal_variable_index_;
            thrust::device_vector<int> bdd_index_;
            thrust::device_vector<int> lo_bdd_node_index_; // = 0
            thrust::device_vector<int> hi_bdd_node_index_; // = 1
            thrust::device_vector<float> hi_cost_;
            thrust::device_vector<float> cost_from_root_;
            thrust::device_vector<float> cost_from_terminal_;

            thrust::device_vector<float> hi_path_cost_; // Cost of shortest path through BDD node where hi arc is used.
            thrust::device_vector<float> lo_path_cost_; // Cost of shortest path through BDD node where lo arc is used. 

            // Other information:
            size_t nr_vars_, nr_bdds_, nr_bdd_nodes_;
            int num_dual_variables_;
            thrust::device_vector<int> primal_variable_counts_; // In how many BDDs does a primal variable appear.
            thrust::device_vector<int> cum_nr_bdd_nodes_per_hop_dist_; // How many BDD nodes (cumulative) are present with a given hop distance away from root node.
            thrust::device_vector<int> num_vars_per_bdd_;
            thrust::device_vector<int> root_indices_, bot_sink_indices_, top_sink_indices_;

            bool forward_state_valid_ = false;
            bool backward_state_valid_ = false;

    };

}
