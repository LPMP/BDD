#pragma once

#include <array>
#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

// For serialization
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>

namespace cereal {
    template<class Archive, class T>
    void save(Archive& ar, const thrust::device_vector<T>& dev_vector)
    {
        std::vector<T> host_vec(dev_vector.size());
        thrust::copy(dev_vector.begin(), dev_vector.end(), host_vec.begin());
        ar << host_vec;
    }

    template<class Archive, class T>
    void load(Archive& ar, thrust::device_vector<T>& dev_vector)
    {
        std::vector<T> host_vec;
        ar >> host_vec;
        dev_vector = thrust::device_vector<T>(host_vec);
    }

    // // Save and load REAL data type as float. Not strictly necessary only because currently
    // // the saved bdd representations are in float and we would like to load it into double.
    // template<class Archive>
    // void save(Archive& ar, const thrust::device_vector<double>& dev_vector)
    // {
    //     std::vector<double> host_vec(dev_vector.size());
    //     thrust::copy(dev_vector.begin(), dev_vector.end(), host_vec.begin());
    //     std::vector<float> host_vec_float(host_vec.begin(), host_vec.end());
    //     ar << host_vec_float;
    // }

    // template<class Archive>
    // void load(Archive& ar, thrust::device_vector<double>& dev_vector)
    // {
    //     std::vector<float> host_vec;
    //     ar >> host_vec;
    //     thrust::device_vector<float> dev_vector_float(host_vec.begin(), host_vec.end());
    //     dev_vector = thrust::device_vector<double>(dev_vector_float.begin(), dev_vector_float.end());
    // }
}

namespace LPMP {
    static constexpr int TOP_SINK_INDICATOR_CUDA = -1;
    static constexpr int BOT_SINK_INDICATOR_CUDA = -2;
    static constexpr int NUM_THREADS_CUDA = 256;

    template<typename REAL>
    class bdd_cuda_base {
        public:
            using value_type = REAL;
            using SOLVER_COSTS_VECS = std::tuple<thrust::device_vector<REAL>, thrust::device_vector<REAL>, thrust::device_vector<REAL>>;
            bdd_cuda_base() {}
            bdd_cuda_base(const BDD::bdd_collection& bdd_col);

            void flush_forward_states();
            void flush_backward_states();

            double lower_bound();
            void lower_bound_per_bdd(thrust::device_ptr<REAL> lb_per_bdd);

            template<typename COST_ITERATOR>
                void update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end);

            template<typename REAL_arg>
            void update_costs(const thrust::device_vector<REAL_arg>& cost_delta_0, const thrust::device_vector<REAL_arg>& cost_delta_1);
            template<typename REAL_arg>
            void update_costs(const thrust::device_ptr<const REAL_arg> cost_delta_0, const size_t delta_0_size,
                            const thrust::device_ptr<const REAL_arg> cost_delta_1, const size_t delta_1_size);

            void set_cost(const double c, const size_t var);
            
            two_dim_variable_array<std::array<double,2>> min_marginals();
            std::tuple<thrust::device_vector<int>, thrust::device_vector<REAL>, thrust::device_vector<REAL>> min_marginals_cuda(bool get_sorted = true);

            void bdds_solution_cuda(thrust::device_ptr<REAL> sol); // Computes argmin for each BDD separately and sets in sol.
            two_dim_variable_array<REAL> bdds_solution(); // Returns the solution on CPU and laid out in similar way as the output of min_marginals()

            void compute_primal_objective_vec(thrust::device_ptr<REAL> primal_obj);
            std::vector<REAL> get_primal_objective_vector_host();

            size_t nr_variables() const { return nr_vars_; }
            size_t nr_variables(const size_t bdd_nr) const { assert(bdd_nr < nr_bdds()); return nr_vars_; }
            size_t nr_bdds() const { return nr_bdds_; }
            size_t nr_bdds(const size_t var) const { assert(var < nr_variables()); return num_bdds_per_var_[var]; }
            size_t nr_layers() const { return cum_nr_layers_per_hop_dist_.back(); }
            size_t nr_layers(const int hop_index) const { 
                return cum_nr_layers_per_hop_dist_[hop_index] - (hop_index > 0 ? cum_nr_layers_per_hop_dist_[hop_index - 1] : 0); 
            }

            size_t nr_bdd_nodes() const { return cum_nr_bdd_nodes_per_hop_dist_.back(); }
            size_t nr_bdd_nodes(const int hop_index) const { 
                return cum_nr_bdd_nodes_per_hop_dist_[hop_index] - (hop_index > 0 ? cum_nr_bdd_nodes_per_hop_dist_[hop_index - 1] : 0); 
            }
            size_t nr_hops() const { return cum_nr_layers_per_hop_dist_.size() - 1; } // ignores terminal nodes.

            void forward_run();
            std::tuple<thrust::device_vector<REAL>, thrust::device_vector<REAL>> backward_run(bool compute_path_costs = true);

            // Return (primal var, BDD) for all dual variables. The ordering is in terms of increasing hop distances i.e. 
            // First all roots nodes, then all nodes at hop distance 1 and so on.
            std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> var_constraint_indices() const;
            
            // All pointers allocate a space of size nr_layers().
            void get_solver_costs(thrust::device_ptr<REAL> lo_cost_out_ptr, 
                                thrust::device_ptr<REAL> hi_cost_out_ptr, 
                                thrust::device_ptr<REAL> deferred_mm_diff_out_ptr) const;

            SOLVER_COSTS_VECS get_solver_costs() const;

            // Set solver costs so perform opposite of get_solver_costs(...).
            void set_solver_costs(const thrust::device_ptr<const REAL> lo_costs, 
                                const thrust::device_ptr<const REAL> hi_costs, 
                                const thrust::device_ptr<const REAL> deferred_mm_diff);

            void set_solver_costs(const SOLVER_COSTS_VECS& costs); // similar to above but takes device_vectors

            const thrust::device_vector<int> get_primal_variable_index() const { return primal_variable_index_; }
            const thrust::device_vector<int> get_bdd_index() const { return bdd_index_; }
            const thrust::device_vector<int>& get_num_bdds_per_var() const {return num_bdds_per_var_; }

            void distribute_delta(thrust::device_ptr<REAL> def_min_marg_diff_ptr);
            void distribute_delta();

            void terminal_layer_indices(thrust::device_ptr<int> indices) const; // indices should point to memory of size nr_bdds()

            void compute_bdd_to_constraint_map(const two_dim_variable_array<size_t>& constraint_to_bdd_map);
            const std::vector<size_t> bdd_to_constraint_map() const {return bdd_to_constraint_map_; }

            // For serialization using cereal:
            template <class Archive>
            void save(Archive& archive) const;

            template <class Archive>
            void load(Archive & archive);

        protected:

            void update_costs(const thrust::device_vector<REAL>& update_vec);
            void flush_costs_from_root();

            // Following arrays have one entry per layer of BDD in each BDD:
            thrust::device_vector<int> primal_variable_index_;
            thrust::device_vector<int> bdd_index_;
            thrust::device_vector<REAL> hi_cost_, lo_cost_, deffered_mm_diff_;

            // Following arrays are allocated for each bdd node:
            thrust::device_vector<int> lo_bdd_node_index_; // = 0
            thrust::device_vector<int> hi_bdd_node_index_; // = 1 // Can possibly be packed with lo_bdd_node_index_ by storing only offset.
            thrust::device_vector<REAL> cost_from_root_;
            thrust::device_vector<REAL> cost_from_terminal_;
            thrust::device_vector<int> bdd_node_to_layer_map_;

            // Other information:
            thrust::device_vector<int> num_bdds_per_var_; // In how many BDDs does a primal variable appear.
            thrust::device_vector<int> root_indices_, bot_sink_indices_, top_sink_indices_;
            thrust::device_vector<int> primal_variable_sorting_order_; // indices to sort primal_variables_indices_
            thrust::device_vector<int> primal_variable_index_sorted_;  // to reduce min-marginals by key.

            thrust::device_vector<int> layer_offsets_; // Similar to CSR representation where row is layer index, and column is bdd node.

            std::vector<int> cum_nr_bdd_nodes_per_hop_dist_; // How many BDD nodes (cumulative) are present with a given hop distance away from root node.
            std::vector<int> cum_nr_layers_per_hop_dist_; // Similar to cum_nr_bdd_nodes_per_hop_dist_ but for BDD layer instead of BDD node.
            std::vector<int> nr_variables_per_hop_dist_;
            size_t nr_vars_, nr_bdds_;
            size_t nr_bdd_nodes_ = 0;
            size_t num_dual_variables_ = 0;
            bool forward_state_valid_ = false; // true means cost from root valid.
            bool backward_state_valid_ = false; // true means cost from terminal are valid.

            std::vector<size_t> bdd_to_constraint_map_;

        private:
            void initialize(const BDD::bdd_collection& bdd_col);
            std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> populate_bdd_nodes(const BDD::bdd_collection& bdd_col);
            void reorder_bdd_nodes(thrust::device_vector<int>& bdd_hop_dist_root, thrust::device_vector<int>& bdd_depth);
            void populate_counts(const BDD::bdd_collection& bdd_col);
            void set_special_nodes_indices(const thrust::device_vector<int>& bdd_hop_dist);
            void set_special_nodes_costs();
            void compress_bdd_nodes_to_layer(const thrust::device_vector<int>& bdd_hop_dist);
            void reorder_within_bdd_layers();
            void print_num_bdd_nodes_per_hop();
            void find_primal_variable_ordering();

    };

}
