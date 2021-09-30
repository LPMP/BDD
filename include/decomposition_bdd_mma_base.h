#pragma once 

#include <vector>
#include <cmath>
#include <cassert>
#include <thread>
#include <queue>
#include <mutex>
#include "decomposition_bdd_mma.h"
#include "bdd_storage.h"
#include "bdd_branch_node_vector.h"
#include "time_measure_util.h"
#include <iostream> // TODO: remove

namespace LPMP {

    class decomposition_bdd_base {
        public:
            decomposition_bdd_base(bdd_storage& stor, decomposition_mma_options opt);

            size_t nr_variables() const;
            void set_cost(const double c, const size_t var);
            void backward_run();
            void solve(const size_t max_iter, const double tolerance, const double time_limit);
            void iteration();
            double lower_bound();

            two_dim_variable_array<std::array<double,2>> min_marginals();
        private:
            void min_marginal_averaging_forward(const size_t interval_nr);
            void min_marginal_averaging_backward(const size_t interval_nr);

            bdd_storage::intervals intervals;
            std::vector<double> costs;

            struct Lagrange_multiplier {
                union {
                    size_t interval_nr;
                    size_t first_node;
                    size_t nr_deltas;
                    double delta;
                };
            }; 

            struct endpoint {
                // bdd branch node delimiters from this interval coming from a split
                size_t first_node;
                size_t last_node; // TODO: possibly not needed, can be inferred from bdd index in bdd base
                // first bdd branch node in opposite interval
                size_t opposite_interval_nr;
                size_t first_node_opposite_interval;
            };

            struct Lagrange_multiplier_queue {
                std::mutex mutex;
                std::queue<Lagrange_multiplier> queue;

                // TODO: possibly not needed?
                // we first write Lagrange multipliers that need to be synchronized via the queue in this cache in order to save synchronization calls
                //std::vector<Lagrange_multiplier> queue_cache;
                //std::vector<size_t> queue_cache_offset;
                //size_t queue_cache_offset_counter;

                // write Lagrange multipliers into queue cache
                //template<typename ITERATOR>
                //    void write_to_cache(const size_t interval_nr, size_t offset, ITERATOR begin, ITERATOR end);

                // write from cache to queue
                //template<typename ITERATOR>
                //    size_t write_to_queue(ITERATOR it);
                template<typename ITERATOR>
                    void write_to_queue(const size_t node_nr, ITERATOR begin, ITERATOR end);

                //void init_cache(const std::vector<endpoint>& endpoints);
                //void reset_cache();
            };

            struct bdd_sub_base {
                bdd_mma_base_vec base;

                void read_in_Lagrange_multipliers(Lagrange_multiplier_queue& queue);

                // TODO: determine whether two_dim_variable_array<endpoint> and interleaved message passing updates are faster.
                // TODO: rename to right/left endpoints?  
                std::vector<endpoint> forward_endpoints;
                std::vector<endpoint> backward_endpoints;
                Lagrange_multiplier_queue forward_queue;
                Lagrange_multiplier_queue backward_queue;
            };

            std::unique_ptr<bdd_sub_base[]> bdd_bases;
            double intra_interval_message_passing_weight;
    };

}
