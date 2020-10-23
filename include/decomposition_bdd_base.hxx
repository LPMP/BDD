#pragma once 

#include <vector>
#include <cmath>
#include <cassert>
#include <thread>
#include <mutex>
#include <queue>
#include "decomposition_bdd_mma.h"
#include "bdd_variable.h"
#include "bdd_branch_node.h"
#include "bdd_mma_base.hxx"
#include <iostream> // TODO: remove

namespace LPMP {

    class decomposition_bdd_base {
        public:
            decomposition_bdd_base(bdd_storage& stor, decomposition_bdd_mma::options opt);

            using bdd_sequential_base = bdd_mma_base_arc_costs<bdd_variable_split, bdd_branch_node_opt_arc_cost>;

            void set_cost(const double c, const size_t var);
            void backward_run();
            void iteration();
            double lower_bound();

        private:
            void min_marginal_averaging_forward(const size_t interval_nr);
            void min_marginal_averaging_backward(const size_t interval_nr);

            bdd_storage::intervals intervals;
            std::vector<double> costs;

            struct bdd_sub_base {
                bdd_sequential_base base;

                // left and right are for Lagrange multipliers for beginning resp. end of split BDDs.
                // Entries are (i) variable, (ii) bdd_index, (iii) # multipliers, (iv) multipliers
                struct Lagrange_multiplier {
                    union {
                        size_t variable;
                        size_t bdd_index;
                        size_t nr_deltas;
                        double delta;
                    };
                };

                std::mutex Lagrange_multipliers_mutex_left;
                std::queue<Lagrange_multiplier> Lagrange_multipliers_queue_left;

                std::mutex Lagrange_multipliers_mutex_right;
                std::queue<Lagrange_multiplier> Lagrange_multipliers_queue_right;

                void read_in_Lagrange_multipliers_from_queue(std::mutex& queue_mutex, std::queue<typename bdd_sub_base::Lagrange_multiplier>& queue);
                void write_Lagrange_multiplers_to_queue(
                        const size_t var, const size_t bdd_index, 
                        const std::vector<double>& multipliers,
                        std::mutex& queue_mutex, std::queue<Lagrange_multiplier>& queue); 

            };

            std::unique_ptr<bdd_sub_base[]> bdd_bases;

            double intra_interval_message_passing_weight;
    };

    inline decomposition_bdd_base::decomposition_bdd_base(bdd_storage& bdd_storage_, decomposition_bdd_mma::options opt)
        : intervals(bdd_storage_.compute_intervals(opt.nr_threads)),
        intra_interval_message_passing_weight(opt.parallel_message_passing_weight)

    {
        std::cout << "decomposing BDDs into " << intervals.nr_intervals() << " intervals: ";
        for(size_t i=0; i<intervals.nr_intervals(); ++i)
        {
            std::cout << "[" << intervals.interval_boundaries[i] << "," << intervals.interval_boundaries[i+1] << "), ";
        }
        std::cout << "\n";

        assert(intra_interval_message_passing_weight >= 0 && intra_interval_message_passing_weight <= 1.0);
        const size_t nr_intervals = opt.nr_threads;
        auto [bdd_storages, duplicated_bdd_variables] = bdd_storage_.split_bdd_nodes(nr_intervals);

        bdd_bases = std::make_unique<bdd_sub_base[]>(nr_intervals);
        two_dim_variable_array<typename bdd_sequential_base::bdd_endpoints_> bdd_endpoints;

        for(size_t i=0; i<nr_intervals; ++i)
        {
            bdd_bases[i].base.init(bdd_storages[i]); 
            const auto current_endpoints = bdd_bases[i].base.bdd_endpoints(bdd_storages[i]);
            bdd_endpoints.push_back(current_endpoints.begin(), current_endpoints.end());
        }

        for(const auto [interval_1, bdd_nr_1, interval_2, bdd_nr_2] : duplicated_bdd_variables)
        {
            assert(interval_1 < interval_2);
            const auto [first_bdd_var_1, first_bdd_index_1, last_bdd_var_1, last_bdd_index_1] = bdd_endpoints(interval_1, bdd_nr_1);
            const auto [first_bdd_var_2, first_bdd_index_2, last_bdd_var_2, last_bdd_index_2] = bdd_endpoints(interval_2, bdd_nr_2);

            //std::cout << first_bdd_var_1 << "," <<  first_bdd_index_1 << "," <<  last_bdd_var_1 << "," <<  last_bdd_index_1 << "\n";
            //std::cout << first_bdd_var_2 << "," <<  first_bdd_index_2 << "," <<  last_bdd_var_2 << "," <<  last_bdd_index_2 << "\n";
            //std::cout << "\n";

            assert(last_bdd_var_1 == first_bdd_var_2);

            auto& var_1 = bdd_bases[interval_1].base.get_bdd_variable(last_bdd_var_1, last_bdd_index_1);
            assert(var_1.split.interval == std::numeric_limits<size_t>::max());
            var_1.split.interval = interval_2;
            var_1.split.bdd_index = first_bdd_index_2;
            var_1.split.is_left_side_of_split = 1;

            auto& var_2 = bdd_bases[interval_2].base.get_bdd_variable(first_bdd_var_2, first_bdd_index_2);
            assert(var_2.split.interval == std::numeric_limits<size_t>::max());
            var_2.split.interval = interval_1;
            var_2.split.bdd_index = last_bdd_index_1;
            var_2.split.is_left_side_of_split = 0;
        }

        costs.clear();
        costs.resize(bdd_storage_.nr_variables(), 0.0);
    }

    void decomposition_bdd_base::set_cost(const double c, const size_t var)
    {
        assert(var < costs.size());
        assert(intervals.interval(var) < intervals.nr_intervals());
        assert(costs[var] == 0.0);
        costs[var] = c;
        size_t nr_bdds = 0;
        for(size_t i=0; i<intervals.nr_intervals(); ++i)
            nr_bdds += bdd_bases[i].base.nr_bdds(i);

        for(size_t i=0; i<intervals.nr_intervals(); ++i)
        {
            const size_t nr_interval_bdds = bdd_bases[i].base.nr_bdds(i);
            if(nr_interval_bdds > 0)
            {
                const double interval_cost = double(nr_interval_bdds)/double(nr_bdds)*c;
                bdd_bases[intervals.interval(var)].base.set_cost(interval_cost, var);
            }
        }
    }

    void decomposition_bdd_base::backward_run()
    {
        // TODO: parallelize
 
            /*
        for(size_t t=0; t<intervals.nr_intervals(); ++t)
        {
            // flush out Lagrange multipliers
            bdd_bases[t].read_in_Lagrange_multipliers_from_queue(
                    bdd_bases[t].Lagrange_multipliers_mutex_left,
                    bdd_bases[t].Lagrange_multipliers_queue_left);

            bdd_bases[t].read_in_Lagrange_multipliers_from_queue(
                bdd_bases[t].Lagrange_multipliers_mutex_right,
                bdd_bases[t].Lagrange_multipliers_queue_right); 

            bdd_bases[t].base.backward_run();
            bdd_bases[t].base.compute_lower_bound();
        }
        */
        for(std::ptrdiff_t interval_nr=intervals.nr_intervals()-1; interval_nr>=0; --interval_nr)
        {
            bdd_bases[interval_nr].read_in_Lagrange_multipliers_from_queue(
                    bdd_bases[interval_nr].Lagrange_multipliers_mutex_right,
                    bdd_bases[interval_nr].Lagrange_multipliers_queue_right);
            bdd_bases[interval_nr].base.backward_run();

            std::vector<double> arc_marginals;
            for(std::ptrdiff_t i=bdd_bases[interval_nr].base.nr_variables()-1; i>=0; --i)
            {
                if(bdd_bases[interval_nr].base.nr_bdds(i) == 0)
                    continue;
                for(size_t bdd_index=0; bdd_index<bdd_bases[interval_nr].base.nr_bdds(i); ++bdd_index)
                {
                    if(bdd_bases[interval_nr].base.first_variable_of_bdd(i, bdd_index)) // check if bdd is split
                    {
                        const auto& bdd_var = bdd_bases[interval_nr].base.get_bdd_variable(i, bdd_index);
                        if(bdd_var.is_right_side_of_split())
                        {
                            const size_t prev_interval_nr = bdd_var.split.interval;
                            assert(prev_interval_nr < interval_nr);
                            const size_t prev_bdd_index = bdd_var.split.bdd_index;
                            bdd_bases[interval_nr].base.get_arc_marginals(i, bdd_index, arc_marginals);
                            const double min_arc_cost = *std::min_element(arc_marginals.begin(), arc_marginals.end());
                            for(auto& x : arc_marginals)
                            {
                                x -= min_arc_cost;
                                x *= -1.0;
                                assert(x <= 0.0);
                            }
                            assert(*std::max_element(arc_marginals.begin(), arc_marginals.end()) == 0.0);
                            bdd_bases[interval_nr].base.update_arc_costs(i, bdd_index, arc_marginals.begin(), arc_marginals.end());

                            for(auto& x : arc_marginals)
                                x *= -1.0;
                            bdd_bases[prev_interval_nr].base.update_arc_costs(i, prev_bdd_index, arc_marginals.begin(), arc_marginals.end());

                            /*
                            bdd_bases[prev_interval_nr].write_Lagrange_multiplers_to_queue(
                                    i, prev_bdd_index, arc_marginals, 
                                    bdd_bases[prev_interval_nr].Lagrange_multipliers_mutex_right,
                                    bdd_bases[prev_interval_nr].Lagrange_multipliers_queue_right
                                    );
                                    */
                        }
                    }
                } 
            }
            bdd_bases[interval_nr].base.compute_lower_bound();
        }
    }

    void decomposition_bdd_base::iteration()
    {
        std::vector<std::thread> threads;
        threads.reserve(intervals.nr_intervals());
        for(size_t t=0; t<intervals.nr_intervals(); ++t)
        {
            auto forward_mma = [&](const size_t thread_nr) {
                this->min_marginal_averaging_forward(thread_nr);
                this->min_marginal_averaging_backward(thread_nr);
                bdd_bases[thread_nr].base.compute_lower_bound();
                //std::cout << "lower bound for interval " << thread_nr << ": " << bdd_bases[thread_nr].base.lower_bound() << "\n";
            };
            threads.push_back(std::thread(forward_mma, t)); 
        }

        for(auto& t : threads)
            t.join(); 

        /*
        for(size_t t=0; t<intervals.nr_intervals(); ++t)
                this->min_marginal_averaging_forward(t);
        for(std::ptrdiff_t t=intervals.nr_intervals()-1; t>=0; --t)
                this->min_marginal_averaging_backward(t);
        for(std::ptrdiff_t t=intervals.nr_intervals()-1; t>=0; --t)
                bdd_bases[t].base.compute_lower_bound();
                */
    }

    void decomposition_bdd_base::bdd_sub_base::read_in_Lagrange_multipliers_from_queue(std::mutex& queue_mutex, std::queue<typename decomposition_bdd_base::bdd_sub_base::Lagrange_multiplier>& queue)
    {
        std::vector<double> deltas;
        std::lock_guard<std::mutex> lock(queue_mutex);
        while(!queue.empty())
        {
            assert(queue.size() >= 4);
            const double variable = queue.front().variable;
            queue.pop();
            const double bdd_index = queue.front().bdd_index;
            queue.pop();
            const double nr_deltas = queue.front().nr_deltas;
            queue.pop();
            assert(queue.size() >= nr_deltas);
            deltas.clear();
            for(size_t c=0; c<nr_deltas; ++c)
            {
                deltas.push_back(queue.front().delta);
                assert(deltas.back() >= 0.0);
                queue.pop();
            }

            base.update_arc_costs(variable, bdd_index, deltas.begin(), deltas.end()); 
        }
    }

    void decomposition_bdd_base::bdd_sub_base::write_Lagrange_multiplers_to_queue(
            const size_t var, const size_t bdd_index, 
            const std::vector<double>& multipliers,
            std::mutex& queue_mutex, std::queue<typename decomposition_bdd_base::bdd_sub_base::Lagrange_multiplier>& queue)
    {
        std::lock_guard<std::mutex> lock(queue_mutex);

        bdd_sub_base::Lagrange_multiplier L;
        L.variable = var;
        queue.push(L);
        L.bdd_index = bdd_index;
        queue.push(L);
        L.nr_deltas = multipliers.size();
        queue.push(L);
        for(const auto x : multipliers)
        {
            assert(x >= 0.0);
            L.delta = x;
            queue.push(L);
        }
    }

    // after forward pass, collect min marginals of all duplicated arcs are computed and sent
    void decomposition_bdd_base::min_marginal_averaging_forward(const size_t interval_nr)
    {
        // read out Lagrange multipliers sent in from different intervals and affected update arc costs
        if(interval_nr == 0)
            assert(bdd_bases[interval_nr].Lagrange_multipliers_queue_left.empty());
        bdd_bases[interval_nr].read_in_Lagrange_multipliers_from_queue(
                bdd_bases[interval_nr].Lagrange_multipliers_mutex_left,
                bdd_bases[interval_nr].Lagrange_multipliers_queue_left);

        std::vector<std::array<double,2>> min_marginals;
        std::vector<double> arc_marginals;
        for(size_t i=0; i<bdd_bases[interval_nr].base.nr_variables(); ++i)
        {
            if(bdd_bases[interval_nr].base.nr_bdds(i) == 0)
                continue;

            bdd_bases[interval_nr].base.min_marginal_averaging_step_forward(i, min_marginals);
            for(size_t bdd_index=0; bdd_index<bdd_bases[interval_nr].base.nr_bdds(i); ++bdd_index)
            {
                if(bdd_bases[interval_nr].base.last_variable_of_bdd(i, bdd_index)) // check if bdd is split
                {
                    const auto& bdd_var = bdd_bases[interval_nr].base.get_bdd_variable(i, bdd_index);
                    if(bdd_var.is_left_side_of_split())
                    {
                        const size_t next_interval_nr = bdd_var.split.interval;
                        assert(next_interval_nr > interval_nr);
                        const size_t next_bdd_index = bdd_var.split.bdd_index;
                        bdd_bases[interval_nr].base.get_arc_marginals(i, bdd_index, arc_marginals);
                        const double min_arc_cost = *std::min_element(arc_marginals.begin(), arc_marginals.end());
                        for(auto& x : arc_marginals)
                        {
                            x -= min_arc_cost;
                            x *= -intra_interval_message_passing_weight;
                            assert(x <= 0.0);
                        }
                        assert(*std::max_element(arc_marginals.begin(), arc_marginals.end()) == 0.0);
                        bdd_bases[interval_nr].base.update_arc_costs(i, bdd_index, arc_marginals.begin(), arc_marginals.end());

                        for(auto& x : arc_marginals)
                            x *= -1.0;

                        //std::cout << "write Lagrange multipliers into queue ->: variable " << i << ", (" << interval_nr << "," << bdd_index << ") -> (" << next_interval_nr << "," << next_bdd_index << ")\n";
                        bdd_bases[next_interval_nr].write_Lagrange_multiplers_to_queue(
                                i, next_bdd_index, arc_marginals, 
                                bdd_bases[next_interval_nr].Lagrange_multipliers_mutex_left,
                                bdd_bases[next_interval_nr].Lagrange_multipliers_queue_left
                                );
                    }
                }
            }
        }
    }

    void decomposition_bdd_base::min_marginal_averaging_backward(const size_t interval_nr)
    {
        // read out Lagrange multipliers sent in from different intervals and affected update arc costs
        if(interval_nr == intervals.nr_intervals()-1)
            assert(bdd_bases[interval_nr].Lagrange_multipliers_queue_right.empty());
        bdd_bases[interval_nr].read_in_Lagrange_multipliers_from_queue(
                bdd_bases[interval_nr].Lagrange_multipliers_mutex_right,
                bdd_bases[interval_nr].Lagrange_multipliers_queue_right);

        std::vector<std::array<double,2>> min_marginals;
        std::vector<double> arc_marginals;
        for(std::ptrdiff_t i=bdd_bases[interval_nr].base.nr_variables()-1; i>=0; --i)
        {
            if(bdd_bases[interval_nr].base.nr_bdds(i) == 0)
                continue;

            for(size_t bdd_index=0; bdd_index<bdd_bases[interval_nr].base.nr_bdds(i); ++bdd_index)
            {
                if(bdd_bases[interval_nr].base.first_variable_of_bdd(i, bdd_index)) // check if bdd is split
                {
                    const auto& bdd_var = bdd_bases[interval_nr].base.get_bdd_variable(i, bdd_index);
                    if(bdd_var.is_right_side_of_split())
                    {
                        const size_t prev_interval_nr = bdd_var.split.interval;
                        assert(prev_interval_nr < interval_nr);
                        const size_t prev_bdd_index = bdd_var.split.bdd_index;
                        bdd_bases[interval_nr].base.get_arc_marginals(i, bdd_index, arc_marginals);
                        const double min_arc_cost = *std::min_element(arc_marginals.begin(), arc_marginals.end());
                        for(auto& x : arc_marginals)
                        {
                            x -= min_arc_cost;
                            x *= -intra_interval_message_passing_weight;
                            assert(x <= 0.0);
                        }
                        assert(*std::max_element(arc_marginals.begin(), arc_marginals.end()) == 0.0);
                        bdd_bases[interval_nr].base.update_arc_costs(i, bdd_index, arc_marginals.begin(), arc_marginals.end());

                        for(auto& x : arc_marginals)
                            x *= -1.0;

                        //std::cout << "write Lagrange multipliers into queue <-: variable " << i << ", (" << interval_nr << "," << bdd_index << ") -> (" << prev_interval_nr << "," << prev_bdd_index << ")\n";
                        bdd_bases[prev_interval_nr].write_Lagrange_multiplers_to_queue(
                                i, prev_bdd_index, arc_marginals, 
                                bdd_bases[prev_interval_nr].Lagrange_multipliers_mutex_right,
                                bdd_bases[prev_interval_nr].Lagrange_multipliers_queue_right
                                );
                    }
                }
            }
            bdd_bases[interval_nr].base.min_marginal_averaging_step_backward(i, min_marginals);
        }
    }

    double decomposition_bdd_base::lower_bound()
    {
        double lb = 0;
        for(size_t i=0; i<intervals.nr_intervals(); ++i)
            lb += bdd_bases[i].base.lower_bound();
        return lb;
    }

}
