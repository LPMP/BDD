#pragma once 

#include <vector>
#include <cmath>
#include <cassert>
#include <thread>
#include <mutex>
#include <condition_variable>
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
            void solve(const size_t max_iter);
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
                        size_t interval;
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

                // we first write Lagrange multipliers that need to be synchronized via the queue in this cache in order to save synchronization calls
                std::vector<Lagrange_multiplier> queue_cache_forward;
                std::vector<size_t> queue_cache_offset_forward;
                size_t queue_cache_offset_counter_forward;

                std::vector<Lagrange_multiplier> queue_cache_backward;
                std::vector<size_t> queue_cache_offset_backward;
                size_t queue_cache_offset_counter_backward;

                void read_in_Lagrange_multipliers_from_queue(std::mutex& queue_mutex, std::queue<typename bdd_sub_base::Lagrange_multiplier>& queue);
                void write_Lagrange_multiplers_to_queue(
                        const size_t var, const size_t bdd_index, 
                        const std::vector<double>& multipliers,
                        std::mutex& queue_mutex, std::queue<Lagrange_multiplier>& queue); 

                void init_queue_cache();
                void init_queue_cache_forward();
                void init_queue_cache_backward();
            };

            std::unique_ptr<bdd_sub_base[]> bdd_bases;
            // for waking up when lower bound is changed
            std::mutex lb_mutex;
            std::condition_variable cv_lb;

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

        for(size_t i=0; i<nr_intervals; ++i)
            bdd_bases[i].init_queue_cache();
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

    void decomposition_bdd_base::solve(const size_t max_iter)
    {
        std::cout << "initial lower bound = " << lower_bound() << "\n";
        std::vector<std::thread> threads;
        
        auto mma = [&](const size_t thread_nr) {
            for(size_t i=0; i<max_iter; ++i)
            {
                this->min_marginal_averaging_forward(thread_nr);
                this->min_marginal_averaging_backward(thread_nr);
                bdd_bases[thread_nr].base.compute_lower_bound();
                if(thread_nr == 0)
                    std::cout << "iteration " << i << ", lower bound = " << this->lower_bound() << "\n";
            }
        };

        for(size_t t=0; t<intervals.nr_intervals(); ++t)
            threads.push_back(std::thread(mma, t)); 

        for(auto& t : threads)
            t.join(); 

        backward_run(); // To flush out the Lagrange multiplier queues and recompute final lower bound
        std::cout << "final lower bound = " << lower_bound() << "\n"; 
    }

    void decomposition_bdd_base::iteration()
    {
        throw std::runtime_error("not supported\n");
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

    void decomposition_bdd_base::bdd_sub_base::init_queue_cache_forward()
    {
        std::vector<size_t> queue_size; // per interval
        for(size_t i=0; i<base.nr_variables(); ++i)
        {
            for(size_t bdd_index=0; bdd_index<base.nr_bdds(i); ++bdd_index)
            {
                const auto& bdd_var = base.get_bdd_variable(i, bdd_index);
                if(base.last_variable_of_bdd(i, bdd_index) && bdd_var.is_left_side_of_split())
                {
                    const size_t next_interval_nr = bdd_var.split.interval;
                    if(queue_size.size() <= next_interval_nr)
                        queue_size.resize(next_interval_nr+1);
                    queue_size[next_interval_nr] += 4 + base.nr_feasible_outgoing_arcs(i, bdd_index);
                }
            }
        }
        if(queue_size.size() == 0)
            return;
        queue_cache_forward.resize(std::accumulate(queue_size.begin(), queue_size.end(), 0));

        // compute offsets
        std::vector<size_t> interval_offsets = {0}; // calculate cumulative sum
        std::partial_sum(queue_size.begin(), queue_size.end()-1, std::back_inserter(interval_offsets));
        for(size_t i=0; i<base.nr_variables(); ++i)
        {
            for(size_t bdd_index=0; bdd_index<base.nr_bdds(i); ++bdd_index)
            {
                const auto& bdd_var = base.get_bdd_variable(i, bdd_index);
                if(base.last_variable_of_bdd(i, bdd_index) && bdd_var.is_left_side_of_split())
                {
                    const size_t next_interval_nr = bdd_var.split.interval;
                    queue_cache_offset_forward.push_back(interval_offsets[next_interval_nr]);
                    interval_offsets[next_interval_nr] += 4 + base.nr_feasible_outgoing_arcs(i, bdd_index);
                }
            }
        }
    }

    void decomposition_bdd_base::bdd_sub_base::init_queue_cache_backward()
    {
        std::vector<size_t> queue_size; // per interval
        for(std::ptrdiff_t i=base.nr_variables()-1; i>=0; --i)
        {
            for(size_t bdd_index=0; bdd_index<base.nr_bdds(i); ++bdd_index)
            {
                const auto& bdd_var = base.get_bdd_variable(i, bdd_index);
                if(base.first_variable_of_bdd(i, bdd_index) && bdd_var.is_right_side_of_split())
                {
                    const size_t prev_interval_nr = bdd_var.split.interval;
                    if(queue_size.size() <= prev_interval_nr)
                        queue_size.resize(prev_interval_nr+1);
                    queue_size[prev_interval_nr] += 4 + base.nr_feasible_outgoing_arcs(i, bdd_index);
                }
            }
        }
        if(queue_size.size() == 0)
            return;
        queue_cache_backward.resize(std::accumulate(queue_size.begin(), queue_size.end(), 0));

        // compute offsets
        std::vector<size_t> interval_offsets = {0}; // calculate cumulative sum
        std::partial_sum(queue_size.begin(), queue_size.end()-1, std::back_inserter(interval_offsets));
        for(std::ptrdiff_t i=base.nr_variables()-1; i>=0; --i)
        {
            for(size_t bdd_index=0; bdd_index<base.nr_bdds(i); ++bdd_index)
            {
                const auto& bdd_var = base.get_bdd_variable(i, bdd_index);
                if(base.first_variable_of_bdd(i, bdd_index) && bdd_var.is_right_side_of_split())
                {
                    const size_t prev_interval_nr = bdd_var.split.interval;
                    queue_cache_offset_backward.push_back(interval_offsets[prev_interval_nr]);
                        interval_offsets[prev_interval_nr] += 4 + base.nr_feasible_outgoing_arcs(i, bdd_index);
                    }
                }
            }
    }

    void decomposition_bdd_base::bdd_sub_base::init_queue_cache()
    {
        init_queue_cache_forward();
        init_queue_cache_backward();
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

        auto& queue_counter = bdd_bases[interval_nr].queue_cache_offset_counter_forward;
        queue_counter = 0;
        auto& queue = bdd_bases[interval_nr].queue_cache_forward;
        const auto& queue_offsets = bdd_bases[interval_nr].queue_cache_offset_forward;

        std::vector<std::array<double,2>> min_marginals;
        std::vector<double> arc_marginals;
        for(size_t i=0; i<bdd_bases[interval_nr].base.nr_variables(); ++i)
        {
            if(bdd_bases[interval_nr].base.nr_bdds(i) == 0)
                continue;

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
                        //bdd_bases[next_interval_nr].base.update_arc_costs(i, next_bdd_index, arc_marginals.begin(), arc_marginals.end());
                        //bdd_bases[next_interval_nr].write_Lagrange_multiplers_to_queue(
                        //        i, next_bdd_index, arc_marginals, 
                        //        bdd_bases[next_interval_nr].Lagrange_multipliers_mutex_left,
                        //        bdd_bases[next_interval_nr].Lagrange_multipliers_queue_left
                        //        );
                        
                        // put Lagrange multipliers into queue
                        bdd_sub_base::Lagrange_multiplier L;
                        assert(queue_counter <= queue.size());
                        size_t queue_offset = queue_offsets[queue_counter++];
                        L.interval = next_interval_nr;
                        queue[queue_offset++] = L;
                        L.variable = i;
                        queue[queue_offset++] = L;
                        L.bdd_index = next_bdd_index;
                        queue[queue_offset++] = L;
                        L.nr_deltas = arc_marginals.size();
                        queue[queue_offset++] = L;
                        for(const auto x : arc_marginals)
                        {
                            assert(x >= 0.0);
                            L.delta = x;
                            queue[queue_offset++] = L;
                        }
                    }
                }
            }
            bdd_bases[interval_nr].base.min_marginal_averaging_step_forward_tmp(i, min_marginals);
        }

        // read out queue cache
        size_t last_interval_nr = std::numeric_limits<size_t>::max();    
        for(size_t l=0; l<queue.size(); )
        {
            const size_t next_interval_nr = queue[l++].interval;
            assert(next_interval_nr < intervals.nr_intervals());
            assert(next_interval_nr != interval_nr);
            if(last_interval_nr != next_interval_nr)
            {
                if(last_interval_nr != std::numeric_limits<size_t>::max())
                    bdd_bases[last_interval_nr].Lagrange_multipliers_mutex_left.unlock();
                bdd_bases[next_interval_nr].Lagrange_multipliers_mutex_left.lock();
                last_interval_nr = next_interval_nr;
            }
            auto& queue_to_push = bdd_bases[next_interval_nr].Lagrange_multipliers_queue_left;
            const size_t var = queue[l].variable;
            assert(var < bdd_bases[interval_nr].base.nr_variables());
            queue_to_push.push(queue[l++]); // variable
            const size_t bdd_index = queue[l].bdd_index;
            assert(bdd_index < bdd_bases[next_interval_nr].base.nr_bdds(var));
            queue_to_push.push(queue[l++]); // bdd index
            const size_t nr_deltas = queue[l].nr_deltas;
            assert(nr_deltas == bdd_bases[next_interval_nr].base.nr_feasible_outgoing_arcs(var, bdd_index));
            queue_to_push.push(queue[l++]);
            for(size_t c=0; c<nr_deltas; ++c)
                queue_to_push.push(queue[l++]);
        }
        if(last_interval_nr != std::numeric_limits<size_t>::max())
            bdd_bases[last_interval_nr].Lagrange_multipliers_mutex_left.unlock();
    }

    void decomposition_bdd_base::min_marginal_averaging_backward(const size_t interval_nr)
    {
        // read out Lagrange multipliers sent in from different intervals and affected update arc costs
        if(interval_nr == intervals.nr_intervals()-1)
            assert(bdd_bases[interval_nr].Lagrange_multipliers_queue_right.empty());
        bdd_bases[interval_nr].read_in_Lagrange_multipliers_from_queue(
                bdd_bases[interval_nr].Lagrange_multipliers_mutex_right,
                bdd_bases[interval_nr].Lagrange_multipliers_queue_right);

        auto& queue_counter = bdd_bases[interval_nr].queue_cache_offset_counter_backward;
        queue_counter = 0;
        auto& queue = bdd_bases[interval_nr].queue_cache_backward;
        const auto& queue_offsets = bdd_bases[interval_nr].queue_cache_offset_backward;

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
                        //bdd_bases[prev_interval_nr].base.update_arc_costs(i, prev_bdd_index, arc_marginals.begin(), arc_marginals.end());
                        //bdd_bases[prev_interval_nr].write_Lagrange_multiplers_to_queue(
                        //        i, prev_bdd_index, arc_marginals, 
                        //        bdd_bases[prev_interval_nr].Lagrange_multipliers_mutex_right,
                        //        bdd_bases[prev_interval_nr].Lagrange_multipliers_queue_right
                        //        );

                        // put Lagrange multipliers into queue
                        bdd_sub_base::Lagrange_multiplier L;
                        assert(queue_counter <= queue.size());
                        size_t queue_offset = queue_offsets[queue_counter++];
                        L.interval = prev_interval_nr;
                        queue[queue_offset++] = L;
                        L.variable = i;
                        queue[queue_offset++] = L;
                        L.bdd_index = prev_bdd_index;
                        queue[queue_offset++] = L;
                        L.nr_deltas = arc_marginals.size();
                        queue[queue_offset++] = L;
                        for(const auto x : arc_marginals)
                        {
                            assert(x >= 0.0);
                            L.delta = x;
                            queue[queue_offset++] = L;
                        }

                    }
                }
            }
            bdd_bases[interval_nr].base.min_marginal_averaging_step_backward(i, min_marginals);
        }

        // read out queue cache
        size_t last_interval_nr = std::numeric_limits<size_t>::max();    
        for(size_t l=0; l<queue.size(); )
        {
            const size_t prev_interval_nr = queue[l++].interval;
            assert(prev_interval_nr < intervals.nr_intervals());
            assert(prev_interval_nr != interval_nr);
            if(last_interval_nr != prev_interval_nr)
            {
                if(last_interval_nr != std::numeric_limits<size_t>::max())
                    bdd_bases[last_interval_nr].Lagrange_multipliers_mutex_right.unlock();
                bdd_bases[prev_interval_nr].Lagrange_multipliers_mutex_right.lock();
                last_interval_nr = prev_interval_nr;
            }
            auto& queue_to_push = bdd_bases[prev_interval_nr].Lagrange_multipliers_queue_right;
            const size_t var = queue[l].variable;
            assert(var < bdd_bases[interval_nr].base.nr_variables());
            queue_to_push.push(queue[l++]); // variable
            const size_t bdd_index = queue[l].bdd_index;
            assert(bdd_index < bdd_bases[prev_interval_nr].base.nr_bdds(var));
            queue_to_push.push(queue[l++]); // bdd index
            const size_t nr_deltas = queue[l].nr_deltas;
            assert(nr_deltas == bdd_bases[prev_interval_nr].base.nr_feasible_outgoing_arcs(var, bdd_index));
            queue_to_push.push(queue[l++]);
            for(size_t c=0; c<nr_deltas; ++c)
                queue_to_push.push(queue[l++]);
        }
        if(last_interval_nr != std::numeric_limits<size_t>::max())
            bdd_bases[last_interval_nr].Lagrange_multipliers_mutex_right.unlock();
    }

    double decomposition_bdd_base::lower_bound()
    {
        double lb = 0;
        for(size_t i=0; i<intervals.nr_intervals(); ++i)
            lb += bdd_bases[i].base.lower_bound();
        return lb;
    }

}
