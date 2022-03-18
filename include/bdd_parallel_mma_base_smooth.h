#pragma once

#include "bdd_sequential_base.h"
#include "bdd_branch_instruction_smooth.h"
#include <limits>
#include <cmath>

namespace LPMP {

    template<typename BDD_BRANCH_NODE>
        class bdd_parallel_mma_base_smooth : public bdd_sequential_base<BDD_BRANCH_NODE> {
            public:
                using value_type = typename BDD_BRANCH_NODE::value_type;
                using base_type = bdd_sequential_base<BDD_BRANCH_NODE>;

                using bdd_sequential_base<BDD_BRANCH_NODE>::bdd_sequential_base;

                void smooth_forward_run();
                void smooth_backward_run();
                void smooth_backward_run(const size_t bdd_nr);
                double smooth_lower_bound();
                void compute_smooth_lower_bound();
                void compute_smooth_lower_bound_after_forward_pass(); 
                void compute_smooth_lower_bound_after_backward_pass(); 
                void set_smoothing(const value_type new_smoothing);
                template<typename COST_ITERATOR>
                    void update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end);

                void smooth_parallel_mma();
                void forward_sm(
                        const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& sms_to_distribute
                        );
                value_type backward_sm(
                        const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& sms_to_distribute
                        );

                void distribute_delta();

            private:
                typename base_type::lower_bound_state smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
                typename base_type::message_passing_state smooth_message_passing_state_ = base_type::message_passing_state::none;
                double smooth_lower_bound_ = -std::numeric_limits<value_type>::infinity();

                value_type smoothing = 1.0;
        };


    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::smooth_forward_run()
        {
            if(smooth_message_passing_state_ == base_type::message_passing_state::after_forward_pass)
                return;
            this->message_passing_state_ = base_type::message_passing_state::none;
            smooth_message_passing_state_ = base_type::message_passing_state::none;

#pragma omp parallel for schedule(static,512)
            for(size_t bdd_nr=0; bdd_nr<this->nr_bdds(); ++bdd_nr)
            {
                // TODO: This only works for non-split BDDs with exactly one root node
                {
                    const auto [first_bdd_node, last_bdd_node] = this->bdd_index_range(bdd_nr,0);
                    assert(first_bdd_node + 1 == last_bdd_node);
                    this->bdd_branch_nodes_[first_bdd_node].m = 1.0;
                    this->bdd_branch_nodes_[first_bdd_node].current_max = 0.0;
                }

                const auto [first_bdd_node, last_bdd_node] = this->bdd_range(bdd_nr);
                // TODO: prepare_smooth_forward_step can be replaced by actually settting all m's and current_max's to appropriate values
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                    this->bdd_branch_nodes_[i].prepare_smooth_forward_step(); 
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                    this->bdd_branch_nodes_[i].smooth_forward_step(); 
            } 
            smooth_message_passing_state_ = base_type::message_passing_state::after_forward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::smooth_backward_run(const size_t bdd_nr)
        {
            const auto [first_bdd_node, last_bdd_node] = this->bdd_range(bdd_nr);
            for(std::ptrdiff_t i=last_bdd_node-1; i>=std::ptrdiff_t(first_bdd_node); --i)
                this->bdd_branch_nodes_[i].smooth_backward_step(); 
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::smooth_backward_run()
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("parallel mma backward_run");
            this->message_passing_state_ = base_type::message_passing_state::none;
            if(smooth_message_passing_state_ == base_type::message_passing_state::after_backward_pass)
                return;
            smooth_message_passing_state_ = base_type::message_passing_state::none;

#pragma omp parallel for schedule(static,512)
            for(std::ptrdiff_t bdd_nr=this->nr_bdds()-1; bdd_nr>=0; --bdd_nr)
                smooth_backward_run(bdd_nr);

            smooth_message_passing_state_ = base_type::message_passing_state::after_backward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        double bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::smooth_lower_bound()
        {
            if(smooth_lower_bound_state_ == base_type::lower_bound_state::invalid)
                compute_smooth_lower_bound();
            assert(smooth_lower_bound_state_ == base_type::lower_bound_state::valid);
            return smooth_lower_bound_; 
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::compute_smooth_lower_bound()
        {
            if(smooth_message_passing_state_ == base_type::message_passing_state::after_backward_pass)
            {
                compute_smooth_lower_bound_after_backward_pass();
            }
            else if(smooth_message_passing_state_ == base_type::message_passing_state::after_forward_pass)
            {
                compute_smooth_lower_bound_after_forward_pass();
            }
            else if(smooth_message_passing_state_ == base_type::message_passing_state::none)
            {
                smooth_backward_run();
                compute_smooth_lower_bound_after_backward_pass();
            }

            assert(smooth_lower_bound_state_ == base_type::lower_bound_state::valid);
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::compute_smooth_lower_bound_after_backward_pass()
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
            assert(smooth_message_passing_state_ == base_type::message_passing_state::after_backward_pass);
            assert(this->constant_ == 0.0);
            double smooth_lb = 0.0;

            // TODO: works only for non-split BDDs
            for(size_t bdd_nr=0; bdd_nr<this->nr_bdds(); ++bdd_nr)
            {
                const auto [first,last] = this->bdd_index_range(bdd_nr, 0);
                assert(first+1 == last);
                smooth_lb += -smoothing * (std::log(this->bdd_branch_nodes_[first].m) + this->bdd_branch_nodes_[first].current_max);
            }

            smooth_lower_bound_state_ = base_type::lower_bound_state::valid;
            smooth_lower_bound_ = smooth_lb;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::compute_smooth_lower_bound_after_forward_pass()
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
            assert(smooth_message_passing_state_ == base_type::message_passing_state::after_forward_pass);
            assert(this->constant_ == 0.0);
            double smooth_lb = 0.0;

            for(size_t bdd_nr=0; bdd_nr<this->nr_bdds(); ++bdd_nr)
            {
                const auto [first,last] = this->bdd_index_range(bdd_nr, this->nr_variables(bdd_nr)-1);
                exp_sum<value_type> s;
                for(size_t idx=first; idx<last; ++idx)
                {
                    const auto sm = this->bdd_branch_nodes_[idx].sum_marginals();
                    s.update(sm[0]);
                    s.update(sm[1]);
                }
                smooth_lb += -smoothing * (std::log(s.sum) + s.max);
            }

            smooth_lower_bound_state_ = base_type::lower_bound_state::valid;
            smooth_lower_bound_ = smooth_lb;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::set_smoothing(const value_type new_smoothing)
        {
            assert(smoothing > 0.0 && new_smoothing > 0.0);
            for(auto& bdd : this->bdd_branch_nodes_)
            {
                bdd.low_cost *= smoothing / new_smoothing;
                bdd.high_cost *= smoothing / new_smoothing;
            }

            smoothing = new_smoothing;
        }

    template<typename BDD_BRANCH_NODE>
        template<typename COST_ITERATOR>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end)
        {
            assert(std::distance(cost_lo_begin, cost_lo_end) <= this->nr_variables());
            assert(std::distance(cost_hi_begin, cost_hi_end) <= this->nr_variables());

            this->message_passing_state_ = base_type::message_passing_state::none;
            this->lower_bound_state_ = base_type::lower_bound_state::invalid;
            smooth_message_passing_state_ = base_type::message_passing_state::none;
            smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;

            auto get_lo_cost = [&](const size_t var) {
                if(var < std::distance(cost_lo_begin, cost_lo_end) && var < this->nr_variables())
                    return *(cost_lo_begin+var)/double(this->nr_bdds(var));
                else
                    return 0.0;
            };
            auto get_hi_cost = [&](const size_t var) {
                if(var < std::distance(cost_hi_begin, cost_hi_end) && var < this->nr_variables())
                    return *(cost_hi_begin+var)/double(this->nr_bdds(var));
                else
                    return 0.0;
            };

            for(size_t bdd_nr=0; bdd_nr<this->nr_bdds(); ++bdd_nr)
            {
                for(size_t bdd_idx=0; bdd_idx<this->nr_variables(bdd_nr); ++bdd_idx)
                {
                    const auto [first_node, last_node] = this->bdd_index_range(bdd_nr, bdd_idx);
                    const size_t var = this->variable(bdd_nr, bdd_idx);
                    const double lo_cost = get_lo_cost(var) / smoothing;
                    assert(std::isfinite(lo_cost));
                    const double hi_cost = get_hi_cost(var) / smoothing;
                    assert(std::isfinite(hi_cost));
                    for(size_t i=first_node; i<last_node; ++i)
                    {
                        if(this->bdd_branch_nodes_[i].offset_low == BDD_BRANCH_NODE::terminal_0_offset)
                            assert(this->bdd_branch_nodes_[i].low_cost == std::numeric_limits<decltype(this->bdd_branch_nodes_[i].low_cost)>::infinity());

                        if(this->bdd_branch_nodes_[i].offset_high == BDD_BRANCH_NODE::terminal_0_offset)
                            assert(this->bdd_branch_nodes_[i].high_cost == std::numeric_limits<decltype(this->bdd_branch_nodes_[i].high_cost)>::infinity());

                        if(this->bdd_branch_nodes_[i].offset_low != BDD_BRANCH_NODE::terminal_0_offset)
                            this->bdd_branch_nodes_[i].low_cost += lo_cost;
                        if(this->bdd_branch_nodes_[i].offset_high != BDD_BRANCH_NODE::terminal_0_offset)
                            this->bdd_branch_nodes_[i].high_cost += hi_cost;
                    }
                }
            }
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::smooth_parallel_mma()
        {
            smooth_backward_run();
            this->message_passing_state_ = base_type::message_passing_state::none;
            this->lower_bound_state_ = base_type::lower_bound_state::invalid;

            auto reset_sms = [&](std::vector<std::array<value_type,2>>& sms) {
                assert(sms.size() == this->nr_variables());
                std::fill(sms.begin(), sms.end(), std::array<value_type,2>{0.0, 0.0});
            };

            auto init_sms = [&](std::vector<std::array<value_type,2>>& sms) {
                if(sms.size() != this->nr_variables())
                {
                    assert(sms.size() == 0.0);
                    sms = std::vector<std::array<value_type,2>>(this->nr_variables(), {0.0, 0.0});
                }
            };

            auto average_sms = [&](std::vector<std::array<value_type,2>>& sms) {
                MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("parallel sum marginal averaging");
#pragma omp parallel for
                for(size_t var=0; var<this->nr_variables(); ++var)
                {
                    assert(this->nr_bdds(var) > 0);
                    sms[var][0] /= value_type(this->nr_bdds(var));
                    sms[var][1] /= value_type(this->nr_bdds(var));
                }
            };

            init_sms(this->delta_out_);
            init_sms(this->delta_in_);

            double lb = this->constant_;

            {
                MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("parallel sum marginal incremental marginal computation");
#pragma omp parallel for schedule(static,256)
                for(size_t bdd_nr=0; bdd_nr<this->nr_bdds(); ++bdd_nr)
                    forward_sm(bdd_nr, 0.5, this->delta_out_, this->delta_in_);
                average_sms(this->delta_out_);
                reset_sms(this->delta_in_);
                std::swap(this->delta_out_, this->delta_in_);
#pragma omp parallel for schedule(static,256) reduction(+:lb)
                for(size_t bdd_nr=0; bdd_nr<this->nr_bdds(); ++bdd_nr)
                    lb += backward_sm(bdd_nr, 0.5, this->delta_out_, this->delta_in_);
                average_sms(this->delta_out_);
                reset_sms(this->delta_in_);
                std::swap(this->delta_out_, this->delta_in_);
            }

            smooth_lower_bound_ = lb;

            smooth_message_passing_state_ = base_type::message_passing_state::after_backward_pass;
            smooth_lower_bound_state_ = base_type::lower_bound_state::valid; 
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::forward_sm(
                const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in)
        {
            assert(delta_out.size() == this->nr_variables());
            assert(delta_in.size() == this->nr_variables());
            assert(omega > 0.0 && omega <= 1.0);
            assert(bdd_nr < this->nr_bdds());

            {
                const auto [first_bdd_node, last_bdd_node] = this->bdd_index_range(bdd_nr, 0);
                assert(first_bdd_node + 1 == last_bdd_node);
                this->bdd_branch_nodes_[first_bdd_node].m = 1.0;
                this->bdd_branch_nodes_[first_bdd_node].current_max = 0.0;
            }

            for(size_t bdd_idx=0; bdd_idx<this->nr_variables(bdd_nr); ++bdd_idx)
            {
                const auto [first_bdd_node, last_bdd_node] = this->bdd_index_range(bdd_nr, bdd_idx);
                const size_t var = this->variable(bdd_nr, bdd_idx);
                std::array<exp_sum<value_type>,2> cur_sm = {exp_sum<value_type>{}, exp_sum<value_type>{}};
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    const auto bdd_sm = this->bdd_branch_nodes_[i].sum_marginals();
                    cur_sm[0].update(bdd_sm[0]);
                    cur_sm[1].update(bdd_sm[1]);
                }

                // TODO: infinity handling
                //if(!std::isfinite(cur_sm[0]))
                //    atomic_store(delta_out[var][0], std::numeric_limits<value_type>::infinity());
                //if(!std::isfinite(cur_sm[1]))
                //    atomic_store(delta_out[var][1], std::numeric_limits<value_type>::infinity());
                const value_type diff_0 = -std::log(cur_sm[0].sum) - cur_sm[0].max; // this gets deleted from low cost
                const value_type diff_1 = -std::log(cur_sm[1].sum) - cur_sm[1].max; // this gets deleted from high cost
                //if(std::isfinite(cur_sm[0]) && std::isfinite(cur_sm[1]))
                {
                    if(diff_0 < diff_1)
                        atomic_add(delta_out[var][1], omega*(diff_1 - diff_0));
                    else
                        atomic_add(delta_out[var][0], omega*(diff_0 - diff_1));
                }

                assert(delta_out[var][0] >= 0.0);
                assert(delta_out[var][1] >= 0.0);

                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    // TODO: infinity handling
                    //if(!std::isfinite(cur_mm[0]))
                    //    bdd_branch_nodes_[i].low_cost = std::numeric_limits<value_type>::infinity();
                    //if(!std::isfinite(cur_mm[1]))
                    //    bdd_branch_nodes_[i].high_cost = std::numeric_limits<value_type>::infinity();
                    //if(std::isfinite(cur_mm[0]) && std::isfinite(cur_mm[1]))
                    {
                        if(diff_0 < diff_1)
                            this->bdd_branch_nodes_[i].high_cost += omega*(diff_0 - diff_1);
                        else
                            this->bdd_branch_nodes_[i].low_cost += omega*(diff_1 - diff_0);
                    }
                }

                if(bdd_idx+1<this->nr_variables(bdd_nr))
                {
                    const auto [next_first_bdd_node, next_last_bdd_node] = this->bdd_index_range(bdd_nr, bdd_idx+1);
                    for(size_t i=next_first_bdd_node; i<next_last_bdd_node; ++i)
                    {
                        this->bdd_branch_nodes_[i].m = 0.0;
                        this->bdd_branch_nodes_[i].current_max = -std::numeric_limits<value_type>::infinity();
                    }
                }
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    this->bdd_branch_nodes_[i].low_cost += delta_in[var][0];
                    this->bdd_branch_nodes_[i].high_cost += delta_in[var][1];
                    this->bdd_branch_nodes_[i].smooth_forward_step(); 
                }
            }
        }

    template<typename BDD_BRANCH_NODE>
        typename BDD_BRANCH_NODE::value_type bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::backward_sm(
                const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in)
        {
            assert(delta_out.size() == this->nr_variables());
            assert(delta_in.size() == this->nr_variables());
            assert(omega > 0.0 && omega <= 1.0);
            assert(bdd_nr < this->nr_bdds());

            for(std::ptrdiff_t bdd_idx=this->nr_variables(bdd_nr)-1; bdd_idx>=0; --bdd_idx)
            {
                const auto [first_bdd_node, last_bdd_node] = this->bdd_index_range(bdd_nr, bdd_idx);
                const size_t var = this->variable(bdd_nr, bdd_idx);
                std::array<exp_sum<value_type>,2> cur_sm = {exp_sum<value_type>{}, exp_sum<value_type>{}};
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    const auto bdd_sm = this->bdd_branch_nodes_[i].sum_marginals();
                    cur_sm[0].update(bdd_sm[0]);
                    cur_sm[1].update(bdd_sm[1]);
                }

                // TODO: infinity handling
                //if(!std::isfinite(cur_sm[0]))
                //    atomic_store(delta_out[var][0], std::numeric_limits<value_type>::infinity());
                //if(!std::isfinite(cur_sm[1]))
                //    atomic_store(delta_out[var][1], std::numeric_limits<value_type>::infinity());
                const value_type diff_0 = -std::log(cur_sm[0].sum) - cur_sm[0].max; // this gets deleted from low cost
                const value_type diff_1 = -std::log(cur_sm[1].sum) - cur_sm[1].max; // this gets deleted from high cost
                //if(std::isfinite(cur_sm[0]) && std::isfinite(cur_sm[1]))
                {
                    if(diff_0 < diff_1)
                        atomic_add(delta_out[var][1], omega*(diff_1 - diff_0));
                    else
                        atomic_add(delta_out[var][0], omega*(diff_0 - diff_1));
                }

                assert(delta_out[var][0] >= 0.0);
                assert(delta_out[var][1] >= 0.0);

                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    // TODO: infinity handling
                    //if(!std::isfinite(cur_mm[0]))
                    //    bdd_branch_nodes_[i].low_cost = std::numeric_limits<value_type>::infinity();
                    //if(!std::isfinite(cur_mm[1]))
                    //    bdd_branch_nodes_[i].high_cost = std::numeric_limits<value_type>::infinity();
                    //if(std::isfinite(cur_mm[0]) && std::isfinite(cur_mm[1]))
                    {
                        if(diff_0 < diff_1)
                            this->bdd_branch_nodes_[i].high_cost += omega*(diff_0 - diff_1);
                        else
                            this->bdd_branch_nodes_[i].low_cost += omega*(diff_1 - diff_0);
                    }
                }

                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    this->bdd_branch_nodes_[i].low_cost += delta_in[var][0];
                    this->bdd_branch_nodes_[i].high_cost += delta_in[var][1];
                    this->bdd_branch_nodes_[i].smooth_backward_step(); 
                }
            }

            const auto [first,last] = this->bdd_index_range(bdd_nr, 0);
            assert(first+1 == last);
            return -smoothing * (std::log(this->bdd_branch_nodes_[first].m) + this->bdd_branch_nodes_[first].current_max);
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::distribute_delta()
        {
            smooth_message_passing_state_ = base_type::message_passing_state::none;
            smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
            base_type::distribute_delta();
        }


}
