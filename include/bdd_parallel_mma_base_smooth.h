#pragma once

#include "bdd_parallel_mma_base.h"
#include "bdd_branch_instruction_smooth.h"
#include "exp_sum.h"
#include <limits>
#include <cmath>

namespace LPMP {

    template<typename BDD_BRANCH_NODE>
        class bdd_parallel_mma_base_smooth : public bdd_parallel_mma_base<BDD_BRANCH_NODE> {
            public:
                using value_type = typename BDD_BRANCH_NODE::value_type;
                using base_type = bdd_parallel_mma_base<BDD_BRANCH_NODE>;

                using bdd_parallel_mma_base<BDD_BRANCH_NODE>::bdd_parallel_mma_base;

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

                void parallel_sma();
                void forward_sm(
                        const typename BDD_BRANCH_NODE::value_type omega,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in
                        );
                void forward_sm(
                        const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in
                        );
                // returns smooth and non-smooth lower bound
                std::tuple<double,double> backward_sm(
                        const typename BDD_BRANCH_NODE::value_type omega,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in
                        );
                std::tuple<double,double> backward_sm(
                        const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                        std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in
                        );

                void distribute_delta();

            private:
                typename base_type::lower_bound_state smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
                typename base_type::message_passing_state smooth_message_passing_state_ = base_type::message_passing_state::none;
                double smooth_lower_bound_ = -std::numeric_limits<value_type>::infinity();

                double delta_smooth_lower_bound(const std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& vec) const;

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
            if(this->delta_out_.size() != 0)
            {
                assert(this->delta_out_.size() == this->nr_variables());
                smooth_lower_bound_ += delta_smooth_lower_bound(this->delta_out_);
            }
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
            if(this->delta_out_.size() != 0)
            {
                assert(this->delta_out_.size() == this->nr_variables());
                smooth_lower_bound_ += delta_smooth_lower_bound(this->delta_out_);
            }
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
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::parallel_sma()
        {
            static int iter = 0;
            iter++;
            if(iter == 50)
            {
                distribute_delta();
                smooth_backward_run();
                compute_smooth_lower_bound_after_backward_pass();
                std::cout << "smooth lower bound after distributing delta = " << smooth_lower_bound_ << "\n";
            }

            smooth_backward_run();
            this->message_passing_state_ = base_type::message_passing_state::none;
            this->lower_bound_state_ = base_type::lower_bound_state::invalid;

            constexpr static value_type omega = 0.051;

            // TODO: rename and reuse functions in sequential base
            auto reset_delta = [&](std::vector<std::array<value_type,2>>& delta) {
                assert(delta.size() == this->nr_variables());
                std::fill(delta.begin(), delta.end(), std::array<value_type,2>{0.0, 0.0});
            };

            auto init_delta = [&](std::vector<std::array<value_type,2>>& delta) {
                if(delta.size() != this->nr_variables())
                {
                    assert(delta.size() == 0.0);
                    delta = std::vector<std::array<value_type,2>>(this->nr_variables(), {0.0, 0.0});
                }
            };

            auto average_delta = [&](std::vector<std::array<value_type,2>>& delta) {
                MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("parallel sum marginal averaging");
#pragma omp parallel for
                for(size_t var=0; var<this->nr_variables(); ++var)
                {
                    assert(this->nr_bdds(var) > 0);
                    delta[var][0] /= value_type(this->nr_bdds(var));
                    delta[var][1] /= value_type(this->nr_bdds(var));
                }
            };

            init_delta(this->delta_out_);
            init_delta(this->delta_in_);

            assert(this->constant_ == 0.0);

            {
                MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("parallel sum marginal incremental marginal computation");
                forward_sm(omega, this->delta_out_, this->delta_in_);
                average_delta(this->delta_out_);
                reset_delta(this->delta_in_);
                std::swap(this->delta_out_, this->delta_in_);

                //distribute_delta();
                //smooth_backward_run();
                //compute_smooth_lower_bound_after_backward_pass();

                //smooth_forward_run();

                std::tie(smooth_lower_bound_, this->lower_bound_) = backward_sm(omega, this->delta_out_, this->delta_in_);
                average_delta(this->delta_out_);
                reset_delta(this->delta_in_);
                std::swap(this->delta_out_, this->delta_in_);

                //distribute_delta();
                //smooth_backward_run();
                //compute_smooth_lower_bound_after_backward_pass();
            }

            std::cout << "smooth lb = " << smooth_lower_bound_ << "\n";
            std::cout << "original lb = " << this->lower_bound_ << "\n";

            smooth_message_passing_state_ = base_type::message_passing_state::after_backward_pass;
            smooth_lower_bound_state_ = base_type::lower_bound_state::valid; 
            this->lower_bound_state_ = base_type::lower_bound_state::valid; 
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::forward_sm(
                const typename BDD_BRANCH_NODE::value_type omega,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in)
        {
            assert(smooth_message_passing_state_ == base_type::message_passing_state::after_backward_pass);

            smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
            this->lower_bound_state_ = base_type::lower_bound_state::invalid;

#pragma omp parallel for schedule(static,256)
            for(size_t bdd_nr=0; bdd_nr<this->nr_bdds(); ++bdd_nr)
                forward_sm(bdd_nr, omega, delta_out, delta_in);
            smooth_message_passing_state_ = base_type::message_passing_state::after_forward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::forward_sm(
                const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in)
        {
            assert(delta_out.size() == this->nr_variables());
            assert(delta_in.size() == this->nr_variables());
            assert(omega >= 0.0 && omega <= 1.0);
            assert(bdd_nr < this->nr_bdds());

            {
                const auto [first_bdd_node, last_bdd_node] = this->bdd_index_range(bdd_nr, 0);
                assert(first_bdd_node + 1 == last_bdd_node);
                this->bdd_branch_nodes_[first_bdd_node].m = 1.0;
                this->bdd_branch_nodes_[first_bdd_node].current_max = 0.0;
            }

            for(size_t bdd_idx=0; bdd_idx<this->nr_variables(bdd_nr); ++bdd_idx)
            {
                auto cur_bdd_lb = [&]() {
                    double lb = 0.0;
                    exp_sum<value_type> cur_sm = exp_sum<value_type>{};
                    const auto [first_bdd_node, last_bdd_node] = this->bdd_index_range(bdd_nr, bdd_idx);
                    for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                    {
                        const auto bdd_sm = this->bdd_branch_nodes_[i].sum_marginals();
                        cur_sm += bdd_sm[0];
                        cur_sm += bdd_sm[1];
                    };
                    return -this->smoothing * cur_sm.log();
                };

                const double before_lb = cur_bdd_lb();

                const auto [first_bdd_node, last_bdd_node] = this->bdd_index_range(bdd_nr, bdd_idx);
                const size_t var = this->variable(bdd_nr, bdd_idx);
                std::array<exp_sum<value_type>,2> cur_sm = {exp_sum<value_type>{}, exp_sum<value_type>{}};
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    const auto bdd_sm = this->bdd_branch_nodes_[i].sum_marginals();
                    cur_sm[0].update(bdd_sm[0]);
                    cur_sm[1].update(bdd_sm[1]);
                    //cur_sm[0] += bdd_sm[0];
                    //cur_sm[1] += bdd_sm[1];
                }

                // TODO: infinity handling
                //if(!std::isfinite(cur_sm[0]))
                //    atomic_store(delta_out[var][0], std::numeric_limits<value_type>::infinity());
                //if(!std::isfinite(cur_sm[1]))
                //    atomic_store(delta_out[var][1], std::numeric_limits<value_type>::infinity());
                //const exp_sum<value_type> cur_total_sm = cur_sm[0] + cur_sm[1];
                //const value_type low_diff = exp_sum_diff_log(value_type(2.0-omega)*cur_sm[0] + omega*cur_sm[1], (value_type(2.0)*cur_sm[0]));
                //const value_type high_diff = exp_sum_diff_log(value_type(2.0-omega)*cur_sm[1] + omega*cur_sm[0], (value_type(2.0)*cur_sm[1]));
                //value_type low_diff = (value_type(2.0-omega)*cur_sm[0] + omega*cur_sm[1]).log() - (value_type(2.0)*cur_sm[0]).log();//;omega * ( cur_total_sm.log() - (cur_sm[0] + cur_sm[0]).log());
                //value_type high_diff = (value_type(2.0-omega)*cur_sm[1] + omega*cur_sm[0]).log() - (value_type(2.0)*cur_sm[1]).log();//;omega * ( cur_total_sm.log() - (cur_sm[0] + cur_sm[0]).log());
                //const value_type high_diff = ((2.0-omega)*cur_sm[1] + omega * cur_sm[0]).log() - (cur_sm[1] + cur_sm[1]).log();//omega * ( cur_total_sm.log() - (cur_sm[1] + cur_sm[1]).log());
                //const value_type diff_0 = -std::log(cur_sm[0].sum) - cur_sm[0].max; // this gets deleted from low cost
                //const value_type diff_1 = -std::log(cur_sm[1].sum) - cur_sm[1].max; // this gets deleted from high cost

                //const value_type low_diff = omega * exp_sum_diff_log(cur_sm[0] + cur_sm[1], value_type(2.0) * cur_sm[0]);
                //const value_type high_diff = omega * exp_sum_diff_log(cur_sm[0] + cur_sm[1], value_type(2.0) * cur_sm[1]);

                const value_type low_diff = omega * std::max(0.0, exp_sum_diff_log(cur_sm[1], cur_sm[0]));
                const value_type high_diff = omega * std::max(0.0, exp_sum_diff_log(cur_sm[0], cur_sm[1]));

                //if(std::isfinite(cur_sm[0]) && std::isfinite(cur_sm[1]))
                {
                    atomic_add(delta_out[var][0], low_diff);
                    atomic_add(delta_out[var][1], high_diff);
                    //if(diff_0 < diff_1)
                    //    atomic_add(delta_out[var][1], omega*(diff_1 - diff_0));
                    //else
                    //    atomic_add(delta_out[var][0], omega*(diff_0 - diff_1));
                }

                //assert(delta_out[var][0] >= 0.0);
                //assert(delta_out[var][1] >= 0.0);

                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    // TODO: infinity handling
                    //if(!std::isfinite(cur_mm[0]))
                    //    bdd_branch_nodes_[i].low_cost = std::numeric_limits<value_type>::infinity();
                    //if(!std::isfinite(cur_mm[1]))
                    //    bdd_branch_nodes_[i].high_cost = std::numeric_limits<value_type>::infinity();
                    //if(std::isfinite(cur_mm[0]) && std::isfinite(cur_mm[1]))
                    {
                        this->bdd_branch_nodes_[i].low_cost -= low_diff;
                        this->bdd_branch_nodes_[i].high_cost -= high_diff;
                        //if(diff_0 < diff_1)
                        //    this->bdd_branch_nodes_[i].high_cost += omega*(diff_0 - diff_1);
                        //else
                        //    this->bdd_branch_nodes_[i].low_cost += omega*(diff_1 - diff_0);
                    }
                }
                const double after_lb = cur_bdd_lb();
                //assert(std::abs(before_lb - after_lb) <= 1e-4);

                if(bdd_idx+1<this->nr_variables(bdd_nr))
                {
                    const auto [next_first_bdd_node, next_last_bdd_node] = this->bdd_index_range(bdd_nr, bdd_idx+1);
                    for(size_t i=next_first_bdd_node; i<next_last_bdd_node; ++i)
                    {
                        //this->bdd_branch_nodes_[i].m = 0.0;
                        //this->bdd_branch_nodes_[i].current_max = -std::numeric_limits<value_type>::infinity();
                    }
                }
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                    this->bdd_branch_nodes_[i].prepare_smooth_forward_step();
                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    this->bdd_branch_nodes_[i].low_cost += delta_in[var][0];
                    this->bdd_branch_nodes_[i].high_cost += delta_in[var][1];
                    this->bdd_branch_nodes_[i].smooth_forward_step(); 
                }
            }
        }

    template<typename BDD_BRANCH_NODE>
        std::tuple<double,double> bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::backward_sm(
                const typename BDD_BRANCH_NODE::value_type omega,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in)
        {
            assert(smooth_message_passing_state_ == base_type::message_passing_state::after_forward_pass);
            double smooth_lb = this->constant_;
            double lb = this->constant_;
#pragma omp parallel for schedule(static,256) reduction(+:smooth_lb,lb)
            for(size_t bdd_nr=0; bdd_nr<this->nr_bdds(); ++bdd_nr)
            {
                const auto [smooth_lb_delta, lb_delta] = backward_sm(bdd_nr, omega, delta_out, delta_in);
                smooth_lb += smooth_lb_delta;
                lb += lb_delta;
            }
            smooth_message_passing_state_ = base_type::message_passing_state::after_backward_pass;
            return {value_type(smooth_lb), value_type(lb)};
        }

    template<typename BDD_BRANCH_NODE>
        std::tuple<double,double> bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::backward_sm(
                const size_t bdd_nr, const typename BDD_BRANCH_NODE::value_type omega,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_out,
                std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& delta_in)
        {
            assert(delta_out.size() == this->nr_variables());
            assert(delta_in.size() == this->nr_variables());
            assert(omega >= 0.0 && omega <= 1.0);
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
                //const exp_sum<value_type> cur_total_sm = cur_sm[0] + cur_sm[1];
                //const value_type low_diff = omega * ( cur_total_sm.log() - (cur_sm[0] + cur_sm[0]).log());
                //const value_type high_diff = omega * ( cur_total_sm.log() - (cur_sm[1] + cur_sm[1]).log());

                //const value_type low_diff = exp_sum_diff_log(value_type(2.0-omega)*cur_sm[0] + omega*cur_sm[1], (value_type(2.0)*cur_sm[0]));
                //const value_type high_diff = exp_sum_diff_log(value_type(2.0-omega)*cur_sm[1] + omega*cur_sm[0], (value_type(2.0)*cur_sm[1]));
                //const value_type diff_0 = -std::log(cur_sm[0].sum) - cur_sm[0].max; // this gets deleted from low cost
                //const value_type diff_1 = -std::log(cur_sm[1].sum) - cur_sm[1].max; // this gets deleted from high cost

                //const value_type low_diff = omega * exp_sum_diff_log(cur_sm[0] + cur_sm[1], value_type(2.0) * cur_sm[0]);
                //const value_type high_diff = omega * exp_sum_diff_log(cur_sm[0] + cur_sm[1], value_type(2.0) * cur_sm[1]);

                const value_type low_diff = omega * std::max(0.0, exp_sum_diff_log(cur_sm[1], cur_sm[0]));
                const value_type high_diff = omega * std::max(0.0, exp_sum_diff_log(cur_sm[0], cur_sm[1]));
                //if(std::isfinite(cur_sm[0]) && std::isfinite(cur_sm[1]))
                {
                    atomic_add(delta_out[var][0], low_diff);
                    atomic_add(delta_out[var][1], high_diff);
                    //if(diff_0 < diff_1)
                    //    atomic_add(delta_out[var][1], omega*(diff_1 - diff_0));
                    //else
                    //    atomic_add(delta_out[var][0], omega*(diff_0 - diff_1));
                }

                //assert(delta_out[var][0] >= 0.0);
                //assert(delta_out[var][1] >= 0.0);

                auto cur_bdd_lb = [&]() {
                    double lb = 0.0;
                    exp_sum<value_type> cur_sm = exp_sum<value_type>{};
                    const auto [first_bdd_node, last_bdd_node] = this->bdd_index_range(bdd_nr, bdd_idx);
                    for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                    {
                        const auto bdd_sm = this->bdd_branch_nodes_[i].sum_marginals();
                        cur_sm += bdd_sm[0];
                        cur_sm += bdd_sm[1];
                    };
                    return -this->smoothing * cur_sm.log();
                };

                const double before_lb = cur_bdd_lb();

                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    // TODO: infinity handling
                    //if(!std::isfinite(cur_mm[0]))
                    //    bdd_branch_nodes_[i].low_cost = std::numeric_limits<value_type>::infinity();
                    //if(!std::isfinite(cur_mm[1]))
                    //    bdd_branch_nodes_[i].high_cost = std::numeric_limits<value_type>::infinity();
                    //if(std::isfinite(cur_mm[0]) && std::isfinite(cur_mm[1]))
                    {
                        this->bdd_branch_nodes_[i].low_cost -= low_diff;
                        this->bdd_branch_nodes_[i].high_cost -= high_diff;
                        //if(diff_0 < diff_1)
                        //    this->bdd_branch_nodes_[i].high_cost += omega*(diff_0 - diff_1);
                        //else
                        //    this->bdd_branch_nodes_[i].low_cost += omega*(diff_1 - diff_0);
                    }
                }
                const double after_lb = cur_bdd_lb();
                //assert(std::abs(before_lb - after_lb) <= 1e-6);

                for(size_t i=first_bdd_node; i<last_bdd_node; ++i)
                {
                    this->bdd_branch_nodes_[i].low_cost += delta_in[var][0];
                    this->bdd_branch_nodes_[i].high_cost += delta_in[var][1];
                    this->bdd_branch_nodes_[i].smooth_backward_step(); 
                }
            }

            const auto [first,last] = this->bdd_index_range(bdd_nr, 0);
            assert(first+1 == last);
            const value_type smooth_lb = -smoothing * (std::log(this->bdd_branch_nodes_[first].m) + this->bdd_branch_nodes_[first].current_max);
            const value_type lb = -smoothing * this->bdd_branch_nodes_[first].current_max;
            return {smooth_lb, lb};
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::distribute_delta()
        {
            std::cout << "distribute delta\n";
            smooth_message_passing_state_ = base_type::message_passing_state::none;
            smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
            base_type::distribute_delta();
        }
    
    template<typename BDD_BRANCH_NODE>
        double bdd_parallel_mma_base_smooth<BDD_BRANCH_NODE>::delta_smooth_lower_bound(const std::vector<std::array<typename BDD_BRANCH_NODE::value_type,2>>& vec) const
        {
            assert(vec.size() == this->nr_variables());
            double lb = 0.0;
            for(size_t i=0; i<vec.size(); ++i)
                lb += std::min(vec[i][0], vec[i][1]);
            return -smoothing * lb;
        }

}
