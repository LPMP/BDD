#pragma once

#include "bdd_mma_base.h"
#include <limits>

namespace LPMP {

    template<typename BDD_BRANCH_NODE>
        class bdd_mma_base_smooth : public bdd_mma_base<BDD_BRANCH_NODE> {
            public:
                using value_type = typename bdd_mma_base<BDD_BRANCH_NODE>::value_type;
                using bdd_mma_base<BDD_BRANCH_NODE>::bdd_mma_base;

                using base_type = bdd_mma_base<BDD_BRANCH_NODE>;

                void sum_marginal_averaging_forward();
                void sum_marginal_averaging_step_forward(const size_t var);
                void smooth_forward_step(const size_t var);

                void sum_marginal_averaging_backward();
                void sum_marginal_averaging_step_backward(const size_t var);
                void smooth_backward_step(const size_t var);

                //std::array<exp_sum<value_type>,2> average_sum_marginals(std::array<exp_sum<value_type>,2>* marginals, const size_t nr_marginals) const;
                std::array<value_type,2> average_sum_marginals(std::array<exp_sum<value_type>,2>* marginals, const size_t nr_marginals) const;

                void smooth_iteration();
                void smooth_backward_run();
                void smooth_forward_run();

                double smooth_lower_bound();
                void compute_smooth_lower_bound_after_forward_pass(); 
                void compute_smooth_lower_bound_after_backward_pass(); 

                void set_smoothing(const value_type new_smoothing);
                void update_cost(const value_type lo_cost, const value_type hi_cost, const size_t var);

            private:
                typename bdd_mma_base<BDD_BRANCH_NODE>::lower_bound_state smooth_lower_bound_state_ = bdd_mma_base<BDD_BRANCH_NODE>::lower_bound_state::invalid;
                typename bdd_mma_base<BDD_BRANCH_NODE>::message_passing_state smooth_message_passing_state_ = bdd_mma_base<BDD_BRANCH_NODE>::message_passing_state::none;
                double smooth_lower_bound_ = -std::numeric_limits<value_type>::infinity();

                value_type smoothing = 1.0;
        };

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::sum_marginal_averaging_step_forward(const size_t var)
        {
            assert(var < this->nr_variables());
            const size_t _nr_bdds = this->nr_bdds(var);
            if(_nr_bdds == 0)
                return;

            std::array<exp_sum<value_type>,2> sum_marginals[_nr_bdds];
            std::fill(sum_marginals, sum_marginals + _nr_bdds, std::array<exp_sum<value_type>,2>{exp_sum<value_type>{}, exp_sum<value_type>{}});

            for(size_t i=this->bdd_branch_node_offsets_[var]; i<this->bdd_branch_node_offsets_[var+1]; ++i)
                this->bdd_branch_nodes_[i].update_sum_marginals(sum_marginals);

            std::array<value_type,2> avg_marginals = this->average_sum_marginals(sum_marginals, _nr_bdds);

            for(size_t i=this->bdd_branch_node_offsets_[var]; i<this->bdd_branch_node_offsets_[var+1]; ++i)
            {
                auto& bdd = this->bdd_branch_nodes_[i];
                bdd.low_cost += sum_marginals[bdd.bdd_index][0].log() - avg_marginals[0];
                bdd.high_cost += sum_marginals[bdd.bdd_index][1].log() - avg_marginals[1];
            }

            smooth_forward_step(var);
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::smooth_forward_step(const size_t var)
        {
            assert(var < this->nr_variables());
            for(size_t i=this->bdd_branch_node_offsets_[var]; i<this->bdd_branch_node_offsets_[var+1]; ++i)
                this->bdd_branch_nodes_[i].prepare_smooth_forward_step();
            for(size_t i=this->bdd_branch_node_offsets_[var]; i<this->bdd_branch_node_offsets_[var+1]; ++i)
                this->bdd_branch_nodes_[i].smooth_forward_step();
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::sum_marginal_averaging_step_backward(const size_t var)
        {
            assert(var < this->nr_variables());
            const size_t _nr_bdds = this->nr_bdds(var);
            if(_nr_bdds == 0)
                return;

            std::array<exp_sum<value_type>,2> sum_marginals[_nr_bdds];
            std::fill(sum_marginals, sum_marginals + _nr_bdds, std::array<exp_sum<value_type>,2>{exp_sum<value_type>{}, exp_sum<value_type>{}});

            for(size_t i=this->bdd_branch_node_offsets_[var]; i<this->bdd_branch_node_offsets_[var+1]; ++i)
                this->bdd_branch_nodes_[i].update_sum_marginals(sum_marginals);

            std::array<value_type,2> avg_marginals = this->average_sum_marginals(sum_marginals, _nr_bdds);

            for(size_t i=this->bdd_branch_node_offsets_[var]; i<this->bdd_branch_node_offsets_[var+1]; ++i)
            {
                auto& bdd = this->bdd_branch_nodes_[i];
                bdd.low_cost += sum_marginals[bdd.bdd_index][0].log() - avg_marginals[0];
                bdd.high_cost += sum_marginals[bdd.bdd_index][1].log() - avg_marginals[1];
            }

            smooth_backward_step(var);
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::smooth_backward_step(const size_t  var)
        {
            assert(var < this->nr_variables());
            for(size_t i=this->bdd_branch_node_offsets_[var]; i<this->bdd_branch_node_offsets_[var+1]; ++i)
                this->bdd_branch_nodes_[i].smooth_backward_step();
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::sum_marginal_averaging_forward()
        {
            if(smooth_message_passing_state_ != base_type::message_passing_state::after_backward_pass)
                smooth_backward_run();
            smooth_message_passing_state_ = base_type::message_passing_state::none;
            this->message_passing_state_ = base_type::message_passing_state::none;
            smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
            this->lower_bound_state_ = base_type::lower_bound_state::invalid;

            for(size_t bdd_index=0; bdd_index<this->first_bdd_node_indices_.size(); ++bdd_index)
            {
                assert(this->first_bdd_node_indices_.size(bdd_index) == 1);
                for(size_t j=0; j<this->first_bdd_node_indices_.size(bdd_index); ++j)
                {
                    this->bdd_branch_nodes_[this->first_bdd_node_indices_(bdd_index,j)].m = 1.0;
                    this->bdd_branch_nodes_[this->first_bdd_node_indices_(bdd_index,j)].current_max = 0.0;
                }
            }

            for(size_t i=0; i<this->nr_variables(); ++i)
                sum_marginal_averaging_step_forward(i);

            smooth_message_passing_state_ = base_type::message_passing_state::after_forward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::sum_marginal_averaging_backward()
        {
            if(smooth_message_passing_state_ != base_type::message_passing_state::after_forward_pass)
                smooth_forward_run();
            smooth_message_passing_state_ = base_type::message_passing_state::none;
            this->message_passing_state_ = base_type::message_passing_state::none;
            smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
            this->lower_bound_state_ = base_type::lower_bound_state::invalid;

            for(std::ptrdiff_t i=this->nr_variables()-1; i>=0; --i)
                sum_marginal_averaging_step_backward(i);

            smooth_message_passing_state_ = base_type::message_passing_state::after_backward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        std::array<typename bdd_mma_base_smooth<BDD_BRANCH_NODE>::value_type,2> bdd_mma_base_smooth<BDD_BRANCH_NODE>::average_sum_marginals(std::array<exp_sum<typename bdd_mma_base_smooth<BDD_BRANCH_NODE>::value_type>,2>* marginals, const size_t nr_marginals) const
        {
            value_type avg_0 = 0.0;
            for(size_t i=0; i<nr_marginals; ++i)
                avg_0 += marginals[i][0].log();
            avg_0 /= value_type(nr_marginals);
            assert(std::isfinite(avg_0));

            value_type avg_1 = 0.0;
            for(size_t i=0; i<nr_marginals; ++i)
                avg_1 += marginals[i][1].log();
            avg_1 /= value_type(nr_marginals);
            assert(std::isfinite(avg_1));

            return {avg_0, avg_1};
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::smooth_iteration()
        {
            sum_marginal_averaging_forward();
            sum_marginal_averaging_backward();
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::smooth_backward_run()
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            if(smooth_message_passing_state_ == base_type::message_passing_state::after_backward_pass)
                return;
            // TODO: if we already have done a backward_run, we do not need to do it again. Check state!
            smooth_message_passing_state_ = base_type::message_passing_state::none;
            this->message_passing_state_ = base_type::message_passing_state::none;
            for(std::ptrdiff_t i=this->bdd_branch_nodes_.size()-1; i>=0; --i)
                this->bdd_branch_nodes_[i].smooth_backward_step();
            smooth_message_passing_state_ = base_type::message_passing_state::after_backward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::smooth_forward_run()
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            if(smooth_message_passing_state_ == base_type::message_passing_state::after_forward_pass)
                return;

            smooth_message_passing_state_ = base_type::message_passing_state::none;
            this->message_passing_state_ = base_type::message_passing_state::none;

            for(size_t bdd_index=0; bdd_index<this->first_bdd_node_indices_.size(); ++bdd_index)
                for(size_t j=0; j<this->first_bdd_node_indices_.size(bdd_index); ++j)
                {
                    this->bdd_branch_nodes_[this->first_bdd_node_indices_(bdd_index,j)].m = 1.0;
                    this->bdd_branch_nodes_[this->first_bdd_node_indices_(bdd_index,j)].current_max = 0.0;
                }

            for(size_t i=0; i<this->nr_variables(); ++i)
                smooth_forward_step(i);
            smooth_message_passing_state_ = base_type::message_passing_state::after_forward_pass;
        }

    template<typename BDD_BRANCH_NODE>
        double bdd_mma_base_smooth<BDD_BRANCH_NODE>::smooth_lower_bound()
        {
            if(smooth_lower_bound_state_ == base_type::lower_bound_state::valid)
                return smooth_lower_bound_ + this->constant_;
            if(smooth_message_passing_state_ == base_type::message_passing_state::after_forward_pass)
                compute_smooth_lower_bound_after_forward_pass();
            else if(smooth_message_passing_state_ == base_type::message_passing_state::after_backward_pass)
                compute_smooth_lower_bound_after_backward_pass();
            else
            {
                smooth_backward_run();
                compute_smooth_lower_bound_after_backward_pass();
            }
            assert(smooth_lower_bound_state_ == base_type::lower_bound_state::valid);
            return smooth_lower_bound_ + this->constant_;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::compute_smooth_lower_bound_after_backward_pass()
        {
            assert(smooth_message_passing_state_ == base_type::message_passing_state::after_backward_pass);
            double smooth_lb = 0.0;
            double lb = 0.0;
            for(size_t i=0; i<this->first_bdd_node_indices_.size(); ++i)
            {
                assert(this->first_bdd_node_indices_.size(i) == 1);
                const auto& bdd_node = this->bdd_branch_nodes_[this->first_bdd_node_indices_(i,0)];
                smooth_lb += -smoothing * bdd_node.get_exp_sum().log();
                lb += -smoothing*bdd_node.current_max;
            }

            smooth_lower_bound_ = smooth_lb;
            // TODO: set non smooth lower bound as well, same for after backward pass
            smooth_lower_bound_state_ = base_type::lower_bound_state::valid;
            this->lower_bound_state_ = base_type::lower_bound_state::valid;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::compute_smooth_lower_bound_after_forward_pass()
        {
            assert(smooth_message_passing_state_ == base_type::message_passing_state::after_forward_pass);
            double smooth_lb = 0.0;
            double lb = 0.0;
            for(size_t i=0; i<this->last_bdd_node_indices_.size(); ++i)
            {
                exp_sum<value_type> s;
                for(size_t j=0; j<this->last_bdd_node_indices_.size(i); ++j)
                {
                    const auto& bdd_node = this->bdd_branch_nodes_[this->last_bdd_node_indices_(i,j)];
                    assert(bdd_node.offset_low == BDD_BRANCH_NODE::terminal_0_offset || bdd_node.offset_low == BDD_BRANCH_NODE::terminal_1_offset);
                    assert(bdd_node.offset_high == BDD_BRANCH_NODE::terminal_0_offset || bdd_node.offset_high == BDD_BRANCH_NODE::terminal_1_offset);

                    const auto sm = bdd_node.sum_marginals();
                    s.update(sm[0]);
                    s.update(sm[1]);
                }
                smooth_lb += -smoothing * s.log();
                lb += -smoothing*s.max;
            }

            smooth_lower_bound_ = smooth_lb;
            smooth_lower_bound_state_ = base_type::lower_bound_state::valid;
            this->lower_bound_state_ = base_type::lower_bound_state::valid;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::set_smoothing(const typename bdd_mma_base<BDD_BRANCH_NODE>::value_type new_smoothing)
        {
            smooth_message_passing_state_ = base_type::message_passing_state::none;
            this->message_passing_state_ = base_type::message_passing_state::none;
            smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
            this->lower_bound_state_ = base_type::lower_bound_state::invalid;

            for(auto& bdd_instr : this->bdd_branch_nodes_)
            {
                bdd_instr.low_cost *= smoothing / new_smoothing;
                bdd_instr.high_cost *= smoothing / new_smoothing;
            }

            smoothing = new_smoothing;
        }

    template<typename BDD_BRANCH_NODE>
        void bdd_mma_base_smooth<BDD_BRANCH_NODE>::update_cost(const typename bdd_mma_base<BDD_BRANCH_NODE>::value_type lo_cost, const typename bdd_mma_base<BDD_BRANCH_NODE>::value_type hi_cost, const size_t var)
        {
            smooth_message_passing_state_ = base_type::message_passing_state::none;
            this->message_passing_state_ = base_type::message_passing_state::none;
            smooth_lower_bound_state_ = base_type::lower_bound_state::invalid;
            this->lower_bound_state_ = base_type::lower_bound_state::invalid;

            assert(this->nr_bdds(var) > 0);
            assert(std::isfinite(std::min(lo_cost, hi_cost)));

            for(size_t i=this->bdd_branch_node_offsets_[var]; i<this->bdd_branch_node_offsets_[var+1]; ++i)
            {
                this->bdd_branch_nodes_[i].low_cost += 1.0 / smoothing * lo_cost / value_type(this->nr_bdds(var));
                this->bdd_branch_nodes_[i].high_cost += 1.0 / smoothing * hi_cost / value_type(this->nr_bdds(var));
            }
        }
}
