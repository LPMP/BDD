#pragma once

#include "bdd_mma_base.hxx"
#include "time_measure_util.h"
#include <vector>
#include <array>
#include <iostream>

namespace LPMP {

    // base class for min marginal averaging with aggressive distribution of excess costs
    template<typename BDD_OPT_BASE>
        class bdd_mma_agg_base : public bdd_mma_base<BDD_OPT_BASE> {
        public:
            using bdd_mma_base<BDD_OPT_BASE>::bdd_mma_base;

            void min_marginal_averaging_forward_aggressive();
            void min_marginal_averaging_backward_aggressive();
            void iteration();
            void solve(const size_t max_iter, const double tolerance, const double time_limit);

        protected: 
            template <typename ITERATOR>
                std::pair<std::array<double, 2>, std::array<double, 2>> average_marginals_forward_aggressive(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t var, const size_t min_last_var_index) const;
            template <typename ITERATOR>
                std::pair<std::array<double, 2>, std::array<double, 2>> average_marginals_backward_aggressive(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t var, const size_t max_first_var_index) const;
            void set_marginal_forward_aggressive(const size_t var, const size_t bdd_index, const std::array<double, 2> marginals_in, const std::array<double, 2> marginals_out, const std::array<double, 2> min_marginals, const size_t min_last_var_index);
            void set_marginal_backward_aggressive(const size_t var, const size_t bdd_index, const std::array<double, 2> marginals_in, const std::array<double, 2> marginals_out, const std::array<double, 2> min_marginals, const size_t max_first_var_index);

    };

    ////////////////////
    // implementation //
    ////////////////////
    
    template<typename BDD_OPT_BASE>
    void bdd_mma_agg_base<BDD_OPT_BASE>::set_marginal_forward_aggressive(const size_t var, const size_t bdd_index, const std::array<double,2> marginals_in, const std::array<double,2> marginals_out, const std::array<double,2> min_marginals, const size_t min_last_var_index)
    {
        const double marginal_diff = min_marginals[1] - min_marginals[0];
        const double marginal_diff_target_in = marginals_in[1] - marginals_in[0];
        const double marginal_diff_target_out = marginals_out[1] - marginals_out[0];
        assert(std::isfinite(marginal_diff));
        assert(std::isfinite(marginal_diff_target_in));
        assert(std::isfinite(marginal_diff_target_out));
        // if(this->bdd_variables_(var,bdd_index).last_var_index == min_last_var_index) {
        if(this->last_variable_of_bdd(var, bdd_index)) {
            this->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target_in);
        } else {
            assert(std::isfinite(marginal_diff_target_out));
            this->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target_out);
        } 
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_agg_base<BDD_OPT_BASE>::set_marginal_backward_aggressive(const size_t var, const size_t bdd_index, const std::array<double,2> marginals_in, const std::array<double,2> marginals_out, const std::array<double,2> min_marginals, const size_t max_first_var_index)
    {
        const double marginal_diff = min_marginals[1] - min_marginals[0];
        const double marginal_diff_target_in = marginals_in[1] - marginals_in[0];
        const double marginal_diff_target_out = marginals_out[1] - marginals_out[0];
        assert(std::isfinite(marginal_diff));
        assert(std::isfinite(marginal_diff_target_in));
        assert(std::isfinite(marginal_diff_target_out));
        // if(this->bdd_variables_(var,bdd_index).first_var_index == max_first_var_index) {
        if(this->first_variable_of_bdd(var, bdd_index)) {
            this->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target_in);
        } else {
            this->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target_out);
        }
    }

    template<typename BDD_OPT_BASE>
    template<typename ITERATOR>
        std::pair<std::array<double,2>, std::array<double,2>> bdd_mma_agg_base<BDD_OPT_BASE>::average_marginals_forward_aggressive(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t var, const size_t min_last_var_index) const
        {
            assert(this->nr_bdds(var) == std::distance(marginals_begin, marginals_end));
            std::array<double,2> avg_marg_out = {0.0, 0.0};
            int divisor_out = 0;
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    avg_marg_out[0] += (*(marginals_begin+bdd_index))[0];
                    avg_marg_out[1] += (*(marginals_begin+bdd_index))[1];
                // if(this->bdd_variables_(var,bdd_index).last_var_index > min_last_var_index)
                if(!this->last_variable_of_bdd(var, bdd_index))
                    ++divisor_out;
            }
            std::array<double,2> avg_marg_in = avg_marg_out;
            int divisor_in = this->nr_bdds(var) - divisor_out;
            int max_divisor = std::max(divisor_in, divisor_out);
            if (divisor_out == 0)
            {
                avg_marg_out[0] = 0;
                avg_marg_out[1] = 0;
                avg_marg_in[0] /= double(divisor_in);
                avg_marg_in[1] /= double(divisor_in);
            }
            else
            // we always have that divisor_in > 0
            {
                avg_marg_out[0] /= double(max_divisor);
                avg_marg_out[1] /= double(max_divisor);
                avg_marg_in[0] *= std::max(0, divisor_in - divisor_out) / (double(divisor_in)*double(divisor_in));
                avg_marg_in[1] *= std::max(0, divisor_in - divisor_out) / (double(divisor_in)*double(divisor_in));
            }

            return std::make_pair(avg_marg_in, avg_marg_out);
        }

    template<typename BDD_OPT_BASE>
    template<typename ITERATOR>
        std::pair<std::array<double,2>, std::array<double,2>> bdd_mma_agg_base<BDD_OPT_BASE>::average_marginals_backward_aggressive(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t var, const size_t max_first_var_index) const
        {
            assert(this->nr_bdds(var) == std::distance(marginals_begin, marginals_end));
            std::array<double,2> avg_marg_out = {0.0, 0.0};
            int divisor_out = 0;
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    avg_marg_out[0] += (*(marginals_begin+bdd_index))[0];
                    avg_marg_out[1] += (*(marginals_begin+bdd_index))[1];
                // if(this->bdd_variables_(var,bdd_index).first_var_index < max_first_var_index)
                if(!this->first_variable_of_bdd(var, bdd_index))
                    ++divisor_out;
            }
            std::array<double,2> avg_marg_in = avg_marg_out;
            int divisor_in = this->nr_bdds(var) - divisor_out;
            int max_divisor = std::max(divisor_in, divisor_out);
            if (divisor_out == 0)
            {
                avg_marg_out[0] = 0;
                avg_marg_out[1] = 0;
                avg_marg_in[0] /= double(divisor_in);
                avg_marg_in[1] /= double(divisor_in);
            }
            else
            // we always have that divisor_in > 0
            {
                avg_marg_out[0] /= double(max_divisor);
                avg_marg_out[1] /= double(max_divisor);
                avg_marg_in[0] *= std::max(0, divisor_in - divisor_out) / (double(divisor_in)*double(divisor_in));
                avg_marg_in[1] *= std::max(0, divisor_in - divisor_out) / (double(divisor_in)*double(divisor_in));
            }
            return std::make_pair(avg_marg_in, avg_marg_out);
        }

    template<typename BDD_OPT_BASE>
    void bdd_mma_agg_base<BDD_OPT_BASE>::min_marginal_averaging_forward_aggressive()
    {
        std::vector<std::array<double,2>> min_marginals;
        for(size_t var=0; var<this->nr_variables(); ++var) {

            // collect min marginals and determine minimum last variable index
            size_t min_last_var_index = std::numeric_limits<size_t>::max();
            min_marginals.clear();
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                auto & bdd_var = this->bdd_variables_(var,bdd_index);
                if (bdd_var.last_var_index < min_last_var_index)
                    min_last_var_index = bdd_var.last_var_index;
                min_marginals.push_back(this->min_marginal(var,bdd_index));
            }

            const auto average_marginals = average_marginals_forward_aggressive(min_marginals.begin(), min_marginals.end(), var, min_last_var_index);
            const std::array<double,2> avg_marg_in = average_marginals.first;
            const std::array<double,2> avg_marg_out = average_marginals.second;

            // set marginals in each bdd so min marginals match each other
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal_forward_aggressive(var,bdd_index,avg_marg_in, avg_marg_out, min_marginals[bdd_index], min_last_var_index);
                this->forward_step(var,bdd_index);
            } 
        }
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_agg_base<BDD_OPT_BASE>::min_marginal_averaging_backward_aggressive()
    {
        double lb = 0.0;
        std::vector<std::array<double,2>> min_marginals;
        for(long int var=this->nr_variables()-1; var>=0; --var) {

            // collect min marginals and determine maximum first variable index
            size_t max_first_var_index = std::numeric_limits<size_t>::min();
            min_marginals.clear();
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                auto & bdd_var = this->bdd_variables_(var,bdd_index);
                if (bdd_var.first_var_index > max_first_var_index)
                    max_first_var_index = bdd_var.first_var_index;
                min_marginals.push_back(this->min_marginal(var,bdd_index)); 
            }

            const auto average_marginals = average_marginals_backward_aggressive(min_marginals.begin(), min_marginals.end(), var, max_first_var_index);
            const std::array<double,2> avg_marg_in = average_marginals.first;
            const std::array<double,2> avg_marg_out = average_marginals.second;

            // set marginals in each bdd so min marginals match each other
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal_backward_aggressive(var,bdd_index,avg_marg_in, avg_marg_out,min_marginals[bdd_index], max_first_var_index);
                this->backward_step(var, bdd_index);
                lb += this->lower_bound_backward(var,bdd_index);
            }
        }

        this->lower_bound_ = lb; 
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_agg_base<BDD_OPT_BASE>::iteration()
    {
        min_marginal_averaging_forward_aggressive();
        min_marginal_averaging_backward_aggressive();
    }

    template<typename BDD_OPT_BASE>
        void bdd_mma_agg_base<BDD_OPT_BASE>::solve(const size_t max_iter, const double tolerance, const double time_limit)
        {
            const auto start_time = std::chrono::steady_clock::now();
            double lb_prev = this->lower_bound();
            double lb_post = lb_prev;
            std::cout << "initial lower bound = " << lb_prev;
            auto time = std::chrono::steady_clock::now();
            std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
            std::cout << "\n";
            size_t iter;
            for(iter=0; iter<max_iter-1; ++iter)
            {
                iteration();
                lb_prev = lb_post;
                lb_post = this->lower_bound();
                std::cout << "iteration " << iter << ", lower bound = " << lb_post;
                time = std::chrono::steady_clock::now();
                std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
                std::cout << "\n";
                if (std::abs(lb_prev-lb_post) < std::abs(tolerance*lb_prev))
                {
                    std::cout << "Relative progress less than tolerance (" << tolerance << ")\n";
                    break;
                }
            }
            bdd_mma_base<BDD_OPT_BASE>::iteration();
            std::cout << "iteration " << iter+1 << ", lower bound = " << this->lower_bound();
            time = std::chrono::steady_clock::now();
            std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
            std::cout << "\n";
            std::cout << "(last iteration with default averaging)" << std::endl;
            std::cout << "final lower bound = " << this->lower_bound() << "\n";

        }

}


