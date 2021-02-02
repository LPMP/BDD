#pragma once

#include "bdd_mma_base.hxx"
#include "time_measure_util.h"
#include <vector>
#include <array>
#include <iostream>

namespace LPMP {

    // 
    template<typename BDD_OPT_BASE>
        class bdd_mma_srmp_rest_base : public bdd_mma_srmp_base<BDD_OPT_BASE> {
        public:
            using bdd_mma_srmp_base<BDD_OPT_BASE>::bdd_mma_srmp_base;

            void restricted_min_marginal_averaging_forward_SRMP(const bool active);
            void restricted_min_marginal_averaging_backward_SRMP(const bool active);
            void restricted_iteration(const bool active);
            void solve(const size_t max_iter, const double tolerance, const double time_limit);

        protected: 

        private:
            std::vector<int> stats_;
            size_t nr_improving_vars_;
    };

    ////////////////////
    // implementation //
    ////////////////////
    
    

    template<typename BDD_OPT_BASE>
    void bdd_mma_srmp_rest_base<BDD_OPT_BASE>::restricted_min_marginal_averaging_forward_SRMP(const bool active)
    {
        std::vector<std::array<double,2>> min_marginals;
        for(size_t var=0; var<this->nr_variables(); ++var) {

            if (active && !stats_[var])
            {
                this->forward_step(var);
                continue;
            }

            double lb_prev = 0;
            double lb_post = 0;

            // collect min marginals
            min_marginals.clear();
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                min_marginals.push_back(this->min_marginal(var,bdd_index));
                lb_prev += std::min(min_marginals.back()[1] - min_marginals.back()[0], 0.0);
                lb_post += min_marginals.back()[1] - min_marginals.back()[0];
            }
            lb_post = std::min(lb_post, 0.0);
            if (lb_post - lb_prev > std::numeric_limits<float>::epsilon())
            {
                stats_[var] = 1;
            }

            const auto average_marginal = this->average_marginals_forward_SRMP(min_marginals.begin(), min_marginals.end(), var);
            const std::array<double,2> avg_marg = average_marginal.first;
            const bool default_averaging = average_marginal.second;

            // set marginals in each bdd so min marginals match each other
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                this->set_marginal_forward_SRMP(var,bdd_index,avg_marg,min_marginals[bdd_index], default_averaging);
                this->forward_step(var,bdd_index);
            } 
        }
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_srmp_rest_base<BDD_OPT_BASE>::restricted_min_marginal_averaging_backward_SRMP(const bool active)
    {
        double lb = 0.0;
        std::vector<std::array<double,2>> min_marginals;
        for(long int var=this->nr_variables()-1; var>=0; --var) {

            if (active && !stats_[var])
            {
                for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    this->backward_step(var, bdd_index);
                    lb += this->lower_bound_backward(var,bdd_index);
                }
                continue;
            }

            double lb_prev = 0;
            double lb_post = 0;

            // collect min marginals
            min_marginals.clear();
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                min_marginals.push_back(this->min_marginal(var,bdd_index));
                lb_prev += std::min(min_marginals.back()[1] - min_marginals.back()[0], 0.0);
                lb_post += min_marginals.back()[1] - min_marginals.back()[0];
            }
            lb_post = std::min(lb_post, 0.0);
            if (lb_post - lb_prev > std::numeric_limits<float>::epsilon())
            {
                stats_[var] = 1;
            }

            const auto average_marginal = this->average_marginals_backward_SRMP(min_marginals.begin(), min_marginals.end(), var);
            const std::array<double,2> avg_marg = average_marginal.first;
            const bool default_averaging = average_marginal.second;

            // set marginals in each bdd so min marginals match each other
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                this->set_marginal_backward_SRMP(var,bdd_index,avg_marg,min_marginals[bdd_index], default_averaging);
                this->backward_step(var, bdd_index);
                lb += this->lower_bound_backward(var,bdd_index);
            }
        }

        this->lower_bound_ = lb; 
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_srmp_rest_base<BDD_OPT_BASE>::restricted_iteration(const bool active)
    {
        const auto begin_time = std::chrono::steady_clock::now();
        restricted_min_marginal_averaging_forward_SRMP(active);
        const auto after_forward = std::chrono::steady_clock::now();
        // std::cout << "forward " <<  std::chrono::duration_cast<std::chrono::milliseconds>(after_forward - begin_time).count() << " ms, " << std::flush;
        const auto before_backward = std::chrono::steady_clock::now();
        restricted_min_marginal_averaging_backward_SRMP(active);
        const auto end_time = std::chrono::steady_clock::now();
        // std::cout << "backward " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end_time - before_backward).count() << " ms, " << std::flush;
    }

    template<typename BDD_OPT_BASE>
        void bdd_mma_srmp_rest_base<BDD_OPT_BASE>::solve(const size_t max_iter, const double tolerance, const double time_limit)
        {
            stats_.resize(this->nr_variables(), 1);
            nr_improving_vars_ = this->nr_variables();
            bool active = true;
            std::cout << "initial lower bound = " << this->lower_bound() << "\n";
            for(size_t iter=0; iter<max_iter-1; ++iter)
            {
                active = !active;
                if (!active)
                    std::fill(stats_.begin(), stats_.end(), 0);
                // bdd_mma_srmp_base<BDD_OPT_BASE>::iteration();
                restricted_iteration(active);
                nr_improving_vars_ = std::accumulate(stats_.begin(), stats_.end(), 0);
                std::cout << "iteration " << iter << ", lower bound = " << this->lower_bound() << ", improving variables = " << nr_improving_vars_ << "\n"; 
            }
            bdd_mma_base<BDD_OPT_BASE>::iteration();
            std::cout << "iteration " << max_iter-1 << ", lower bound = " << this->lower_bound() << "\n";
            std::cout << "(last iteration with default averaging)" << std::endl;
            std::cout << "final lower bound = " << this->lower_bound() << "\n";

        }

}


