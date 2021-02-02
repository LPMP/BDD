#pragma once

#include "bdd_opt_base.hxx"
#include "time_measure_util.h"
#include <vector>
#include <array>
#include <iostream>

namespace LPMP {

    // base class for min marginal averaging. It either derived from bdd_mma_base_node_costs or bdd_mma_base_arc_costs
    template<typename BDD_OPT_BASE>
        class bdd_mma_base : public BDD_OPT_BASE
    {
        public:
            using BDD_OPT_BASE::BDD_OPT_BASE;

            void min_marginal_averaging_forward();
            void min_marginal_averaging_backward();

            void iteration();
            void solve(const size_t max_iter, const double tolerance, const double time_limit);

            void min_marginal_averaging_step_forward(const size_t var);
            void min_marginal_averaging_step_backward(const size_t var); 
    };

    ////////////////////
    // implementation //
    ////////////////////
    
    // use only outgoing arc pointers
    template<typename BDD_OPT_BASE>
    void bdd_mma_base<BDD_OPT_BASE>::min_marginal_averaging_step_forward(const size_t var)
    {
        if(this->nr_bdds(var) == 0)
            return;

        std::array<double,2> min_marginals[this->nr_bdds(var)];

        for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index)
        {
            min_marginals[bdd_index] = this->min_marginal(var,bdd_index);
        }
 
        const std::array<double,2> average_marginal = this->average_marginals(min_marginals, min_marginals + this->nr_bdds(var));

        // set marginals in each bdd so min marginals match each other
        for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
            this->set_marginal(var,bdd_index,average_marginal,min_marginals[bdd_index]);
            this->forward_step(var,bdd_index);
        } 
    }

    // min marginal averaging
    template<typename BDD_OPT_BASE>
    void bdd_mma_base<BDD_OPT_BASE>::min_marginal_averaging_forward()
    {
        // MEASURE_FUNCTION_EXECUTION_TIME
        for(size_t var=0; var<this->nr_variables(); ++var) {
            min_marginal_averaging_step_forward(var);
        }
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_base<BDD_OPT_BASE>::min_marginal_averaging_step_backward(const size_t var)
    {
        if(this->nr_bdds(var) > 1)
        {
            // collect min marginals
            std::array<double,2> min_marginals[this->nr_bdds(var)];

            for(std::ptrdiff_t bdd_index=this->nr_bdds(var)-1; bdd_index >=0; --bdd_index) {
                min_marginals[bdd_index] = this->min_marginal(var,bdd_index);
            }

            const std::array<double,2> average_marginal = this->average_marginals(min_marginals, min_marginals + this->nr_bdds(var));

            // set marginals in each bdd so min marginals match each other
            for(std::ptrdiff_t bdd_index=this->nr_bdds(var)-1; bdd_index >=0; --bdd_index) {
                this->set_marginal(var,bdd_index,average_marginal,min_marginals[bdd_index]);
                this->backward_step(var, bdd_index);
                // TODO: reintroduce
                //lb += lower_bound_backward(var,bdd_index);
            }
        }
        else
        {
            for(std::ptrdiff_t bdd_index=this->nr_bdds(var)-1; bdd_index >=0; --bdd_index) {
                this->backward_step(var, bdd_index);
                // TODO: reintroduce
                //lb += lower_bound_backward(var,bdd_index);
            } 
        }
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_base<BDD_OPT_BASE>::min_marginal_averaging_backward()
    {
        // MEASURE_FUNCTION_EXECUTION_TIME;
        double lb = 0.0;

        for(std::ptrdiff_t var=this->nr_variables()-1; var>=0; --var) {
            min_marginal_averaging_step_backward(var);
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index)
                lb += this->lower_bound_backward(var,bdd_index);
        }

        this->lower_bound_ = lb; 
    }

    template<typename BDD_OPT_BASE>
        void bdd_mma_base<BDD_OPT_BASE>::iteration()
        {
            min_marginal_averaging_forward();
            min_marginal_averaging_backward();
        }

    template<typename BDD_OPT_BASE>
        void bdd_mma_base<BDD_OPT_BASE>::solve(const size_t max_iter, const double tolerance, const double time_limit)
        {
            const auto start_time = std::chrono::steady_clock::now();
            double lb_prev = this->lower_bound();
            double lb_post = lb_prev;
            std::cout << "initial lower bound = " << lb_prev;
            auto time = std::chrono::steady_clock::now();
            std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
            std::cout << "\n";
            for(size_t iter=0; iter<max_iter; ++iter)
            {
                iteration();
                lb_prev = lb_post;
                lb_post = this->lower_bound();
                std::cout << "iteration " << iter << ", lower bound = " << lb_post;
                time = std::chrono::steady_clock::now();
                double time_spent = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
                std::cout << ", time = " << time_spent << " s";
                std::cout << "\n";
                if (time_spent > time_limit)
                {
                    std::cout << "Time limit reached." << std::endl;
                    break;
                }
                if (std::abs(lb_prev-lb_post) < std::abs(tolerance*lb_prev))
                {
                    std::cout << "Relative progress less than tolerance (" << tolerance << ")\n";
                    break;
                }
            }
            std::cout << "final lower bound = " << this->lower_bound() << "\n"; 
        }

} 
