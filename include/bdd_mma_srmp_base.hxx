#pragma once

#include "bdd_mma_base.hxx"
#include "time_measure_util.h"
#include <vector>
#include <array>
#include <iostream>

namespace LPMP {

    // base class for min marginal averaging with SRMP-like distribution of excess costs
    template<typename BDD_OPT_BASE>
        class bdd_mma_srmp_base : public bdd_mma_base<BDD_OPT_BASE> {
        public:
            using bdd_mma_base<BDD_OPT_BASE>::bdd_mma_base;

            void min_marginal_averaging_forward_SRMP();
            void min_marginal_averaging_backward_SRMP();
            void iteration();
            void solve(const size_t max_iter);

        protected: 
            template <typename ITERATOR>
                std::pair<std::array<double, 2>, bool> average_marginals_forward_SRMP(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t var) const;
            template <typename ITERATOR>
                std::pair<std::array<double, 2>, bool> average_marginals_backward_SRMP(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t var) const;
            void set_marginal_forward_SRMP(const size_t var, const size_t bdd_index, const std::array<double, 2> marginals, const std::array<double, 2> min_marginals, const bool default_avg);
            void set_marginal_backward_SRMP(const size_t var, const size_t bdd_index, const std::array<double, 2> marginals, const std::array<double, 2> min_marginals, const bool default_avg);

    };

    ////////////////////
    // implementation //
    ////////////////////
    
    template<typename BDD_OPT_BASE>
    void bdd_mma_srmp_base<BDD_OPT_BASE>::set_marginal_forward_SRMP(const size_t var, const size_t bdd_index, const std::array<double,2> marginals, const std::array<double,2> min_marginals, const bool default_avg)
    {
        const double marginal_diff = min_marginals[1] - min_marginals[0];
        const double marginal_diff_target = marginals[1] - marginals[0];
        assert(std::isfinite(marginal_diff));
        assert(std::isfinite(marginal_diff_target));
        if (default_avg)
        {
            this->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
            //bdd_var.cost += -marginal_diff + marginal_diff_target;
        }
        else if(this->last_variable_of_bdd(var, bdd_index)) {
            this->update_cost(var, bdd_index, -marginal_diff);
            //bdd_var.cost -= marginal_diff;
        } else {
            assert(std::isfinite(marginal_diff_target));
            this->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
            //bdd_var.cost += -marginal_diff + marginal_diff_target; 
        } 
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_srmp_base<BDD_OPT_BASE>::set_marginal_backward_SRMP(const size_t var, const size_t bdd_index, const std::array<double,2> marginals, const std::array<double,2> min_marginals, const bool default_avg)
    {
        auto& bdd_var = this->bdd_variables_(var,bdd_index);
        const double marginal_diff = min_marginals[1] - min_marginals[0];
        const double marginal_diff_target = marginals[1] - marginals[0];
        assert(std::isfinite(marginal_diff));
        assert(std::isfinite(marginal_diff_target));
        if (default_avg)
        {
            this->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
            //bdd_var.cost += -marginal_diff + marginal_diff_target;
        }
        else if(this->first_variable_of_bdd(var, bdd_index)) {
            this->update_cost(var, bdd_index, -marginal_diff);
            //bdd_var.cost -= marginal_diff;
        } else {
            assert(std::isfinite(marginal_diff_target));
            this->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
            //bdd_var.cost += -marginal_diff + marginal_diff_target; 
        }
    }

    template<typename BDD_OPT_BASE>
    template<typename ITERATOR>
        std::pair<std::array<double,2>, bool> bdd_mma_srmp_base<BDD_OPT_BASE>::average_marginals_forward_SRMP(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t var) const
        {
            assert(this->nr_bdds(var) == std::distance(marginals_begin, marginals_end));
            std::array<double,2> average_marginal = {0.0, 0.0};
            size_t divisor = 0;
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    average_marginal[0] += (*(marginals_begin+bdd_index))[0];
                    average_marginal[1] += (*(marginals_begin+bdd_index))[1];
                if(!this->last_variable_of_bdd(var, bdd_index))
                    ++divisor;
            }
            // if no BDD satisfies forward condition, resort to averaging over all BDDs
            bool default_avg = false;
            if (divisor == 0)
            {
                divisor = this->nr_bdds(var);
                default_avg = true;
            }

            average_marginal[0] /= double(divisor);
            average_marginal[1] /= double(divisor);

            return std::make_pair(average_marginal, default_avg);
        }

    template<typename BDD_OPT_BASE>
    template<typename ITERATOR>
        std::pair<std::array<double,2>, bool> bdd_mma_srmp_base<BDD_OPT_BASE>::average_marginals_backward_SRMP(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t var) const
        {
            assert(this->nr_bdds(var) == std::distance(marginals_begin, marginals_end));
            std::array<double,2> average_marginal = {0.0, 0.0};
            size_t divisor = 0;
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    average_marginal[0] += (*(marginals_begin+bdd_index))[0];
                    average_marginal[1] += (*(marginals_begin+bdd_index))[1];
                if(!this->first_variable_of_bdd(var, bdd_index))
                    ++divisor;
            }
            // if no BDD satisfies forward condition, resort to averaging over all BDDs
            bool default_avg = false;
            if (divisor == 0)
            {
                divisor = this->nr_bdds(var);
                default_avg = true;
            }

            average_marginal[0] /= double(divisor);
            average_marginal[1] /= double(divisor);

            assert(std::isfinite(average_marginal[0]));
            assert(std::isfinite(average_marginal[1]));
            return std::make_pair(average_marginal, default_avg);
        }

    template<typename BDD_OPT_BASE>
    void bdd_mma_srmp_base<BDD_OPT_BASE>::min_marginal_averaging_forward_SRMP()
    {
        std::vector<std::array<double,2>> min_marginals;
        for(size_t var=0; var<this->nr_variables(); ++var) {

            // collect min marginals
            min_marginals.clear();
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                min_marginals.push_back(this->min_marginal(var,bdd_index));
            }

            const auto average_marginal = average_marginals_forward_SRMP(min_marginals.begin(), min_marginals.end(), var);
            const std::array<double,2> avg_marg = average_marginal.first;
            const bool default_averaging = average_marginal.second;

            // set marginals in each bdd so min marginals match each other
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal_forward_SRMP(var,bdd_index,avg_marg,min_marginals[bdd_index], default_averaging);
                this->forward_step(var,bdd_index);
            } 
        }
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_srmp_base<BDD_OPT_BASE>::min_marginal_averaging_backward_SRMP()
    {
        double lb = 0.0;
        std::vector<std::array<double,2>> min_marginals;
        for(long int var=this->nr_variables()-1; var>=0; --var) {

            // collect min marginals
            min_marginals.clear();
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                min_marginals.push_back(this->min_marginal(var,bdd_index)); 
            }

            const auto average_marginal = average_marginals_backward_SRMP(min_marginals.begin(), min_marginals.end(), var);
            const std::array<double,2> avg_marg = average_marginal.first;
            const bool default_averaging = average_marginal.second;

            // set marginals in each bdd so min marginals match each other
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal_backward_SRMP(var,bdd_index,avg_marg,min_marginals[bdd_index], default_averaging);
                this->backward_step(var, bdd_index);
                lb += this->lower_bound_backward(var,bdd_index);
            }
        }

        this->lower_bound_ = lb; 
    }

    template<typename BDD_OPT_BASE>
    void bdd_mma_srmp_base<BDD_OPT_BASE>::iteration()
    {
        const auto begin_time = std::chrono::steady_clock::now();
        min_marginal_averaging_forward_SRMP();
        const auto after_forward = std::chrono::steady_clock::now();
        // std::cout << "forward " <<  std::chrono::duration_cast<std::chrono::milliseconds>(after_forward - begin_time).count() << " ms, " << std::flush;
        const auto before_backward = std::chrono::steady_clock::now();
        min_marginal_averaging_backward_SRMP();
        const auto end_time = std::chrono::steady_clock::now();
        // std::cout << "backward " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end_time - before_backward).count() << " ms, " << std::flush;
    }

    template<typename BDD_OPT_BASE>
        void bdd_mma_srmp_base<BDD_OPT_BASE>::solve(const size_t max_iter)
        {
            std::cout << "initial lower bound = " << this->lower_bound() << "\n";
            for(size_t iter=0; iter<max_iter-1; ++iter)
            {
                iteration();
                std::cout << "iteration " << iter << ", lower bound = " << this->lower_bound() << "\n";
            }
            bdd_mma_base<BDD_OPT_BASE>::iteration();
            std::cout << "iteration " << max_iter-1 << ", lower bound = " << this->lower_bound() << "\n";
            std::cout << "(last iteration with default averaging)" << std::endl;
            std::cout << "final lower bound = " << this->lower_bound() << "\n";

        }

}


