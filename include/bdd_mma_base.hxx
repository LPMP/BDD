#pragma once

#include "bdd_base.hxx"
#include "time_measure_util.h"
#include <vector>
#include <array>
#include <chrono>
#include <iostream>

namespace LPMP {

    // base class for min marginal averaging. Hold costs additionally to bdds.
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
        class bdd_mma_base : public bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>
    {
        public:
            using bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_base;

            void init(bdd_storage& bdd_storage);

            // TODO: implement set_cost( const std::vector<double>& costs);
            void set_cost(const double c, const size_t var_);
            template<typename ITERATOR>
                void set_costs(ITERATOR begin, ITERATOR end);
            void set_avg_type(const bdd_mma::averaging_type avg_type);

            template<typename ITERATOR>
                double evaluate(ITERATOR var_begin, ITERATOR var_end) const;

            template <typename ITERATOR>
                bool check_feasibility(ITERATOR var_begin, ITERATOR var_end) const;

            double lower_bound() const { return lower_bound_; }
            double lower_bound_backward(const std::size_t var, const std::size_t bdd_index);
            double lower_bound_forward(const std::size_t var, const std::size_t bdd_index);
            double compute_lower_bound();

            void min_marginal_averaging_iteration();
            void min_marginal_averaging_forward();
            void min_marginal_averaging_backward();
            void min_marginal_averaging_iteration_SRMP();
            void min_marginal_averaging_forward_SRMP();
            void min_marginal_averaging_backward_SRMP();

            void iteration();
            void solve(const size_t max_iter);

            void min_marginal_averaging_step_forward(const size_t var, std::vector<std::array<double,2>>& min_marginals);
            void min_marginal_averaging_step_forward_tmp(const size_t var, std::vector<std::array<double,2>>& min_marginals);
            void min_marginal_averaging_step_backward(const size_t var, std::vector<std::array<double,2>>& min_marginals); 

        protected: 
            std::array<double, 2> min_marginal(const std::size_t var, const std::size_t bdd_index) const;
            template <typename ITERATOR>
                static std::array<double, 2> average_marginals(ITERATOR marginals_begin, ITERATOR marginals_end, const std::size_t nr_marginals_to_distribute = std::numeric_limits<std::size_t>::max());
            template <typename ITERATOR>
                std::pair<std::array<double, 2>, bool> average_marginals_forward_SRMP(ITERATOR marginals_begin, ITERATOR marginals_end, const std::size_t var) const;
            template <typename ITERATOR>
                std::pair<std::array<double, 2>, bool> average_marginals_backward_SRMP(ITERATOR marginals_begin, ITERATOR marginals_end, const std::size_t var) const;
            void set_marginal(const std::size_t var, const std::size_t bdd_index, const std::array<double, 2> marginals, const std::array<double, 2> min_marginals);
            void set_marginal_forward_SRMP(const std::size_t var, const std::size_t bdd_index, const std::array<double, 2> marginals, const std::array<double, 2> min_marginals, const bool default_avg);
            void set_marginal_backward_SRMP(const std::size_t var, const std::size_t bdd_index, const std::array<double, 2> marginals, const std::array<double, 2> min_marginals, const bool default_avg);

            std::vector<double> costs_;
            double lower_bound_ = -std::numeric_limits<double>::infinity();

            bdd_mma::averaging_type avg_type_ = bdd_mma::averaging_type::classic;
    };

    // this class expects as bdd branch node something derived from bdd_branch_node_opt_arc_cost
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    class bdd_mma_base_arc_costs : public bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, bdd_mma_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>>
    {
    public:
        using base = bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, bdd_mma_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>>;
        using base::base;

        void update_cost(const size_t var, const size_t bdd_index, const double delta);

        void get_arc_marginals(const size_t var, const size_t bdd_index, std::vector<double>& marginals);

        template<typename ITERATOR>
            void update_arc_costs(const size_t var, const size_t bdd_index, ITERATOR cost_begin, ITERATOR cost_end);
    };

    // this class expects as bdd branch node bdd_branch_node_opt_node_cost and as bdd variable bdd_variable_opt or derived from these
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    class bdd_mma_base_node_costs : public bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, bdd_mma_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>>
    {
    public:
        using base = bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, bdd_mma_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>>;

        bdd_mma_base_node_costs() : base() {}
        bdd_mma_base_node_costs(bdd_storage& stor);

        // additionally set pointers of costs in bdd branch nodes
        void init(bdd_storage& bdd_storage_);

        void update_cost(const size_t var, const size_t bdd_index, const double delta);
    };

    ////////////////////
    // implementation //
    ////////////////////
    
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::init(bdd_storage& bdd_storage_)
    {
        bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::init(bdd_storage_);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    std::array<double,2> bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal(const std::size_t var, const std::size_t bdd_index) const
    {
        assert(var < this->nr_variables());
        assert(bdd_index < this->nr_bdds(var));
        std::array<double,2> m = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        const auto& bdd_var = this->bdd_variables_(var,bdd_index);
        for(size_t bdd_node_index=bdd_var.first_node_index; bdd_node_index<bdd_var.last_node_index; ++bdd_node_index) {
            const auto& bdd = this->bdd_branch_nodes_[bdd_node_index];
            const auto [m0,m1] = bdd.min_marginal();
            m[0] = std::min(m[0], m0);
            m[1] = std::min(m[1], m1);
        }
        assert(std::isfinite(m[0]));
        assert(std::isfinite(m[1]));
        return m;
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_marginal(const std::size_t var, const std::size_t bdd_index, const std::array<double,2> marginals, const std::array<double,2> min_marginals)
    {
        assert(var < this->nr_variables());
        assert(bdd_index < this->nr_bdds(var));
        assert(min_marginals == min_marginal(var,bdd_index));
        assert(std::isfinite(marginals[0]) && std::isfinite(marginals[1]));
        assert(std::isfinite(min_marginals[0]) && std::isfinite(min_marginals[1]));

        const double marginal_diff = min_marginals[1] - min_marginals[0];
        const double marginal_diff_target = marginals[1] - marginals[0];
        assert(std::isfinite(marginal_diff));
        assert(std::isfinite(marginal_diff_target));
        static_cast<DERIVED*>(this)->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_marginal_forward_SRMP(const std::size_t var, const std::size_t bdd_index, const std::array<double,2> marginals, const std::array<double,2> min_marginals, const bool default_avg)
    {
        const double marginal_diff = min_marginals[1] - min_marginals[0];
        const double marginal_diff_target = marginals[1] - marginals[0];
        assert(std::isfinite(marginal_diff));
        assert(std::isfinite(marginal_diff_target));
        if (default_avg)
        {
            static_cast<DERIVED*>(this)->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
            //bdd_var.cost += -marginal_diff + marginal_diff_target;
        }
        else if(this->last_variable_of_bdd(var, bdd_index)) {
            static_cast<DERIVED*>(this)->update_cost(var, bdd_index, -marginal_diff);
            //bdd_var.cost -= marginal_diff;
        } else {
            assert(std::isfinite(marginal_diff_target));
            static_cast<DERIVED*>(this)->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
            //bdd_var.cost += -marginal_diff + marginal_diff_target; 
        } 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_marginal_backward_SRMP(const std::size_t var, const std::size_t bdd_index, const std::array<double,2> marginals, const std::array<double,2> min_marginals, const bool default_avg)
    {
        auto& bdd_var = this->bdd_variables_(var,bdd_index);
        const double marginal_diff = min_marginals[1] - min_marginals[0];
        const double marginal_diff_target = marginals[1] - marginals[0];
        assert(std::isfinite(marginal_diff));
        assert(std::isfinite(marginal_diff_target));
        if (default_avg)
        {
            static_cast<DERIVED*>(this)->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
            //bdd_var.cost += -marginal_diff + marginal_diff_target;
        }
        else if(this->first_variable_of_bdd(var, bdd_index)) {
            static_cast<DERIVED*>(this)->update_cost(var, bdd_index, -marginal_diff);
            //bdd_var.cost -= marginal_diff;
        } else {
            assert(std::isfinite(marginal_diff_target));
            static_cast<DERIVED*>(this)->update_cost(var, bdd_index, -marginal_diff + marginal_diff_target);
            //bdd_var.cost += -marginal_diff + marginal_diff_target; 
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    double bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::compute_lower_bound()
    {
        double lb = 0.0;
        for(long int var=this->nr_variables()-1; var>=0; --var)
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index)
                lb += lower_bound_backward(var,bdd_index);
        this->lower_bound_ = lb;
        return lb;
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    double bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::lower_bound_backward(const std::size_t var, const std::size_t bdd_index)
    {
        const auto& bdd_var = this->bdd_variables_(var,bdd_index);
        if(this->first_variable_of_bdd(var, bdd_index)) {
            //assert(bdd_var.nr_bdd_nodes() == 1); // need not hold in decomposition bdd
            double lb = std::numeric_limits<double>::infinity();
            const std::size_t first_node_index = bdd_var.first_node_index;
            const std::size_t last_node_index = bdd_var.last_node_index;
            for (std::size_t i = first_node_index; i < last_node_index; ++i)
            {
                const auto& node = this->bdd_branch_nodes_[i];
                lb = std::min(node.m, lb);
            }
            return lb;
        } else {
            return 0.0;
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    double bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::lower_bound_forward(const std::size_t var, const std::size_t bdd_index)
    {
        const auto& bdd_var = this->bdd_variables_(var,bdd_index);
        if(this->last_variable_of_bdd(var, bdd_index)) 
        {
            double lb = std::numeric_limits<double>::infinity();
            const std::size_t first_node_index = bdd_var.first_node_index;
            const std::size_t last_node_index = bdd_var.last_node_index;
            for (std::size_t i = first_node_index; i < last_node_index; ++i)
            {
                const auto& node = this->bdd_branch_nodes_[i];
                assert(node.low_outgoing == node.terminal_1() || node.high_outgoing == node.terminal_1());
                const double node_lb = [&]() {
                    double lb = std::numeric_limits<double>::infinity();
                    if(node.low_outgoing == node.terminal_1())
                        lb = std::min(lb, node.m);
                    if(node.high_outgoing == node.terminal_1())
                        lb = std::min(lb, node.m + *node.variable_cost);
                    return lb;
                }();
                lb = std::min(lb, node_lb); 
            }
            return lb;
        }
        else
        {
            return 0.0;
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_iteration()
    {
        const auto begin_time = std::chrono::steady_clock::now();
        min_marginal_averaging_forward();
        const auto after_forward = std::chrono::steady_clock::now();
        // std::cout << "forward " <<  std::chrono::duration_cast<std::chrono::milliseconds>(after_forward - begin_time).count() << " ms, " << std::flush;
        const auto before_backward = std::chrono::steady_clock::now();
        min_marginal_averaging_backward();
        const auto end_time = std::chrono::steady_clock::now();
        // std::cout << "backward " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end_time - before_backward).count() << " ms, " << std::flush;
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    template<typename ITERATOR>
        std::array<double,2> bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::average_marginals(ITERATOR marginals_begin, ITERATOR marginals_end, const std::size_t nr_marginals_to_distribute)
        {
            std::array<double,2> average_marginal = {0.0, 0.0};
            for(auto m_iter=marginals_begin; m_iter!=marginals_end; ++m_iter) {
                average_marginal[0] += (*m_iter)[0];
                average_marginal[1] += (*m_iter)[1];
            }
            if(nr_marginals_to_distribute == std::numeric_limits<std::size_t>::max()) 
            {
                const double no_marginals = std::distance(marginals_begin, marginals_end);
                average_marginal[0] /= no_marginals;
                average_marginal[1] /= no_marginals;
            } 
            else
            {
                average_marginal[0] /= nr_marginals_to_distribute;
                average_marginal[1] /= nr_marginals_to_distribute; 
            }

            assert(std::isfinite(average_marginal[0]));
            assert(std::isfinite(average_marginal[1]));
            return average_marginal;
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    template<typename ITERATOR>
        std::pair<std::array<double,2>, bool> bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::average_marginals_forward_SRMP(ITERATOR marginals_begin, ITERATOR marginals_end, const std::size_t var) const
        {
            assert(this->nr_bdds(var) == std::distance(marginals_begin, marginals_end));
            std::array<double,2> average_marginal = {0.0, 0.0};
            std::size_t nr_averaged_marginals = 0;
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    average_marginal[0] += (*(marginals_begin+bdd_index))[0];
                    average_marginal[1] += (*(marginals_begin+bdd_index))[1];
                if(!this->last_variable_of_bdd(var, bdd_index))
                    ++nr_averaged_marginals;
            }
            // if no BDD satisfies forward condition, resort to averaging over all BDDs
            bool default_avg = false;
            if (nr_averaged_marginals == 0)
            {
                nr_averaged_marginals = this->nr_bdds(var);
                default_avg = true;
            }

            average_marginal[0] /= double(nr_averaged_marginals);
            average_marginal[1] /= double(nr_averaged_marginals);

            return std::make_pair(average_marginal, default_avg);
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    template<typename ITERATOR>
        std::pair<std::array<double,2>, bool> bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::average_marginals_backward_SRMP(ITERATOR marginals_begin, ITERATOR marginals_end, const std::size_t var) const
        {
            assert(this->nr_bdds(var) == std::distance(marginals_begin, marginals_end));
            std::array<double,2> average_marginal = {0.0, 0.0};
            std::size_t nr_averaged_marginals = 0;
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    average_marginal[0] += (*(marginals_begin+bdd_index))[0];
                    average_marginal[1] += (*(marginals_begin+bdd_index))[1];
                if(!this->first_variable_of_bdd(var, bdd_index))
                    ++nr_averaged_marginals;
            }
            // if no BDD satisfies forward condition, resort to averaging over all BDDs
            bool default_avg = false;
            if (nr_averaged_marginals == 0)
            {
                nr_averaged_marginals = this->nr_bdds(var);
                default_avg = true;
            }

            average_marginal[0] /= double(nr_averaged_marginals);
            average_marginal[1] /= double(nr_averaged_marginals);

            assert(std::isfinite(average_marginal[0]));
            assert(std::isfinite(average_marginal[1]));
            return std::make_pair(average_marginal, default_avg);
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_step_forward(const size_t var, std::vector<std::array<double,2>>& min_marginals)
    {
        assert(this->nr_bdds(var) > 0);
        //std::cout << "variable " << var << " of " << this->nr_variables() << std::endl;
        // collect min marginals
        if(this->nr_bdds(var) == 0)
            return;

        min_marginals.clear();
        for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
            this->forward_step(var,bdd_index);
            min_marginals.push_back(min_marginal(var,bdd_index)); 
        }

        const std::array<double,2> average_marginal = average_marginals(min_marginals.begin(), min_marginals.end());

        // set marginals in each bdd so min marginals match each other
        for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
            set_marginal(var,bdd_index,average_marginal,min_marginals[bdd_index]);
        } 
    }

    // use only outgoing arc pointers
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_step_forward_tmp(const size_t var, std::vector<std::array<double,2>>& min_marginals)
    {
        if(this->nr_bdds(var) == 0)
            return;
        min_marginals.clear();

        for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index)
            min_marginals.push_back(min_marginal(var,bdd_index)); 
 
        const std::array<double,2> average_marginal = average_marginals(min_marginals.begin(), min_marginals.end());

        // set marginals in each bdd so min marginals match each other
        for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
            set_marginal(var,bdd_index,average_marginal,min_marginals[bdd_index]);
            this->forward_step_tmp(var,bdd_index);
        } 
    }


    // min marginal averaging
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_forward()
    {
        std::vector<std::array<double,2>> min_marginals;
        for(std::size_t var=0; var<this->nr_variables(); ++var) {
            min_marginals.clear();
            min_marginal_averaging_step_forward_tmp(var, min_marginals);
            continue;

            // collect min marginals
            min_marginals.clear();
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                this->forward_step(var,bdd_index);
                min_marginals.push_back(min_marginal(var,bdd_index)); 
            }

            const std::array<double,2> average_marginal = average_marginals(min_marginals.begin(), min_marginals.end());

            // set marginals in each bdd so min marginals match each other
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal(var,bdd_index,average_marginal,min_marginals[bdd_index]);
            } 
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_forward_SRMP()
    {
        std::vector<std::array<double,2>> min_marginals;
        for(std::size_t var=0; var<this->nr_variables(); ++var) {

            // collect min marginals
            min_marginals.clear();
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                this->forward_step(var,bdd_index);
                min_marginals.push_back(min_marginal(var,bdd_index));
            }


            const auto average_marginal = average_marginals_forward_SRMP(min_marginals.begin(), min_marginals.end(), var);
            const std::array<double,2> avg_marg = average_marginal.first;
            const bool default_averaging = average_marginal.second;

            // set marginals in each bdd so min marginals match each other
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal_forward_SRMP(var,bdd_index,avg_marg,min_marginals[bdd_index], default_averaging);
            } 
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_step_backward(const size_t var, std::vector<std::array<double,2>>& min_marginals)
    {
        //std::cout << "variable " << var << " of " << this->nr_variables() << std::endl;
        // collect min marginals
        if(this->nr_bdds(var) > 0)
        {
            min_marginals.clear();
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                min_marginals.push_back(min_marginal(var,bdd_index)); 
            }

            const std::array<double,2> average_marginal = average_marginals(min_marginals.begin(), min_marginals.end());

            // set marginals in each bdd so min marginals match each other
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal(var,bdd_index,average_marginal,min_marginals[bdd_index]);
                this->backward_step(var, bdd_index);
                // TODO: reintroduce
                //lb += lower_bound_backward(var,bdd_index);
            }
        }
        else
        {
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                this->backward_step(var, bdd_index);
                // TODO: reintroduce
                //lb += lower_bound_backward(var,bdd_index);
            } 
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_backward()
    {
        double lb = 0.0;
        std::vector<std::array<double,2>> min_marginals;

        for(long int var=this->nr_variables()-1; var>=0; --var) {
            min_marginals.clear();
            min_marginal_averaging_step_backward(var, min_marginals);
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index)
                lb += lower_bound_backward(var,bdd_index);
            continue;

            // collect min marginals
            min_marginals.clear();
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                min_marginals.push_back(min_marginal(var,bdd_index)); 
            }

            if(min_marginals.size() > 1) {
                //std::cout << "min marginals in backward pass: ";
                //for(const auto m : min_marginals)
                //    std::cout << "(" << m[0] << "," << m[1] << "), ";
                //std::cout << "\n";
            }

            const std::array<double,2> average_marginal = average_marginals(min_marginals.begin(), min_marginals.end());

            // set marginals in each bdd so min marginals match each other
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal(var,bdd_index,average_marginal,min_marginals[bdd_index]);
                this->backward_step(var, bdd_index);
                lb += lower_bound_backward(var,bdd_index);
            }
        }

        lower_bound_ = lb; 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_backward_SRMP()
    {
        double lb = 0.0;
        std::vector<std::array<double,2>> min_marginals;
        for(long int var=this->nr_variables()-1; var>=0; --var) {

            // collect min marginals
            min_marginals.clear();
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                min_marginals.push_back(min_marginal(var,bdd_index)); 
            }

            const auto average_marginal = average_marginals_backward_SRMP(min_marginals.begin(), min_marginals.end(), var);
            const std::array<double,2> avg_marg = average_marginal.first;
            const bool default_averaging = average_marginal.second;

            // set marginals in each bdd so min marginals match each other
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                set_marginal_backward_SRMP(var,bdd_index,avg_marg,min_marginals[bdd_index], default_averaging);
                this->backward_step(var, bdd_index);
                lb += lower_bound_backward(var,bdd_index);
            }
        }

        lower_bound_ = lb; 
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal_averaging_iteration_SRMP()
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



    //const bdd_branch_instruction& bdd_min_marginal_averaging::get_bdd_branch_instruction(const std::size_t var, const std::size_t bdd_index, const std::size_t bdd_node_index) const
   
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::iteration()
    {
        if(this->avg_type_ == bdd_mma::averaging_type::classic)
            min_marginal_averaging_iteration();
        else if(this->avg_type_ == bdd_mma::averaging_type::srmp)
            min_marginal_averaging_iteration_SRMP();
        else
            assert(false);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::solve(const size_t max_iter)
    {
        for(size_t iter=0; iter<max_iter; ++iter)
        {
            iteration();
            std::cout << "iteration " << iter << ", lower bound = " << lower_bound() << "\n";
        }
        std::cout << "final lower bound = " << lower_bound() << "\n";

    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    template<typename ITERATOR>
        void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_costs(ITERATOR begin, ITERATOR end)
        {
            // TODO: remove costs_ array
            std::fill(costs_.begin(), costs_.end(), 0.0);
            assert(std::distance(begin,end) <= this->nr_variables());
            std::copy(begin, end, costs_.begin());

            // distribute costs to bdds uniformly
            for(std::size_t v=0; v<this->nr_variables(); ++v)
            {
                const double cost = v < std::distance(begin,end) ? *(begin+v) : 0.0; 
                set_cost(cost, v);
            }
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
        void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_cost(const double c, const size_t var)
        {
            assert(var < this->nr_variables());
            assert(this->nr_bdds(var) > 0);
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                static_cast<DERIVED*>(this)->update_cost(var, bdd_index, c/double(this->nr_bdds(var)));
            }
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
        void bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_avg_type(const bdd_mma::averaging_type avg_type)
        {
            avg_type_ = avg_type;
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
        template<typename ITERATOR>
        bool bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::check_feasibility(ITERATOR var_begin, ITERATOR var_end) const
        {
            assert(std::distance(var_begin, var_end) == this->nr_variables());

            std::vector<char> bdd_nbranch_node_marks(this->bdd_branch_nodes_.size(), 0);

            std::size_t var = 0;
            for(auto var_iter=var_begin; var_iter!=var_end; ++var_iter, ++var) {
                const bool val = *(var_begin+var);
                for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                    const auto& bdd_var = this->bdd_variables_(var, bdd_index);
                    if(bdd_var.is_first_bdd_variable()) {
                        bdd_nbranch_node_marks[bdd_var.first_node_index] = 1;
                    }
                    for(std::size_t bdd_node_index=bdd_var.first_node_index; bdd_node_index<bdd_var.last_node_index; ++bdd_node_index) {
                        if(bdd_nbranch_node_marks[bdd_node_index] == 1) {
                            const auto& bdd = this->bdd_branch_nodes_[bdd_node_index];
                            const auto* bdd_next_index = [&]() {
                                if(val == false)
                                    return bdd.low_outgoing;
                                else 
                                    return bdd.high_outgoing;
                            }();

                            if(bdd_next_index == BDD_BRANCH_NODE::terminal_0())
                                return false;
                            if(bdd_next_index == BDD_BRANCH_NODE::terminal_1()) {
                            } else { 
                                bdd_nbranch_node_marks[ this->bdd_branch_node_index(bdd_next_index) ] = 1;
                            }
                        }
                    }
                }
            }

            return true;
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    template<typename ITERATOR>
        double bdd_mma_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::evaluate(ITERATOR var_begin, ITERATOR var_end) const
        {
            assert(std::distance(var_begin, var_end) == this->nr_variables());

            if(!check_feasibility(var_begin, var_end))
                return std::numeric_limits<double>::infinity();

            double cost = 0.0;
            std::size_t var = 0;
            for(auto var_iter=var_begin; var_iter!=var_end; ++var_iter, ++var)
                cost += *var_iter * costs_[var];
            return cost;
        }

    ////////////////////////////
    // bdd_mma_base_arc_costs //
    ////////////////////////////
    
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_mma_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::update_cost(const size_t var, const size_t bdd_index, const double delta)
    {
        assert(var < this->nr_variables());
        assert(bdd_index < this->nr_bdds(var));

        const auto& bdd_var = this->bdd_variables_(var, bdd_index);
        const size_t first_node_index = bdd_var.first_node_index;
        const size_t last_node_index = bdd_var.last_node_index;
        for(size_t i=first_node_index; i<last_node_index; ++i)
        {
            auto& node = this->bdd_branch_nodes_[i];
            node.high_cost += delta;
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
        void bdd_mma_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::get_arc_marginals(const size_t var, const size_t bdd_index, std::vector<double>& marginals)
        {
            assert(var < this->nr_variables());
            assert(bdd_index < this->nr_bdds(var));
            marginals.clear();
            
            const auto & bdd_var = this->bdd_variables_(var, bdd_index);
            const std::size_t first_node_index = bdd_var.first_node_index;
            const std::size_t last_node_index = bdd_var.last_node_index;
            auto cost_it = marginals.begin();
            for(size_t i = first_node_index; i < last_node_index; ++i)
            {
                auto& node = this->bdd_branch_nodes_[i];
                const auto margs = node.min_marginal();
                if(node.low_outgoing != node.terminal_0())
                {
                    marginals.push_back(margs[0]);
                }
                else
                {
                    assert(margs[0] == std::numeric_limits<double>::infinity());
                    //assert(node.low_cost == std::numeric_limits<double>::infinity());
                }
                if(node.high_outgoing != node.terminal_0())
                {
                    marginals.push_back(margs[1]);
                }
                else
                {
                    assert(margs[1] == std::numeric_limits<double>::infinity());
                    //assert(node.high_cost == std::numeric_limits<double>::infinity());
                }

            }

            assert(marginals.size() == this->nr_feasible_outgoing_arcs(var, bdd_index));
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    template<typename ITERATOR>
        void bdd_mma_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::update_arc_costs(const size_t var, const size_t bdd_index, ITERATOR cost_begin, ITERATOR cost_end)
        {
            assert(var < this->nr_variables());
            assert(bdd_index < this->nr_bdds(var));
            assert(std::distance(cost_begin, cost_end) == this->nr_feasible_outgoing_arcs(var, bdd_index));
            
            const auto & bdd_var = this->bdd_variables_(var, bdd_index);
            const size_t first_node_index = bdd_var.first_node_index;
            const size_t last_node_index = bdd_var.last_node_index;
            auto cost_it = cost_begin;
            for(size_t i=first_node_index; i<last_node_index; ++i)
            {
                auto& node = this->bdd_branch_nodes_[i];
                if(node.low_outgoing != node.terminal_0())
                {
                    node.low_cost += *cost_it;
                    ++cost_it;
                }
                else
                {
                    //assert(node.low_cost == std::numeric_limits<double>::infinity());
                }
                if(node.high_outgoing != node.terminal_0())
                {
                    node.high_cost += *cost_it;
                    ++cost_it;
                }
                else
                { 
                    //assert(node.high_cost == std::numeric_limits<double>::infinity());
                }

            }

            assert(cost_it == cost_end);
        }

    /////////////////////////////
    // bdd_mma_base_node_costs //
    /////////////////////////////
    
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    bdd_mma_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_mma_base_node_costs(bdd_storage& bdd_storage_)
    {
        init(bdd_storage_); 
    } 

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_mma_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::init(bdd_storage& bdd_storage_)
    {
        base::init(bdd_storage_);

        for(size_t var = 0; var < this->nr_variables(); var++)
        {
            for (size_t bdd_index = 0; bdd_index < this->nr_bdds(var); bdd_index++)
            {
                auto & bdd_var = this->bdd_variables_(var, bdd_index);
                for (size_t node_index = bdd_var.first_node_index; node_index < bdd_var.last_node_index; node_index++)
                {
                    this->bdd_branch_nodes_[node_index].variable_cost = & bdd_var.cost;
                }
            }
        }
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    void bdd_mma_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::update_cost(const size_t var, const size_t bdd_index, const double delta)
    {
        assert(var < this->nr_variables());
        assert(bdd_index < this->nr_bdds(var));
        auto& bdd_var = this->bdd_variables_(var, bdd_index);
        bdd_var.cost += delta;
    }

}

