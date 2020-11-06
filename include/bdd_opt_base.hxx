#pragma once

#include "bdd_base.hxx"
#include <vector>
#include <array>

namespace LPMP {

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
        class bdd_opt_base : public bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>
    {
        public:
            using bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_base;

            void init(bdd_storage& bdd_storage);

            // TODO: implement set_cost( const std::vector<double>& costs);
            void set_cost(const double c, const size_t var_);
            template<typename ITERATOR>
                void set_costs(ITERATOR begin, ITERATOR end);
            template<typename ITERATOR>
                double evaluate(ITERATOR var_begin, ITERATOR var_end) const;

            double lower_bound() const { return lower_bound_; }
            double lower_bound_backward(const std::size_t var, const std::size_t bdd_index);
            double lower_bound_forward(const std::size_t var, const std::size_t bdd_index);
            double compute_lower_bound();

        protected: 
            std::array<double, 2> min_marginal(const size_t var, const size_t bdd_index) const;
            std::array<double, 2> min_marginal(const BDD_VARIABLE& bdd_var) const;
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
    };

    // this class expects as bdd branch node something derived from bdd_branch_node_opt_arc_cost
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    class bdd_opt_base_arc_costs : public bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, bdd_opt_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>>
    {
    public:
        using base = bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, bdd_opt_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>>;
        using base::base;

        void update_cost(const size_t var, const size_t bdd_index, const double delta);
        void update_cost(const BDD_VARIABLE& bdd_var, const double delta);

        void get_arc_marginals(const size_t var, const size_t bdd_index, std::vector<double>& marginals);

        template<typename ITERATOR>
            void update_arc_costs(const size_t var, const size_t bdd_index, ITERATOR cost_begin, ITERATOR cost_end);
    };

    // this class expects as bdd branch node bdd_branch_node_opt_node_cost and as bdd variable bdd_variable_opt or derived from these
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
    class bdd_opt_base_node_costs : public bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, bdd_opt_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>>
    {
    public:
        using base = bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, bdd_opt_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>>;

        bdd_opt_base_node_costs() : base() {}
        bdd_opt_base_node_costs(bdd_storage& stor);

        // additionally set pointers of costs in bdd branch nodes
        void init(bdd_storage& bdd_storage_);

        void update_cost(const size_t var, const size_t bdd_index, const double delta);
        void update_cost(BDD_VARIABLE& bdd_var, const double delta);
    };

    ////////////////////
    // implementation //
    ////////////////////
    
    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    void bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::init(bdd_storage& bdd_storage_)
    {
        bdd_base<BDD_VARIABLE, BDD_BRANCH_NODE>::init(bdd_storage_);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    std::array<double,2> bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal(const std::size_t var, const std::size_t bdd_index) const
    {
        assert(var < this->nr_variables());
        assert(bdd_index < this->nr_bdds(var));
        const auto& bdd_var = this->bdd_variables_(var,bdd_index);
        return min_marginal(bdd_var);
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
        std::array<double, 2> bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::min_marginal(const BDD_VARIABLE& bdd_var) const
        {
            std::array<double,2> m = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
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
    void bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_marginal(const std::size_t var, const std::size_t bdd_index, const std::array<double,2> marginals, const std::array<double,2> min_marginals)
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
    void bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_marginal_forward_SRMP(const std::size_t var, const std::size_t bdd_index, const std::array<double,2> marginals, const std::array<double,2> min_marginals, const bool default_avg)
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
    void bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_marginal_backward_SRMP(const std::size_t var, const std::size_t bdd_index, const std::array<double,2> marginals, const std::array<double,2> min_marginals, const bool default_avg)
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
    double bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::compute_lower_bound()
    {
        double lb = 0.0;
        for(long int var=this->nr_variables()-1; var>=0; --var)
            for(std::size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index)
                lb += lower_bound_backward(var,bdd_index);
        this->lower_bound_ = lb;
        return lb;
    }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
    double bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::lower_bound_backward(const std::size_t var, const std::size_t bdd_index)
    {
        const auto& bdd_var = this->bdd_variables_(var,bdd_index);
        if(this->first_variable_of_bdd(var, bdd_index)) {
            //assert(bdd_var.nr_bdd_nodes() == 1); // need not hold in decomposition bdd
            double lb = std::numeric_limits<double>::infinity();
            const size_t first_node_index = bdd_var.first_node_index;
            const size_t last_node_index = bdd_var.last_node_index;
            for (size_t i = first_node_index; i < last_node_index; ++i)
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
    double bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::lower_bound_forward(const std::size_t var, const std::size_t bdd_index)
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
    template<typename ITERATOR>
        std::array<double,2> bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::average_marginals(ITERATOR marginals_begin, ITERATOR marginals_end, const size_t nr_marginals_to_distribute)
        {
            std::array<double,2> average_marginal = {0.0, 0.0};
            for(auto m_iter=marginals_begin; m_iter!=marginals_end; ++m_iter) {
                average_marginal[0] += (*m_iter)[0];
                average_marginal[1] += (*m_iter)[1];
            }
            if(nr_marginals_to_distribute == std::numeric_limits<std::size_t>::max()) 
            {
                const double nr_marginals = std::distance(marginals_begin, marginals_end);
                average_marginal[0] /= nr_marginals;
                average_marginal[1] /= nr_marginals;
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
        void bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_costs(ITERATOR begin, ITERATOR end)
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
        void bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::set_cost(const double c, const size_t var)
        {
            assert(var < this->nr_variables());
            assert(this->nr_bdds(var) > 0);
            for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index) {
                static_cast<DERIVED*>(this)->update_cost(var, bdd_index, c/double(this->nr_bdds(var)));
            }
        } 

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE, typename DERIVED>
        template<typename ITERATOR>
        double bdd_opt_base<BDD_VARIABLE, BDD_BRANCH_NODE, DERIVED>::evaluate(ITERATOR var_begin, ITERATOR var_end) const
        {
            assert(std::distance(var_begin, var_end) == this->nr_variables());

            if(!this->check_feasibility(var_begin, var_end))
                return std::numeric_limits<double>::infinity();

            double cost = 0.0;
            std::size_t var = 0;
            for(auto var_iter=var_begin; var_iter!=var_end; ++var_iter, ++var)
                cost += *var_iter * costs_[var];
            return cost;
        }

    ////////////////////////////
    // bdd_opt_base_arc_costs //
    ////////////////////////////

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
        void bdd_opt_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::update_cost(const size_t var, const size_t bdd_index, const double delta)
        {
            assert(var < this->nr_variables());
            assert(bdd_index < this->nr_bdds(var));

            const auto& bdd_var = this->bdd_variables_(var, bdd_index);
            update_cost(bdd_var, delta);
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
        void bdd_opt_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::update_cost(const BDD_VARIABLE& bdd_var, const double delta)
        {
            const size_t first_node_index = bdd_var.first_node_index;
            const size_t last_node_index = bdd_var.last_node_index;
            for(size_t i=first_node_index; i<last_node_index; ++i)
            {
                auto& node = this->bdd_branch_nodes_[i];
                node.high_cost += delta;
            }
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
        void bdd_opt_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::get_arc_marginals(const size_t var, const size_t bdd_index, std::vector<double>& marginals)
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
        void bdd_opt_base_arc_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::update_arc_costs(const size_t var, const size_t bdd_index, ITERATOR cost_begin, ITERATOR cost_end)
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
    // bdd_opt_base_node_costs //
    /////////////////////////////

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
        bdd_opt_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::bdd_opt_base_node_costs(bdd_storage& bdd_storage_)
        {
            init(bdd_storage_); 
        } 

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
        void bdd_opt_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::init(bdd_storage& bdd_storage_)
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
        void bdd_opt_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::update_cost(const size_t var, const size_t bdd_index, const double delta)
        {
            assert(var < this->nr_variables());
            assert(bdd_index < this->nr_bdds(var));
            auto& bdd_var = this->bdd_variables_(var, bdd_index);
            update_cost(bdd_var, delta);
        }

    template<typename BDD_VARIABLE, typename BDD_BRANCH_NODE>
        void bdd_opt_base_node_costs<BDD_VARIABLE, BDD_BRANCH_NODE>::update_cost(BDD_VARIABLE& bdd_var, const double delta)
        {
            bdd_var.cost += delta; 
        }

}
