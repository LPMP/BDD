#pragma once

#include "bdd_mma_base.hxx"

namespace LPMP {

    template<typename BDD_OPT_BASE>
        class bdd_mma_anisotropic_base : public BDD_OPT_BASE {
            public: 
                using BDD_OPT_BASE::BDD_OPT_BASE;

                void min_marginal_averaging_forward();
                void min_marginal_averaging_backward();
                void iteration();
                void solve(const size_t max_iter, const double tolerance); 
        };

    ////////////////////
    // implementation //
    ////////////////////

    template<typename BDD_OPT_BASE>
        void bdd_mma_anisotropic_base<BDD_OPT_BASE>::min_marginal_averaging_forward()
        {
            for(size_t var=0; var<this->nr_variables(); ++var) {

                // collect min marginals
                this->min_marginal_averaging_step_forward(var);

                // check if some of the BDDs of the current variable are last and propagate min-marginals of preceeding variables to bdds that are used later on
                for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index)
                {
                    auto& bdd_var = this->bdd_variables_(var, bdd_index);
                    if(bdd_var.is_last_bdd_variable()) {
                        const size_t last_var_index = bdd_var.last_var_index;
                        assert(last_var_index == bdd_var.var_index);
                        auto* var = &bdd_var;
                        while(var != nullptr)
                        {
                            const size_t cur_variable_index = var->var_index;
                            assert(cur_variable_index <= last_var_index);
                            const std::array<double,2> min_marg = this->min_marginal(*var);

                            size_t nr_bdds_to_push = 0;
                            for(size_t bdd_index=0; bdd_index<this->nr_bdds(cur_variable_index); ++bdd_index)
                            {
                                const auto& bdd_var = this->bdd_variables_(cur_variable_index, bdd_index);
                                if(bdd_var.last_var_index > last_var_index)
                                    ++nr_bdds_to_push;
                            }

                            if(nr_bdds_to_push > 0)
                            {
                                const double delta = (min_marg[1] - min_marg[0])/double(nr_bdds_to_push);
                                this->update_cost(*var, -(min_marg[1] - min_marg[0]));

                                for(size_t bdd_index=0; bdd_index<this->nr_bdds(cur_variable_index); ++bdd_index)
                                {
                                    const auto& bdd_var = this->bdd_variables_(cur_variable_index, bdd_index);
                                    if(bdd_var.last_var_index > last_var_index)
                                    {
                                        this->update_cost(cur_variable_index, bdd_index, delta);
                                    }
                                }
                            }

                            this->backward_step(*var);
                            var = var->prev;
                        }
                    } 
                } 
            }
        }

    template<typename BDD_OPT_BASE>
        void bdd_mma_anisotropic_base<BDD_OPT_BASE>::min_marginal_averaging_backward()
        {

            for(std::ptrdiff_t var=this->nr_variables()-1; var>=0; --var) {
                this->min_marginal_averaging_step_backward(var);
                for(size_t bdd_index=0; bdd_index<this->nr_bdds(var); ++bdd_index)
                {
                    const auto& bdd_var = this->bdd_variables_(var, bdd_index);
                    if(bdd_var.is_first_bdd_variable()) {
                        const size_t first_var_index = bdd_var.first_var_index;
                        auto* var = bdd_var.prev;
                        while(var != nullptr)
                        {
                            const size_t cur_variable_index = var->var_index;
                            const std::array<double,2> min_marg = this->min_marginal(*var);

                            size_t nr_bdds_to_push = 0;
                            for(size_t bdd_index=0; bdd_index<this->nr_bdds(cur_variable_index); ++bdd_index)
                            {
                                const auto& bdd_var = this->bdd_variables_(cur_variable_index, bdd_index);
                                if(bdd_var.first_var_index < first_var_index)
                                    ++nr_bdds_to_push;
                            }

                            if(nr_bdds_to_push > 0)
                            {
                                const double delta = (min_marg[1] - min_marg[0])/double(nr_bdds_to_push);
                                this->update_cost(*var, -(min_marg[1] - min_marg[0]));

                                for(size_t bdd_index=0; bdd_index<this->nr_bdds(cur_variable_index); ++bdd_index)
                                {
                                    const auto& bdd_var = this->bdd_variables_(cur_variable_index, bdd_index);
                                    if(bdd_var.first_var_index < first_var_index)
                                    {
                                        this->update_cost(cur_variable_index, bdd_index, delta);
                                    }
                                }
                            }

                            this->forward_step(*var);
                            var = var->next;
                        }

                    }
                }
            }
        }

    template<typename BDD_OPT_BASE>
        void bdd_mma_anisotropic_base<BDD_OPT_BASE>::iteration()
        {
            min_marginal_averaging_forward();
            this->forward_run();
            min_marginal_averaging_backward(); 
            this->backward_run();
        }

    template<typename BDD_OPT_BASE>
        void bdd_mma_anisotropic_base<BDD_OPT_BASE>::solve(const size_t max_iter, const double tolerance)
        {
            for(size_t iter=0; iter<max_iter; ++iter)
            {
                iteration(); 
                this->compute_lower_bound();
                std::cout << "iteration " << iter << ", lower bound = " << this->lower_bound() << "\n";
            }
        }

} 
