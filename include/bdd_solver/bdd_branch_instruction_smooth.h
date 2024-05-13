#pragma once 

#include <numeric>
#include <array>
#include <limits>
#include <type_traits>
#include "bdd_branch_instruction.h"
#include "exp_sum.h"

namespace LPMP {

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED, template<class,class,class> class BASE>
        class bdd_branch_instruction_smooth_base : public BASE<REAL,OFFSET_TYPE,DERIVED> 
    {
        public:
            static constexpr bool with_sum_exp_normalization = true;
            REAL current_max = -std::numeric_limits<REAL>::infinity();
            void prepare_smooth_forward_step();
            void smooth_forward_step();
            void smooth_backward_step();
            std::array<exp_sum<REAL>,2> sum_marginals() const;

            exp_sum<REAL> get_exp_sum() const { return {this->m, current_max}; }
            bdd_branch_instruction_smooth_base& assign(const exp_sum<REAL> es);

            void check_instruction() const;
    };

    template<typename REAL, typename OFFSET_TYPE>
        class bdd_branch_instruction_smooth : 
            public bdd_branch_instruction_smooth_base<REAL, OFFSET_TYPE,
            bdd_branch_instruction_smooth<REAL, OFFSET_TYPE>,
            bdd_branch_instruction_base> 
    {};

    template<typename REAL, typename OFFSET_TYPE>
        class bdd_branch_instruction_smooth_bdd_index :
            public bdd_branch_instruction_smooth_base<REAL, OFFSET_TYPE,
            bdd_branch_instruction_smooth_bdd_index<REAL, OFFSET_TYPE>,
            bdd_branch_instruction_bdd_index_base> 
    {
        public:
            void update_sum_marginals(std::array<exp_sum<REAL>,2>* sum_marginals) const;
            void set_sum_marginals(std::array<exp_sum<REAL>,2>* sum_marginals, const std::array<exp_sum<REAL>,2>& avg_marginals);
    };

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED, template<class,class,class> class BASE>
        void bdd_branch_instruction_smooth_base<REAL,OFFSET_TYPE,DERIVED,BASE>::smooth_backward_step()
        {
            assert(std::isfinite(std::min(this->low_cost, this->high_cost)));

            // version without numerical stabilization
            if(!with_sum_exp_normalization)
            {
                this->current_max = 0.0;

                if(this->offset_low == this->terminal_0_offset)
                    this->m = 0;
                else if(this->offset_low == this->terminal_1_offset)
                    this->m = std::exp(-this->low_cost);
                else
                    this->m = this->address(this->offset_low)->m * std::exp(-this->low_cost);

                if(this->offset_high == this->terminal_0_offset)
                    this->m += 0;
                else if(this->offset_high == this->terminal_1_offset)
                    this->m += std::exp(-this->high_cost);
                else
                    this->m += this->address(this->offset_high)->m * std::exp(-this->high_cost);

                assert(!std::isnan(this->m));

                return;
            }

            // low edge
            const auto low = [&]() -> exp_sum<REAL> {
                if(this->offset_low == this->terminal_0_offset)
                    return {0.0, -std::numeric_limits<REAL>::infinity()};
                if(this->offset_low == this->terminal_1_offset)
                    return {std::exp(REAL(0.0)), REAL(0.0)};
                else
                    return this->address(this->offset_low)->get_exp_sum();//{this->address(this->offset_low)->m, this->address(this->offset_low)->current_max};
            }();
            // high edge
            const auto high = [&]() -> exp_sum<REAL> {
                if(this->offset_high == this->terminal_0_offset)
                    return {REAL(0.0), -std::numeric_limits<REAL>::infinity()};
                if(this->offset_high == this->terminal_1_offset)
                    return {std::exp(REAL(0.0)), REAL(0.0)};
                else
                    return this->address(this->offset_high)->get_exp_sum();
            }();

            std::tuple<REAL,REAL> t;
            t = (low * exp_sum<REAL>(-this->low_cost)) + (high * exp_sum<REAL>(-this->high_cost));
            std::tie(this->m,current_max) = t;

            assert(std::isfinite(this->m));
            assert(std::isfinite(current_max));
        }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED, template<class,class,class> class BASE>
        void bdd_branch_instruction_smooth_base<REAL,OFFSET_TYPE,DERIVED,BASE>::smooth_forward_step()
        {
            check_instruction();

            // version without numerical stabilization
            if(!with_sum_exp_normalization)
            {
                assert(this->current_max == 0);
                if(this->offset_low != this->terminal_0_offset && this->offset_low != this->terminal_1_offset)
                {
                    auto* low_node = this->address(this->offset_low);
                    low_node->m += this->m * std::exp(-this->low_cost);
                    assert(!std::isnan(low_node->m));
                }

                if(this->offset_high != this->terminal_0_offset && this->offset_high != this->terminal_1_offset)
                {
                    auto* high_node = this->address(this->offset_high);
                    high_node->m += this->m * std::exp(-this->high_cost);
                    assert(!std::isnan(high_node->m));
                }

                return;
            }

            if(this->offset_low != this->terminal_0_offset && this->offset_low != this->terminal_1_offset)
            {
                auto* low = this->address(this->offset_low);
                low->check_instruction();
                low->assign(this->get_exp_sum() * exp_sum<REAL>(- this->low_cost) + low->get_exp_sum());
                
                /*
                if(low->m == 0.0)
                {
                    assert(low->current_max == -std::numeric_limits<REAL>::infinity());
                    low->m = this->m;
                    low->current_max = current_max - this->low_cost;
                }
                else if(low->current_max > current_max - this->low_cost)
                {
                    low->m += std::exp(current_max - low->current_max - this->low_cost) * this->m; 
                }
                else
                {
                    low->m *= std::exp(low->current_max - (current_max - this->low_cost));
                    low->m += this->m;
                    low->current_max = current_max - this->low_cost;
                }
                */
                low->check_instruction();
            }

            if(this->offset_high != this->terminal_0_offset && this->offset_high != this->terminal_1_offset)
            {
                auto* high = this->address(this->offset_high);
                high->check_instruction();
                high->assign(this->get_exp_sum() * exp_sum<REAL>(- this->high_cost) + high->get_exp_sum());

                /*
                if(high->m == 0.0)
                {
                    assert(high->current_max == -std::numeric_limits<REAL>::infinity());
                    high->m = this->m;
                    high->current_max = current_max - this->high_cost;
                }
                else if(high->current_max > current_max - this->high_cost)
                {
                    high->m += std::exp(current_max - high->current_max - this->high_cost) * this->m;
                }
                else
                {
                    high->m *= std::exp(high->current_max - (current_max - this->high_cost));
                    high->m += this->m;
                    high->current_max = current_max - this->high_cost;
                }
                */
                high->check_instruction();
            }
        }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED, template<class,class,class> class BASE>
        void bdd_branch_instruction_smooth_base<REAL,OFFSET_TYPE,DERIVED,BASE>::prepare_smooth_forward_step()
        {
            check_instruction();
            assert(this->offset_low > 0 && this->offset_high > 0);

            if(!with_sum_exp_normalization)
            {
                if(this->offset_low != this->terminal_0_offset && this->offset_low != this->terminal_1_offset)
                {
                    auto* low = this->address(this->offset_low);
                    low->m = 0.0;
                    low->current_max = 0.0;
                }

                if(this->offset_high != this->terminal_0_offset && this->offset_high != this->terminal_1_offset)
                {
                    auto* high = this->address(this->offset_high);
                    high->m = 0.0;
                    high->current_max = 0.0;
                }

                return;
            }

            if(this->offset_low != this->terminal_0_offset && this->offset_low != this->terminal_1_offset)
            {
                auto* low = this->address(this->offset_low);
                low->m = 0.0;
                low->current_max = -std::numeric_limits<double>::infinity();
            }

            if(this->offset_high != this->terminal_0_offset && this->offset_high != this->terminal_1_offset)
            {
                auto* high = this->address(this->offset_high);
                high->m = 0.0;
                high->current_max = -std::numeric_limits<double>::infinity();
            }
        }


    template<typename REAL, typename OFFSET_TYPE, typename DERIVED, template<class,class,class> class BASE>
        std::array<exp_sum<REAL>,2> bdd_branch_instruction_smooth_base<REAL,OFFSET_TYPE,DERIVED,BASE>::sum_marginals() const
        {
            // without stabilization
            if(!with_sum_exp_normalization)
            {
                assert(this->current_max == 0.0);

                const REAL sum_0 = [&]() -> REAL {
                    if(this->offset_low == this->terminal_0_offset)
                        return 0.0;
                    else if(this->offset_low == this->terminal_1_offset)
                        return this->m * std::exp(-this->low_cost);
                    else
                        return this->m * std::exp(-this->low_cost) * this->address(this->offset_low)->m;
                }();
                assert(!std::isnan(sum_0));

                const REAL sum_1 = [&]() -> REAL {
                    if(this->offset_high == this->terminal_0_offset)
                        return 0.0;
                    else if(this->offset_high == this->terminal_1_offset)
                        return this->m * std::exp(-this->high_cost);
                    else
                        return this->m * std::exp(-this->high_cost) * this->address(this->offset_high)->m;
                }();
                assert(!std::isnan(sum_1));

                return {exp_sum<REAL>(sum_0, 0.0), exp_sum<REAL>(sum_1,0.0)};
            }

            check_instruction();

            const exp_sum<REAL> low_sum = [&]() -> exp_sum<REAL> {
                if(this->offset_low == this->terminal_0_offset)
                    return {0.0, -std::numeric_limits<REAL>::infinity()};
                else if(this->offset_low == this->terminal_1_offset)
                    return {this->m, current_max - this->low_cost};
                else
                {
                    auto* low = this->address(this->offset_low);
                    return (this->get_exp_sum() * low->get_exp_sum()) * exp_sum<REAL>(-this->low_cost);
                }
            }();

            const exp_sum<REAL> high_sum = [&]() -> exp_sum<REAL> {
                if(this->offset_high == this->terminal_0_offset)
                    return {0.0, -std::numeric_limits<REAL>::infinity()};
                else if(this->offset_high == this->terminal_1_offset)
                    return {this->m, current_max - this->high_cost};
                else
                {
                    auto* high = this->address(this->offset_high);
                    return (this->get_exp_sum() * high->get_exp_sum()) * exp_sum<REAL>(-this->high_cost);
                }
            }();

            return {low_sum, high_sum};
        }
    template<typename REAL, typename OFFSET_TYPE, typename DERIVED, template<class,class,class> class BASE>
            bdd_branch_instruction_smooth_base<REAL,OFFSET_TYPE,DERIVED,BASE>& bdd_branch_instruction_smooth_base<REAL,OFFSET_TYPE,DERIVED,BASE>::assign(const exp_sum<REAL> es)
            {
                this->m = es.sum;
                this->current_max = es.max;
                return *this;
            }

    template<typename REAL, typename OFFSET_TYPE, typename DERIVED, template<class,class,class> class BASE>
        void bdd_branch_instruction_smooth_base<REAL,OFFSET_TYPE,DERIVED,BASE>::check_instruction() const
        {
            assert(!std::isnan(current_max));
            assert(std::isfinite(this->m));
            assert(!std::isnan(this->low_cost));
            assert(!std::isnan(this->high_cost));
        }

    template<typename REAL, typename OFFSET_TYPE>
        void bdd_branch_instruction_smooth_bdd_index<REAL,OFFSET_TYPE>::update_sum_marginals(std::array<exp_sum<REAL>,2>* reduced_sum_marginals) const
        {
            const auto sm = this->sum_marginals();
            if(!this->with_sum_exp_normalization)
            {
                reduced_sum_marginals[this->bdd_index][0].max = 0.0;
                reduced_sum_marginals[this->bdd_index][1].max = 0.0;
                assert(sm[0].max == 0.0);
                assert(sm[1].max == 0.0);
                reduced_sum_marginals[this->bdd_index][0].sum += sm[0].sum;
                reduced_sum_marginals[this->bdd_index][1].sum += sm[1].sum;
                return;
            }

            reduced_sum_marginals[this->bdd_index][0].update(sm[0]);
            reduced_sum_marginals[this->bdd_index][1].update(sm[1]);
        }

    template<typename REAL, typename OFFSET_TYPE>
        void bdd_branch_instruction_smooth_bdd_index<REAL,OFFSET_TYPE>::set_sum_marginals(std::array<exp_sum<REAL>,2>* sum_marginals, const std::array<exp_sum<REAL>,2>& avg_marginals)
        {
            this->check_instruction();
            assert(false);

            this->low_cost -= sum_marginals[this->bdd_index][0].sum + sum_marginals[this->bdd_index][0].max; 
            this->low_cost += avg_marginals[0].sum + avg_marginals[0].max;

            this->high_cost -= sum_marginals[this->bdd_index][1].sum + sum_marginals[this->bdd_index][1].max; 
            this->high_cost += avg_marginals[1].sum + avg_marginals[1].max;

            this->check_instruction();
        }

}
