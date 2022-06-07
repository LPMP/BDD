#include "bdd_mma.h"
#include "bdd_mma_base.h"
#include "bdd_branch_instruction.h"
#include "bdd_tightening.h"
#include "time_measure_util.h"
#include "two_dimensional_variable_array.hxx"

namespace LPMP {

    template<typename REAL>
    class bdd_mma<REAL>::impl {
        public:
            impl(BDD::bdd_collection& bdd_col)
                : mma(bdd_col)
            {};

            bdd_mma_base<bdd_branch_instruction_bdd_index<REAL,uint32_t>> mma;
    };

    template<typename REAL>
    bdd_mma<REAL>::bdd_mma(BDD::bdd_collection& bdd_col)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col);
    }

    template<typename REAL>
    bdd_mma<REAL>::bdd_mma(bdd_mma&& o)
        : pimpl(std::move(o.pimpl))
    {}

    template<typename REAL>
    bdd_mma<REAL>& bdd_mma<REAL>::operator=(bdd_mma<REAL>&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    template<typename REAL>
    bdd_mma<REAL>::~bdd_mma()
    {}

    template<typename REAL>
        size_t bdd_mma<REAL>::nr_variables() const
        {
            return pimpl->mma.nr_variables();
        }

    template<typename REAL>
        size_t bdd_mma<REAL>::nr_bdds(const size_t var) const
        {
            return pimpl->mma.nr_bdds(var);
        }

    template<typename REAL>
    void bdd_mma<REAL>::update_costs(const two_dim_variable_array<std::array<double,2>>& delta)
    {
        pimpl->mma.update_costs(delta);
    }

    template<typename REAL>
    void bdd_mma<REAL>::update_cost(const double lo_cost, const double hi_cost, const size_t var)
    {
        pimpl->mma.update_cost(lo_cost, hi_cost, var);
    }

    template<typename REAL>
    void bdd_mma<REAL>::add_to_constant(const double c)
    {
        pimpl->mma.add_to_constant(c);
    }

    template<typename REAL>
    void bdd_mma<REAL>::backward_run()
    {
        pimpl->mma.backward_run();
    }

    template<typename REAL>
    void bdd_mma<REAL>::iteration()
    {
        pimpl->mma.iteration();
    }

    template<typename REAL>
    double bdd_mma<REAL>::lower_bound()
    {
        return pimpl->mma.lower_bound();
    }

    template<typename REAL>
    two_dim_variable_array<std::array<double,2>> bdd_mma<REAL>::min_marginals()
    {
        return pimpl->mma.min_marginals();
    }

    template<typename REAL>
        template<typename ITERATOR>
        two_dim_variable_array<char> bdd_mma<REAL>::bdd_feasibility(ITERATOR sol_begin, ITERATOR sol_end)
        {
            return pimpl->mma.bdd_feasibility(sol_begin, sol_end);
        }

    template<typename REAL>
    void bdd_mma<REAL>::fix_variable(const size_t var, const bool value)
    {
        pimpl->mma.fix_variable(var, value);
    }

    template<typename REAL>
    void bdd_mma<REAL>::tighten()
    {
        return LPMP::tighten(pimpl->mma, 0.1); 
    }

    // explicitly instantiate templates
    template class bdd_mma<float>;
    template class bdd_mma<double>;

    template two_dim_variable_array<char> bdd_mma<double>::bdd_feasibility(char*, char*);
    template two_dim_variable_array<char> bdd_mma<double>::bdd_feasibility(std::vector<char>::iterator, std::vector<char>::iterator);
    template two_dim_variable_array<char> bdd_mma<double>::bdd_feasibility(std::vector<char>::const_iterator, std::vector<char>::const_iterator);

    template two_dim_variable_array<char> bdd_mma<float>::bdd_feasibility(char*, char*);
    template two_dim_variable_array<char> bdd_mma<float>::bdd_feasibility(std::vector<char>::iterator, std::vector<char>::iterator);
    template two_dim_variable_array<char> bdd_mma<float>::bdd_feasibility(std::vector<char>::const_iterator, std::vector<char>::const_iterator);
}
