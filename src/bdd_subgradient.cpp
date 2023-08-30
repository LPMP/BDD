#include "bdd_subgradient.h"
#include "subgradient.h"
#include "bdd_parallel_mma_base.h"
#include "bdd_branch_instruction.h"
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
    class bdd_subgradient<REAL>::impl {
        public:
            impl(BDD::bdd_collection& bdd_col);

            subgradient<bdd_parallel_mma_base<bdd_branch_instruction<REAL, uint16_t>>, REAL> mma;
    };

    template<typename REAL>
    bdd_subgradient<REAL>::impl::impl(BDD::bdd_collection& bdd_col)
    : mma(bdd_col)
    {
    }

    template<typename REAL>
    bdd_subgradient<REAL>::bdd_subgradient(BDD::bdd_collection& bdd_col)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col);
    }

    template<typename REAL>
    bdd_subgradient<REAL>::bdd_subgradient(bdd_subgradient&& o)
        : pimpl(std::move(o.pimpl))
    {}

    template<typename REAL>
    bdd_subgradient<REAL>& bdd_subgradient<REAL>::operator=(bdd_subgradient<REAL>&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    template<typename REAL>
    bdd_subgradient<REAL>::~bdd_subgradient()
    {}

    template<typename REAL>
    template<typename COST_ITERATOR>
    void bdd_subgradient<REAL>::update_costs(COST_ITERATOR costs_lo_begin, COST_ITERATOR costs_lo_end, COST_ITERATOR costs_hi_begin, COST_ITERATOR costs_hi_end)
    {
        pimpl->mma.update_costs(costs_lo_begin, costs_lo_end, costs_hi_begin, costs_hi_end);
    }

    // Need to have explicit instantiation in the base.
    template void bdd_subgradient<float>::update_costs(double*, double*, double*, double*);
    template void bdd_subgradient<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_subgradient<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_subgradient<float>::update_costs(float*, float*, float*, float*);
    template void bdd_subgradient<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_subgradient<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template void bdd_subgradient<double>::update_costs(double*, double*, double*, double*);
    template void bdd_subgradient<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_subgradient<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_subgradient<double>::update_costs(float*, float*, float*, float*);
    template void bdd_subgradient<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_subgradient<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template<typename REAL>
    void bdd_subgradient<REAL>::backward_run()
    {
        pimpl->mma.backward_run();
    }

    template<typename REAL>
    void bdd_subgradient<REAL>::iteration()
    {
        pimpl->mma.iteration();
    }

    template<typename REAL>
    double bdd_subgradient<REAL>::lower_bound()
    {
        return pimpl->mma.lower_bound();
    } 

    template<typename REAL>
    size_t bdd_subgradient<REAL>::nr_variables()
    {
        return pimpl->mma.nr_variables();
    } 

    template<typename REAL>
    two_dim_variable_array<std::array<double,2>> bdd_subgradient<REAL>::min_marginals()
    {
        return pimpl->mma.min_marginals();
    }

    template class bdd_subgradient<float>;
    template class bdd_subgradient<double>;
}

