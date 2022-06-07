#include "bdd_cuda.h"
#ifdef WITH_CUDA
#include "bdd_cuda_parallel_mma.h"
#include "incremental_mm_agreement_rounding_cuda.h"
#endif
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
    class bdd_cuda<REAL>::impl {
        public:
            impl(BDD::bdd_collection& bdd_col);

#ifdef WITH_CUDA
            bdd_cuda_parallel_mma<REAL> pmma;
#endif
    };

    template<typename REAL>
    bdd_cuda<REAL>::impl::impl(BDD::bdd_collection& bdd_col)
#ifdef WITH_CUDA
    : pmma(bdd_col)
#endif
    {
#ifndef WITH_CUDA
        throw std::runtime_error("bdd solver not compiled with CUDA support");
#endif
    }

    template<typename REAL>
    bdd_cuda<REAL>::bdd_cuda(BDD::bdd_collection& bdd_col)
    {
#ifdef WITH_CUDA
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col);
#else
        throw std::runtime_error("bdd_solver not compiled with CUDA support");
#endif
    }

    template<typename REAL>
    bdd_cuda<REAL>::bdd_cuda(bdd_cuda&& o)
        : pimpl(std::move(o.pimpl))
    {}

    template<typename REAL>
    bdd_cuda<REAL>& bdd_cuda<REAL>::operator=(bdd_cuda<REAL>&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    template<typename REAL>
    bdd_cuda<REAL>::~bdd_cuda()
    {}

    template<typename REAL>
    template<typename COST_ITERATOR>
    void bdd_cuda<REAL>::update_costs(COST_ITERATOR costs_lo_begin, COST_ITERATOR costs_lo_end, COST_ITERATOR costs_hi_begin, COST_ITERATOR costs_hi_end)
    {
#ifdef WITH_CUDA
        pimpl->pmma.update_costs(costs_lo_begin, costs_lo_end, costs_hi_begin, costs_hi_end);
#endif
    }

    // Need to have explicit instantiation in the base.
    template void bdd_cuda<float>::update_costs(double*, double*, double*, double*);
    template void bdd_cuda<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_cuda<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_cuda<float>::update_costs(float*, float*, float*, float*);
    template void bdd_cuda<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_cuda<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template void bdd_cuda<double>::update_costs(double*, double*, double*, double*);
    template void bdd_cuda<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_cuda<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_cuda<double>::update_costs(float*, float*, float*, float*);
    template void bdd_cuda<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_cuda<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template<typename REAL>
    void bdd_cuda<REAL>::backward_run()
    {
#ifdef WITH_CUDA
        pimpl->pmma.backward_run();
#endif
    }

    template<typename REAL>
    void bdd_cuda<REAL>::iteration()
    {
#ifdef WITH_CUDA
        pimpl->pmma.iteration();
#endif
    }

    template<typename REAL>
    double bdd_cuda<REAL>::lower_bound()
    {
#ifdef WITH_CUDA
        return pimpl->pmma.lower_bound();
#endif
        return -std::numeric_limits<double>::infinity();
    } 

    template<typename REAL>
    two_dim_variable_array<std::array<double,2>> bdd_cuda<REAL>::min_marginals()
    {
#ifdef WITH_CUDA
        return pimpl->pmma.min_marginals();
#endif
        return {};
    }

    template<typename REAL>
    std::vector<char> bdd_cuda<REAL>::incremental_mm_agreement_rounding(const double init_delta, const double delta_growth_rate, const int num_itr_lb)
    {
#ifdef WITH_CUDA
        return incremental_mm_agreement_rounding_cuda(pimpl->pmma, init_delta, delta_growth_rate, num_itr_lb);
#endif
        return {};
    }

    template class bdd_cuda<float>;
    template class bdd_cuda<double>;

}
