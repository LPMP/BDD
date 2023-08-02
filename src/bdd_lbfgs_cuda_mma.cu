#include "bdd_lbfgs_cuda_mma.h"
#ifdef WITH_CUDA
#include "lbfgs.h"
#include "bdd_cuda_parallel_mma.h"
#include "incremental_mm_agreement_rounding_cuda.h"
#endif
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
    class bdd_lbfgs_cuda_mma<REAL>::impl {
        public:
            impl(BDD::bdd_collection& bdd_col, const int _history_size,
                const double _init_step_size, const double _req_rel_lb_increase,
                const double _step_size_decrease_factor, const double _step_size_increase_factor);
#ifdef WITH_CUDA
            lbfgs<bdd_cuda_parallel_mma<REAL>, thrust::device_vector<REAL>, REAL> mma;
#endif
    };

    template<typename REAL>
    bdd_lbfgs_cuda_mma<REAL>::impl::impl(BDD::bdd_collection& bdd_col, const int _history_size,
                    const double _init_step_size, const double _req_rel_lb_increase,
                    const double _step_size_decrease_factor, const double _step_size_increase_factor)
#ifdef WITH_CUDA
    : mma(bdd_col, _history_size,
        _init_step_size, _req_rel_lb_increase,
        _step_size_decrease_factor, _step_size_increase_factor)
#endif
    {
#ifndef WITH_CUDA
        throw std::runtime_error("bdd solver not compiled with CUDA support");
#endif
    }

    template<typename REAL>
    bdd_lbfgs_cuda_mma<REAL>::bdd_lbfgs_cuda_mma(BDD::bdd_collection& bdd_col, const int _history_size,
                    const double _init_step_size, const double _req_rel_lb_increase,
                    const double _step_size_decrease_factor, const double _step_size_increase_factor)
    {
#ifdef WITH_CUDA
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col, _history_size,
                _init_step_size, _req_rel_lb_increase,
                _step_size_decrease_factor, _step_size_increase_factor);
#else
        throw std::runtime_error("bdd_solver not compiled with CUDA support");
#endif
    }

    template<typename REAL>
    bdd_lbfgs_cuda_mma<REAL>::bdd_lbfgs_cuda_mma(bdd_lbfgs_cuda_mma<REAL>&& o)
        : pimpl(std::move(o.pimpl))
    {}

    template<typename REAL>
    bdd_lbfgs_cuda_mma<REAL>& bdd_lbfgs_cuda_mma<REAL>::operator=(bdd_lbfgs_cuda_mma<REAL>&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    template<typename REAL>
    bdd_lbfgs_cuda_mma<REAL>::~bdd_lbfgs_cuda_mma()
    {}

    template<typename REAL>
    template<typename COST_ITERATOR>
    void bdd_lbfgs_cuda_mma<REAL>::update_costs(COST_ITERATOR costs_lo_begin, COST_ITERATOR costs_lo_end, COST_ITERATOR costs_hi_begin, COST_ITERATOR costs_hi_end)
    {
#ifdef WITH_CUDA
        pimpl->mma.update_costs(costs_lo_begin, costs_lo_end, costs_hi_begin, costs_hi_end);
#endif
    }

    // Need to have explicit instantiation in the base.
    template void bdd_lbfgs_cuda_mma<float>::update_costs(double*, double*, double*, double*);
    template void bdd_lbfgs_cuda_mma<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_lbfgs_cuda_mma<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_lbfgs_cuda_mma<float>::update_costs(float*, float*, float*, float*);
    template void bdd_lbfgs_cuda_mma<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_lbfgs_cuda_mma<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template void bdd_lbfgs_cuda_mma<double>::update_costs(double*, double*, double*, double*);
    template void bdd_lbfgs_cuda_mma<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_lbfgs_cuda_mma<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_lbfgs_cuda_mma<double>::update_costs(float*, float*, float*, float*);
    template void bdd_lbfgs_cuda_mma<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_lbfgs_cuda_mma<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template<typename REAL>
    void bdd_lbfgs_cuda_mma<REAL>::backward_run()
    {
#ifdef WITH_CUDA
        pimpl->mma.backward_run();
#endif
    }

    template<typename REAL>
    void bdd_lbfgs_cuda_mma<REAL>::iteration()
    {
#ifdef WITH_CUDA
        pimpl->mma.iteration();
#endif
    }

    template<typename REAL>
    double bdd_lbfgs_cuda_mma<REAL>::lower_bound()
    {
#ifdef WITH_CUDA
        return pimpl->mma.lower_bound();
#endif
        return -std::numeric_limits<double>::infinity();
    } 

    template<typename REAL>
    two_dim_variable_array<std::array<double,2>> bdd_lbfgs_cuda_mma<REAL>::min_marginals()
    {
#ifdef WITH_CUDA
        return pimpl->mma.min_marginals();
#endif
        return {};
    }

    template<typename REAL>
    std::vector<char> bdd_lbfgs_cuda_mma<REAL>::incremental_mm_agreement_rounding(const double init_delta, const double delta_growth_rate, const int num_itr_lb, const int num_rounds)
    {
#ifdef WITH_CUDA
        return incremental_mm_agreement_rounding_cuda(pimpl->mma, init_delta, delta_growth_rate, num_itr_lb, true, num_rounds);
#endif
        return {};
    }

    template class bdd_lbfgs_cuda_mma<float>;
    template class bdd_lbfgs_cuda_mma<double>;
}
