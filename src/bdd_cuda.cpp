#include "bdd_cuda.h"
#ifdef WITH_CUDA
#include "bdd_cuda_parallel_mma_sorting.h"
#endif
#include "time_measure_util.h"

namespace LPMP {

    class bdd_cuda::impl {
        public:
            impl(BDD::bdd_collection& bdd_col);

#ifdef WITH_CUDA
            bdd_cuda_parallel_mma_sorting pmma;
#endif
    };

    bdd_cuda::impl::impl(BDD::bdd_collection& bdd_col)
#ifdef WITH_CUDA
    : pmma(bdd_col)
#endif
    {
#ifndef WITH_CUDA
        throw std::runtime_error("bdd_solver not compiled with CUDA support");
#endif
    }

    bdd_cuda::bdd_cuda(BDD::bdd_collection& bdd_col)
    {
#ifdef WITH_CUDA
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col);
#else
        throw std::runtime_error("bdd_solver not compiled with CUDA support");
#endif
    }

    bdd_cuda::bdd_cuda(bdd_cuda&& o)
        : pimpl(std::move(o.pimpl))
    {}

    bdd_cuda& bdd_cuda::operator=(bdd_cuda&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    bdd_cuda::~bdd_cuda()
    {}

    void bdd_cuda::set_cost(const double c, const size_t var)
    {
#ifdef WITH_CUDA
        pimpl->pmma.set_cost(c, var);
#endif
    }

    void bdd_cuda::backward_run()
    {
#ifdef WITH_CUDA
        pimpl->pmma.backward_run();
#endif
    }

    void bdd_cuda::solve(const size_t max_iter, const double tolerance, const double time_limit)
    {
#ifdef WITH_CUDA
        MEASURE_FUNCTION_EXECUTION_TIME;
        pimpl->pmma.solve(max_iter, tolerance, time_limit);
#endif
    }

    double bdd_cuda::lower_bound()
    {
#ifdef WITH_CUDA
        return pimpl->pmma.lower_bound();
#endif
        return -std::numeric_limits<double>::infinity();
    } 

    two_dim_variable_array<std::array<double,2>> bdd_cuda::min_marginals()
    {
        throw std::runtime_error("not implemented");
#ifdef WITH_CUDA
        //return pimpl->pmma.min_marginals();
#endif
    }

}
