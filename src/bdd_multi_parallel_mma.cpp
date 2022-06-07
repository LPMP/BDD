#include "bdd_multi_parallel_mma.h"
#include <limits>
#ifdef WITH_CUDA
#include "bdd_multi_parallel_mma_base.h"
#endif
#include "time_measure_util.h"
#include "two_dimensional_variable_array.hxx"

namespace LPMP {

    template<typename REAL>
        class bdd_multi_parallel_mma<REAL>::impl {
            public:
                impl(BDD::bdd_collection& bdd_col);

#ifdef WITH_CUDA
                bdd_multi_parallel_mma_base<REAL> mma;
#endif
        };

    template<typename REAL>
    bdd_multi_parallel_mma<REAL>::impl::impl(BDD::bdd_collection& bdd_col)
#ifdef WITH_CUDA
    : mma(bdd_col)
#endif
    {
#ifndef WITH_CUDA
        throw std::runtime_error("bdd solver not compiled with CUDA support");
#endif
    }

    template<typename REAL>
        bdd_multi_parallel_mma<REAL>::bdd_multi_parallel_mma(BDD::bdd_collection& bdd_col)
        {
#ifdef WITH_CUDA
            MEASURE_FUNCTION_EXECUTION_TIME; 
            pimpl = std::make_unique<impl>(bdd_col);
#else
            throw std::runtime_error("bdd_solver not compiled with CUDA support");
#endif
        }

    template<typename REAL>
        bdd_multi_parallel_mma<REAL>::bdd_multi_parallel_mma(bdd_multi_parallel_mma&& o)
        : pimpl(std::move(o.pimpl))
        {}

    template<typename REAL>
        bdd_multi_parallel_mma<REAL>& bdd_multi_parallel_mma<REAL>::operator=(bdd_multi_parallel_mma<REAL>&& o)
        {
            pimpl = std::move(o.pimpl);
            return *this;
        }

    template<typename REAL>
        bdd_multi_parallel_mma<REAL>::~bdd_multi_parallel_mma()
        {}

    template<typename REAL>
        size_t bdd_multi_parallel_mma<REAL>::nr_variables() const
        {
#ifdef WITH_CUDA
            return pimpl->mma.nr_variables();
#else
            return 0;
#endif
        }

    template<typename REAL>
        size_t bdd_multi_parallel_mma<REAL>::nr_bdds(const size_t var) const
        {
#ifdef WITH_CUDA
            return pimpl->mma.nr_bdds(var);
#else
            return 0;
#endif
        }

    template<typename REAL>
        void bdd_multi_parallel_mma<REAL>::update_costs(const two_dim_variable_array<std::array<double,2>>& delta)
        {
            throw std::runtime_error("not implemented yet");
            //pimpl->mma.update_costs(delta);
        }

    template<typename REAL>
        template<typename COST_ITERATOR> 
        void bdd_multi_parallel_mma<REAL>::update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end)
        {
#ifdef WITH_CUDA
            pimpl->mma.update_costs(cost_lo_begin, cost_lo_end, cost_hi_begin, cost_hi_end);
#endif
        }

    template<typename REAL>
        void bdd_multi_parallel_mma<REAL>::add_to_constant(const double c)
        {
#ifdef WITH_CUDA
            pimpl->mma.add_to_constant(c);
#endif
        }

    template<typename REAL>
        void bdd_multi_parallel_mma<REAL>::iteration()
        {
#ifdef WITH_CUDA
            pimpl->mma.parallel_mma();
#endif
        }

    template<typename REAL>
        double bdd_multi_parallel_mma<REAL>::lower_bound()
        {
#ifdef WITH_CUDA
            return pimpl->mma.lower_bound();
#endif
            return -std::numeric_limits<double>::infinity();
        }

    template<typename REAL>
        two_dim_variable_array<std::array<double,2>> bdd_multi_parallel_mma<REAL>::min_marginals()
        {
#ifdef WITH_CUDA
            return pimpl->mma.min_marginals();
#endif
            return {};
        }

    template<typename REAL>
        template<typename ITERATOR>
        two_dim_variable_array<char> bdd_multi_parallel_mma<REAL>::bdd_feasibility(ITERATOR sol_begin, ITERATOR sol_end)
        {
            throw std::runtime_error("not implemented yet");
            //return pimpl->mma.bdd_feasibility(sol_begin, sol_end);
        }

    template<typename REAL>
        void bdd_multi_parallel_mma<REAL>::fix_variable(const size_t var, const bool value)
        {
#ifdef WITH_CUDA
            pimpl->mma.fix_variable(var, value);
#endif
        }

    // explicitly instantiate templates
    template class bdd_multi_parallel_mma<float>;
    template class bdd_multi_parallel_mma<double>;

    template void bdd_multi_parallel_mma<float>::update_costs(float*, float*, float*, float*);
    template void bdd_multi_parallel_mma<float>::update_costs(double*, double*, double*, double*);
    template void bdd_multi_parallel_mma<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_multi_parallel_mma<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_multi_parallel_mma<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_multi_parallel_mma<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_multi_parallel_mma<double>::update_costs(float*, float*, float*, float*);
    template void bdd_multi_parallel_mma<double>::update_costs(double*, double*, double*, double*);
    template void bdd_multi_parallel_mma<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_multi_parallel_mma<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_multi_parallel_mma<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_multi_parallel_mma<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template two_dim_variable_array<char> bdd_multi_parallel_mma<double>::bdd_feasibility(char*, char*);
    template two_dim_variable_array<char> bdd_multi_parallel_mma<double>::bdd_feasibility(std::vector<char>::iterator, std::vector<char>::iterator);
    template two_dim_variable_array<char> bdd_multi_parallel_mma<double>::bdd_feasibility(std::vector<char>::const_iterator, std::vector<char>::const_iterator);

    template two_dim_variable_array<char> bdd_multi_parallel_mma<float>::bdd_feasibility(char*, char*);
    template two_dim_variable_array<char> bdd_multi_parallel_mma<float>::bdd_feasibility(std::vector<char>::iterator, std::vector<char>::iterator);
    template two_dim_variable_array<char> bdd_multi_parallel_mma<float>::bdd_feasibility(std::vector<char>::const_iterator, std::vector<char>::const_iterator);
}
