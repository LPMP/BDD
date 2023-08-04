#include "bdd_lbfgs_parallel_mma.h"
#include "lbfgs.h"
#include "bdd_parallel_mma_base.h"
#include "bdd_branch_instruction.h"
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
    class bdd_lbfgs_parallel_mma<REAL>::impl {
        public:
            impl(BDD::bdd_collection& bdd_col, const int _history_size,
                const double _init_step_size, const double _req_rel_lb_increase,
                const double _step_size_decrease_factor, const double _step_size_increase_factor);

            lbfgs<bdd_parallel_mma_base<bdd_branch_instruction<REAL, uint16_t>>, std::vector<REAL>, REAL> mma;
    };

    template<typename REAL>
    bdd_lbfgs_parallel_mma<REAL>::impl::impl(BDD::bdd_collection& bdd_col, const int _history_size,
                const double _init_step_size, const double _req_rel_lb_increase,
                const double _step_size_decrease_factor, const double _step_size_increase_factor)
    : mma(bdd_col, _history_size,
        _init_step_size, _req_rel_lb_increase,
        _step_size_decrease_factor, _step_size_increase_factor)
    {
    }

    template<typename REAL>
    bdd_lbfgs_parallel_mma<REAL>::bdd_lbfgs_parallel_mma(BDD::bdd_collection& bdd_col, const int _history_size,
                const double _init_step_size, const double _req_rel_lb_increase,
                const double _step_size_decrease_factor, const double _step_size_increase_factor)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col, _history_size,
            _init_step_size, _req_rel_lb_increase,
            _step_size_decrease_factor, _step_size_increase_factor);
    }

    template<typename REAL>
    bdd_lbfgs_parallel_mma<REAL>::bdd_lbfgs_parallel_mma(bdd_lbfgs_parallel_mma&& o)
        : pimpl(std::move(o.pimpl))
    {}

    template<typename REAL>
    bdd_lbfgs_parallel_mma<REAL>& bdd_lbfgs_parallel_mma<REAL>::operator=(bdd_lbfgs_parallel_mma<REAL>&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    template<typename REAL>
    bdd_lbfgs_parallel_mma<REAL>::~bdd_lbfgs_parallel_mma()
    {}

    template<typename REAL>
    template<typename COST_ITERATOR>
    void bdd_lbfgs_parallel_mma<REAL>::update_costs(COST_ITERATOR costs_lo_begin, COST_ITERATOR costs_lo_end, COST_ITERATOR costs_hi_begin, COST_ITERATOR costs_hi_end)
    {
        pimpl->mma.update_costs(costs_lo_begin, costs_lo_end, costs_hi_begin, costs_hi_end);
    }

    // Need to have explicit instantiation in the base.
    template void bdd_lbfgs_parallel_mma<float>::update_costs(double*, double*, double*, double*);
    template void bdd_lbfgs_parallel_mma<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_lbfgs_parallel_mma<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_lbfgs_parallel_mma<float>::update_costs(float*, float*, float*, float*);
    template void bdd_lbfgs_parallel_mma<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_lbfgs_parallel_mma<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template void bdd_lbfgs_parallel_mma<double>::update_costs(double*, double*, double*, double*);
    template void bdd_lbfgs_parallel_mma<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_lbfgs_parallel_mma<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_lbfgs_parallel_mma<double>::update_costs(float*, float*, float*, float*);
    template void bdd_lbfgs_parallel_mma<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_lbfgs_parallel_mma<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template<typename REAL>
    void bdd_lbfgs_parallel_mma<REAL>::backward_run()
    {
        pimpl->mma.backward_run();
    }

    template<typename REAL>
    void bdd_lbfgs_parallel_mma<REAL>::iteration()
    {
        pimpl->mma.iteration();
    }

    template<typename REAL>
    double bdd_lbfgs_parallel_mma<REAL>::lower_bound()
    {
        return pimpl->mma.lower_bound();
    } 

    template<typename REAL>
    size_t bdd_lbfgs_parallel_mma<REAL>::nr_variables()
    {
        return pimpl->mma.nr_variables();
    } 

    template<typename REAL>
    two_dim_variable_array<std::array<double,2>> bdd_lbfgs_parallel_mma<REAL>::min_marginals()
    {
        return pimpl->mma.min_marginals();
    }

    template class bdd_lbfgs_parallel_mma<float>;
    template class bdd_lbfgs_parallel_mma<double>;
}
