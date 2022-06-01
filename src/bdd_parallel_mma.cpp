#include "bdd_parallel_mma.h"
#include "bdd_parallel_mma_base.h"
#include "bdd_branch_instruction.h"
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
    class bdd_parallel_mma<REAL>::impl {
        public:
            impl(BDD::bdd_collection& bdd_col)
                : base(bdd_col)
            {};

            bdd_parallel_mma_base<bdd_branch_instruction<REAL,uint16_t>> base;
    };

    template<typename REAL>
    bdd_parallel_mma<REAL>::bdd_parallel_mma(BDD::bdd_collection& bdd_col)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col);
    }

    template<typename REAL>
    bdd_parallel_mma<REAL>::bdd_parallel_mma(bdd_parallel_mma<REAL>&& o)
        : pimpl(std::move(o.pimpl))
    {}

    template<typename REAL>
    bdd_parallel_mma<REAL>& bdd_parallel_mma<REAL>::operator=(bdd_parallel_mma<REAL>&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    template<typename REAL>
    bdd_parallel_mma<REAL>::~bdd_parallel_mma()
    {}

    template<typename REAL>
    template<typename ITERATOR>
    void bdd_parallel_mma<REAL>::update_costs(ITERATOR cost_lo_begin, ITERATOR cost_lo_end, ITERATOR cost_hi_begin, ITERATOR cost_hi_end)
    {
        pimpl->base.update_costs(cost_lo_begin, cost_lo_end, cost_hi_begin, cost_hi_end);
    }

    template<typename REAL>
        void bdd_parallel_mma<REAL>::add_to_constant(const double c)
        {
            return pimpl->base.add_to_constant(c);
        }

    template<typename REAL>
        size_t bdd_parallel_mma<REAL>::nr_variables() const
        {
            return pimpl->base.nr_variables();
        }

    template<typename REAL>
        size_t bdd_parallel_mma<REAL>::nr_bdds(const size_t var) const
        {
            return pimpl->base.nr_bdds(var);
        }

    template<typename REAL>
    void bdd_parallel_mma<REAL>::backward_run()
    {
        pimpl->base.backward_run();
    }

    template<typename REAL>
    void bdd_parallel_mma<REAL>::iteration()
    {
        pimpl->base.parallel_mma();
    }

    template<typename REAL>
    void bdd_parallel_mma<REAL>::distribute_delta()
    {
        pimpl->base.distribute_delta();
    }

    template<typename REAL>
    double bdd_parallel_mma<REAL>::lower_bound()
    {
        return pimpl->base.lower_bound();
    }

    template<typename REAL>
    two_dim_variable_array<std::array<double,2>> bdd_parallel_mma<REAL>::min_marginals()
    {
        return pimpl->base.min_marginals();
    }

    template<typename REAL>
    void bdd_parallel_mma<REAL>::fix_variable(const size_t var, const bool value)
    {
        size_t v = var;
        if(value == false)
            fix_variables(&v, (&v)+1, &v, &v);
        else
            fix_variables(&v, &v, &v, (&v)+1);
    }

    template<typename REAL>
        template<typename ITERATOR>
    void bdd_parallel_mma<REAL>::fix_variables(ITERATOR zero_fixations_begin, ITERATOR zero_fixations_end, ITERATOR one_fixations_begin, ITERATOR one_fixations_end)
    {
        pimpl->base.fix_variables(zero_fixations_begin, zero_fixations_end, one_fixations_begin, one_fixations_end);
    }

    template<typename REAL>
    void bdd_parallel_mma<REAL>::tighten()
    {
        throw std::runtime_error("not implemented");
    }

    // explicitly instantiate templates
    template class bdd_parallel_mma<float>;
    template class bdd_parallel_mma<double>;

    template void bdd_parallel_mma<float>::update_costs(float*, float*, float*, float*);
    template void bdd_parallel_mma<float>::update_costs(double*, double*, double*, double*);
    template void bdd_parallel_mma<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_parallel_mma<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_parallel_mma<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_parallel_mma<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_parallel_mma<float>::fix_variables(size_t*, size_t*, size_t*, size_t*);
    template void bdd_parallel_mma<float>::fix_variables(std::vector<size_t>::iterator,std::vector<size_t>::iterator,std::vector<size_t>::iterator,   std::vector<size_t>::iterator);
    template void bdd_parallel_mma<float>::fix_variables(std::vector<size_t>::const_iterator,std::vector<size_t>::const_iterator,std::vector<size_t>::const_iterator,std::vector<size_t>::const_iterator);

    template void bdd_parallel_mma<double>::update_costs(float*, float*, float*, float*);
    template void bdd_parallel_mma<double>::update_costs(double*, double*, double*, double*);
    template void bdd_parallel_mma<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_parallel_mma<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_parallel_mma<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_parallel_mma<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_parallel_mma<double>::fix_variables(size_t*, size_t*, size_t*, size_t*);
    template void bdd_parallel_mma<double>::fix_variables(std::vector<size_t>::iterator,std::vector<size_t>::iterator,std::vector<size_t>::iterator,   std::vector<size_t>::iterator);
    template void bdd_parallel_mma<double>::fix_variables(std::vector<size_t>::const_iterator,std::vector<size_t>::const_iterator,std::vector<size_t>::const_iterator,std::vector<size_t>::const_iterator);
}
