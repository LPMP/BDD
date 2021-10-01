#include "bdd_parallel_mma.h"
#include "bdd_sequential_base.h"
#include "bdd_branch_node_vector.h"
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
    class bdd_parallel_mma<REAL>::impl {
        public:
            impl(BDD::bdd_collection& bdd_col)
                : base(bdd_col)
            {};

            bdd_sequential_base<bdd_branch_instruction<REAL>> base;
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
    void bdd_parallel_mma<REAL>::set_costs(ITERATOR cost_begin, ITERATOR cost_end)
    {
        pimpl->base.set_costs(cost_begin, cost_end);
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
        throw std::runtime_error("not implemented");
    }

    template<typename REAL>
    void bdd_parallel_mma<REAL>::tighten()
    {
        throw std::runtime_error("not implemented");
    }

    // explicitly instantiate templates
    template class bdd_parallel_mma<float>;
    template class bdd_parallel_mma<double>;

    template void bdd_parallel_mma<float>::set_costs(float*, float*);
    template void bdd_parallel_mma<float>::set_costs(double*, double*);
    template void bdd_parallel_mma<float>::set_costs(std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_parallel_mma<float>::set_costs(std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_parallel_mma<float>::set_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_parallel_mma<float>::set_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator);

    template void bdd_parallel_mma<double>::set_costs(float*, float*);
    template void bdd_parallel_mma<double>::set_costs(double*, double*);
    template void bdd_parallel_mma<double>::set_costs(std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_parallel_mma<double>::set_costs(std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_parallel_mma<double>::set_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_parallel_mma<double>::set_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator);
}
