#include "bdd_mma_vec.h"
#include "bdd_branch_node_vector.h"
#include "bdd_branch_instruction.h"
#include "bdd_tightening.h"
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
    class bdd_mma_vec<REAL>::impl {
        public:
            impl(BDD::bdd_collection& bdd_col)
                : mma(bdd_col)
            {};

            bdd_mma_base<bdd_branch_instruction_bdd_index<REAL,uint32_t>> mma;
    };

    template<typename REAL>
    bdd_mma_vec<REAL>::bdd_mma_vec(BDD::bdd_collection& bdd_col)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col);
    }

    template<typename REAL>
    bdd_mma_vec<REAL>::bdd_mma_vec(bdd_mma_vec&& o)
        : pimpl(std::move(o.pimpl))
    {}

    template<typename REAL>
    bdd_mma_vec<REAL>& bdd_mma_vec<REAL>::operator=(bdd_mma_vec<REAL>&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    template<typename REAL>
    bdd_mma_vec<REAL>::~bdd_mma_vec()
    {}

    template<typename REAL>
    void bdd_mma_vec<REAL>::update_cost(const double c, const size_t var)
    {
        pimpl->mma.update_cost(c, var);
    }

    template<typename REAL>
    void bdd_mma_vec<REAL>::backward_run()
    {
        pimpl->mma.backward_run();
    }

    template<typename REAL>
    void bdd_mma_vec<REAL>::iteration()
    {
        pimpl->mma.iteration();
    }

    template<typename REAL>
    double bdd_mma_vec<REAL>::lower_bound()
    {
        return pimpl->mma.lower_bound();
    }

    template<typename REAL>
    two_dim_variable_array<std::array<double,2>> bdd_mma_vec<REAL>::min_marginals()
    {
        return pimpl->mma.min_marginals();
    }

    template<typename REAL>
    void bdd_mma_vec<REAL>::fix_variable(const size_t var, const bool value)
    {
        pimpl->mma.fix_variable(var, value);
    }

    template<typename REAL>
    void bdd_mma_vec<REAL>::tighten()
    {
        return LPMP::tighten(pimpl->mma, 0.1); 
    }

    // explicitly instantiate templates
    template class bdd_mma_vec<float>;
    template class bdd_mma_vec<double>;
}
