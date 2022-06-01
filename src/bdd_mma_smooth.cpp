#include "bdd_mma_smooth.h"
#include "bdd_branch_instruction_smooth.h"
#include "bdd_mma_base_smooth.h"
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
        class bdd_mma_smooth<REAL>::impl {
            public:
                impl(BDD::bdd_collection& bdd_col)
                    : mma(bdd_col)
                {};

                bdd_mma_base_smooth<bdd_branch_instruction_smooth_bdd_index<REAL,uint32_t> > mma;
        };

    template<typename REAL>
        bdd_mma_smooth<REAL>::bdd_mma_smooth(BDD::bdd_collection& bdd_col)
        {
            MEASURE_FUNCTION_EXECUTION_TIME; 
            pimpl = std::make_unique<impl>(bdd_col);
        }

    template<typename REAL>
        bdd_mma_smooth<REAL>::bdd_mma_smooth(bdd_mma_smooth&& o)
        : pimpl(std::move(o.pimpl))
        {}

    template<typename REAL>
        bdd_mma_smooth<REAL>& bdd_mma_smooth<REAL>::operator=(bdd_mma_smooth<REAL>&& o)
        { 
            pimpl = std::move(o.pimpl);
            return *this;
        }

    template<typename REAL>
        bdd_mma_smooth<REAL>::~bdd_mma_smooth()
        {}

    template<typename REAL>
        void bdd_mma_smooth<REAL>::update_cost(const double lo_cost, const double hi_cost, const size_t var)
        {
            pimpl->mma.update_cost(lo_cost, hi_cost, var);
        }

    template<typename REAL>
        void bdd_mma_smooth<REAL>::add_to_constant(const double c)
        {
            throw std::runtime_error("not implemented yet");
            //pimpl->mma.add_to_constant(c);
        }

    template<typename REAL>
        void bdd_mma_smooth<REAL>::backward_run()
        {
            pimpl->mma.smooth_backward_run();
        }

    template<typename REAL>
        void bdd_mma_smooth<REAL>::iteration()
        {
            pimpl->mma.smooth_iteration();
        }

    template<typename REAL>
        double bdd_mma_smooth<REAL>::lower_bound()
        {
            return pimpl->mma.smooth_lower_bound();
        }

    template<typename REAL>
        two_dim_variable_array<std::array<double,2>> bdd_mma_smooth<REAL>::min_marginals()
        {
            return pimpl->mma.min_marginals();
        }

    template<typename REAL>
        void bdd_mma_smooth<REAL>::fix_variable(const size_t var, const bool value)
        {
            pimpl->mma.fix_variable(var, value);
        }

    template<typename REAL>
        void bdd_mma_smooth<REAL>::set_smoothing(const double smoothing)
        {
            pimpl->mma.set_smoothing(smoothing);
        }

    // explicitly instantiate templates
    template class bdd_mma_smooth<float>;
    template class bdd_mma_smooth<double>;
}
