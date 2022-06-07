#include "bdd_parallel_mma_smooth.h"
#include "bdd_branch_instruction_smooth.h"
#include "bdd_parallel_mma_base_smooth.h"
#include "time_measure_util.h"

namespace LPMP {

    template<typename REAL>
        class bdd_parallel_mma_smooth<REAL>::impl {
            public:
                impl(BDD::bdd_collection& bdd_col)
                    : parallel_mma(bdd_col)
                {};

                bdd_parallel_mma_base_smooth<bdd_branch_instruction_smooth<REAL,uint32_t> > parallel_mma;
        };

    template<typename REAL>
        bdd_parallel_mma_smooth<REAL>::bdd_parallel_mma_smooth(BDD::bdd_collection& bdd_col)
        {
            MEASURE_FUNCTION_EXECUTION_TIME; 
            pimpl = std::make_unique<impl>(bdd_col);
        }

    template<typename REAL>
        bdd_parallel_mma_smooth<REAL>::bdd_parallel_mma_smooth(bdd_parallel_mma_smooth&& o)
        : pimpl(std::move(o.pimpl))
        {}

    template<typename REAL>
        bdd_parallel_mma_smooth<REAL>& bdd_parallel_mma_smooth<REAL>::operator=(bdd_parallel_mma_smooth<REAL>&& o)
        { 
            pimpl = std::move(o.pimpl);
            return *this;
        }

    template<typename REAL>
        bdd_parallel_mma_smooth<REAL>::~bdd_parallel_mma_smooth()
        {
            if(pimpl != nullptr)
            {
                //pimpl->parallel_mma.distribute_delta();
                //pimpl->parallel_mma.backward_run();
                const double lb = pimpl->parallel_mma.lower_bound();
                std::cout << "[parallel mma smooth] final original lower bound = " << lb << "\n";
            }
        }

    template<typename REAL>
        template<typename ITERATOR>
        void bdd_parallel_mma_smooth<REAL>::update_costs(ITERATOR cost_lo_begin, ITERATOR cost_lo_end, ITERATOR cost_hi_begin, ITERATOR cost_hi_end)
        {
            pimpl->parallel_mma.update_costs(cost_lo_begin, cost_lo_end, cost_hi_begin, cost_hi_end);
        }

    template<typename REAL>
        void bdd_parallel_mma_smooth<REAL>::add_to_constant(const double c)
        {
            pimpl->parallel_mma.add_to_constant(c);
        }

    template<typename REAL>
        void bdd_parallel_mma_smooth<REAL>::backward_run()
        {
            pimpl->parallel_mma.smooth_backward_run();
        }

    template<typename REAL>
        void bdd_parallel_mma_smooth<REAL>::iteration()
        {
            pimpl->parallel_mma.parallel_sma();
        }

    template<typename REAL>
        double bdd_parallel_mma_smooth<REAL>::lower_bound()
        {
            return pimpl->parallel_mma.smooth_lower_bound();
        }

    template<typename REAL>
        two_dim_variable_array<std::array<double,2>> bdd_parallel_mma_smooth<REAL>::min_marginals()
        {
            return pimpl->parallel_mma.min_marginals();
        }

    template<typename REAL>
        void bdd_parallel_mma_smooth<REAL>::fix_variable(const size_t var, const bool value)
        {
            pimpl->parallel_mma.fix_variable(var, value);
        }

    template<typename REAL>
        void bdd_parallel_mma_smooth<REAL>::set_smoothing(const double smoothing)
        {
            pimpl->parallel_mma.set_smoothing(smoothing);
        }

    // explicitly instantiate templates
    template class bdd_parallel_mma_smooth<float>;
    template class bdd_parallel_mma_smooth<double>;

    template void bdd_parallel_mma_smooth<float>::update_costs(float*, float*, float*, float*);
    template void bdd_parallel_mma_smooth<float>::update_costs(double*, double*, double*, double*);
    template void bdd_parallel_mma_smooth<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_parallel_mma_smooth<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_parallel_mma_smooth<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_parallel_mma_smooth<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_parallel_mma_smooth<double>::update_costs(float*, float*, float*, float*);
    template void bdd_parallel_mma_smooth<double>::update_costs(double*, double*, double*, double*);
    template void bdd_parallel_mma_smooth<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_parallel_mma_smooth<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_parallel_mma_smooth<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_parallel_mma_smooth<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);

}

