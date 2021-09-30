#include "bdd_parallel_mma.h"
#include "bdd_sequential_base.h"
#include "bdd_branch_node_vector.h"
#include "time_measure_util.h"

namespace LPMP {

    class bdd_parallel_mma::impl {
        public:
            impl(BDD::bdd_collection& bdd_col)
                : base(bdd_col)
            {};

            bdd_sequential_base<bdd_branch_instruction<float>> base;
    };

    bdd_parallel_mma::bdd_parallel_mma(BDD::bdd_collection& bdd_col)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(bdd_col);
    }

    bdd_parallel_mma::bdd_parallel_mma(bdd_parallel_mma&& o)
        : pimpl(std::move(o.pimpl))
    {}

    bdd_parallel_mma& bdd_parallel_mma::operator=(bdd_parallel_mma&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    bdd_parallel_mma::~bdd_parallel_mma()
    {}

    template<typename ITERATOR>
    void bdd_parallel_mma::set_costs(ITERATOR cost_begin, ITERATOR cost_end)
    {
        pimpl->base.set_costs(cost_begin, cost_end);
    }
    // explicitly instantiate templates
    template void bdd_parallel_mma::set_costs(float*, float*);
    template void bdd_parallel_mma::set_costs(double*, double*);
    template void bdd_parallel_mma::set_costs(std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_parallel_mma::set_costs(std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_parallel_mma::set_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator);
    template void bdd_parallel_mma::set_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator);


    void bdd_parallel_mma::backward_run()
    {
        pimpl->base.backward_run();
    }

    void bdd_parallel_mma::iteration()
    {
        pimpl->base.parallel_mma();
    }

    double bdd_parallel_mma::lower_bound()
    {
        return pimpl->base.lower_bound();
    }

    two_dim_variable_array<std::array<double,2>> bdd_parallel_mma::min_marginals()
    {
        return pimpl->base.min_marginals();
    }

    void bdd_parallel_mma::fix_variable(const size_t var, const bool value)
    {
        throw std::runtime_error("not implemented");
    }

    void bdd_parallel_mma::tighten()
    {
        throw std::runtime_error("not implemented");
    }
}
