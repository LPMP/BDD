#include "bdd_mma_anisotropic.h"
#include "time_measure_util.h"
#include "bdd_branch_node.h"
#include "bdd_variable.h"
#include "bdd_mma_anisotropic_base.hxx"

namespace LPMP {

    class bdd_mma_anisotropic::impl {
        public:
            using bdd_mma_base_type = bdd_mma_base<bdd_opt_base_node_costs<bdd_variable_with_indices, bdd_branch_node_opt>>;
            using bdd_mma_anisotropic_base_type = bdd_mma_anisotropic_base<bdd_mma_base_type>;

            impl(bdd_storage& bdd_storage_)
                : mma(bdd_storage_)
            {};

            bdd_mma_anisotropic_base_type mma;
    };

    bdd_mma_anisotropic::bdd_mma_anisotropic(bdd_storage& stor)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(stor);
    }

    bdd_mma_anisotropic::bdd_mma_anisotropic(bdd_mma_anisotropic&& o)
        : pimpl(std::move(o.pimpl))
    {}

    bdd_mma_anisotropic& bdd_mma_anisotropic::operator=(bdd_mma_anisotropic&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    bdd_mma_anisotropic::~bdd_mma_anisotropic()
    {}

    void bdd_mma_anisotropic::set_cost(const double c, const size_t var)
    {
        pimpl->mma.set_cost(c, var);
    }

    void bdd_mma_anisotropic::backward_run()
    {
        pimpl->mma.backward_run();
        pimpl->mma.compute_lower_bound();
    }

    void bdd_mma_anisotropic::iteration()
    {
        pimpl->mma.iteration();
    }

    void bdd_mma_anisotropic::solve(const size_t max_iter, const double tolerance, const double time_limit)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        pimpl->mma.solve(max_iter, tolerance, time_limit);
    }

    double bdd_mma_anisotropic::lower_bound()
    {
        return pimpl->mma.lower_bound();
    }

    two_dim_variable_array<std::array<double,2>> bdd_mma_anisotropic::min_marginals()
    {
        return pimpl->mma.min_marginals();
    }

}
