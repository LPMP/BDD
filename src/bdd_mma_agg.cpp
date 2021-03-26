#include "bdd_mma_agg.h"
#include "bdd_mma_agg_base.hxx"
#include "bdd_branch_node.h"
#include "bdd_variable.h"
#include "time_measure_util.h"

namespace LPMP {

    class bdd_mma_agg::impl {
        public:
            using bdd_mma_agg_base_type = bdd_mma_agg_base<bdd_opt_base_node_costs<bdd_variable_with_indices, bdd_branch_node_opt>>;

            impl(bdd_storage& bdd_storage_)
                : mma(bdd_storage_)
            {};

            bdd_mma_agg_base_type mma;
    };

    bdd_mma_agg::bdd_mma_agg(bdd_storage& stor)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(stor);
    }

    bdd_mma_agg::bdd_mma_agg(bdd_mma_agg&& o)
        : pimpl(std::move(o.pimpl))
    {}

    bdd_mma_agg& bdd_mma_agg::operator=(bdd_mma_agg&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    bdd_mma_agg::~bdd_mma_agg()
    {}

    void bdd_mma_agg::set_cost(const double c, const size_t var)
    {
        pimpl->mma.set_cost(c, var);
    }

    void bdd_mma_agg::backward_run()
    {
        pimpl->mma.backward_run();
    }

    void bdd_mma_agg::solve(const size_t max_iter, const double tolerance, const double time_limit)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        pimpl->mma.solve(max_iter, tolerance, time_limit);
    }

    double bdd_mma_agg::lower_bound()
    {
        return pimpl->mma.lower_bound();
    }

    std::vector<double> bdd_mma_agg::total_min_marginals()
    {
        return pimpl->mma.total_min_marginals();
    } 

}

