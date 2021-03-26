#include "bdd_mma_vec.h"
#include "bdd_branch_node_vector.h"
#include "bdd_branch_node_vec8f.h"
#include "time_measure_util.h"

namespace LPMP {

    class bdd_mma_vec::impl {
        public:
            impl(bdd_storage& bdd_storage_)
                : mma(bdd_storage_)
            {};

            bdd_mma_base_vec mma;
            //bdd_mma_base_8f mma;
    };

    bdd_mma_vec::bdd_mma_vec(bdd_storage& stor)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(stor);
    }

    bdd_mma_vec::bdd_mma_vec(bdd_mma_vec&& o)
        : pimpl(std::move(o.pimpl))
    {}

    bdd_mma_vec& bdd_mma_vec::operator=(bdd_mma_vec&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    bdd_mma_vec::~bdd_mma_vec()
    {}

    void bdd_mma_vec::set_cost(const double c, const size_t var)
    {
        pimpl->mma.set_cost(c, var);
    }

    void bdd_mma_vec::backward_run()
    {
        pimpl->mma.backward_run();
    }

    void bdd_mma_vec::solve(const size_t max_iter, const double tolerance, const double time_limit)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        pimpl->mma.solve(max_iter, tolerance, time_limit);
    }

    double bdd_mma_vec::lower_bound()
    {
        return pimpl->mma.lower_bound();
    }

    std::vector<double> bdd_mma_vec::total_min_marginals()
    {
        return pimpl->mma.total_min_marginals();
    }

}


