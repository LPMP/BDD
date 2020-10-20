#include "bdd_mma.h"
#include "bdd_mma_base.hxx"
#include "bdd_branch_node.h"
#include "bdd_variable.h"

namespace LPMP {

    class bdd_mma::impl {
        public:
            using bdd_mma_base_type = bdd_mma_base_node_costs<bdd_variable_mma, bdd_branch_node_opt>;

            impl(bdd_storage& bdd_storage_)
                : mma(bdd_storage_)
            {};

            bdd_mma_base_type mma;
    };

    bdd_mma::bdd_mma(bdd_storage& stor)
        : pimpl(new impl(stor))
    {}

    bdd_mma::~bdd_mma()
    {}

    void bdd_mma::set_cost(const double c, const size_t var)
    {
        pimpl->mma.set_cost(c, var);
    }

    void bdd_mma::backward_run()
    {
        pimpl->mma.backward_run();
    }

    void bdd_mma::iteration()
    {
        pimpl->mma.iteration();
    }

    double bdd_mma::lower_bound()
    {
        return pimpl->mma.lower_bound();
    } 

}
