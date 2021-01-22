#include "bdd_fix.h"
#include "bdd_fix_base.hxx"
#include "time_measure_util.h"

namespace LPMP {

    class bdd_fix::impl {
        public:
            using bdd_fix_base_type = bdd_fix_base;

            impl(bdd_storage& bdd_storage_)
                : fix(bdd_storage_)
            {
                fix.init_pointers();
            };

            impl(bdd_storage& bdd_storage_, bdd_fix_options opts)
                : fix(bdd_storage_)
            {
                fix.init_pointers();
                fix.set_options(opts);
            };

            bdd_fix_base_type fix;
    };

    bdd_fix::bdd_fix(bdd_storage& stor)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(stor);
    }

    bdd_fix::bdd_fix(bdd_storage& stor, bdd_fix_options opts) 
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(stor, opts);
    }

    bdd_fix::bdd_fix(bdd_fix&& o)
        : pimpl(std::move(o.pimpl))
    {}

    bdd_fix& bdd_fix::operator=(bdd_fix&& o)
    { 
        pimpl = std::move(o.pimpl);
        return *this;
    }

    bdd_fix::~bdd_fix()
    {}
 
    bool bdd_fix::round(const std::vector<double> total_min_marginals)
    {
        pimpl->fix.set_total_min_marginals(total_min_marginals);
        return pimpl->fix.fix_variables(); 
    }

    std::vector<char> bdd_fix::primal_solution()
    {
        return pimpl->fix.primal_solution(); 
    }

}
