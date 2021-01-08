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

            bdd_fix_base_type fix;
    };

    bdd_fix::bdd_fix(bdd_storage& stor)
    {
        MEASURE_FUNCTION_EXECUTION_TIME; 
        pimpl = std::make_unique<impl>(stor);
    }

    bdd_fix::bdd_fix(bdd_storage& stor, const int var_order, const int var_value) 
    : bdd_fix(stor)
    {
        set_var_order(var_order);
        set_var_value(var_value);
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

    void bdd_fix::set_var_order(const int var_order)
    {
        pimpl->fix.set_var_order(static_cast<bdd_fix_base::variable_order>(var_order)); 
    }

    void bdd_fix::set_var_value(const int var_value)
    {
        pimpl->fix.set_var_value(static_cast<bdd_fix_base::variable_value>(var_value));
    }
 
    bool bdd_fix::round(const std::vector<double> total_min_marginals)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        pimpl->fix.set_total_min_marginals(total_min_marginals);
        return pimpl->fix.fix_variables(); 
    }

    std::vector<char> bdd_fix::primal_solution()
    {
        return pimpl->fix.primal_solution(); 
    }

}
