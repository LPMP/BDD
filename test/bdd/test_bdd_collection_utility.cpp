#include "bdd_collection/bdd_collection.h"
#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    BDD::bdd_mgr bdd_mgr;
    BDD::bdd_collection bdd_col;

    std::vector<BDD::node_ref> bdd_mgr_vars;
    for(size_t i=0; i<10; ++i)
        bdd_mgr_vars.push_back(bdd_mgr.projection(i));

    // simplex
    for(size_t i=1; i<10; ++i)
    {
        BDD::node_ref bdd_mgr_simplex = bdd_mgr.simplex(bdd_mgr_vars.begin(), bdd_mgr_vars.begin()+i);
        const size_t bdd_nr = bdd_col.simplex_constraint(i);
        BDD::node_ref bdd_col_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        test(bdd_col_exported == bdd_mgr_simplex);
    }

    // not all false
    for(size_t i=1; i<10; ++i)
    {
        BDD::node_ref bdd_mgr_not_all_false = bdd_mgr.negate(bdd_mgr.all_false(bdd_mgr_vars.begin(), bdd_mgr_vars.begin()+i));
        const size_t bdd_nr = bdd_col.not_all_false_constraint(i);
        BDD::node_ref bdd_col_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        test(bdd_col_exported == bdd_mgr_not_all_false);
    }

    // all equal
    for(size_t i=2; i<10; ++i)
    {
        BDD::node_ref bdd_mgr_all_equal = bdd_mgr.all_equal(bdd_mgr_vars.begin(), bdd_mgr_vars.begin()+i);
        const size_t bdd_nr = bdd_col.all_equal_constraint(i);
        BDD::node_ref bdd_col_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        test(bdd_col_exported == bdd_mgr_all_equal);
    }
}
