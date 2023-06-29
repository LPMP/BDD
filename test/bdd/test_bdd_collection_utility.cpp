#include "bdd_collection/bdd_collection.h"
#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    BDD::bdd_mgr bdd_mgr;
    BDD::bdd_collection bdd_col;

    std::vector<BDD::node_ref> bdd_mgr_vars;
    for(size_t i=0; i<15; ++i)
        bdd_mgr_vars.push_back(bdd_mgr.projection(i));

    // simplex
    for(size_t i=1; i<10; ++i)
    {
        BDD::node_ref bdd_mgr_simplex = bdd_mgr.simplex(bdd_mgr_vars.begin(), bdd_mgr_vars.begin()+i);
        const size_t bdd_nr = bdd_col.simplex_constraint(i);
        BDD::node_ref bdd_col_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        test(bdd_col_exported == bdd_mgr_simplex);

        bdd_col.negate(bdd_nr);
        BDD::node_ref bdd_col_negate_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        BDD::node_ref neg_bdd_mgr_simplex = bdd_mgr.negate(bdd_mgr_simplex);
        test(bdd_col_negate_exported == neg_bdd_mgr_simplex);
    }

    // not all false
    for(size_t i=1; i<10; ++i)
    {
        BDD::node_ref bdd_mgr_not_all_false = bdd_mgr.negate(bdd_mgr.all_false(bdd_mgr_vars.begin(), bdd_mgr_vars.begin()+i));
        const size_t bdd_nr = bdd_col.not_all_false_constraint(i);
        BDD::node_ref bdd_col_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        test(bdd_col_exported == bdd_mgr_not_all_false);

        bdd_col.negate(bdd_nr);
        BDD::node_ref bdd_col_negate_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        BDD::node_ref neg_bdd_mgr_not_all_false = bdd_mgr.negate(bdd_mgr_not_all_false);
        test(bdd_col_negate_exported == neg_bdd_mgr_not_all_false);
    }

    // all equal
    for(size_t i=2; i<10; ++i)
    {
        BDD::node_ref bdd_mgr_all_equal = bdd_mgr.all_equal(bdd_mgr_vars.begin(), bdd_mgr_vars.begin()+i);
        const size_t bdd_nr = bdd_col.all_equal_constraint(i);
        BDD::node_ref bdd_col_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        test(bdd_col_exported == bdd_mgr_all_equal);

        bdd_col.negate(bdd_nr);
        BDD::node_ref bdd_col_negate_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
        BDD::node_ref neg_bdd_mgr_all_equal = bdd_mgr.negate(bdd_mgr_all_equal);
        test(bdd_col_negate_exported == neg_bdd_mgr_all_equal);
    }

    // cardinality constraint
    for(size_t i=2; i<15; ++i)
    {
        for(size_t k=1; k<=i; ++k)
        {
            BDD::node_ref bdd_mgr_cardinality = bdd_mgr.cardinality(bdd_mgr_vars.begin(), bdd_mgr_vars.begin() + i, k);
            const size_t bdd_nr = bdd_col.cardinality_constraint(i, k);
            BDD::node_ref bdd_col_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
            test(bdd_col_exported == bdd_mgr_cardinality);

            bdd_col.negate(bdd_nr);
            BDD::node_ref bdd_col_negate_exported = bdd_col.export_bdd(bdd_mgr, bdd_nr);
            BDD::node_ref neg_bdd_mgr_cardinality = bdd_mgr.negate(bdd_mgr_cardinality);
            test(bdd_col_negate_exported == neg_bdd_mgr_cardinality);
        }
    }
}
