#include "bdd_collection/bdd_collection.h"
#include "../test.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    BDD::bdd_collection bdd_col;

    for(size_t i=2; i<17; ++i)
    {
        const size_t not_all_false_nr = bdd_col.not_all_false_constraint(i);
        const size_t not_all_false_qbdd = bdd_col.make_qbdd(not_all_false_nr);
        test(bdd_col.is_qbdd(not_all_false_qbdd));
        test(!bdd_col.is_qbdd(not_all_false_nr));
        test(bdd_col.nr_bdd_nodes(not_all_false_nr) < bdd_col.nr_bdd_nodes(not_all_false_qbdd));
    }
}
