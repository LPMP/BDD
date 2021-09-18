#include "bdd_collection/bdd_collection.h"
#include "../test.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    BDD::bdd_collection bdd_col;

    const size_t simplex_nr_3 = bdd_col.simplex_constraint(3);
    test(bdd_col.nr_bdd_nodes(simplex_nr_3) == 2*3 - 1 + 2);
    const size_t simplex_nr_4 = bdd_col.simplex_constraint(4);
    test(bdd_col.nr_bdd_nodes(simplex_nr_4) == 2*4 - 1 + 2);

    test(bdd_col.nr_bdds() == 2);
    bdd_col.remove(0);
    test(bdd_col.nr_bdds() == 1);
    test(bdd_col.nr_bdd_nodes(0) == 2*4-1 + 2);
}
