#include "bdd_collection/bdd_collection.h"
#include "../test.h"
#include <vector>
#include <array>

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_collection bdd_col;

    for(size_t i=2; i<29; ++i)
    {
        std::vector<size_t> var_indices(i);
        std::iota(var_indices.begin(), var_indices.end(), 0);

        const size_t simplex_nr = bdd_col.simplex_constraint(i);
        test(bdd_col.variables(simplex_nr) == var_indices);
        test(bdd_col.min_max_variables(simplex_nr) == std::array<size_t,2>{0,var_indices.back()});

        const size_t not_all_false_nr = bdd_col.not_all_false_constraint(i);
        test(bdd_col.variables(not_all_false_nr) == var_indices);
        test(bdd_col.min_max_variables(not_all_false_nr) == std::array<size_t,2>{0,var_indices.back()});
    }
}

