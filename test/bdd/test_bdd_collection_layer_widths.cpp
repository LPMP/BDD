#include "bdd_collection/bdd_collection.h"
#include "../test.h"

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_collection bdd_col;

    for (size_t i = 2; i < 15; ++i)
    {
        std::vector<size_t> var_indices(i);
        std::iota(var_indices.begin(), var_indices.end(), 0);

        const size_t simplex_nr = bdd_col.simplex_constraint(i);
        const auto layer_widths = bdd_col.layer_widths(simplex_nr);
        test(layer_widths.size() == i);
        for(size_t v=1; v<i; ++v)
        {
            test(layer_widths[v] == 2);
        }
    }
}
