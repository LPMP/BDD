#include "bdd_collection/bdd_collection.h"
#include "../test.h"
#include <vector>
#include <array>
#include <random>
#include <algorithm>

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_collection bdd_col;

    // test variables of various constraints encoded as BDDs
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

        std::reverse(var_indices.begin(), var_indices.end());

        bdd_col.rebase(simplex_nr, var_indices.begin(), var_indices.end());
        test(bdd_col.variables(simplex_nr) == var_indices);

        bdd_col.rebase(not_all_false_nr, var_indices.begin(), var_indices.end());
        test(bdd_col.variables(not_all_false_nr) == var_indices);
    }

    // test for sorting
    {
        std::vector<size_t> var_indices(42);
        std::iota(var_indices.begin(), var_indices.end(), 0);
        const size_t bdd_nr = bdd_col.simplex_constraint(var_indices.size());
        test(bdd_col.variables_sorted(bdd_nr));
        test(bdd_col.variables(bdd_nr) == var_indices);

        std::reverse(var_indices.begin(), var_indices.end());;
        bdd_col.rebase(bdd_nr, var_indices.begin(), var_indices.end());
        test(!bdd_col.variables_sorted(bdd_nr));
        test(bdd_col.variables(bdd_nr) == var_indices);
    }

    // test for random variables
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> distrib(1, 42);

        std::vector<size_t> var_indices = {distrib(gen)};
        for(size_t i=0; i<41; ++i)
            var_indices.push_back(var_indices.back() + distrib(gen));
        std::shuffle(var_indices.begin(), var_indices.end(), gen);

        const size_t bdd_nr = bdd_col.simplex_constraint(var_indices.size());
        test(bdd_col.variables_sorted(bdd_nr));
        bdd_col.rebase(bdd_nr, var_indices.begin(), var_indices.end());
        test(bdd_col.variables(bdd_nr) == var_indices);
    }
}

