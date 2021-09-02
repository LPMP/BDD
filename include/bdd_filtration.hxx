#pragma once

#include "bdd_manager/bdd.h"
#include "bdd_branch_instruction.h"

namespace LPMP {

    using min_marg_diff_type = char; 
    constexpr static char negative_min_marg_diff = 0; // means in solution value 1 is preferred
    constexpr static char positive_min_marg_diff = 1; // means in solution value 0 is preferred
    constexpr static char zero_min_marg_diff = 2;

    struct bdd_node {
        constexpr static size_t terminal_0 = std::numeric_limits<size_t>::max()-1;
        constexpr static size_t terminal_1 = std::numeric_limits<size_t>::max();
        size_t var;
        size_t low;
        size_t high;
    };

    /*
    class BDD_tightening {


        two_dim_variable_array<bdd_node> bdds;
        std::vector<char> min_marg_diffs;

        BDD::bdd_collection filtrated_BDDs();
    };
    */

}
