#pragma once

#include <numeric>

namespace LPMP {

    struct lineq_bdd_node {

        lineq_bdd_node() {}
        lineq_bdd_node(int lb, int ub, lineq_bdd_node* zero_kid, lineq_bdd_node* one_kid)
        : lb_(lb), ub_(ub), zero_kid_(zero_kid), one_kid_(one_kid)
        {}

        int lb_;
        int ub_;

        lineq_bdd_node* zero_kid_;
        lineq_bdd_node* one_kid_;
    };

    struct lineq_bdd {

        lineq_bdd() {}
        lineq_bdd(const size_t dim) : inverted(dim), levels(dim),
            topsink(0, std::numeric_limits<int>::max(), nullptr, nullptr), 
            botsink(std::numeric_limits<int>::min(), -1, nullptr, nullptr)
        {}

        lineq_bdd_node* root_node;

        std::vector<char> inverted; // flags inverted variables

        lineq_bdd_node topsink;
        lineq_bdd_node botsink;
        std::vector<std::vector<lineq_bdd_node>> levels;
    };
}
