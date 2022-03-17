#pragma once

#include <array>
#include <vector>
#include <string>
#include <limits>
#include <unordered_map>
#include "hash_helper.hxx"
#include "ILP_input.h"

namespace LPMP {

    struct graph_matching_instance {
        constexpr static size_t no_assignment = std::numeric_limits<size_t>::max();
        struct linear_assignment { size_t i; size_t j; double cost; };
        struct quadratic_assignment { std::array<size_t,2> i; std::array<size_t,2> j; double cost; };
        std::vector<linear_assignment> linear_assignments;
        std::vector<quadratic_assignment> quadratic_assignments;
    };

    // return ILP and map of linear and quadratic variables
    std::tuple<ILP_input,
        std::unordered_map<std::array<size_t,2>, size_t>,
        std::unordered_map<std::array<size_t,4>, size_t>>
            construct_graph_matching_ILP(const graph_matching_instance& gm_instance);

    ILP_input parse_graph_matching_file(const std::string& filename);
    ILP_input parse_graph_matching_string(const std::string& string);
}
