#pragma once

#include <unordered_map>
#include "hash_helper.hxx"
#include "specialized_solvers/graph_matching_input.h"
#include "ILP_input.h"

namespace LPMP {

    using multi_graph_matching_instance = std::unordered_map<std::array<size_t,2>, graph_matching_instance>;

    ILP_input construct_multi_graph_matching_ILP(const multi_graph_matching_instance& gm);

    ILP_input parse_multi_graph_matching_file(const std::string& filename);
    ILP_input parse_multi_graph_matching_string(const std::string& mgm_string);

}
