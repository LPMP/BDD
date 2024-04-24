#include "specialized_solvers/graph_matching_solver.h"
#include "specialized_solvers/graph_matching_input.h"

namespace LPMP {

ILP_input graph_matching_bdd_solver::read_ILP(const nlohmann::json &config)
{
    if (!config.contains("input"))
        throw std::runtime_error("no input specified");

    // determine whether input file or input string is specified
    const std::string input = config["input"].template get<std::string>();
    std::ifstream f(input);
    if (f)
    {
        bdd_log << "[graph matching solver] Read input file " << config["input"] << "\n";
        return LPMP::parse_graph_matching_file(input);
    }
    else // input might be ILP in string
    {
        bdd_log << "[graph matching solver] Read input string " << config["input"] << "\n";
        return LPMP::parse_graph_matching_string(input);
    }
}

}