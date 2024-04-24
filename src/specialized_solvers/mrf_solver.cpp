#include "bdd_solver.h"
#include "specialized_solvers/mrf_solver.h"
#include "specialized_solvers/mrf_input.h"

using namespace LPMP;


ILP_input mrf_bdd_solver::read_ILP(const nlohmann::json &config)
{
    if (!config.contains("input"))
        throw std::runtime_error("no input specified");

    // determine whether input file or input string is specified
    const std::string input = config["input"].template get<std::string>();
    std::ifstream f(input);
    if (f)
    {
        bdd_log << "[mrf solver] Read input file " << config["input"] << "\n";
        return LPMP::parse_mrf_uai_file(input).convert_to_ilp();
    }
    else // input might be ILP in string
    {
        bdd_log << "[mrf solver] Read input string " << config["input"] << "\n";
        return LPMP::parse_mrf_uai_string(input).convert_to_ilp();
    }
}

