#include "bdd_solver.h"
#include "specialized_solvers/mrf_input.h"

int main(int argc, char** argv)
{
    auto parse_and_convert_uai_file = [](const std::string filename) { return LPMP::parse_mrf_uai_file(filename).convert_to_ilp(); };
    auto parse_and_convert_uai_string = [](const std::string string) { return LPMP::parse_mrf_uai_string(string).convert_to_ilp(); };
    LPMP::bdd_solver_options opts(argc, argv, parse_and_convert_uai_file, parse_and_convert_uai_string);
    LPMP::bdd_solver solver(opts);
    solver.solve();
    solver.round();
}
