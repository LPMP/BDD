#include "bdd_solver.h"
#include "specialized_solvers/multi_graph_matching_input.h"

int main(int argc, char** argv)
{
    LPMP::bdd_solver_options opts(argc, argv, LPMP::parse_multi_graph_matching_file, LPMP::parse_multi_graph_matching_string);
    LPMP::bdd_solver solver(opts);
    solver.solve();
    solver.round();
}
