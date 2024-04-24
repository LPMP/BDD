#include "specialized_solvers/multi_graph_matching_solver.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    if(argc != 2)
        throw std::runtime_error("one argument (json config file) expected");
    multi_graph_matching_bdd_solver solver(argv[1]);
    solver.solve();
}
