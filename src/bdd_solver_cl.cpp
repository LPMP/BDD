#include "bdd_solver.h"

int main(int argc, char** argv)
{ 
    LPMP::bdd_solver_options options(argc, argv);
    LPMP::bdd_solver solver(options);
    solver.solve();
    solver.round();
}
