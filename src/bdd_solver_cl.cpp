#include "bdd_solver.h"

int main(int argc, char** argv)
{ 
    LPMP::bdd_solver solver(argc, argv);
    solver.solve();
    solver.round();
}
