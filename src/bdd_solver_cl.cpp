#include "bdd_solver.h"

int main(int argc, char** argv)
{ 
    if(argc != 2)
        throw std::runtime_error("one argument (json config file) expected");
    LPMP::bdd_solver solver;
    auto config = solver.read_config(argv[1]);
    solver.solve(config);
    //LPMP::bdd_solver_options options(argc, argv);
    //LPMP::bdd_solver solver(options);
    //solver.solve();
    //solver.round();
}
