#include "bdd_solver.h"
#include "bdd_branch_node.h"

int main(int argc, char** argv)
{
    std::cout << "static " << LPMP::bdd_branch_node_opt::terminal_0()->m << "\n";
    std::cout << "static " << LPMP::bdd_branch_node_opt_arc_cost::terminal_0()->m << "\n";
    LPMP::bdd_solver solver(argc, argv);
    solver.solve();
}
