#include <string>
#include "specialized_solvers/multi_graph_matching_input.h"
#include "../test.h"
#include "bdd_solver.h"
#include <iostream>

using namespace LPMP;

const std::string minimal_synchronization_example =
R"(gm 0 1
p 2 2 0 0
a 0 0 0 -1
a 1 0 1 -10
a 2 1 0 -10
a 3 1 1 -1

gm 0 2
p 2 2 0 0
a 0 0 0 -1
a 1 0 1 -10
a 2 1 0 -10
a 3 1 1 -1

gm 1 2
p 2 2 0 0
a 0 0 0 -1
a 1 0 1 -10
a 2 1 0 -10
a 3 1 1 -1
)";

int main(int argc, char** argv)
{
    const auto mgm_instance = parse_multi_graph_matching_string(minimal_synchronization_example);
    mgm_instance.write_lp(std::cout);
    bdd_solver_options opts;
    opts.ilp = mgm_instance;
    opts.bdd_solver_impl_ = bdd_solver_options::bdd_solver_impl::parallel_mma;
    bdd_solver s(opts);
    s.solve();
    test(std::abs(s.lower_bound() - (-42.0)) <= 1e-4);
}
