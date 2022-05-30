#include <vector>
#include <string>
#include "bdd_solver.h"
#include "test.h"

using namespace LPMP;

std::string test_instance = R"(Minimize
x1 + x2 + x3 + x4 + x5 + x6
Subject To
x1 + x2 + x4 >= 1
x1 + x3 + x5 >= 1
x2 + x3 + x6 >= 1
Bounds
Binaries
x1
x2
x3
x4
x5
x6
End)";

std::string test_instance_tightened = R"(Minimize
x1 + x2 + x3 + x4 + x5 + x6
Subject To
x1 + x2 + x4 >= 1
x1 + x3 + x5 >= 1
x2 + x3 + x6 >= 1
x1 + x2 + x3 + x4 + x5 + x6 >= 2
Bounds
Binaries
x1
x2
x3
x4
x5
x6
End)";

std::string test_instance_coalesced = R"(Minimize
x1 + x2 + x3 + x4 + x5 + x6
Subject To
ineq1: x1 + x2 + x4 >= 1
ineq2: x1 + x3 + x5 >= 1
ineq3: x2 + x3 + x6 >= 1
Coalesce
ineq1 ineq2 ineq3
Bounds
Binaries
x1
x2
x3
x4
x5
x6
End)";

int main(int argc, char** argv)
{
    std::cout << "Solve original covering problem\n";

    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance,
            "-s", "mma",
            "--max_iter", "1000"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(std::abs(solver.lower_bound() - 1.5) <= 1e-4);
    }

    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance,
            "-s", "mma",
            "--max_iter", "1000",
            "--tighten"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(std::abs(solver.lower_bound() - 2.0) <= 1e-4);
    }


    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance_coalesced,
            "-s", "mma",
            "--max_iter", "1000"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(std::abs(solver.lower_bound() - 2.0) <= 1e-4);
    }

    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance_tightened,
            "-s", "mma",
            "--max_iter", "1000"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(solver.lower_bound() > 1.5); // does not solve exactly
    }

}

