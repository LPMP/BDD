#include <vector>
#include <string>
#include "bdd_solver.h"
#include "test.h"

using namespace LPMP;

//mu_12_00 + mu_12_11 + mu_13_00 + mu_13_11 + mu_23_00 + mu_23_11
// binary graphical model triplet with negative Potts
std::string test_instance = R"(Minimize
- mu_12_01 - mu_12_10 - mu_13_01 - mu_13_10 - mu_23_01 - mu_23_10
Subject To
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_3_0 + mu_3_1 = 1
mu_12_00 + mu_12_01 + mu_12_10 + mu_12_11 = 1
mu_13_00 + mu_13_01 + mu_13_10 + mu_13_11 = 1
mu_23_00 + mu_23_01 + mu_23_10 + mu_23_11 = 1
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_1_0 - mu_13_00 - mu_13_01 = 0
mu_1_1 - mu_13_10 - mu_13_11 = 0
mu_3_0 - mu_13_00 - mu_13_10 = 0
mu_3_1 - mu_13_01 - mu_13_11 = 0
mu_2_0 - mu_23_00 - mu_23_01 = 0
mu_2_1 - mu_23_10 - mu_23_11 = 0
mu_3_0 - mu_23_00 - mu_23_10 = 0
mu_3_1 - mu_23_01 - mu_23_11 = 0
Bounds
Binaries
mu_0_0
mu_0_1
mu_1_0
mu_1_1
mu_2_0
mu_2_1
mu_01_00
mu_01_01
mu_01_10
mu_01_11
mu_02_00
mu_02_01
mu_02_10
mu_02_11
mu_12_00
mu_12_01
mu_12_10
mu_12_11
End)";

std::string test_instance_single_bdd = R"(Minimize
- mu_12_01 - mu_12_10 - mu_13_01 - mu_13_10 - mu_23_01 - mu_23_10
Subject To
simplex_1: mu_1_0 + mu_1_1 = 1
simplex_2: mu_2_0 + mu_2_1 = 1
simplex_3: mu_3_0 + mu_3_1 = 1
simplex_12: mu_12_00 + mu_12_01 + mu_12_10 + mu_12_11 = 1
simplex_13: mu_13_00 + mu_13_01 + mu_13_10 + mu_13_11 = 1
simplex_23: mu_23_00 + mu_23_01 + mu_23_10 + mu_23_11 = 1
marginalization_1_12_0: mu_1_0 - mu_12_00 - mu_12_01 = 0
marginalization_1_12_1: mu_1_1 - mu_12_10 - mu_12_11 = 0
marginalization_2_12_0: mu_2_0 - mu_12_00 - mu_12_10 = 0
marginalization_2_12_1: mu_2_1 - mu_12_01 - mu_12_11 = 0
marginalization_1_13_0: mu_1_0 - mu_13_00 - mu_13_01 = 0
marginalization_1_13_1: mu_1_1 - mu_13_10 - mu_13_11 = 0
marginalization_3_13_0: mu_3_0 - mu_13_00 - mu_13_10 = 0
marginalization_3_13_1: mu_3_1 - mu_13_01 - mu_13_11 = 0
marginalization_2_23_0: mu_2_0 - mu_23_00 - mu_23_01 = 0
marginalization_2_23_1: mu_2_1 - mu_23_10 - mu_23_11 = 0
marginalization_3_23_0: mu_3_0 - mu_23_00 - mu_23_10 = 0
marginalization_3_23_1: mu_3_1 - mu_23_01 - mu_23_11 = 0
Coalesce
simplex_1 simplex_2 simplex_3 simplex_12 simplex_13 simplex_23 marginalization_1_12_0 marginalization_1_12_1 marginalization_2_12_0 marginalization_2_12_1 marginalization_1_13_0 marginalization_1_13_1 marginalization_3_13_0 marginalization_3_13_1 marginalization_2_23_0 marginalization_2_23_1 marginalization_3_23_0 marginalization_3_23_1
Bounds
Bounds
Binaries
mu_0_0
mu_0_1
mu_1_0
mu_1_1
mu_2_0
mu_2_1
mu_01_00
mu_01_01
mu_01_10
mu_01_11
mu_02_00
mu_02_01
mu_02_10
mu_02_11
mu_12_00
mu_12_01
mu_12_10
mu_12_11
End)";

std::string test_instance_tight_reduced_bdd = R"(Minimize
- mu_12_01 - mu_12_10 - mu_13_01 - mu_13_10 - mu_23_01 - mu_23_10
Subject To
mu_12_00 + mu_12_01 + mu_12_10 + mu_12_11 = 1
mu_13_00 + mu_13_01 + mu_13_10 + mu_13_11 = 1
mu_23_00 + mu_23_01 + mu_23_10 + mu_23_11 = 1
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_1_0 - mu_13_00 - mu_13_01 = 0
mu_1_1 - mu_13_10 - mu_13_11 = 0
mu_3_0 - mu_13_00 - mu_13_10 = 0
mu_3_1 - mu_13_01 - mu_13_11 = 0
mu_2_0 - mu_23_00 - mu_23_01 = 0
mu_2_1 - mu_23_10 - mu_23_11 = 0
mu_3_0 - mu_23_00 - mu_23_10 = 0
mu_3_1 - mu_23_01 - mu_23_11 = 0
simplex_1: mu_1_0 + mu_1_1 = 1
simplex_2: mu_2_0 + mu_2_1 = 1
simplex_3: mu_3_0 + mu_3_1 = 1
simplex_12: mu_12_01 + mu_12_10 <= 1
simplex_13: mu_13_01 + mu_13_10 <= 1
simplex_23: mu_23_01 + mu_23_10 <= 1
marginalization_1_12_0: mu_1_0 - mu_12_01 >= 0
marginalization_1_12_1: mu_1_1 - mu_12_10 >= 0
marginalization_2_12_0: mu_2_0 - mu_12_10 >= 0
marginalization_2_12_1: mu_2_1 - mu_12_01 >= 0
marginalization_1_13_0: mu_1_0 - mu_13_01 >= 0
marginalization_1_13_1: mu_1_1 - mu_13_10 >= 0
marginalization_3_13_0: mu_3_0 - mu_13_10 >= 0
marginalization_3_13_1: mu_3_1 - mu_13_01 >= 0
marginalization_2_23_0: mu_2_0 - mu_23_01 >= 0
marginalization_2_23_1: mu_2_1 - mu_23_10 >= 0
marginalization_3_23_0: mu_3_0 - mu_23_10 >= 0
marginalization_3_23_1: mu_3_1 - mu_23_01 >= 0
Coalesce
simplex_1 simplex_2 simplex_3 simplex_12 simplex_13 simplex_23 marginalization_1_12_0 marginalization_1_12_1 marginalization_2_12_0 marginalization_2_12_1 marginalization_1_13_0 marginalization_1_13_1 marginalization_3_13_0 marginalization_3_13_1 marginalization_2_23_0 marginalization_2_23_1 marginalization_3_23_0 marginalization_3_23_1
Bounds
Binaries
mu_0_0
mu_0_1
mu_1_0
mu_1_1
mu_2_0
mu_2_1
mu_01_00
mu_01_01
mu_01_10
mu_01_11
mu_02_00
mu_02_01
mu_02_10
mu_02_11
mu_12_00
mu_12_01
mu_12_10
mu_12_11
End)";

std::string test_instance_single_reduced_bdd = R"(Minimize
- mu_12_01 - mu_12_10 - mu_13_01 - mu_13_10 - mu_23_01 - mu_23_10
Subject To
simplex_1: mu_1_0 + mu_1_1 = 1
simplex_2: mu_2_0 + mu_2_1 = 1
simplex_3: mu_3_0 + mu_3_1 = 1
simplex_12: mu_12_01 + mu_12_10 <= 1
simplex_13: mu_13_01 + mu_13_10 <= 1
simplex_23: mu_23_01 + mu_23_10 <= 1
marginalization_1_12_0: mu_1_0 - mu_12_01 >= 0
marginalization_1_12_1: mu_1_1 - mu_12_10 >= 0
marginalization_2_12_0: mu_2_0 - mu_12_10 >= 0
marginalization_2_12_1: mu_2_1 - mu_12_01 >= 0
marginalization_1_13_0: mu_1_0 - mu_13_01 >= 0
marginalization_1_13_1: mu_1_1 - mu_13_10 >= 0
marginalization_3_13_0: mu_3_0 - mu_13_10 >= 0
marginalization_3_13_1: mu_3_1 - mu_13_01 >= 0
marginalization_2_23_0: mu_2_0 - mu_23_01 >= 0
marginalization_2_23_1: mu_2_1 - mu_23_10 >= 0
marginalization_3_23_0: mu_3_0 - mu_23_10 >= 0
marginalization_3_23_1: mu_3_1 - mu_23_01 >= 0
Coalesce
simplex_1 simplex_2 simplex_3 simplex_12 simplex_13 simplex_23 marginalization_1_12_0 marginalization_1_12_1 marginalization_2_12_0 marginalization_2_12_1 marginalization_1_13_0 marginalization_1_13_1 marginalization_3_13_0 marginalization_3_13_1 marginalization_2_23_0 marginalization_2_23_1 marginalization_3_23_0 marginalization_3_23_1
Bounds
Binaries
mu_0_0
mu_0_1
mu_1_0
mu_1_1
mu_2_0
mu_2_1
mu_01_00
mu_01_01
mu_01_10
mu_01_11
mu_02_00
mu_02_01
mu_02_10
mu_02_11
mu_12_00
mu_12_01
mu_12_10
mu_12_11
End)";

std::string test_instance_tight_large = R"(Minimize
- mu_12_01 - mu_12_10 - mu_13_01 - mu_13_10 - mu_23_01 - mu_23_10
Subject To
mu_12_00 + mu_12_01 + mu_12_10 + mu_12_11 = 1
mu_13_00 + mu_13_01 + mu_13_10 + mu_13_11 = 1
mu_23_00 + mu_23_01 + mu_23_10 + mu_23_11 = 1
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_1_0 - mu_13_00 - mu_13_01 = 0
mu_1_1 - mu_13_10 - mu_13_11 = 0
mu_3_0 - mu_13_00 - mu_13_10 = 0
mu_3_1 - mu_13_01 - mu_13_11 = 0
mu_2_0 - mu_23_00 - mu_23_01 = 0
mu_2_1 - mu_23_10 - mu_23_11 = 0
mu_3_0 - mu_23_00 - mu_23_10 = 0
mu_3_1 - mu_23_01 - mu_23_11 = 0
simplex_1: mu_1_0 + mu_1_1 = 1
simplex_2: mu_2_0 + mu_2_1 = 1
simplex_3: mu_3_0 + mu_3_1 = 1
simplex_12: mu_12_00 + mu_12_01 + mu_12_10 + mu_12_11 = 1
simplex_13: mu_13_00 + mu_13_01 + mu_13_10 + mu_13_11 = 1
simplex_23: mu_23_00 + mu_23_01 + mu_23_10 + mu_23_11 = 1
marginalization_1_12_0: mu_1_0 - mu_12_00 - mu_12_01 = 0
marginalization_1_12_1: mu_1_1 - mu_12_10 - mu_12_11 = 0
marginalization_2_12_0: mu_2_0 - mu_12_00 - mu_12_10 = 0
marginalization_2_12_1: mu_2_1 - mu_12_01 - mu_12_11 = 0
marginalization_1_13_0: mu_1_0 - mu_13_00 - mu_13_01 = 0
marginalization_1_13_1: mu_1_1 - mu_13_10 - mu_13_11 = 0
marginalization_3_13_0: mu_3_0 - mu_13_00 - mu_13_10 = 0
marginalization_3_13_1: mu_3_1 - mu_13_01 - mu_13_11 = 0
marginalization_2_23_0: mu_2_0 - mu_23_00 - mu_23_01 = 0
marginalization_2_23_1: mu_2_1 - mu_23_10 - mu_23_11 = 0
marginalization_3_23_0: mu_3_0 - mu_23_00 - mu_23_10 = 0
marginalization_3_23_1: mu_3_1 - mu_23_01 - mu_23_11 = 0
Coalesce
simplex_1 simplex_2 simplex_3 simplex_12 simplex_13 simplex_23 marginalization_1_12_0 marginalization_1_12_1 marginalization_2_12_0 marginalization_2_12_1 marginalization_1_13_0 marginalization_1_13_1 marginalization_3_13_0 marginalization_3_13_1 marginalization_2_23_0 marginalization_2_23_1 marginalization_3_23_0 marginalization_3_23_1
Bounds
Binaries
mu_0_0
mu_0_1
mu_1_0
mu_1_1
mu_2_0
mu_2_1
mu_01_00
mu_01_01
mu_01_10
mu_01_11
mu_02_00
mu_02_01
mu_02_10
mu_02_11
mu_12_00
mu_12_01
mu_12_10
mu_12_11
End)";

int main(int argc, char** argv)
{
    std::cout << "Solve original loose LP\n";
    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance,
            "-s", "mma",
            "--max_iter", "35"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(std::abs(solver.lower_bound() - (-3.0)) <= 1e-3);
    }

    std::cout << "Solve tight LP with single BDD of intersection of all BDDs\n";
    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance_single_bdd,
            "-s", "mma",
            "--max_iter", "35"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(std::abs(solver.lower_bound() - (-2.0)) <= 1e-3);
    }

    std::cout << "Solve tight LP with single BDD of intersection of all reduced BDDs\n";
    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance_single_reduced_bdd,
            "-s", "mma",
            "--max_iter", "35"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(std::abs(solver.lower_bound() - (-2.0)) <= 1e-3);
    }

    std::cout << "Solve tight LP original BDDs and intersected BDD\n";
    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance_tight_large,
            "-s", "mma",
            "--max_iter", "1000"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(std::abs(solver.lower_bound() - (-2.0)) <= 1e-4);
    }

    std::cout << "Solve tight LP original BDDs and intersected reduced BDD\n";
    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance_tight_reduced_bdd,
            "-s", "mma",
            "--max_iter", "1000"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();
        test(std::abs(solver.lower_bound() - (-2.0)) <= 1e-4);
    }
}
