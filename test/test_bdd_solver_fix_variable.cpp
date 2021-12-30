#include "bdd_solver.h"
#include "test.h"

using namespace LPMP;

const char * short_mrf_chain = 
R"(Minimize
3 mu_1_0 + 1 mu_1_1
- 1 mu_2_0 + 0 mu_2_1
+ 1 mu_00 + 2 mu_10 + 1 mu_01 + 0 mu_11
Subject To
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_00 + mu_10 + mu_01 + mu_11 = 1
mu_1_0 - mu_00 - mu_01 = 0
mu_1_1 - mu_10 - mu_11 = 0
mu_2_0 - mu_00 - mu_10 = 0
mu_2_1 - mu_01 - mu_11 = 0
End)";

void variable_fixation_test(const std::string& solver_name)
{
    std::vector<std::string> args {
        "--input_string", short_mrf_chain,
        "-s", solver_name
    };

    bdd_solver solver((bdd_solver_options(args)));
    solver.solve();
    // optimal solution is (1,1);
    test(std::abs(solver.lower_bound() - (1.0 + 0.0 + 0.0)) <= 1e-6);

    solver.fix_variable("mu_2_1", false);
    solver.solve();
    // optimal solution (1,0);
    test(std::abs(solver.lower_bound() - (1.0 - 1.0 + 2.0)) <= 1e-6);

    solver.fix_variable("mu_1_1", false);
    solver.solve();
    // optimal solution (0,0);
    test(std::abs(solver.lower_bound() - (3.0 - 1.0 + 1.0)) <= 1e-6);
}

int main(int argc, char** argv)
{
    variable_fixation_test("sequential_mma");
    variable_fixation_test("parallel_mma");
}
