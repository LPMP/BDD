#include "bdd_solver.h"
#include <vector>
#include <string>
#include <iostream>
#include "test.h"

using namespace LPMP;

void test_problem(const std::string input_string, const double expected_lb, std::vector<std::string> args)
{
    args.push_back("--input_string");
    args.push_back(input_string);
    bdd_solver solver((bdd_solver_options(args)));
    const double initial_lb = solver.lower_bound();
    solver.solve();
    const double lb = solver.lower_bound();

    test(std::abs(lb - expected_lb) <= 1e-6);
}

const char * matching_3x3_diag = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-1 x_21 - 2 x_22 - 1 x_23
-1 x_31 - 1 x_32 - 2 x_33
Subject To
x_11 + x_12 + x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
x_13 + x_23 + x_33 = 1
End)";

const char * matching_3x3_first_row = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-2 x_21 - 1 x_22 - 1 x_23
-2 x_31 - 1 x_32 - 1 x_33
Subject To
x_11 + x_12 + x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
x_13 + x_23 + x_33 = 1
End)";


int main(int argc, char** arv)
{
//    test_problem(matching_3x3_diag, -6.0, {"-s", "mma", "--max_iter", "20"});
//    test_problem(matching_3x3_diag, -6.0, {"-s", "decomposition_mma", "--nr_threads", "2", "--max_iter", "1000"});
//    test_problem(matching_3x3_diag, -6.0, {"-s", "anisotropic_mma", "--max_iter", "20"});
//    test_problem(matching_3x3_diag, -6.0, {"-s", "mma_srmp", "--max_iter", "20"});
    test_problem(matching_3x3_diag, -6.0, {"-s", "mma", "--max_iter", "20"});

//    test_problem(matching_3x3_first_row, -4.0, {"-s", "mma", "--max_iter", "20"});
//    test_problem(matching_3x3_first_row, -4.0, {"-s", "decomposition_mma", "--nr_threads", "2", "--max_iter", "1000"});
//    test_problem(matching_3x3_first_row, -4.0, {"-s", "anisotropic_mma", "--max_iter", "20"});
//    test_problem(matching_3x3_first_row, -4.0, {"-s", "mma_srmp", "--max_iter", "20"});
    test_problem(matching_3x3_first_row, -4.0, {"-s", "mma", "--max_iter", "20"});
}
