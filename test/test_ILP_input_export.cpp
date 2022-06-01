#include "ILP_parser.h"
#include "OPB_parser.h"
#include "bdd_solver.h"
#include <string>
#include <sstream>
#include "test.h"

using namespace LPMP;

const char * matching_3x3 = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-1 x_21 - 2 x_22 - 1 x_23
-1 x_31 - 1 x_32 - 2 x_33
Subject To
ineq_1x: + 1 x_11 + 1 x_12 + 1 x_13 = 1
ineq_2x: x_21 + x_22 + x_23 = 1
ineq_3x: x_31 + x_32 + x_33 = 1
ineq_x1: x_11 + x_21 + x_31 = 1
ineq_x2: x_12 + x_22 + x_32 = 1
ineq_x3: - x_13 - x_23 - x_33 = -1
End)";

const char * covering_problem_3x3 = 
R"(Minimize
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

const char * covering_problem_2_3x3 = 
R"(Minimize
x1 + x2 + x3 + x4 + x5 + x6
Subject To
-x1 - x2 - x4 <= -1
-x1 - x3 - x5 <= -1
-x2 - x3 - x6 <= -1
Bounds
Binaries
x1
x2
x3
x4
x5
x6
End)";

const std::string export_lp(const std::string& problem)
{
    const ILP_input input_orig = ILP_parser::parse_string(problem);
    std::stringstream lp_exported;
    input_orig.write_lp(lp_exported);
    return lp_exported.str();
}

const std::string export_opb(const std::string& problem)
{
    const ILP_input input_orig = ILP_parser::parse_string(problem);
    std::stringstream opb_exported;
    input_orig.write_opb(opb_exported);
    return opb_exported.str();
}

void test_export(const std::string& problem, const double lb)
{
    auto compute_lp = [&](const std::string& problem) {
        std::vector<std::string> solver_input = {
            "--input_string", problem,
            "-s", "mma",
            "--max_iter", "1000",
            "--tolerance", "1e-10"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();

        return solver.lower_bound();
    };

    auto compute_opb = [&](const std::string& problem) {
        std::vector<std::string> solver_input = {
            "--input_string", problem,
            "-s", "mma",
            "--max_iter", "1000",
            "--tolerance", "1e-10"
        };

        bdd_solver solver((bdd_solver_options(solver_input))); 
        solver.solve();

        return solver.lower_bound();
    };

    std::cout << "original problem:\n" << problem << "\n";
    std::cout << "exported lp:\n" << export_lp(problem) << "\n";
    std::cout << "exported opb:\n" << export_opb(problem) << "\n";

    test(ILP_parser::parse_string(problem).nr_variables() == ILP_parser::parse_string(export_lp(problem)).nr_variables());

    const double orig_lb = compute_lp(problem);
    const double exported_lp_lb = compute_lp(export_lp(problem));
    const double exported_opb_lb = compute_opb(export_opb(problem));

    test(std::abs(orig_lb - lb) <= 1e-6);
    test(std::abs(exported_lp_lb - lb) <= 1e-6);
    test(std::abs(exported_opb_lb - lb) <= 1e-6);
}

int main(int argc, char** arv)
{
    test_export(matching_3x3, -6.0);
    test_export(covering_problem_3x3, 1.5);
    test_export(covering_problem_2_3x3, 1.5);
}
