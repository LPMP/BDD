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
+ 1 x_11 + 1 x_12 + 1 x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
- x_13 - x_23 - x_33 = -1
End)";

void test_input_export()
{
    const ILP_input input_orig = ILP_parser::parse_string(matching_3x3);

    std::stringstream lp_exported;
    input_orig.write_lp(lp_exported);

    std::stringstream opb_exported;
    input_orig.write_opb(opb_exported);

    auto produce_lb = [&](const std::string& input) {
        std::vector<std::string> solver_input = {
            "--input_string", input ,
            "-s", "mma_vec",
            "--max_iter", "1000"
        };

        bdd_solver solver(solver_input); 
        solver.solve();

        return solver.lower_bound();
    };

    test(std::abs(produce_lb(matching_3x3) - produce_lb(lp_exported.str())) <= 1e-8);
    test(std::abs(produce_lb(matching_3x3) - produce_lb(opb_exported.str())) <= 1e-8);
}

int main(int argc, char** arv)
{
    test_input_export();
} 
