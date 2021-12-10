#include "ILP_parser.h"
#include <string>
#include "test.h"
#include <iostream>

using namespace LPMP;

const std::string ILP_example =
R"(Minimize
x1 + 2*x2 + 1.5 * x3 - 0.5*x4 - x5
Subject To
x1 + 2*x2 + 3 * x3 - 5*x4 - x5 >= 1
Bounds
 x1 <= 1
 x2 >= 0
End)";

const std::string nonlinear_example =
R"(Minimize
x1 + 2*x2 + 1.5 * x3 - 0.5*x4 - x5
Subject To
x1*x2 + 2*x3*x4 + 3 * x5*x6 - 5*x7*x8 - x9*x10 >= 1
End)";

int main(int argc, char** argv)
{
    // ILP_example
    {
    const ILP_input input = ILP_parser::parse_string(ILP_example);

    test(input.nr_variables() == 5);
    test(input.var_exists("x1"));
    test(input.var_exists("x2"));
    test(input.var_exists("x3"));
    test(input.var_exists("x4"));
    test(input.var_exists("x5"));

    test(input.objective("x1") == 1.0);
    test(input.objective("x2") == 2.0);
    test(input.objective("x3") == 1.5);
    test(input.objective("x4") == -0.5);
    test(input.objective("x5") == -1.0);

    input.write_lp(std::cout);
    test(input.nr_constraints() == 1);
    }

    // nonlinear_example
    {
    const ILP_input input = ILP_parser::parse_string(nonlinear_example);
    test(input.nr_variables() == 10);
    test(input.var_exists("x1"));
    test(input.var_exists("x2"));
    test(input.var_exists("x3"));
    test(input.var_exists("x4"));
    test(input.var_exists("x5"));
    test(input.var_exists("x6"));
    test(input.var_exists("x7"));
    test(input.var_exists("x8"));
    test(input.var_exists("x8"));
    test(input.var_exists("x10"));

    test(input.objective("x1") == 1.0);
    test(input.objective("x2") == 2.0);
    test(input.objective("x3") == 1.5);
    test(input.objective("x4") == -0.5);
    test(input.objective("x5") == -1.0);
    test(input.objective("x6") == 0.0);
    test(input.objective("x7") == 0.0);
    test(input.objective("x8") == 0.0);
    test(input.objective("x9") == 0.0);
    test(input.objective("x10") == 0.0);
    test(input.nr_constraints() == 1);
    }
}
