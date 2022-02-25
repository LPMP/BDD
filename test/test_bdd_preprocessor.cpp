#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "two_dimensional_variable_array.hxx"
#include "test.h"

using namespace LPMP;

const std::string ilp_string = 
R"(Minimize
Subject To
a + b + c + d = 1
a + b + c  <= 1
b + c <= 1
End)";

int main(int argc, char** argv)
{
    // test if inequalities were correctly translated in bdd preprocessor
    ILP_input ilp = ILP_parser::parse_string(ilp_string);
    bdd_preprocessor bdd_pre;
    const two_dim_variable_array<size_t> ineq_to_bdd_nrs = bdd_pre.add_ilp(ilp);
    test(ineq_to_bdd_nrs.size() == 3);
    test(ineq_to_bdd_nrs.size(0) == 1 && ineq_to_bdd_nrs(0,0) == 0);
    test(ineq_to_bdd_nrs.size(1) == 1 && ineq_to_bdd_nrs(1,0) == 1);
    test(ineq_to_bdd_nrs.size(2) == 1 && ineq_to_bdd_nrs(2,0) == 2);
}
