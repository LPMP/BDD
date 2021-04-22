#include "bdd_sequential_base.h"
#include "bdd_branch_instruction.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "bdd_storage.h"
#include "test.h"

using namespace LPMP;

const char * two_simplex_problem = 
R"(Minimize
2 x_1 + 1 x_2 + 1 x_3
+1 x_4 + 2 x_5 - 1 x_6
Subject To
x_1 + x_2 + x_3 = 1
x_4 + x_5 + x_6 = 2
End)";

int main(int argc, char** argv)
{
    using bdd_base_type = bdd_sequential_base<bdd_branch_instruction<float>>;
    const ILP_input ilp = ILP_parser::parse_string(two_simplex_problem);
    bdd_preprocessor pre(ilp);
    bdd_storage stor(pre);
    bdd_base_type solver(stor);
    solver.set_costs(ilp.objective().begin(), ilp.objective().end());
    solver.backward_run();
    const double lb = solver.lower_bound();
    test(lb == 1 + 0); 

    const auto mm = solver.min_marginals();
    test(mm.size() == 2);
    test(mm.size(0) == 3);
    test(mm.size(1) == 3);

    test(mm(0,0) == std::array<float,2>{1.0,2.0});
    test(mm(0,1) == std::array<float,2>{1.0,1.0});
    test(mm(0,2) == std::array<float,2>{1.0,1.0});

    test(mm(1,0) == std::array<float,2>{1.0,0.0});
    test(mm(1,1) == std::array<float,2>{0.0,1.0});
    test(mm(1,2) == std::array<float,2>{3.0,0.0});

}
