#include "bdd_sequential_base.h"
#include "bdd_branch_instruction.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "test.h"

using namespace LPMP;

const char * matching_3x3 = 
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

int main(int argc, char** argv)
{
    using bdd_base_type = bdd_sequential_base<bdd_branch_instruction<float>>;
    const ILP_input ilp = ILP_parser::parse_string(matching_3x3);
    bdd_preprocessor pre(ilp);
    bdd_base_type solver(pre.get_bdd_collection());
    solver.set_costs(ilp.objective().begin(), ilp.objective().end());

    solver.backward_run();

    std::vector<std::array<float,2>> mms(ilp.nr_variables(), {0.0,0.0});
    solver.forward_mms(mms.begin(), 0.5);

    for(size_t i=0; i<mms.size(); ++i)
    {
        std::cout << "var " << i << ", mms = (" << mms[i][0] << "," << mms[i][1] << ")\n";
    }

    for(size_t i=0; i<mms.size(); ++i)
        test(mms[i][0] >= 0.0 && mms[i][1] >= 0.0);

}
