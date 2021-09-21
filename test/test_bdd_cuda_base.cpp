#include "bdd_cuda_base.h"
#include "ILP_parser.h"
#include "bdd_collection/bdd_collection.h"
#include "bdd_preprocessor.h"
#include "test.h"

using namespace LPMP;
using namespace BDD;

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
    ILP_input ilp = ILP_parser::parse_string(matching_3x3);
    bdd_preprocessor bdd_pre(ilp);
    bdd_collection bdd_col = bdd_pre.get_bdd_collection();
    bdd_cuda_base bcb(bdd_col);

    test(bcb.nr_variables() == 9);
    test(bcb.nr_bdds() == 6);

    for(size_t i=0; i<bcb.nr_variables(); ++i)
        bcb.set_cost(ilp.objective()[i], i);

    const auto mms = bcb.min_marginals();
    // test for correct min marginals
}
