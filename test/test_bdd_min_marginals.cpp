#include "bdd_solver/bdd_mma_base.h"
#include "bdd_solver/bdd_parallel_mma_base.h"
#include "ILP/ILP_parser.h"
#include "bdd_conversion/bdd_preprocessor.h"
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

template<typename SOLVER_TYPE>
void test_min_marginals()
{
    const ILP_input ilp = ILP_parser::parse_string(two_simplex_problem);
    bdd_preprocessor pre(ilp);

    SOLVER_TYPE solver(pre.get_bdd_collection(), ilp.objective());

    const auto mms = solver.min_marginals();
    test(mms.size() == 6);
    for(size_t i=0; i<6; ++i)
        test(mms.size(i) == 1);

    test(mms(0,0)[0] == 1 && mms(0,0)[1] == 2);
    test(mms(1,0)[0] == 1 && mms(1,0)[1] == 1);
    test(mms(2,0)[0] == 1 && mms(2,0)[1] == 1);

    test(mms(3,0)[0] == 1 && mms(3,0)[1] == 0);
    test(mms(4,0)[0] == 0 && mms(4,0)[1] == 1);
    test(mms(5,0)[0] == 3 && mms(5,0)[1] == 0);
}

int main(int argc, char** argv)
{
    test_min_marginals<bdd_mma_base<bdd_branch_instruction_bdd_index<float, uint32_t>>>();
    test_min_marginals<bdd_mma_base<bdd_branch_instruction_bdd_index<double, uint32_t>>>();
    test_min_marginals<bdd_parallel_mma_base<bdd_branch_instruction<float, uint16_t>>>();
    test_min_marginals<bdd_parallel_mma_base<bdd_branch_instruction<double, uint16_t>>>();
}
