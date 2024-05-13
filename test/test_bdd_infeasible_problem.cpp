#include "ILP/ILP_parser.h"
#include "bdd_conversion/bdd_preprocessor.h"
#include "bdd_solver/bdd_parallel_mma_base.h"
#include "bdd_solver/subgradient.h"
#include "bdd_solver/bdd_mma_base.h"
#include "bdd_solver/bdd_branch_instruction.h"
#include "test.h"

using namespace LPMP;

const char * infeasible_problem = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-1 x_21 - 2 x_22 - 1 x_23 + 3 x_24
-1 x_31 - 1 x_32 - 2 x_33 - 3.5 x_24
Subject To
x_11 + x_12 + x_13 = 1 
x_21 + x_22 + x_23 + x_24 = 2 
x_31 + x_32 + x_33 + x_24 = 1 
x_11 + x_21 + x_31 = 1 
x_12 + x_22 + x_32 = 1 
x_13 + x_23 + x_33 = 1 
End)";

// Once bug: this problem would be detected infeasible?
const char * infeasible_problem_2 = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-1 x_21 - 2 x_22 - 1 x_23 + 3 x_24
-1 x_31 - 1 x_32 - 2 x_33 - 3.5 x_24
Subject To
x_11 + x_12 + x_13 = 1 
x_21 + x_22 + x_23 + x_24 = 2 
x_31 + x_32 + x_33 + x_24 = 1 
x_11 + x_21 + x_31 = 1 
x_12 + x_22 + x_32 = 1 
x_13 + x_23 + x_33 = 1 
Bounds
Binaries
x_11
x_12
x_13
x_21
x_22
x_23
x_24
x_31
x_32
x_33
x_24
End)";

int main(int argc, char** argv)
{
    {
        ILP_input ilp = ILP_parser::parse_string(infeasible_problem);
        ilp.normalize();
        bdd_preprocessor pre(ilp, false, true);
        //bdd_mma<float> solver(pre.get_bdd_collection(), ilp.objective().begin(), ilp.objective().end());
        subgradient<bdd_parallel_mma_base<bdd_branch_instruction<float, uint16_t>>, float> solver(pre.get_bdd_collection(), ilp.objective());
        for(size_t i=0; i<5000; ++i)
        {
            solver.iteration();
            std::cout << "iteration " << i << " lower bound = " << solver.lower_bound() << "\n";
        }
        std::cout << "final lower bound = " << solver.lower_bound() << "\n";
        //test(solver.lower_bound() == std::numeric_limits<double>::infinity());
    }

    {
        ILP_input ilp = ILP_parser::parse_string(infeasible_problem_2);
        ilp.normalize();
        bdd_preprocessor pre(ilp, false, true);
        bdd_mma_base<bdd_branch_instruction_bdd_index<float,uint32_t>> solver(pre.get_bdd_collection(), ilp.objective());
        for(size_t i=0; i<5; ++i)
        {
            solver.iteration();
            std::cout << "iteration " << i << " lower bound = " << solver.lower_bound() << "\n";
        }
        std::cout << "final lower bound = " << solver.lower_bound() << "\n";
        //test(solver.lower_bound() == std::numeric_limits<double>::infinity());
    }
}
