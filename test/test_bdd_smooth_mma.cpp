#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "test.h"
#include "bdd_branch_instruction_smooth.h"
#include "bdd_mma_base_smooth.h"

using namespace LPMP;

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

const char * matching_3x3_non_diag = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-2 x_21 - 2 x_22 - 1 x_23
-1 x_31 - 1 x_32 - 2 x_33
Subject To
x_11 + x_12 + x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
x_13 + x_23 + x_33 = 1
End)";

void run_smooth_instance(const std::string& input)
{
    const ILP_input ilp = ILP_parser::parse_string(input);
    bdd_preprocessor bdd_pre(ilp);
    using smooth_bdd_instr_type = bdd_branch_instruction_smooth_bdd_index<double,uint32_t>;
    bdd_mma_base_smooth<smooth_bdd_instr_type> solver(bdd_pre.get_bdd_collection());
    for(size_t i=0; i<ilp.nr_variables(); ++i)
        solver.update_cost(0, ilp.objective(i), i);
    for(size_t iter=0; iter<100; ++iter)
    {
        solver.smooth_iteration();
        const double solver_lb = solver.smooth_lower_bound();
        std::cout << "iteration = " << iter << " smooth lb = " << solver_lb << "\n";
    }
    const double solver_lb = solver.smooth_lower_bound();
    std::cout << "final smooth lb = " << solver_lb << "\n";
    std::cout << "final lb = " << solver.lower_bound() << "\n";
}

int main(int argc, char** argv)
{
    run_smooth_instance(matching_3x3_diag);
    run_smooth_instance(matching_3x3_non_diag);
}
