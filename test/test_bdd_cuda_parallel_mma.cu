#include "bdd_cuda_parallel_mma.h"
#include "ILP_parser.h"
#include "bdd_collection/bdd_collection.h"
#include "bdd_preprocessor.h"
#include "test.h"

using namespace LPMP;
using namespace BDD;

const char * problem_with_fixed_var = 
R"(Minimize
2 x_1 + 1 x_2 + 1 x_3 + 1 x_4 + 2 x_5 + x_6
Subject To
x_1 + x_2 + 2 x_3 = 1
x_3 + x_4 + x_5 = 2
x_5 + x_6 = 2
End)";

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

const char * short_chain_shuffled = 
R"(Minimize
+ 1 mu_2_1 + 1 mu_10 + 0 mu_1_1 + 0 mu_11
-1 mu_1_0 + 1 mu_00 + 2 mu_01 + 2 mu_2_0
Subject To
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_00 + mu_10 + mu_01 + mu_11 = 1
mu_1_0 - mu_00 - mu_01 = 0
mu_1_1 - mu_10 - mu_11 = 0
mu_2_0 - mu_00 - mu_10 = 0
mu_2_1 - mu_01 - mu_11 = 0
End)";

const char * long_chain = 
R"(Minimize
2 mu_0_0 - 1 mu_0_1 + 3 mu_1_0 - 1 mu_1_1
+ 3 mu_2_0 + 2 mu_2_1 - 1 mu_3_0 - 2 mu_3_1
- 2 mu_4_0 - 1 mu_4_1 + 1 mu_5_0 - 1 mu_5_1
+ 1 mu_6_0 + 1 mu_6_1 - 3 mu_7_0 + 2 mu_7_1
+ 0 mu_8_0 + 2 mu_8_1
+ 1 mu_01_00 - 2 mu_01_01 + 2 mu_01_10 - 1 mu_01_11
+ 0 mu_12_00 - 1 mu_12_01 + 1 mu_12_10 + 0 mu_12_11
- 1 mu_23_00 + 2 mu_23_01 + 1 mu_23_10 - 2 mu_23_11
+ 2 mu_34_00 + 0 mu_34_01 + 2 mu_34_10 + 2 mu_34_11
+ 1 mu_45_00 - 2 mu_45_01 - 3 mu_45_10 - 1 mu_45_11
- 2 mu_56_00 + 0 mu_56_01 + 1 mu_56_10 + 3 mu_56_11
- 1 mu_67_00 - 2 mu_67_01 - 1 mu_67_10 - 1 mu_67_11
+ 2 mu_78_00 + 0 mu_78_01 + 2 mu_78_10 + 3 mu_78_11
Subject To
mu_0_0 + mu_0_1 = 1
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_3_0 + mu_3_1 = 1
mu_4_0 + mu_4_1 = 1
mu_5_0 + mu_5_1 = 1
mu_6_0 + mu_6_1 = 1
mu_7_0 + mu_7_1 = 1
mu_8_0 + mu_8_1 = 1
mu_01_00 + mu_01_10 + mu_01_01 + mu_01_11 = 1
mu_12_00 + mu_12_10 + mu_12_01 + mu_12_11 = 1
mu_23_00 + mu_23_10 + mu_23_01 + mu_23_11 = 1
mu_34_00 + mu_34_10 + mu_34_01 + mu_34_11 = 1
mu_45_00 + mu_45_10 + mu_45_01 + mu_45_11 = 1
mu_56_00 + mu_56_10 + mu_56_01 + mu_56_11 = 1
mu_67_00 + mu_67_10 + mu_67_01 + mu_67_11 = 1
mu_78_00 + mu_78_10 + mu_78_01 + mu_78_11 = 1
mu_0_0 - mu_01_00 - mu_01_01 = 0
mu_0_1 - mu_01_10 - mu_01_11 = 0
mu_1_0 - mu_01_00 - mu_01_10 = 0
mu_1_1 - mu_01_01 - mu_01_11 = 0
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_2_0 - mu_23_00 - mu_23_01 = 0
mu_2_1 - mu_23_10 - mu_23_11 = 0
mu_3_0 - mu_23_00 - mu_23_10 = 0
mu_3_1 - mu_23_01 - mu_23_11 = 0
mu_3_0 - mu_34_00 - mu_34_01 = 0
mu_3_1 - mu_34_10 - mu_34_11 = 0
mu_4_0 - mu_34_00 - mu_34_10 = 0
mu_4_1 - mu_34_01 - mu_34_11 = 0
mu_4_0 - mu_45_00 - mu_45_01 = 0
mu_4_1 - mu_45_10 - mu_45_11 = 0
mu_5_0 - mu_45_00 - mu_45_10 = 0
mu_5_1 - mu_45_01 - mu_45_11 = 0
mu_5_0 - mu_56_00 - mu_56_01 = 0
mu_5_1 - mu_56_10 - mu_56_11 = 0
mu_6_0 - mu_56_00 - mu_56_10 = 0
mu_6_1 - mu_56_01 - mu_56_11 = 0
mu_6_0 - mu_67_00 - mu_67_01 = 0
mu_6_1 - mu_67_10 - mu_67_11 = 0
mu_7_0 - mu_67_00 - mu_67_10 = 0
mu_7_1 - mu_67_01 - mu_67_11 = 0
mu_7_0 - mu_78_00 - mu_78_01 = 0
mu_7_1 - mu_78_10 - mu_78_11 = 0
mu_8_0 - mu_78_00 - mu_78_10 = 0
mu_8_1 - mu_78_01 - mu_78_11 = 0
End)";

const char * grid_graph_3x3 = 
R"(Minimize
2 mu_0_0 - 1 mu_0_1 + 3 mu_1_0 - 1 mu_1_1
+ 3 mu_2_0 + 2 mu_2_1 - 1 mu_3_0 - 2 mu_3_1
- 2 mu_4_0 - 1 mu_4_1 + 3 mu_5_0 - 1 mu_5_1
+ 1 mu_6_0 + 1 mu_6_1 - 3 mu_7_0 + 2 mu_7_1
+ 0 mu_8_0 + 2 mu_8_1
+ 1 mu_01_00 - 2 mu_01_01 + 2 mu_01_10 - 1 mu_01_11
+ 0 mu_12_00 + 1 mu_12_01 + 1 mu_12_10 + 0 mu_12_11
- 1 mu_03_00 + 2 mu_03_01 + 0 mu_03_10 - 2 mu_03_11
+ 2 mu_14_00 + 0 mu_14_01 + 2 mu_14_10 + 2 mu_14_11
+ 1 mu_25_00 - 2 mu_25_01 - 3 mu_25_10 - 1 mu_25_11
+ 0 mu_34_00 + 1 mu_34_01 + 1 mu_34_10 + 1 mu_34_11
- 1 mu_45_00 - 2 mu_45_01 + 4 mu_45_10 - 2 mu_45_11
- 2 mu_36_00 + 0 mu_36_01 + 1 mu_36_10 + 3 mu_36_11
+ 3 mu_47_00 - 2 mu_47_01 - 2 mu_47_10 - 1 mu_47_11
+ 0 mu_58_00 + 1 mu_58_01 + 1 mu_58_10 + 1 mu_58_11
- 1 mu_67_00 + 2 mu_67_01 - 1 mu_67_10 - 1 mu_67_11
+ 2 mu_78_00 + 0 mu_78_01 + 2 mu_78_10 + 2 mu_78_11
Subject To
mu_0_0 + mu_0_1 = 1
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_3_0 + mu_3_1 = 1
mu_4_0 + mu_4_1 = 1
mu_5_0 + mu_5_1 = 1
mu_6_0 + mu_6_1 = 1
mu_7_0 + mu_7_1 = 1
mu_8_0 + mu_8_1 = 1
mu_01_00 + mu_01_10 + mu_01_01 + mu_01_11 = 1
mu_12_00 + mu_12_10 + mu_12_01 + mu_12_11 = 1
mu_03_00 + mu_03_10 + mu_03_01 + mu_03_11 = 1
mu_14_00 + mu_14_10 + mu_14_01 + mu_14_11 = 1
mu_25_00 + mu_25_10 + mu_25_01 + mu_25_11 = 1
mu_34_00 + mu_34_10 + mu_34_01 + mu_34_11 = 1
mu_45_00 + mu_45_10 + mu_45_01 + mu_45_11 = 1
mu_36_00 + mu_36_10 + mu_36_01 + mu_36_11 = 1
mu_47_00 + mu_47_10 + mu_47_01 + mu_47_11 = 1
mu_58_00 + mu_58_10 + mu_58_01 + mu_58_11 = 1
mu_67_00 + mu_67_10 + mu_67_01 + mu_67_11 = 1
mu_78_00 + mu_78_10 + mu_78_01 + mu_78_11 = 1
mu_0_0 - mu_01_00 - mu_01_01 = 0
mu_0_1 - mu_01_10 - mu_01_11 = 0
mu_0_0 - mu_03_00 - mu_03_01 = 0
mu_0_1 - mu_03_10 - mu_03_11 = 0
mu_1_0 - mu_01_00 - mu_01_10 = 0
mu_1_1 - mu_01_01 - mu_01_11 = 0
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_1_0 - mu_14_00 - mu_14_01 = 0
mu_1_1 - mu_14_10 - mu_14_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_2_0 - mu_25_00 - mu_25_01 = 0
mu_2_1 - mu_25_10 - mu_25_11 = 0
mu_3_0 - mu_03_00 - mu_03_10 = 0
mu_3_1 - mu_03_01 - mu_03_11 = 0
mu_3_0 - mu_34_00 - mu_34_01 = 0
mu_3_1 - mu_34_10 - mu_34_11 = 0
mu_3_0 - mu_36_00 - mu_36_01 = 0
mu_3_1 - mu_36_10 - mu_36_11 = 0
mu_4_0 - mu_14_00 - mu_14_10 = 0
mu_4_1 - mu_14_01 - mu_14_11 = 0
mu_4_0 - mu_34_00 - mu_34_10 = 0
mu_4_1 - mu_34_01 - mu_34_11 = 0
mu_4_0 - mu_45_00 - mu_45_01 = 0
mu_4_1 - mu_45_10 - mu_45_11 = 0
mu_4_0 - mu_47_00 - mu_47_01 = 0
mu_4_1 - mu_47_10 - mu_47_11 = 0
mu_5_0 - mu_25_00 - mu_25_10 = 0
mu_5_1 - mu_25_01 - mu_25_11 = 0
mu_5_0 - mu_45_00 - mu_45_10 = 0
mu_5_1 - mu_45_01 - mu_45_11 = 0
mu_5_0 - mu_58_00 - mu_58_01 = 0
mu_5_1 - mu_58_10 - mu_58_11 = 0
mu_6_0 - mu_36_00 - mu_36_10 = 0
mu_6_1 - mu_36_01 - mu_36_11 = 0
mu_6_0 - mu_67_00 - mu_67_01 = 0
mu_6_1 - mu_67_10 - mu_67_11 = 0
mu_7_0 - mu_47_00 - mu_47_10 = 0
mu_7_1 - mu_47_01 - mu_47_11 = 0
mu_7_0 - mu_67_00 - mu_67_10 = 0
mu_7_1 - mu_67_01 - mu_67_11 = 0
mu_7_0 - mu_78_00 - mu_78_01 = 0
mu_7_1 - mu_78_10 - mu_78_11 = 0
mu_8_0 - mu_58_00 - mu_58_10 = 0
mu_8_1 - mu_58_01 - mu_58_11 = 0
mu_8_0 - mu_78_00 - mu_78_10 = 0
mu_8_1 - mu_78_01 - mu_78_11 = 0
End)";


void test_problem(const char* instance, const double expected_lb, const double tol = 1e-12, const bool test_primal_obj = true)
{
    ILP_input ilp = ILP_parser::parse_string(instance);
    bdd_preprocessor bdd_pre(ilp);
    bdd_collection bdd_col = bdd_pre.get_bdd_collection();
    bdd_cuda_parallel_mma<double> solver(bdd_col);

    for(size_t i=0; i<solver.nr_variables(); ++i)
        solver.set_cost(ilp.objective()[i], i);

    std::vector<double> cost_vector_before = solver.compute_primal_objective_vector();
    for(size_t i=0; i<solver.nr_variables(); ++i)
    {
        const auto diff = std::abs(ilp.objective()[i] - cost_vector_before[i]);
        std::stringstream buffer;
        buffer<<i<<" "<<ilp.objective()[i]<<" "<<cost_vector_before[i]<<" "<<diff<<"\n";
        test(diff <= tol, buffer.str());
    }

    for(size_t iter=0; iter<5; ++iter)
    {
        solver.iteration();
        std::cout<<"Iter: "<<iter<<", LB: "<<solver.lower_bound()<<std::endl;
    }
    
    std::cout<<"LB before distribute delta: "<<solver.lower_bound()<<", Expected: "<<expected_lb<<"\n";
    solver.distribute_delta();
    std::cout<<"Final lower bound: "<<solver.lower_bound()<<", Expected: "<<expected_lb<<"\n";
    test(std::abs(solver.lower_bound() - expected_lb) <= tol);

    if (test_primal_obj)
    {
        std::vector<double> cost_vector_after = solver.compute_primal_objective_vector();
        for(size_t i=0; i<solver.nr_variables(); ++i)
        {
            const auto diff = std::abs(ilp.objective()[i] - cost_vector_after[i]);
            std::stringstream buffer;
            buffer<<i<<" "<<ilp.objective()[i]<<" "<<cost_vector_before[i]<<" "<<diff<<"\n";
            test(diff <= tol, buffer.str());
        }
    }
}

int main(int argc, char** argv)
{
    std::cout<<"problem_with_fixed_var\n";
    test_problem(problem_with_fixed_var, 5.0, false); // Primal objective vector cannot be reconstructed as some variables assignments are infeasible.
    std::cout<<"matching_3x3"<<"\n";
    test_problem(matching_3x3, -6.0);
    std::cout<<"short_chain_shuffled"<<"\n";
    test_problem(short_chain_shuffled, 1.0);
    std::cout<<"long_chain"<<"\n";
    test_problem(long_chain, -9.0);
    std::cout<<"grid_graph_3x3"<<"\n";
    test_problem(grid_graph_3x3, -8.0);
}

