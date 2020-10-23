#include "bdd_solver.h"
#include "test.h"

using namespace LPMP;

std::string test_instance = R"(Minimize
- 2 x1 - 2 x2 - x3 - x4
Subject To
2 x1 + 3 x2 + 4 x3 + x4 <= 5
Bounds
Binaries
x1
x2
x3
x4
End)";

void test_ILP(const std::string ilp, const double expected_lb, const permutation order)
{
    ILP_input ilp_input = ILP_parser::parse_string(ilp);
    ilp_input.reorder(order);
    std::stringstream ss;
    ilp_input.write(ss);
    const std::string ilp_string = ss.str();
    bdd_solver solver({
            "--input_string", ilp_string,
            "-s", "decomposition_mma",
            "--nr_threads", "2",
            "--max_iter", "20",
            "--parallel_message_passing_weight", "1.0"
            });

        solver.solve();
        test(std::abs(solver.lower_bound() - expected_lb) <= 1e-8);
}

int main(int argc, char** argv)
{
    test_ILP(test_instance, -4, {0,1,2,3});
    test_ILP(test_instance, -4, {0,1,2,3});
    test_ILP(test_instance, -4, {0,1,3,2});
    test_ILP(test_instance, -4, {0,2,1,3});
    test_ILP(test_instance, -4, {0,2,3,1});
    test_ILP(test_instance, -4, {0,3,1,2});
    test_ILP(test_instance, -4, {0,3,2,1});
    test_ILP(test_instance, -4, {1,0,2,3});
    test_ILP(test_instance, -4, {1,0,3,2});
    test_ILP(test_instance, -4, {1,2,0,3});
    test_ILP(test_instance, -4, {1,2,3,0});
    test_ILP(test_instance, -4, {1,3,0,2});
    test_ILP(test_instance, -4, {1,3,2,0});
    test_ILP(test_instance, -4, {2,0,1,3});
    test_ILP(test_instance, -4, {2,0,3,1});
    test_ILP(test_instance, -4, {2,1,0,3});
    test_ILP(test_instance, -4, {2,1,3,0});
    test_ILP(test_instance, -4, {2,3,0,1});
    test_ILP(test_instance, -4, {2,3,1,0});
    test_ILP(test_instance, -4, {3,0,1,2});
    test_ILP(test_instance, -4, {3,0,2,1});
    test_ILP(test_instance, -4, {3,1,0,2});
    test_ILP(test_instance, -4, {3,1,2,0});
    test_ILP(test_instance, -4, {3,2,0,1});
    test_ILP(test_instance, -4, {3,2,1,0});
}
