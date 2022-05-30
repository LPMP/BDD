#include "bdd_solver.h"
#include <cstdlib>
#include "test.h"

using namespace LPMP;

const char * short_mrf_chain = 
R"(Minimize
3 mu_1_0 + 1 mu_1_1
- 1 mu_2_0 + 0 mu_2_1
+ 1 mu_00 + 2 mu_10 + 1 mu_01 + 0 mu_11
Subject To
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_00 + mu_10 + mu_01 + mu_11 = 1
mu_1_0 - mu_00 - mu_01 = 0
mu_1_1 - mu_10 - mu_11 = 0
mu_2_0 - mu_00 - mu_10 = 0
mu_2_1 - mu_01 - mu_11 = 0
End)";

int main(int argc, char** argv)
{
    std::vector<std::string> args_input_order = {
        "--input_string", short_mrf_chain,
        "-s", "mma",
        "-o", "input"
    };
    bdd_solver solver_input_order((bdd_solver_options(args_input_order)));
    solver_input_order.solve();

    std::vector<std::string> args_bfs_order = {
        "--input_string", short_mrf_chain,
        "-s", "mma",
        "-o", "bfs"
    };
    bdd_solver solver_bfs_order((bdd_solver_options(args_bfs_order)));
    solver_bfs_order.solve();

    test(std::abs(solver_input_order.lower_bound() - solver_bfs_order.lower_bound()) <= 1e-6);

    // check min_marginals invariance
    const auto mm_input_order = solver_input_order.min_marginals();
    const auto mm_bfs_order = solver_input_order.min_marginals();
    test(mm_input_order.size() == mm_bfs_order.size());
    for(size_t i=0; i<mm_input_order.size(); ++i)
    {
        test(mm_input_order.size(i) == mm_bfs_order.size(i));
        for(size_t j=0; j<mm_input_order.size(i); ++j)
        {
            auto sign = [](auto val) -> int {
                return (decltype(val)(0) < val) - (val < decltype(val)(0));
            };

            const double cur_mm_input_order = mm_input_order(i,j)[1] - mm_input_order(i,j)[0];
            const double cur_mm_bfs_order = mm_bfs_order(i,j)[1] - mm_bfs_order(i,j)[0];
            test(
                    (std::abs(cur_mm_input_order) <= 1e-6 && std::abs(cur_mm_bfs_order) <= 1e-6)
                ||
                sign(cur_mm_input_order) == sign(cur_mm_bfs_order)
                );
        }
    }

    // test variable fixation invariance
    solver_input_order.fix_variable("mu_2_1", false);
    solver_input_order.solve();
    solver_bfs_order.fix_variable("mu_2_1", false);
    solver_bfs_order.solve();
    // optimal solution (1,0);
    test(std::abs(solver_input_order.lower_bound() - (1.0 - 1.0 + 2.0)) <= 1e-6);
    test(std::abs(solver_bfs_order.lower_bound() - (1.0 - 1.0 + 2.0)) <= 1e-6);

    solver_input_order.fix_variable("mu_1_1", false);
    solver_input_order.solve();
    solver_bfs_order.fix_variable("mu_1_1", false);
    solver_bfs_order.solve();
    // optimal solution (0,0);
    test(std::abs(solver_input_order.lower_bound() - (3.0 - 1.0 + 1.0)) <= 1e-6);
    test(std::abs(solver_bfs_order.lower_bound() - (3.0 - 1.0 + 1.0)) <= 1e-6);
}
