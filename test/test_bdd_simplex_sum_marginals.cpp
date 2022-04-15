#include "bdd_collection/bdd_collection.h"
#include "test.h"
#include "bdd_branch_instruction_smooth.h"
#include "bdd_mma_base_smooth.h"
#include "bdd_parallel_mma_base_smooth.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    std::vector<double> smoothing_values = {1.0, 0.5, 1.5, 0.1, 7.5};

    for(size_t i=2; i<42; ++i)
    {
        for(const double smoothing_val : smoothing_values)
        {
            std::cout << "testing sum marginals for " << i << " variables with smoothing of " << smoothing_val << "\n";
            BDD::bdd_collection bdd_col;
            const size_t bdd_nr = bdd_col.simplex_constraint(i);

            bdd_mma_base_smooth<bdd_branch_instruction_smooth_bdd_index<double,uint32_t>> solver(bdd_col);
            bdd_parallel_mma_base_smooth<bdd_branch_instruction_smooth<double,uint32_t>> parallel_solver(bdd_col);

            std::vector<double> costs;
            for(size_t j=0; j<i; ++j)
            {
                costs.push_back(j+1);
                solver.update_cost(0, j+1, j);
            }
            parallel_solver.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());
            solver.set_smoothing(smoothing_val);
            parallel_solver.set_smoothing(smoothing_val);

            const double solver_lb = solver.smooth_lower_bound();
            const double parallel_solver_lb = parallel_solver.smooth_lower_bound();
            // lb = -log(exp(-1) + ... + exp(-i))
            double expected_lb = 0;
            for(size_t j=1; j<=i; ++j)
                expected_lb += std::exp(-double(j)/smoothing_val);
            expected_lb = -smoothing_val*std::log(expected_lb);

            test(std::abs(expected_lb - solver_lb) <= 1e-6);
            test(std::abs(expected_lb - parallel_solver_lb) <= 1e-6);

            solver.smooth_forward_run();
            solver.compute_smooth_lower_bound_after_forward_pass();
            const double forward_solver_lb = solver.smooth_lower_bound();
            test(std::abs(expected_lb - forward_solver_lb) <= 1e-6);

            // try out same costs but reverse
            for(size_t j=0; j<i; ++j)
                solver.update_cost(0, -double(j+1) + double(i-j), j);
            for(auto& x : costs)
                x *= -1.0;
            parallel_solver.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());
            for(auto& x : costs)
                x *= -1.0;
            std::reverse(costs.begin(), costs.end());
            parallel_solver.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());

            solver.smooth_backward_run();
            parallel_solver.smooth_backward_run();
            const double solver_lb_2 = solver.smooth_lower_bound();
            const double parallel_solver_lb_2 = parallel_solver.smooth_lower_bound();
            test(std::abs(expected_lb - solver_lb_2) <= 1e-6);
            test(std::abs(expected_lb - parallel_solver_lb_2) <= 1e-6);

            solver.smooth_forward_run();
            solver.compute_smooth_lower_bound_after_forward_pass();
            const double forward_solver_lb_2 = solver.smooth_lower_bound();
            test(std::abs(expected_lb - forward_solver_lb_2) <= 1e-6);
        }
    }
    return 0;
}
