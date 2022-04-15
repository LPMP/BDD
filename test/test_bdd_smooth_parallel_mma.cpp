#include "bdd_parallel_mma_base_smooth.h"
#include "bdd_branch_instruction_smooth.h"
#include "test.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    std::vector<double> omegas = {0.01, 0.1, 0.3, 0.5, 0.75, 1.0};
    for(const double omega : omegas) {
        for(size_t i=2; i<42; ++i)
        {
            BDD::bdd_collection bdd_col;
            const size_t bdd_nr = bdd_col.simplex_constraint(i);
            std::vector<double> costs(i);
            std::iota(costs.begin(), costs.end(), 1);

            // test if lower bound is correctly computed even when substracting deltas

            // test if forward_sm and backward_sm improve lower bound (and omega=1.0 does not change lower bound)
            {
                bdd_parallel_mma_base_smooth<bdd_branch_instruction_smooth<double,uint32_t>> solver(bdd_col);
                solver.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());
                solver.smooth_backward_run();
                solver.compute_smooth_lower_bound_after_backward_pass();
                const double lb_before = solver.smooth_lower_bound();
                std::vector<std::array<double,2>> delta_in(i, {0.0,0.0});
                std::vector<std::array<double,2>> delta_out(i, {0.0,0.0});
                solver.forward_sm(omega, delta_out, delta_in);
                solver.compute_smooth_lower_bound_after_forward_pass();
                const double lb_after = solver.smooth_lower_bound();
                //if(omega == 1.0)
                    test(std::abs(lb_before - lb_after) <= 1e-4);
                //else
                //    test(lb_before <= lb_after + 1e-4);
            }

            {
                bdd_parallel_mma_base_smooth<bdd_branch_instruction_smooth<double,uint32_t>> solver(bdd_col);
                solver.update_costs(costs.begin(), costs.begin(), costs.begin(), costs.end());
                solver.smooth_forward_run();
                solver.compute_smooth_lower_bound_after_forward_pass();
                const double lb_before = solver.smooth_lower_bound();
                std::vector<std::array<double,2>> delta_in(i, {0.0,0.0});
                std::vector<std::array<double,2>> delta_out(i, {0.0,0.0});
                solver.backward_sm(omega, delta_out, delta_in);
                solver.compute_smooth_lower_bound_after_backward_pass();
                const double lb_after = solver.smooth_lower_bound();
                //if(omega == 1.0)
                    test(std::abs(lb_before - lb_after) <= 1e-6);
                //else
                //    test(lb_before <= lb_after + 1e-6);
            }
        }
    }
}

