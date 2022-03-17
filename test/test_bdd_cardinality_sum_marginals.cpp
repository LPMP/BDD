#include "bdd_collection/bdd_collection.h"
#include "bdd_manager/bdd.h"
#include "test.h"
#include "bdd_branch_instruction_smooth.h"
#include "bdd_mma_base_smooth.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    std::vector<double> smoothing_values = {1.0, 0.5, 1.5, 0.1, 7.5};

    for(size_t i=2; i<42; ++i)
    {
        for(const double smoothing_val : smoothing_values)
        {
            std::cout << "testing sum marginals for " << i << " variables with smoothing of " << smoothing_val << "\n";
            BDD::bdd_mgr bdd_mgr;
            std::vector<BDD::node_ref> bdd_vars;
            for(size_t j=0; j<i; ++j)
                bdd_vars.push_back(bdd_mgr.projection(j));
            auto cardinality_bdd = bdd_mgr.cardinality(bdd_vars.begin(), bdd_vars.end(), 2);
            BDD::bdd_collection bdd_col;
            const size_t bdd_nr = bdd_col.add_bdd(cardinality_bdd);
            bdd_col.reorder(bdd_nr);
            using smooth_bdd_instr_type = bdd_branch_instruction_smooth_bdd_index<double,uint32_t>;
            bdd_mma_base_smooth<smooth_bdd_instr_type> solver(bdd_col);
            for(size_t j=0; j<i; ++j)
                solver.update_cost(0, j+1, j);
            solver.set_smoothing(smoothing_val);
            const double solver_lb = solver.smooth_lower_bound();
            // lb = -log(exp(-1-2) + ... + exp(-(i-1) -i))
            double expected_lb = 0;
            for(size_t j0=0; j0<i; ++j0)
                for(size_t j1=j0+1; j1<i; ++j1)
                    expected_lb += std::exp((-double(j0+1) - double(j1+1))/smoothing_val);
            expected_lb = -smoothing_val*std::log(expected_lb);
            test(std::abs(expected_lb - solver_lb) <= 1e-6);

            solver.smooth_forward_run();
            solver.compute_smooth_lower_bound_after_forward_pass();
            const double forward_solver_lb = solver.smooth_lower_bound();
            test(std::abs(expected_lb - forward_solver_lb) <= 1e-6);

            // try out same costs but reverse
            for(size_t j=0; j<i; ++j)
                solver.update_cost(0, -double(j+1) + double(i-j), j);

            solver.smooth_backward_run();
            const double solver_lb_2 = solver.smooth_lower_bound();
            test(std::abs(expected_lb - solver_lb_2) <= 1e-6);

            solver.smooth_forward_run();
            solver.compute_smooth_lower_bound_after_forward_pass();
            const double forward_solver_lb_2 = solver.smooth_lower_bound();
            test(std::abs(expected_lb - forward_solver_lb_2) <= 1e-6);
        }
    }
    return 0;
}

