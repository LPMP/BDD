#include "bdd_sequential_base.h"
#include "bdd_branch_instruction.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "test_problem_generator.h"
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

int main(int argc, char** argv)
{
    using bdd_base_type = bdd_sequential_base<bdd_branch_instruction<float>>;
    const ILP_input ilp = ILP_parser::parse_string(two_simplex_problem);
    bdd_preprocessor pre(ilp);

    // forward incremental mm
    {
        bdd_base_type solver(pre.get_bdd_collection());
        solver.set_costs(ilp.objective().begin(), ilp.objective().end());

        solver.backward_run();
        const double lb_before = solver.lower_bound();

        std::vector<std::array<std::atomic<float>,2>> mms(ilp.nr_variables());
        for(auto& x : mms)
        {
            x[0] = 0.0;
            x[1] = 0.0;
        }
        for(size_t bdd_nr=0; bdd_nr<solver.nr_bdds(); ++bdd_nr)
            solver.forward_mm(bdd_nr, 1.0, mms);

        const double lb_after = solver.lower_bound();
        test(std::abs(lb_before - lb_after) <= 1e-6);

        for(size_t i=0; i<mms.size(); ++i)
            test(mms[i][0] >= 0.0 && mms[i][1] >= 0.0);

        test(std::abs(mms[0][1] - mms[0][0] - (1 - 0)) <= 1e-6);
        test(std::abs(mms[1][1] - mms[1][0] - (1 - 1)) <= 1e-6);
        test(std::abs(mms[2][1] - mms[2][0] - (1 - 1)) <= 1e-6);

        test(std::abs(mms[3][1] - mms[3][0] - (1-1 - (2-1))) <= 1e-6); // cost of x_4 becomes 2
        test(std::abs(mms[4][1] - mms[4][0] - (2-1 - (2-1))) <= 1e-6);
        test(std::abs(mms[5][1] - mms[5][0] - (2-1 - (2+2))) <= 1e-6);
    }

    // backward incremental mm
    {
        bdd_base_type solver(pre.get_bdd_collection());
        solver.set_costs(ilp.objective().begin(), ilp.objective().end());

        solver.forward_run();
        const double lb_before = solver.lower_bound();

        std::vector<std::array<std::atomic<float>,2>> mms(ilp.nr_variables());
        for(auto& x : mms)
        {
            x[0] = 0.0;
            x[1] = 0.0;
        }

        for(size_t bdd_nr=0; bdd_nr<solver.nr_bdds(); ++bdd_nr)
            solver.backward_mm(bdd_nr, 1.0, mms);

        const double lb_after = solver.lower_bound();
        test(std::abs(lb_before - lb_after) <= 1e-6);

        test(std::abs(mms[2][1] - mms[2][0] - (1 - 1)) <= 1e-6);
        test(std::abs(mms[1][1] - mms[1][0] - (1 - 1)) <= 1e-6);
        test(std::abs(mms[0][1] - mms[0][0] - (1 - 0)) <= 1e-6);

        test(std::abs(mms[5][1] - mms[5][0] - (1-1 - (2+1))) <= 1e-6); // x_6 has now cost 1
        test(std::abs(mms[4][1] - mms[4][0] - (2+1 - (2+1))) <= 1e-6);
        test(std::abs(mms[3][1] - mms[3][0] - (1+1 - (2+1))) <= 1e-6);
    }

    // random inequalities
    for(size_t nr_vars=2; nr_vars<50; ++nr_vars)
    {
        const auto [coefficients, ineq, rhs] = generate_random_inequality(nr_vars);
        ILP_input ilp = generate_ILP(coefficients, ineq, rhs);
        bdd_preprocessor pre(ilp);

        // forward incremental mm
        {
            bdd_base_type solver(pre.get_bdd_collection());
            solver.set_costs(ilp.objective().begin(), ilp.objective().end());

            solver.backward_run();
            const double lb_before = solver.lower_bound();

            std::vector<std::array<std::atomic<float>,2>> mms(ilp.nr_variables());
            for(auto& x : mms)
            {
                x[0] = 0.0;
                x[1] = 0.0;
            }
            for(size_t bdd_nr=0; bdd_nr<solver.nr_bdds(); ++bdd_nr)
                solver.forward_mm(bdd_nr, 1.0, mms);

            const double lb_after = solver.lower_bound();
            test(std::abs(lb_before - lb_after) <= 1e-6);

        }

        // backward incremental mm
        {
            bdd_base_type solver(pre.get_bdd_collection());
            solver.set_costs(ilp.objective().begin(), ilp.objective().end());

            solver.forward_run();
            const double lb_before = solver.lower_bound();

            std::vector<std::array<std::atomic<float>,2>> mms(ilp.nr_variables());
            for(auto& x : mms)
            {
                x[0] = 0.0;
                x[1] = 0.0;
            }
            for(size_t bdd_nr=0; bdd_nr<solver.nr_bdds(); ++bdd_nr)
                solver.backward_mm(bdd_nr, 1.0, mms);

            const double lb_after = solver.lower_bound();
            test(std::abs(lb_before - lb_after) <= 1e-6);
        }
    }
}
