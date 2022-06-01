#include "bdd_parallel_mma_base.h"
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

std::vector<std::array<float,2>> test_mm(const ILP_input& ilp, const std::string direction = "forward")
{
    using bdd_base_type = bdd_parallel_mma_base<bdd_branch_instruction<float,uint16_t>>;

    bdd_preprocessor pre(ilp);
    bdd_base_type solver(pre.get_bdd_collection());
    solver.update_costs(ilp.objective().begin(), ilp.objective().begin(), ilp.objective().begin(), ilp.objective().end());

    if(direction == "forward")
        solver.backward_run();
    else
        solver.forward_run();

    const double lb_before = solver.lower_bound();

    std::vector<std::array<float,2>> mms_to_collect(solver.nr_variables());
    std::vector<std::array<float,2>> mms_to_collect2(solver.nr_variables());
    std::vector<std::array<float,2>> mms_to_distribute(solver.nr_variables());
    for(auto& x : mms_to_collect)
    {
        x[0] = 0.0;
        x[1] = 0.0;
    }
    for(auto& x : mms_to_collect2)
    {
        x[0] = 0.0;
        x[1] = 0.0;
    }
    for(auto& x : mms_to_distribute)
    {
        x[0] = 0.0;
        x[1] = 0.0;
    }

    for(size_t bdd_nr=0; bdd_nr<solver.nr_bdds(); ++bdd_nr)
        if(direction == "forward")
            solver.forward_mm(bdd_nr, 1.0, mms_to_collect, mms_to_distribute);
        else
            solver.backward_mm(bdd_nr, 1.0, mms_to_collect, mms_to_distribute);

    const double lb_after = solver.lower_bound();
    test(std::abs(lb_before - lb_after) <= 1e-6);

    for(size_t i=0; i<mms_to_collect.size(); ++i)
    {
        test(mms_to_collect[i][0] >= 0.0 ||  mms_to_collect[i][0] == std::numeric_limits<float>::infinity());
        test(mms_to_collect[i][1] >= 0.0 ||  mms_to_collect[i][1] == std::numeric_limits<float>::infinity());
    }

    for(size_t bdd_nr=0; bdd_nr<solver.nr_bdds(); ++bdd_nr)
        if(direction == "forward")
            solver.backward_mm(bdd_nr, 1.0, mms_to_collect2, mms_to_distribute);
        else
            solver.forward_mm(bdd_nr, 1.0, mms_to_collect2, mms_to_distribute);

    for(size_t i=0; i<mms_to_collect2.size(); ++i)
    {
        test(std::abs(mms_to_collect2[i][0] - 0.0) <= 1e-6 || mms_to_collect2[i][0] == std::numeric_limits<float>::infinity());
        test(std::abs(mms_to_collect2[i][1] - 0.0) <= 1e-6 || mms_to_collect2[i][1] == std::numeric_limits<float>::infinity());
    }

    const double lb_after2 = solver.lower_bound();
    test(std::abs(lb_before - lb_after2) <= 1e-6);

    return mms_to_collect;
}

int main(int argc, char** argv)
{
    using bdd_base_type = bdd_parallel_mma_base<bdd_branch_instruction<float,uint16_t>>;
    const ILP_input ilp = ILP_parser::parse_string(two_simplex_problem);
    bdd_preprocessor pre(ilp);

    // forward incremental mm
    {
        const auto mms = test_mm(ilp, "forward");

        test(std::abs(mms[0][1] - mms[0][0] - (1 - 0)) <= 1e-6);
        test(std::abs(mms[1][1] - mms[1][0] - (1 - 1)) <= 1e-6);
        test(std::abs(mms[2][1] - mms[2][0] - (1 - 1)) <= 1e-6);

        test(std::abs(mms[3][1] - mms[3][0] - (1-1 - (2-1))) <= 1e-6); // cost of x_4 becomes 2
        test(std::abs(mms[4][1] - mms[4][0] - (2-1 - (2-1))) <= 1e-6);
        test(std::abs(mms[5][1] - mms[5][0] - (2-1 - (2+2))) <= 1e-6);
    }

    // backward incremental mm
    {
        const auto mms = test_mm(ilp, "backward");

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
        test_mm(ilp, "forward");
        test_mm(ilp, "backward");
    }
}
