#include "bdd_cuda_base.h"
#include "ILP_parser.h"
#include "bdd_collection/bdd_collection.h"
#include "bdd_preprocessor.h"
#include "test.h"

using namespace LPMP;
using namespace BDD;

const char * two_simplex_diff_size_problem = 
R"(Minimize
2 x_1 + 1 x_2 + 1.5 x_3
+2 x_4 + 2 x_5 + 3 x_6
Subject To
x_1 + x_2 + x_3 + x_4 = 1
x_4 + x_5 + x_6 = 2
End)";

const char * two_simplex_non_unique_sols = 
R"(Minimize
1 x_1 + 1 x_2 + 1 x_3
+2 x_4 + 1 x_5 + 1 x_6
Subject To
x_1 + x_2 + x_3 + x_4 = 1
x_4 + x_5 + x_6 = 2
End)";

int main(int argc, char** argv)
{
    {
        ILP_input ilp = ILP_parser::parse_string(two_simplex_diff_size_problem);
        bdd_preprocessor bdd_pre(ilp);
        bdd_collection bdd_col = bdd_pre.get_bdd_collection();
        bdd_cuda_base<float> bcb(bdd_col);

        test(bcb.nr_variables() == 6);
        test(bcb.nr_bdds() == 2);

        for(size_t i=0; i<bcb.nr_variables(); ++i)
            bcb.set_cost(ilp.objective()[i], i);

        const auto bdd_sol = bcb.bdds_solution();
        test(bdd_sol.size() == 6);
        for(size_t i=0; i<6; ++i)
        {
            if (i != 3)
                test(bdd_sol.size(i) == 1);
            else
                test(bdd_sol.size(i) == 2);
        }

        test(bdd_sol(0,0) == 0);
        test(bdd_sol(1,0) == 1);
        test(bdd_sol(2,0) == 0);
        test(bdd_sol(3,0) == 0);

        test(bdd_sol(3,1) == 1);
        test(bdd_sol(4,0) == 1);
        test(bdd_sol(5,0) == 0);
    }

    {
        ILP_input ilp = ILP_parser::parse_string(two_simplex_non_unique_sols);
        bdd_preprocessor bdd_pre(ilp);
        bdd_collection bdd_col = bdd_pre.get_bdd_collection();
        bdd_cuda_base<float> bcb(bdd_col);

        test(bcb.nr_variables() == 6);
        test(bcb.nr_bdds() == 2);

        for(size_t i=0; i<bcb.nr_variables(); ++i)
            bcb.set_cost(ilp.objective()[i], i);

        const auto bdd_sol = bcb.bdds_solution();
        test(bdd_sol.size() == 6);
        for(size_t i=0; i<6; ++i)
        {
            if (i != 3)
                test(bdd_sol.size(i) == 1);
            else
                test(bdd_sol.size(i) == 2);
        }

        test(bdd_sol(0,0) + bdd_sol(1,0) + bdd_sol(2,0) + bdd_sol(3,0) == 1);
        test(bdd_sol(3,1) + bdd_sol(4,0) + bdd_sol(5,0) == 2);
    }
}
