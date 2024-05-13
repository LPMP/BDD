#include "bdd_solver/bdd_cuda_base.h"
#include "ILP/ILP_parser.h"
#include "bdd_conversion/bdd_preprocessor.h"
#include "test.h"

using namespace LPMP;

const char * one_simplex_problem = 
R"(Minimize
1 x_1 + 2 x_2 + 1 x_3
Subject To
x_1 + x_2 + x_3 = 1
End
)";

const char * two_simplex_problem = 
R"(Minimize
2 x_1 + 3 x_2 + 4 x_3
+1 x_4 + 2 x_5 - 1 x_6
Subject To  
x_1 + x_2 + x_3 = 2
x_4 + x_5 + x_6 = 1
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

int main(int argc, char** argv)
{
    {
        const ILP_input ilp = ILP_parser::parse_string(one_simplex_problem);
        bdd_preprocessor pre(ilp);

        bdd_cuda_base<float> solver(pre.get_bdd_collection(), ilp.objective());

        const auto mms = solver.sum_marginals(true);
    }

    {
        const ILP_input ilp = ILP_parser::parse_string(two_simplex_problem);
        bdd_preprocessor pre(ilp);

        bdd_cuda_base<float> solver(pre.get_bdd_collection(), ilp.objective());

        {
            const auto mms = solver.sum_marginals(false); // unnormalized probabilities
            test(mms.size() == 6);
            for(size_t i=0; i<6; ++i)
                test(mms.size(i) == 1);

            test(std::abs(mms(0,0)[0] - std::exp(-3 - 4)) <= 1e-5 && std::abs(mms(0,0)[1] - (std::exp(-2 - 3) + std::exp(-2 - 4))) <= 1e-5);
            test(std::abs(mms(1,0)[0] - std::exp(-2 - 4)) <= 1e-5 && std::abs(mms(1,0)[1] - (std::exp(-3 - 2) + std::exp(-3 - 4))) <= 1e-5);
            test(std::abs(mms(2,0)[0] - std::exp(-2 - 3)) <= 1e-5 && std::abs(mms(2,0)[1] - (std::exp(-4 - 2) + std::exp(-4 - 3))) <= 1e-5);

            test(std::abs(mms(3,0)[0] - (std::exp(-2) + std::exp(1)))  <= 1e-5 && std::abs(mms(3,0)[1] - std::exp(-1.0)) <= 1e-5);
            test(std::abs(mms(4,0)[0] - (std::exp(-1) + std::exp(+1))) <= 1e-5 && std::abs(mms(4,0)[1] - std::exp(-2.0)) <= 1e-5);
            test(std::abs(mms(5,0)[0] - (std::exp(-1) + std::exp(-2))) <= 1e-5 && std::abs(mms(5,0)[1] - std::exp(1.0)) <= 1e-5);
        }

        {
            const auto mms = solver.sum_marginals(true); // logits
            test(mms.size() == 6);
            for(size_t i=0; i<6; ++i)
                test(mms.size(i) == 1);

            test(std::abs(std::exp(mms(0,0)[0]) - std::exp(-3 - 4)) <= 1e-5 && std::abs(std::exp(mms(0,0)[1]) - (std::exp(-2 - 3) + std::exp(-2 - 4))) <= 1e-5);
            test(std::abs(std::exp(mms(1,0)[0]) - std::exp(-2 - 4)) <= 1e-5 && std::abs(std::exp(mms(1,0)[1]) - (std::exp(-3 - 2) + std::exp(-3 - 4))) <= 1e-5);
            test(std::abs(std::exp(mms(2,0)[0]) - std::exp(-2 - 3)) <= 1e-5 && std::abs(std::exp(mms(2,0)[1]) - (std::exp(-4 - 2) + std::exp(-4 - 3))) <= 1e-5);

            test(std::abs(std::exp(mms(3,0)[0]) - (std::exp(-2) + std::exp(1)))  <= 1e-5 && std::abs(std::exp(mms(3,0)[1]) - std::exp(-1.0)) <= 1e-5);
            test(std::abs(std::exp(mms(4,0)[0]) - (std::exp(-1) + std::exp(+1))) <= 1e-5 && std::abs(std::exp(mms(4,0)[1]) - std::exp(-2.0)) <= 1e-5);
            test(std::abs(std::exp(mms(5,0)[0]) - (std::exp(-1) + std::exp(-2))) <= 1e-5 && std::abs(std::exp(mms(5,0)[1]) - std::exp(1.0)) <= 1e-5);
        }
    }

    {
        const ILP_input ilp = ILP_parser::parse_string(short_chain_shuffled);
        bdd_preprocessor pre(ilp);

        bdd_cuda_base<float> solver(pre.get_bdd_collection(), ilp.objective());

        const auto mms = solver.sum_marginals(false); // unnormalized probabilities
    }
}