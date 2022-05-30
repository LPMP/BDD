#include "bdd_mma.h"
#include "bdd_parallel_mma.h"
#include "bdd_cuda.h"
#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "test.h"
#include "test_problems.h"

using namespace LPMP;

template<typename SOLVER>
void test_bdd_feasibility_on_short_mrf_chain()
{
    ILP_input ilp = ILP_parser::parse_string(short_mrf_chain);
    bdd_preprocessor bdd_pre(ilp);
    SOLVER s(bdd_pre.get_bdd_collection());

    std::vector<char> sol(8, 0);

    sol[0] = 1; // mu_1_0 = 1
    sol[2] = 1; // mu_2_0 = 1
    {
        const auto bdd_feas = s.bdd_feasibility(sol.begin(), sol.end());
        test(bdd_feas(0,0) == true && bdd_feas(1,0) == true);
        test(bdd_feas(2,0) == true && bdd_feas(3,0) == true);
        test(bdd_feas(4,0) == false && bdd_feas(5,0) == false && bdd_feas(6,0) == false && bdd_feas(7,0) == false);
        test(bdd_feas(0,1) == false && bdd_feas(4,1) == false && bdd_feas(6,1) == false);
        test(bdd_feas(1,1) == true && bdd_feas(5,1) == true && bdd_feas(7,1) == true);
        test(bdd_feas(2,1) == false && bdd_feas(4,2) == false && bdd_feas(5,2) == false);
        test(bdd_feas(3,1) == true && bdd_feas(6,2) == true && bdd_feas(7,2) == true);
    }


    sol[4] = true;
    {
        const auto bdd_feas = s.bdd_feasibility(sol.begin(), sol.end());
        for(size_t i=0; i<bdd_feas.size(); ++i)
            for(size_t j=0; j<bdd_feas.size(i); ++j)
                test(bdd_feas(i,j) == true);
    }
}

int main(int argc, char** argv)
{
    test_bdd_feasibility_on_short_mrf_chain<bdd_mma<double>>();
    test_bdd_feasibility_on_short_mrf_chain<bdd_mma<float>>();
    //test_bdd_feasibility_on_short_mrf_chain<bdd_parallel_mma<double>>();
    //test_bdd_feasibility_on_short_mrf_chain<bdd_parallel_mma<float>>();
    //test_bdd_feasibility_on_short_mrf_chain<bdd_cuda_parallel_mma<double>>();
    //test_bdd_feasibility_on_short_mrf_chain<bdd_cuda_parallel_mma<float>>();
}
