#include <string>
#include "specialized_solvers/mrf_input.h"
#include "../test.h"
#include "bdd_solver.h"
#include <iostream>

using namespace LPMP;

const std::string uai_test_input_1 =
R"(MARKOV
3
2 2 3
5
1 0
1 1
1 2
2 0 1
2 1 2
2
 0.436 0.564
2
 0.69 0.69
3
 0.42 0.42 0.42
4
 0.128 0.872
 0.920 0.080
6
 0.210 0.333 0.457
 0.811 0.000 0.189)";


std::string uai_test_input_2 = 
R"(MARKOV
3
2 2 2 
5
1 0 
1 1 
1 2 
2 0 1 
2 1 2 
2
0.000000 0.000000 
2
0.000000 0.000000 
2
0.000000 0.000000 
4
30.000000 9.000000
10.000000 19.000000 
4
9.000000 17.000000 
38.000000 8.000000 
)";

int main(int argc, char** argv)
{
    {
        mrf_input mrf = parse_mrf_uai_string(uai_test_input_1);

        test(mrf.nr_variables() == 3);

        // unary potentials
        test(mrf.nr_labels(0) == 2);
        test(mrf.unary(0,0) == 0.436 && mrf.unary(0,1) == 0.564);
        test(mrf.nr_labels(1) == 2);
        test(mrf.unary(1,0) == 0.69 && mrf.unary(1,1) == 0.69);
        test(mrf.nr_labels(2) == 3);
        test(mrf.unary(2,0) == 0.42 && mrf.unary(2,1) == 0.42 && mrf.unary(2,1) == 0.42);


        // pairwise potentials
        test(mrf.nr_pairwise_potentials() == 2);
        test(mrf.pairwise_variables(0)[0] == 0 && mrf.pairwise_variables(0)[1] == 1);
        test(mrf.pairwise(0,{0,0}) == 0.128 && mrf.pairwise(0,{0,1}) == 0.872 && mrf.pairwise(0,{1,0}) == 0.920 && mrf.pairwise(0,{1,1}) == 0.080);
        test(mrf.pairwise_variables(1)[0] == 1 && mrf.pairwise_variables(1)[1] == 2);
        test(mrf.pairwise(1,{0,0}) == 0.210);

        // LB = 0.644 + 0.69 + 0.42
        bdd_solver_options opts;
        opts.ilp = mrf.convert_to_ilp();
        opts.bdd_solver_impl_ = bdd_solver_options::bdd_solver_impl::sequential_mma;
        bdd_solver s(opts);
        s.solve();
        test(std::abs(s.lower_bound() - (0.644 + 0.69 + 0.42)) <= 1e-4);
    }

    // LB = 17
    {
        mrf_input mrf = parse_mrf_uai_string(uai_test_input_2);
        bdd_solver_options opts;
        opts.ilp = mrf.convert_to_ilp();
        opts.bdd_solver_impl_ = bdd_solver_options::bdd_solver_impl::sequential_mma;
        bdd_solver s(opts);
        s.solve();
        test(std::abs(s.lower_bound() - 17) <= 1e-4);
    }
}

