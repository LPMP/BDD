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
void test_update_costs(const std::string& problem_string)
{
    ILP_input ilp = ILP_parser::parse_string(problem_string);
    ilp.normalize();
    bdd_preprocessor bdd_pre(ilp);
    SOLVER s(bdd_pre.get_bdd_collection());

    two_dim_variable_array<std::array<double,2>> bdd_costs = s.min_marginals();
    test(bdd_costs.size() == s.nr_variables());
    for(size_t i=0; i<bdd_costs.size(); ++i)
    {
        test(bdd_costs.size(i) == s.nr_bdds(i));

        test(bdd_costs.size(i) == s.nr_bdds(i));

        for(size_t j=0; j<bdd_costs.size(i); ++j)
        {
            test(std::abs(bdd_costs(i,j)[0]) <= 1e-6);
            test(std::abs(bdd_costs(i,j)[1]) <= 1e-6);
        }
    }

    std::vector<double> var_costs(s.nr_variables(), 0.0);

    for(size_t i=0; i<bdd_costs.size(); ++i)
    {
        var_costs[i] = -double(s.nr_bdds(i)*(i+1));
        for(size_t j=0; j<bdd_costs.size(i); ++j)
        {
            bdd_costs(i,j) = {0.0, double(i+1)};
        }
    }
    test(std::abs(s.lower_bound() - 0.0) <= 1e-6);

    s.update_costs(bdd_costs);
    test(s.lower_bound() > 0.5); // otherwise there exists an all zero solution, which is not the case in the test_problems
    test(s.lower_bound() < ilp.nr_constraints() * (ilp.nr_variables()*(ilp.nr_variables()-1))/2); // not all BDDs may have all one assignments
    s.update_costs(var_costs.begin(), var_costs.begin(), var_costs.begin(), var_costs.end());

    test(std::abs(s.lower_bound() - 0.0) <= 1e-6);
}

int main(int argc, char** argv)
{
    for(const std::string problem_str : test_problems)
    {
        test_update_costs<bdd_mma<double>>(problem_str);
        //test_update_costs<bdd_parallel_mma<double>>(problem_str);
        //test_update_costs<bdd_cuda<double>>(problem_str);

    }

}
