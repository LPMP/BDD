#include "bdd_parallel_mma_base.h"
#include "bdd_cuda_parallel_mma.h"
#include "bdd_branch_instruction.h"
#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "test_problems.h"
#include "test.h"
#include <random>

using namespace LPMP;

void test_problem(const std::string& problem_input, const bool with_additional_gaps)
{
    ILP_input ilp = ILP_parser::parse_string(problem_input);
    bdd_preprocessor bdd_pre(ilp);

    std::mt19937 gen(0);
    std::uniform_int_distribution<> distrib(0, 6);
    // map variables to new ones
    BDD::bdd_collection bdd_col = bdd_pre.get_bdd_collection();
    std::vector<size_t> var_map;
    if(with_additional_gaps)
    {
        var_map.push_back(distrib(gen));
        for(size_t i=1; i<ilp.nr_variables(); ++i)
            var_map.push_back(var_map.back() + 1 + distrib(gen));
    }
    else
    {
        var_map.resize(ilp.nr_variables());
        std::iota(var_map.begin(), var_map.end(), 0);
    }
    bdd_col.rebase(var_map.begin(), var_map.end());
    std::vector<double> mapped_obj(var_map.back() + 1, 0.0);
    for(size_t i=0; i<ilp.nr_variables(); ++i)
        mapped_obj[var_map[i]] = ilp.objective()[i];

    bdd_parallel_mma_base<bdd_branch_instruction<double,uint32_t>> parallel_mma(bdd_col);
    bdd_cuda_parallel_mma<double> cuda_mma(bdd_col);

    test(parallel_mma.nr_variables() == var_map.back()+1);
    test(parallel_mma.nr_variables() == cuda_mma.nr_variables());

    test(parallel_mma.nr_bdds() == cuda_mma.nr_bdds());

    for(size_t i=0; i<parallel_mma.nr_variables(); ++i)
        test(parallel_mma.nr_bdds(i) == cuda_mma.nr_bdds(i));

    {
        const double parallel_mma_lb = parallel_mma.lower_bound(); 
        const double cuda_mma_lb = cuda_mma.lower_bound(); 
        std::cout << "before cost update: parallel mma lb = " << parallel_mma_lb << ", cuda mma lb = " << cuda_mma_lb << "\n";
        test(std::abs(parallel_mma_lb - cuda_mma_lb) < 1e-6);
    }

    parallel_mma.update_costs(mapped_obj.begin(), mapped_obj.begin(), mapped_obj.begin(), mapped_obj.end());
    cuda_mma.update_costs(mapped_obj.begin(), mapped_obj.begin(), mapped_obj.begin(), mapped_obj.end());

    // initial lb after setting costs
    {
        const double parallel_mma_lb = parallel_mma.lower_bound(); 
        const double cuda_mma_lb = cuda_mma.lower_bound(); 
        std::cout << "initial lb: parallel mma lb = " << parallel_mma_lb << ", cuda mma lb = " << cuda_mma_lb << "\n";
        test(std::abs(parallel_mma_lb - cuda_mma_lb) < 1e-6);
    }

    // lb and deltas after forward and backward passes
    {
        std::vector<std::array<double,2>> cpu_delta(parallel_mma.nr_variables(), std::array<double,2>{0.0, 0.0});
        thrust::device_vector<double> cuda_delta(2*parallel_mma.nr_variables(), 0.0);

        for(size_t iter=0; iter<10; ++iter)
        {
            parallel_mma.forward_mm(0.5, cpu_delta);
            cuda_mma.forward_mm(0.5, cuda_delta);

            for(size_t i=0; i<parallel_mma.nr_variables(); ++i)
            {
                std::cout << i << " <" << cpu_delta[i][0] << "," << cpu_delta[i][1] << ">"
                    << " <" << cuda_delta[2*i] << "," << cuda_delta[2*i+1] << ">\n";
                test(std::abs(cpu_delta[i][0] - cuda_delta[2*i]) < 1e-6);
                test(std::abs(cpu_delta[i][1] - cuda_delta[2*i+1]) < 1e-6);
            }

            parallel_mma.backward_mm(0.5, cpu_delta);
            cuda_mma.backward_mm(0.5, cuda_delta);

            for(size_t i=0; i<parallel_mma.nr_variables(); ++i)
            {
                std::cout << i << " <" << cpu_delta[i][0] << "," << cpu_delta[i][1] << ">"
                    << " <" << cuda_delta[2*i] << "," << cuda_delta[2*i+1] << ">\n";
                test(std::abs(cpu_delta[i][0] - cuda_delta[2*i]) < 1e-6);
                test(std::abs(cpu_delta[i][1] - cuda_delta[2*i+1]) < 1e-6);
            }

            const double parallel_mma_lb = parallel_mma.lower_bound(); 
            const double cuda_mma_lb = cuda_mma.lower_bound(); 
            std::cout << "lb after iteration " << iter << ": parallel mma lb = " << parallel_mma_lb << ", cuda mma lb = " << cuda_mma_lb << "\n";
            test(std::abs(parallel_mma_lb - cuda_mma_lb) < 1e-6);
        }
    }
}

// test whether values produced by parallel mma and cuda solver are equal
// do this also for degenerate cases like variables with no BDD covering them.
int main(int argc, char** argv)
{
    for(const auto problem : test_problems)
    {
        test_problem(problem, false);
        test_problem(problem, true);
    }
}
