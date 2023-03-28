#include "bdd_multi_parallel_mma_base.h"
#include "bdd_parallel_mma_base.h"
#include "bdd_branch_instruction.h"
#include "ILP_input.h"
#include "ILP_parser.h"
#include "bdd_preprocessor.h"
#include "test_problems.h"
#include "test.h"

using namespace LPMP;

// check if parallel mma and hybrid parallel mma give identical results
void test_problem(const std::string& problem_input)
{
    ILP_input ilp = ILP_parser::parse_string(problem_input);
    bdd_preprocessor bdd_pre(ilp);

    bdd_parallel_mma_base<bdd_branch_instruction<double,uint32_t>> parallel_mma(bdd_pre.get_bdd_collection());
    bdd_cuda_parallel_mma<double> cuda_mma(bdd_pre.get_bdd_collection()); 
    const auto [size_shortest_ineq, size_longest_ineq] = [&]() -> std::tuple<size_t, size_t> {
        size_t size_shortest_ineq = std::numeric_limits<size_t>::max();
        size_t size_longest_ineq = 0;
        for(size_t c=0; c<ilp.constraints().size(); ++c)
        {
            size_shortest_ineq = std::min(size_shortest_ineq, ilp.constraints()[c].coefficients.size());
            size_longest_ineq = std::max(size_longest_ineq, ilp.constraints()[c].coefficients.size());
        }
        std::cout << "shortest inequality has " << size_shortest_ineq << " variables, longest has " << size_longest_ineq << " ones\n";
        return {size_shortest_ineq, size_longest_ineq};
    }();
    const size_t hybrid_gpu_th = size_shortest_ineq + 1;
    const size_t hybrid_cpu_th = std::max(size_longest_ineq - 1, hybrid_gpu_th + 1);
    assert(hybrid_gpu_th < hybrid_cpu_th);
    auto [cpu_bdds, gpu_bdds] = split_bdd_collection(bdd_pre.get_bdd_collection(), hybrid_gpu_th, hybrid_cpu_th);

    {
        std::vector<size_t> nr_bdds_per_gpu(ilp.nr_variables(), 0);
        std::vector<size_t> nr_bdds_per_cpu(ilp.nr_variables(), 0);
        for(size_t bdd_nr=0; bdd_nr<gpu_bdds.nr_bdds(); ++bdd_nr)
        {
            const auto vars = gpu_bdds.variables(bdd_nr);
            for(const size_t v : vars)
                nr_bdds_per_gpu[v]++;
        }
        for(size_t bdd_nr=0; bdd_nr<cpu_bdds.nr_bdds(); ++bdd_nr)
        {
            const auto vars = cpu_bdds.variables(bdd_nr);
            for(const size_t v : vars)
                nr_bdds_per_cpu[v]++;
        }

        std::cout << "nr (gpu,cpu) bdds per var:\n";
        for(size_t i=0; i<ilp.nr_variables(); ++i)
            std::cout << "(" << nr_bdds_per_gpu[i] << "," << nr_bdds_per_cpu[i] << ") ";
        std::cout << "\n";
    }

    BDD::bdd_collection empty_bdd_col;
    bdd_multi_parallel_mma_base<double> hybrid_parallel_mma(cpu_bdds, gpu_bdds);
    //std::cout << "only gpu in hybrid\n";
    //bdd_multi_parallel_mma_base<double> hybrid_parallel_mma(empty_bdd_col, bdd_pre.get_bdd_collection());

    test(parallel_mma.nr_variables() == hybrid_parallel_mma.nr_variables());
    test(cuda_mma.nr_variables() == hybrid_parallel_mma.nr_variables());

    test(parallel_mma.nr_bdds() == hybrid_parallel_mma.nr_bdds());
    test(cuda_mma.nr_bdds() == hybrid_parallel_mma.nr_bdds());

    for(size_t i=0; i<ilp.nr_variables(); ++i)
    {
        test(parallel_mma.nr_bdds(i) == hybrid_parallel_mma.nr_bdds(i));
        test(cuda_mma.nr_bdds(i) == hybrid_parallel_mma.nr_bdds(i));
    }

    {
        const double parallel_mma_lb = parallel_mma.lower_bound(); 
        const double cuda_mma_lb = cuda_mma.lower_bound(); 
        const double hybrid_parallel_mma_lb = hybrid_parallel_mma.lower_bound(); 
        std::cout << "before cost update: parallel mma lb = " << parallel_mma_lb << ", hybrid parallel mma lb = " << hybrid_parallel_mma_lb << ", cuda mma lb = " << cuda_mma_lb << "\n";
        test(std::abs(parallel_mma_lb - hybrid_parallel_mma_lb) < 1e-6);
    }

    parallel_mma.update_costs(ilp.objective().begin(), ilp.objective().begin(), ilp.objective().begin(), ilp.objective().end());
    cuda_mma.update_costs(ilp.objective().begin(), ilp.objective().begin(), ilp.objective().begin(), ilp.objective().end());
    hybrid_parallel_mma.update_costs(ilp.objective().begin(), ilp.objective().begin(), ilp.objective().begin(), ilp.objective().end());

    // initial lb after setting costs
    {
        const double parallel_mma_lb = parallel_mma.lower_bound(); 
        const double cuda_mma_lb = cuda_mma.lower_bound(); 
        const double hybrid_parallel_mma_lb = hybrid_parallel_mma.lower_bound(); 
        std::cout << "initial lb: parallel mma lb = " << parallel_mma_lb << ", hybrid parallel mma lb = " << hybrid_parallel_mma_lb << ", cuda mma lb = " << cuda_mma_lb << "\n";
        test(std::abs(parallel_mma_lb - hybrid_parallel_mma_lb) < 1e-6);
    }

    // lb and deltas after forward and backward passes
    {
        std::vector<std::array<double,2>> cpu_delta(ilp.nr_variables(), std::array<double,2>{0.0, 0.0});
        thrust::device_vector<double> cuda_delta(2*ilp.nr_variables(), 0.0);
        thrust::device_vector<double> hybrid_delta(2*ilp.nr_variables(), 0.0);

        for(size_t iter=0; iter<10; ++iter)
        {
            parallel_mma.forward_mm(0.5, cpu_delta);
            cuda_mma.forward_mm(0.5, cuda_delta);
            hybrid_parallel_mma.forward_mm(0.5, hybrid_delta);

            for(size_t i=0; i<ilp.nr_variables(); ++i)
            {
                std::cout << i << ": <" << cpu_delta[i][0] << "," << cpu_delta[i][1] << ">"
                    << " <" << cuda_delta[2*i] << "," << cuda_delta[2*i+1] << ">"
                    << " <" << hybrid_delta[2*i] << "," << hybrid_delta[2*i+1] << ">\n";
                test(std::abs(cpu_delta[i][0] - cuda_delta[2*i]) < 1e-6);
                test(std::abs(cpu_delta[i][1] - cuda_delta[2*i+1]) < 1e-6);
                test(std::abs(cpu_delta[i][0] - hybrid_delta[2*i]) < 1e-6);
                test(std::abs(cpu_delta[i][1] - hybrid_delta[2*i+1]) < 1e-6);
            }

            // TODO: reactivate again!
            //{
            //    const double parallel_mma_lb = parallel_mma.lower_bound(); 
            //    const double cuda_mma_lb = cuda_mma.lower_bound(); 
            //    const double hybrid_parallel_mma_lb = hybrid_parallel_mma.lower_bound(); 
            //    std::cout << "lb after " << iter << " iterations: parallel mma lb = " << parallel_mma_lb << ", hybrid parallel mma lb = " << hybrid_parallel_mma_lb << ", cuda mma lb = " << cuda_mma_lb << "\n";
            //    test(std::abs(parallel_mma_lb - hybrid_parallel_mma_lb) < 1e-6);
            //}

            parallel_mma.backward_mm(0.5, cpu_delta);
            cuda_mma.backward_mm(0.5, cuda_delta);
            hybrid_parallel_mma.backward_mm(0.5, hybrid_delta);

            for(size_t i=0; i<ilp.nr_variables(); ++i)
            {
                std::cout << i << ": <" << cpu_delta[i][0] << "," << cpu_delta[i][1] << ">"
                    << " <" << cuda_delta[2*i] << "," << cuda_delta[2*i+1] << ">"
                    << " <" << hybrid_delta[2*i] << "," << hybrid_delta[2*i+1] << ">\n";
                test(std::abs(cpu_delta[i][0] - cuda_delta[2*i]) < 1e-6);
                test(std::abs(cpu_delta[i][1] - cuda_delta[2*i+1]) < 1e-6);
                test(std::abs(cpu_delta[i][0] - hybrid_delta[2*i]) < 1e-6);
                test(std::abs(cpu_delta[i][1] - hybrid_delta[2*i+1]) < 1e-6);
            }

            {
                const double parallel_mma_lb = parallel_mma.lower_bound(); 
                const double cuda_mma_lb = cuda_mma.lower_bound(); 
                const double hybrid_parallel_mma_lb = hybrid_parallel_mma.lower_bound(); 
                std::cout << "lb after " << iter << " iterations: parallel mma lb = " << parallel_mma_lb << ", hybrid parallel mma lb = " << hybrid_parallel_mma_lb << ", cuda mma lb = " << cuda_mma_lb << "\n";
                test(std::abs(parallel_mma_lb - hybrid_parallel_mma_lb) < 1e-6);
            }
        }
    }

    //  lb after iterations
    for(size_t iter=0; iter<10; ++iter)
    {
        parallel_mma.parallel_mma();
        cuda_mma.iteration();
        hybrid_parallel_mma.parallel_mma();

        const double parallel_mma_lb = parallel_mma.lower_bound(); 
        const double cuda_mma_lb = cuda_mma.lower_bound(); 
        const double hybrid_parallel_mma_lb = hybrid_parallel_mma.lower_bound(); 
        std::cout << "lb after " << iter << " iterations: parallel mma lb = " << parallel_mma_lb << ", hybrid parallel mma lb = " << hybrid_parallel_mma_lb << ", cuda mma lb = " << cuda_mma_lb << "\n";
        //test(std::abs(parallel_mma_lb - hybrid_parallel_mma_lb) < 1e-6);
    }
}

int main(int argc, char** argv)
{
    for(const auto problem : test_problems)
        test_problem(problem);
}
