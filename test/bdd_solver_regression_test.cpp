#include <vector>
#include <string>
#include "bdd_solver.h"
#include "test.h"

using namespace LPMP;

struct test_instance {
    std::string filename;
    // results reached by LP solver & ILP optimum
    double lower_bound_lp;
    double opt_sol_cost;
    // results that should be attained by approximative message passing
    double lower_bound_mp;
    double app_sol_cost;
};

std::vector<test_instance> test_instances = {
    // from hotel
    test_instance({"energy_hotel_frame15frame99.lp", -3.860424e+03, -3.860424367000e+03, 1.01 * -3.860424e+03, 0.99 * -3.860424367000e+03}),
    // from house
    test_instance({"energy_house_frame15frame105.lp", -3.745160e+03, -3.745159569000e+03, 1.02 * -3.745159569000e+03, 0.99 * -3.745159569000e+03}),
    // from cell tracking AISTATS
    test_instance({"drosophila.lp", -1.297201e+07, -1.297200572591e+07, 1.01 * -1.297201e+07, 0.99 * -1.297200572591e+07}),
    // from shape matching TOSCA, gurobi could not find optimal solution
    // gurobi could not verify optimality of solution, I use best lb obtained instead
    test_instance({"000000880800_241600_cat0_200_cat6_200_.lp", 3.07763340e+02, 3.07763340e+02, 0.92 * 3.07763340e+02, 1.2 * 3.07763340e+02}),
    // from color-seg-n4
    test_instance({"pfau-small.lp", 2.25273057e+04, 2.423444787696e+04, 0.95 * 2.25273057e+04, 1.2 * 2.423444787696e+04}),
    // from worms graph matching Kainmueller et al
    test_instance({"worm01-16-03-11-1745.lp", -4.632114e+04, -4.631054884800e+04, 1.1 * -4.632114e+04, 0.95 * -4.631054884800e+04}),
    // from protein-folding
    test_instance({"1CKK.lp", 1.284017e+04, -1.271240947404e+04, 0.98 * 1.284017e+04, 1.05 * 12712.41})
};

void test_solver(const test_instance& instance, std::vector<std::string> opts_vec)
{
    const std::string full_path = BDD_SOLVER_REGRESSION_TEXT_DIR + instance.filename;
    opts_vec.push_back("-i");
    opts_vec.push_back(full_path);
    bdd_solver_options opts(opts_vec);
    bdd_solver s(opts);
    s.solve();
    const double lb = s.lower_bound();
    s.round();
    test(instance.lower_bound_mp <= instance.lower_bound_lp);
    test(lb >= instance.lower_bound_mp - 1e-6,
            "Lower bound computed by solver " + std::to_string(lb) + " smaller than required one " + std::to_string(instance.lower_bound_mp));
    test(lb <= instance.lower_bound_lp + 1e-3,
            "Lower bound computed by solver " + std::to_string(lb) + " larger than optimal lower bound " + std::to_string(instance.lower_bound_lp));
    test(instance.app_sol_cost >= instance.opt_sol_cost);
    const double sol_cost = s.round();
    test(sol_cost <= instance.app_sol_cost + 1e-6,
            "Solution computed by solver " + std::to_string(sol_cost) + " has higher cost than required one " + std::to_string(instance.app_sol_cost));
    test(sol_cost >= instance.opt_sol_cost - 1e-6,
            "Solution computed by solver " + std::to_string(sol_cost) + " has lower cost than optimal one " + std::to_string(instance.opt_sol_cost));
    std::cout << "done and results pass\n";
}

void test_mma(const test_instance& instance)
{
    std::cout << "solving " << instance.filename << " with mma\n";
    std::vector<std::string> opts = {
            "-s", "mma",
            "--incremental_primal",
            "--incremental_initial_perturbation", "1.0",
            "--incremental_perturbation_growth_rate", "1.05",
            "--precision", "double"
            };
    test_solver(instance, opts);
}

void test_parallel_mma(const test_instance& instance)
{
    std::cout << "solving " << instance.filename << " with parallel mma\n";
    std::vector<std::string> opts = {
            "-s", "parallel_mma",
            "--nr_threads", "4",
            "--incremental_primal",
            "--incremental_initial_perturbation", "1.0",
            "--incremental_perturbation_growth_rate", "1.05",
            "--precision", "double"
            };
    test_solver(instance, opts);
}

void test_cuda_mma(const test_instance& instance)
{
    std::cout << "solving " << instance.filename << " with cuda mma\n";
    std::vector<std::string> opts = {
            "-s", "cuda_mma",
            "--incremental_primal",
            "--incremental_initial_perturbation", "1.0",
            "--incremental_perturbation_growth_rate", "1.05",
            "--precision", "double"
            };
    test_solver(instance, opts);
}

int main(int argc, char** argv)
{
    for(const auto& instance : test_instances) 
    {
        test_mma(instance);
        test_parallel_mma(instance);
#ifdef WITH_CUDA
        test_cuda_mma(instance);
#endif
    }
}
