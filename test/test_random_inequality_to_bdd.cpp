#include "bdd_solver.h"
//#include "bdd/bdd_min_marginal_averaging_smoothed.h"
//#include "bdd/bdd_anisotropic_diffusion.h"
#include "convert_pb_to_bdd.h"
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include "test_problem_generator.h"
#include "test.h"

// TODO: rename single to random

using namespace LPMP;


void test_random_inequality_min_sum()
{
    for(size_t nr_vars=3; nr_vars<=15; ++nr_vars)
    {
        const auto [coefficients, ineq, rhs] = generate_random_inequality(nr_vars);
        ILP_input ilp = generate_ILP(coefficients, ineq, rhs);
        std::stringstream ss;
        ilp.write(ss);
        const std::string ilp_string = ss.str();

        bdd_solver decomp_mma({
            "--input_string", ilp_string,
            "-s", "decomposition_mma",
            "--nr_threads", "2",
            "--max_iter", "20",
            "--parallel_message_passing_weight", "1.0" 
                });
        decomp_mma.solve();

        bdd_solver mma({
            "--input_string", ilp_string,
            "-s", "mma",
            "--max_iter", "20"
            });
        mma.solve();

        test(std::abs(decomp_mma.lower_bound() - mma.lower_bound()) <= 1e-8);

        const auto [enumeration_lb, sol] = min_cost(coefficients.begin(), coefficients.end(), ineq, rhs, ilp.objective().begin(), ilp.objective().end());
        std::cout << "enumeration lb = " << enumeration_lb << ", backward lb = " << mma.lower_bound() << "\n";
        test(std::abs(mma.lower_bound() - enumeration_lb) <= 1e-8);
        std::cout << "cost of primal = " << ilp.evaluate(sol.begin(), sol.end()) << "\n";
        std::cout << "primal size = " << sol.size() << "\n";
        for(const auto x : sol)
            std::cout << int(x) << " ";
        std::cout << "\n";
        test(std::abs(enumeration_lb - ilp.evaluate(sol.begin(), sol.end())) <= 1e-8);

    } 
}

/*
void test_random_inequality_log_exp()
{
    BDD::bdd_mgr bdd_mgr;
    bdd_converter converter(bdd_mgr);

    for(std::size_t nr_vars = 3; nr_vars <= 15; ++nr_vars) {
        const auto [coefficients, ineq, rhs] = generate_random_inequality(nr_vars);
        for(const auto c : coefficients) {
            std::cout << c << " ";
        }
        if(ineq == ILP_input::inequality_type::equal)
            std::cout << " = ";
        if(ineq == ILP_input::inequality_type::smaller_equal)
            std::cout << " <= ";
        if(ineq == ILP_input::inequality_type::greater_equal)
            std::cout << " >= ";
        std::cout << rhs << "\n";

        auto bdd = converter.convert_to_bdd(coefficients.begin(), coefficients.end(), ineq, rhs);
        if(bdd.nr_nodes() < 2) 
            continue;
        bdd_min_marginal_averaging_smoothed bdds;
        std::vector<std::size_t> vars(nr_vars);
        std::iota (std::begin(vars), std::end(vars), 0);
        bdds.add_bdd(bdd, vars.begin(), vars.end(), bdd_mgr);
        //bdds.export_dot(std::cout);
        bdds.init(); 
        const std::vector<double> costs = generate_random_costs(nr_vars);
        std::cout << "cost: ";
        for(const auto x : costs)
            std::cout << x << " ";
        std::cout << "\n"; 
        bdds.set_costs(costs.begin(), costs.end());
        const double backward_lb = bdds.compute_smooth_lower_bound();
        bdds.forward_run();
        const double forward_lb = bdds.compute_smooth_lower_bound_forward();
        const double enumeration_lb = log_exp(coefficients.begin(), coefficients.end(), ineq, rhs, costs.begin(), costs.end());
        std::cout << "enumeration lb = " << enumeration_lb << ", backward lb = " << backward_lb << ", forward lb = " << forward_lb << "\n";
        test(std::abs(backward_lb - forward_lb) <= 1e-8);
        test(std::abs(backward_lb - enumeration_lb) <= 1e-8);
    } 
}
*/

int main(int argc, char** arv)
{
    //test_random_inequality_log_exp();
    test_random_inequality_min_sum();
    //test_random_inequality_min_sum<bdd_anisotropic_diffusion>();
}
