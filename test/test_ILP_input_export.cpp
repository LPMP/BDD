#include "ILP_parser.h"
#include "OPB_parser.h"
#include "bdd_solver.h"
#include <string>
#include <sstream>
#include "test_instances.h"
#include "test.h"

using namespace LPMP;

const std::string export_lp(const std::string& problem)
{
    const ILP_input input_orig = ILP_parser::parse_string(problem);
    std::stringstream lp_exported;
    input_orig.write_lp(lp_exported);
    return lp_exported.str();
}

const std::string export_opb(const std::string& problem)
{
    const ILP_input input_orig = ILP_parser::parse_string(problem);
    std::stringstream opb_exported;
    input_orig.write_opb(opb_exported);
    return opb_exported.str();
}

void test_export(const std::string& problem, const double lb)
{
    auto compute_lp = [&](const std::string& problem) {
        std::vector<std::string> solver_input = {
            "--lp_input_string", problem,
            "-s", "mma_vec",
            "--max_iter", "1000"
        };

        bdd_solver solver(solver_input); 
        solver.solve();

        return solver.lower_bound();
    };

    auto compute_opb = [&](const std::string& problem) {
        std::vector<std::string> solver_input = {
            "--opb_input_string", problem,
            "-s", "mma_vec",
            "--max_iter", "1000"
        };

        bdd_solver solver(solver_input); 
        solver.solve();

        return solver.lower_bound();
    };

    std::cout << "original problem:\n" << problem;
    std::cout << "exported lp:\n" << export_lp(problem);;
    std::cout << "exported opb:\n" << export_opb(problem);;

    const double orig_lb = compute_lp(problem);
    const double exported_lp_lb = compute_lp(export_lp(problem));
    const double exported_opb_lb = compute_opb(export_opb(problem));

    test(std::abs(orig_lb - lb) <= 1e-6);
    test(std::abs(exported_lp_lb - lb) <= 1e-6);
    test(std::abs(exported_opb_lb - lb) <= 1e-6);
}

int main(int argc, char** arv)
{
    test_export(matching_3x3, -6.0);
    test_export(covering_problem_3x3, 1.5);
    test_export(covering_problem_2_3x3, 1.5);
} 
