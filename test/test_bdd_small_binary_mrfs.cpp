#include "bdd_solver.h"
#include <vector>
#include <string>
#include <iostream>
#include "test.h"
#include "test_problems.h"

using namespace LPMP;

void test_problem(const std::string input_string, const double expected_lb, std::vector<std::string> args)
{
	for (auto it = args.begin(); it != args.end(); it++)
		std::cout << *it << " " << std::flush;
	std::cout << std::endl;
	args.push_back("-t");
	args.push_back("1e-16");
	args.push_back("-p");
    args.push_back("--input_string");
    args.push_back(input_string);
    bdd_solver solver((bdd_solver_options(args)));
    const double initial_lb = solver.lower_bound();
    solver.solve();
    const double lb = solver.lower_bound();

    test(std::abs(lb - expected_lb) <= 1e-01);

    solver.round();
}

int main(int argc, char** arv)
{
	std::cout << "--- Short chain shuffled ---" << std::endl;

	test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma", "-o", "input", "--max_iter", "5"});
    return 0;
	test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma", "-o", "bfs", "--max_iter", "5"});
	test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma", "-o", "cuthill", "--max_iter", "5"});
	// // test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma", "-o", "mindegree", "--max_iter", "5"});
	// test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma_srmp", "-o", "input", "--max_iter", "5"});
	// // test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma_srmp", "-o", "bfs", "--max_iter", "5"});
	// test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma_srmp", "-o", "cuthill", "--max_iter", "5"});
	// // test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma_srmp", "-o", "mindegree", "--max_iter", "5"});
	// test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma_agg", "-o", "input", "--max_iter", "5"});
	// // test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma_agg", "-o", "bfs", "--max_iter", "5"});
	// test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma_agg", "-o", "cuthill", "--max_iter", "5"});
	// // test_problem(short_mrf_chain_shuffled, 1.0, {"-s", "mma_agg", "-o", "mindegree", "--max_iter", "5"});

	std::cout << "--- Long chain ---" << std::endl;

	// test_problem(long_mrf_chain, -9.0, {"-s", "mma", "-o", "input", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma", "-o", "bfs", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma", "-o", "cuthill", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma", "-o", "mindegree", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma_srmp", "-o", "input", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma_srmp", "-o", "bfs", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma_srmp", "-o", "cuthill", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma_srmp", "-o", "mindegree", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma_agg", "-o", "input", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma_agg", "-o", "bfs", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma_agg", "-o", "cuthill", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "mma_agg", "-o", "mindegree", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "anisotropic_mma", "-o", "input", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "anisotropic_mma", "-o", "bfs", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "anisotropic_mma", "-o", "cuthill", "--max_iter", "10"});
	// test_problem(long_mrf_chain, -9.0, {"-s", "anisotropic_mma", "-o", "mindegree", "--max_iter", "10"});

	std::cout << "--- Grid graph ---" << std::endl;

	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma", "-o", "input", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma", "-o", "bfs", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma", "-o", "cuthill", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma", "-o", "mindegree", "--max_iter", "20"});
	test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma_srmp", "-o", "input", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma_srmp", "-o", "bfs", "--max_iter", "20"});
	test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma_srmp", "-o", "cuthill", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma_srmp", "-o", "mindegree", "--max_iter", "20"});
	test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma_agg", "-o", "input", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma_agg", "-o", "bfs", "--max_iter", "20"});
	test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma_agg", "-o", "cuthill", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "mma_agg", "-o", "mindegree", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "anisotropic_mma", "-o", "input", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "anisotropic_mma", "-o", "bfs", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "anisotropic_mma", "-o", "cuthill", "--max_iter", "20"});
	// test_problem(mrf_grid_graph_3x3, -8.0, {"-s", "anisotropic_mma", "-o", "mindegree", "--max_iter", "20"});
}
