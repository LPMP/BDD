#include "bdd_solver.h"
#include <vector>
#include <string>
#include <iostream>
#include "test.h"
#include "test_problems.h"

using namespace LPMP;

void test_problem(const std::string input_string, const std::string& relaxation_solver, const std::string& var_order, const double expected_lb)
{
	auto config_json = nlohmann::json::parse(R""""({"precision": "double",
      "termination criteria": { 
        "maximum iterations": 200,
         "improvement slope": 0.0,
          "minimum improvement": 0.0,
           "time limit": 1e10 
           }
})"""");
	config_json["relaxation solver"] = relaxation_solver;
	config_json["variable order"] = var_order;
    config_json["input"] = input_string;

	bdd_solver solver(config_json);
    solver.solve();
    const double lb = solver.lower_bound();

    test(std::abs(lb - expected_lb) <= 1e-01);
}

int main(int argc, char** arv)
{
	std::cout << "--- Short chain shuffled ---" << std::endl;

	test_problem(short_mrf_chain_shuffled, "sequential mma", "input", 1.0);
	test_problem(short_mrf_chain_shuffled, "sequential mma", "bfs", 1.0);
	test_problem(short_mrf_chain_shuffled, "sequential mma", "cuthill", 1.0);
	test_problem(short_mrf_chain_shuffled, "sequential mma", "minimum degree", 1.0);
	test_problem(short_mrf_chain_shuffled, "parallel mma", "input", 1.0);
	test_problem(short_mrf_chain_shuffled, "parallel mma", "bfs", 1.0);
	test_problem(short_mrf_chain_shuffled, "parallel mma", "cuthill", 1.0);
	test_problem(short_mrf_chain_shuffled, "parallel mma", "minimum degree", 1.0);

	std::cout << "--- Long chain ---" << std::endl;

	test_problem(long_mrf_chain, "sequential mma", "input", -9.0);
	test_problem(long_mrf_chain, "sequential mma", "bfs", -9.0);
	test_problem(long_mrf_chain, "sequential mma", "cuthill", -9.0);
	test_problem(long_mrf_chain, "sequential mma", "minimum degree", -9.0);
	test_problem(long_mrf_chain, "parallel mma", "input", -9.0);
	test_problem(long_mrf_chain, "parallel mma", "bfs", -9.0);
	test_problem(long_mrf_chain, "parallel mma", "cuthill", -9.0);
	test_problem(long_mrf_chain, "parallel mma", "minimum degree", -9.0);

	std::cout << "--- Grid graph ---" << std::endl;

	test_problem(mrf_grid_graph_3x3, "sequential mma", "input", -8.0);
	test_problem(mrf_grid_graph_3x3, "sequential mma", "bfs", -8.0);
	test_problem(mrf_grid_graph_3x3, "sequential mma", "cuthill", -8.0);
	test_problem(mrf_grid_graph_3x3, "sequential mma", "minimum degree", -8.0);
	test_problem(mrf_grid_graph_3x3, "parallel mma", "input", -8.0);
	test_problem(mrf_grid_graph_3x3, "parallel mma", "bfs", -8.0);
	test_problem(mrf_grid_graph_3x3, "parallel mma", "cuthill", -8.0);
	test_problem(mrf_grid_graph_3x3, "parallel mma", "minimum degree", -8.0);
}
