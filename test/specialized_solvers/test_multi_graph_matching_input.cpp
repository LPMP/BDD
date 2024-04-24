#include <string>
#include "specialized_solvers/multi_graph_matching_solver.h"
#include "specialized_solvers/multi_graph_matching_input.h"
#include "../test.h"
#include <iostream>

using namespace LPMP;

const std::string minimal_synchronization_example =
R"(gm 0 1
p 2 2 0 0
a 0 0 0 -1
a 1 0 1 -10
a 2 1 0 -10
a 3 1 1 -1

gm 0 2
p 2 2 0 0
a 0 0 0 -1
a 1 0 1 -10
a 2 1 0 -10
a 3 1 1 -1

gm 1 2
p 2 2 0 0
a 0 0 0 -1
a 1 0 1 -10
a 2 1 0 -10
a 3 1 1 -1
)";

int main(int argc, char** argv)
{
    auto config_json = nlohmann::json::parse(R""""({"precision": "double",
     "relaxation solver": "parallel mma",
     "variable order": "bfs",
      "termination criteria": { 
        "maximum iterations": 200,
         "improvement slope": 0.0,
          "minimum improvement": 0.0,
           "time limit": 1e10 
           }
})"""");
    config_json["input"] = minimal_synchronization_example;
    multi_graph_matching_bdd_solver mgm_solver(config_json);
    mgm_solver.solve();
    
    const auto mgm_instance = parse_multi_graph_matching_string(minimal_synchronization_example);
    config_json["input"] = mgm_instance.write_lp();
    bdd_solver solver(config_json);
    solver.solve();

    test(std::abs(mgm_solver.lower_bound() - (-42.0)) <= 1e-4);
    test(std::abs(solver.lower_bound() - (-42.0)) <= 1e-4);
}
