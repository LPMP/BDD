#include <vector>
#include <string>
#include "bdd_solver/bdd_solver.h"
#include "test.h"

using namespace LPMP;

std::string test_instance = R"(Minimize
x1 + x2 + x3 + x4 + x5 + x6
Subject To
x1 + x2 + x4 >= 1
x1 + x3 + x5 >= 1
x2 + x3 + x6 >= 1
Bounds
Binaries
x1
x2
x3
x4
x5
x6
End)";

std::string test_instance_tightened = R"(Minimize
x1 + x2 + x3 + x4 + x5 + x6
Subject To
x1 + x2 + x4 >= 1
x1 + x3 + x5 >= 1
x2 + x3 + x6 >= 1
x1 + x2 + x3 + x4 + x5 + x6 >= 2
Bounds
Binaries
x1
x2
x3
x4
x5
x6
End)";

int main(int argc, char** argv)
{
    std::cout << "Solve original covering problem\n";

    {
        auto config_json = nlohmann::json::parse(R""""({"precision": "double",
      "relaxation solver": "sequential mma",
      "termination criteria": { 
       "maximum iterations": 200,
         "improvement slope": 0.0,
          "minimum improvement": 0.0,
           "time limit": 1e10 
           }
})"""");
    config_json["input"] = test_instance;

        bdd_solver solver(config_json); 
        solver.solve();
        test(std::abs(solver.lower_bound() - 1.5) <= 1e-4);
    }

    // TODO: once tightening is implemented, tighten on the test_instance problem and show that it has lb 2.0

    {
        std::vector<std::string> solver_input = {
            "--input_string", test_instance_tightened,
            "-s", "mma",
            "--max_iter", "1000"
        };

         auto config_json = nlohmann::json::parse(R""""({"precision": "double",
      "relaxation solver": "sequential mma",
      "termination criteria": { 
       "maximum iterations": 200,
         "improvement slope": 0.0,
          "minimum improvement": 0.0,
           "time limit": 1e10 
           }
})"""");
    config_json["input"] = test_instance_tightened;

        bdd_solver solver(config_json); 
        solver.solve();
        test(solver.lower_bound() > 1.5); // does not solve exactly
    }

}

