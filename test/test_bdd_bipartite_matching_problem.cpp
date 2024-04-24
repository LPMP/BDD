#include "bdd_solver.h"
#include <string>
#include <nlohmann/json.hpp>
#include "test.h"

using namespace LPMP;

void test_problem(const std::string input_string, const double expected_lb)
{
    auto config_json = nlohmann::json::parse(R""""({"precision": "double",
     "relaxation solver": "sequential mma",
      "termination criteria": { 
        "maximum iterations": 20,
         "improvement slope": 0.0,
          "minimum improvement": 0.0,
           "time limit": 1e10 
           }
})"""");
    config_json["input"] = input_string;
    bdd_solver solver(config_json);
    solver.solve();
    const double lb = solver.lower_bound();
    test(std::abs(lb - expected_lb) <= 1e-6);
}

const char * matching_3x3_diag = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-1 x_21 - 2 x_22 - 1 x_23
-1 x_31 - 1 x_32 - 2 x_33
Subject To
x_11 + x_12 + x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
x_13 + x_23 + x_33 = 1
End)";

const char * matching_3x3_first_row = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-2 x_21 - 1 x_22 - 1 x_23
-2 x_31 - 1 x_32 - 1 x_33
Subject To
x_11 + x_12 + x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
x_13 + x_23 + x_33 = 1
End)";


int main(int argc, char** arv)
{
    test_problem(matching_3x3_diag, -6.0);
    test_problem(matching_3x3_first_row, -4.0);
}
