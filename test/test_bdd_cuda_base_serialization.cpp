#include "bdd_cuda_base.h"
#include "ILP_parser.h"
#include "bdd_collection/bdd_collection.h"
#include "bdd_preprocessor.h"
#include "test.h"

#include <cereal/archives/binary.hpp>
#include <sstream>

using namespace LPMP;
using namespace BDD;

const char * one_simplex_problem = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
Subject To
x_11 + x_12 + x_13 = 1
End)";

const char * two_simplex_problem = 
R"(Minimize
2 x_1 + 1 x_2 + 1 x_3
+1 x_4 + 2 x_5 - 1 x_6
Subject To
x_1 + x_2 + x_3 = 1
x_4 + x_5 + x_6 = 2
End)";

const char * two_simplex_diff_size_problem = 
R"(Minimize
2 x_1 + 1 x_2 + 1 x_3
+2 x_4 + 2 x_5 + 3 x_6
Subject To
x_1 + x_2 + x_3 + x_4 = 1
x_4 + x_5 + x_6 = 2
End)";

const char * matching_3x3 = 
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

void test_problem(const char* instance)
{
    std::stringstream ss;
    ILP_input ilp = ILP_parser::parse_string(instance);
    bdd_preprocessor bdd_pre(ilp);
    bdd_collection bdd_col = bdd_pre.get_bdd_collection();
    bdd_cuda_base<float> bcb_orig(bdd_col);

    for(size_t i=0; i<bcb_orig.nr_variables(); ++i)
        bcb_orig.set_cost(ilp.objective()[i], i);

    {
        cereal::BinaryOutputArchive oarchive(ss);
        oarchive(bcb_orig);
    } // archive goes out of scope, ensuring all contents are flushed

    bdd_cuda_base<float> bcb_from_ss;
    {
        cereal::BinaryInputArchive iarchive(ss);
        iarchive(bcb_from_ss); 
    }
    test(bcb_from_ss.nr_variables() == bcb_orig.nr_variables());
    test(bcb_from_ss.nr_bdds() == bcb_orig.nr_bdds());
    test(bcb_from_ss.lower_bound() == bcb_orig.lower_bound());
}

int main(int argc, char** argv)
{
    test_problem(one_simplex_problem);
    test_problem(two_simplex_problem);
    test_problem(two_simplex_diff_size_problem);
    test_problem(matching_3x3);
}