#include "ILP_input.h"
#include <unordered_set>
#include "test.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    {
        ILP_input::constraint simplex;
        std::array<size_t,1> var = {0};
        simplex.monomials.push_back(var.begin(), var.end());
        var[0] = 1;
        simplex.monomials.push_back(var.begin(), var.end());
        var[0] = 2;
        simplex.monomials.push_back(var.begin(), var.end());

        simplex.right_hand_side = 1;
        simplex.ineq = ILP_input::inequality_type::equal;
        simplex.coefficients = {1,1,1};

        ILP_input ilp;

        ilp.add_new_variable("x0");
        ilp.add_new_variable("x1");
        ilp.add_new_variable("x2");
        ilp.add_constraint(simplex);

        std::unordered_set<size_t> zero_fixations;
        std::unordered_set<size_t> one_fixations;
        zero_fixations.insert(ilp.get_var_index("x1"));
        ILP_input reduced_ilp = ilp.reduce(zero_fixations, one_fixations);

        test(reduced_ilp.nr_variables() == 2);
        test(reduced_ilp.constraints()[0].is_linear());
        test(reduced_ilp.constraints()[0].is_simplex());

    }

}

