#include "ILP_input.h"
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

        ilp.fix_variable("x1",0);
        ilp.substitute_fixed_variables();

        test(ilp.nr_variables() == 2);
        test(ilp.constraints()[0].is_linear());
        test(ilp.constraints()[0].is_simplex());

    }

}

