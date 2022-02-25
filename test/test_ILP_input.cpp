#include "ILP_input.h"
#include "test.h"

using namespace LPMP;

int main(int argc, char** argv)
{
    {
        ILP_input::constraint simplex;
        std::array<size_t,1> var = {1};
        simplex.monomials.push_back(var.begin(), var.end());
        var[0] = 2;
        simplex.monomials.push_back(var.begin(), var.end());
        var[0] = 3;
        simplex.monomials.push_back(var.begin(), var.end());

        simplex.right_hand_side = 1;
        simplex.ineq = ILP_input::inequality_type::equal;
        simplex.coefficients = {1,1,1};

        test(simplex.is_linear());
        test(simplex.is_simplex());

        simplex.right_hand_side = 2;
        simplex.ineq = ILP_input::inequality_type::equal;
        simplex.coefficients = {1,1,1};

        test(simplex.is_linear());
        test(!simplex.is_simplex());

        simplex.right_hand_side = -2;
        simplex.ineq = ILP_input::inequality_type::equal;
        simplex.coefficients = {-2,-2,-2};

        test(simplex.is_linear());
        test(simplex.is_simplex());
    }

}
