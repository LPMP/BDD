#include "convert_pb_to_bdd.h"
#include <iostream> // TODO: remove

namespace LPMP {

    BDD::node_ref bdd_converter::convert_to_bdd(const std::vector<int> coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side)
    {
        if(coefficients.size() == 0)
            throw std::runtime_error("Expected non-empty coefficients");
        return convert_to_bdd(coefficients.begin(), coefficients.end(), ineq_type, right_hand_side);
    }


}
