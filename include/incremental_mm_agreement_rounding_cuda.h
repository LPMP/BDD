#pragma once

#include <vector>

namespace LPMP {

    template<typename SOLVER>
        std::vector<char> incremental_mm_agreement_rounding_cuda(SOLVER& s, double init_delta, const double delta_growth_rate, const int num_itr_lb, const bool verbose = true);

}
