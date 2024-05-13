#include "lbfgs_impl.h"
#include "bdd_solver/bdd_parallel_mma_base.h"
#include "bdd_solver/bdd_branch_instruction.h"

namespace LPMP {
    template class lbfgs<bdd_parallel_mma_base<bdd_branch_instruction<float, uint16_t>>, std::vector<float>, float, std::vector<char>, false>;
    template class lbfgs<bdd_parallel_mma_base<bdd_branch_instruction<double, uint16_t>>, std::vector<double>, double, std::vector<char>, false>;
}