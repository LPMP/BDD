#include "lbfgs_impl.h"
#include "bdd_solver/bdd_cuda_parallel_mma.h"

namespace LPMP {
    template class lbfgs<bdd_cuda_parallel_mma<float>, thrust::device_vector<float>, float, thrust::device_vector<char>, true>;
    template class lbfgs<bdd_cuda_parallel_mma<double>, thrust::device_vector<double>, double, thrust::device_vector<char>, true>;
}