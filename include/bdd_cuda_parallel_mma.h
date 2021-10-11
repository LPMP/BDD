#pragma once

#include "bdd_cuda_base.h"

namespace LPMP {
    class bdd_cuda_parallel_mma : public bdd_cuda_base {
        public:
            using bdd_cuda_base::bdd_cuda_base; // inherit base constructor

            void iteration();
            void solve(const size_t max_iter, const double tolerance, const double time_limit);

            // about memory coalescing:
            // https://forums.developer.nvidia.com/t/accessing-same-global-memory-address-within-warps/66574/2
            // https://on-demand.gputechconf.com/gtc-express/2011/presentations/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf

        private:
            void forward_iteration(const float omega);
            void backward_iteration(const float omega);

            thrust::device_vector<float> delta_lo, delta_hi; // One entry in each per primal variable.


    };
}
