#pragma once

#include "bdd_cuda_base.h"

namespace LPMP {
    // Solving happens here. 
    // Take top-k variables with highest abs. min marginal difference but also prioritize the variables which 
    // have not been updated recently. Update only these dual variable by parallel mma. 
    class bdd_cuda_parallel_mma_sorting : public bdd_cuda_base {
        public:
            bdd_cuda_base::bdd_cuda_base; // inherit base constructor

            void iteration();

            // about memory coalescing:
            // https://forums.developer.nvidia.com/t/accessing-same-global-memory-address-within-warps/66574/2
            // https://on-demand.gputechconf.com/gtc-express/2011/presentations/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf

        private:

    }
}
