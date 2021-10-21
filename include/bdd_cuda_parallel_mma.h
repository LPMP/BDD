#pragma once

#include "bdd_cuda_base.h"

namespace LPMP {

    template<typename REAL>
    class bdd_cuda_parallel_mma : public bdd_cuda_base<REAL> {
        public:
            // using bdd_cuda_base::bdd_cuda_base; // inherit base constructor

            bdd_cuda_parallel_mma(const BDD::bdd_collection& bdd_col);

            void iteration();

            void distribute_delta();

            // about memory coalescing:
            // https://forums.developer.nvidia.com/t/accessing-same-global-memory-address-within-warps/66574/2
            // https://on-demand.gputechconf.com/gtc-express/2011/presentations/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf

        private:
            void forward_iteration(const REAL omega);
            void forward_iteration_layer_based(const REAL omega);

            void backward_iteration(const REAL omega);

            void min_marginals_from_directional_costs(const int hop_index);

            void flush_delta_out();
            void flush_mm();

            thrust::device_vector<REAL> delta_lo_in_, delta_hi_in_, delta_lo_out_, delta_hi_out_; // One entry in each per primal variable.
            thrust::device_vector<REAL> mm_lo_, mm_hi_, hi_cost_out_, lo_cost_out_; // One entry per BDD layer.
    };
}
