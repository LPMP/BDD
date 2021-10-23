#pragma once

#include "bdd_cuda_base.h"

namespace LPMP {

    template<typename REAL>
    class bdd_cuda_parallel_mma : public bdd_cuda_base<REAL> {
        public:
            // using bdd_cuda_base::bdd_cuda_base; // inherit base constructor

            bdd_cuda_parallel_mma(const BDD::bdd_collection& bdd_col);

            void iteration();

            // about memory coalescing:
            // https://forums.developer.nvidia.com/t/accessing-same-global-memory-address-within-warps/66574/2
            // https://on-demand.gputechconf.com/gtc-express/2011/presentations/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf

        private:
            void forward_iteration(const REAL omega);
            void forward_iteration_layer_based(const REAL omega);

            void backward_iteration(const REAL omega);

            void min_marginals_from_directional_costs(const int hop_index, const REAL omega);

            void distribute_delta();
            void normalize_delta();
            void compute_delta();
            void flush_mm();
            void flush_delta_out();

            thrust::device_vector<REAL> delta_lo_, delta_hi_; // One entry in each per primal variable.
            thrust::device_vector<REAL> mm_lo_, mm_diff_, hi_cost_out_, lo_cost_out_; // One entry per BDD layer.
            thrust::device_vector<int> primal_variable_sorting_order_; // indices to sort primal_variables_indices_
            thrust::device_vector<int> primal_variable_index_sorted_;  // to reduce min-marginals by key.
    };
}
