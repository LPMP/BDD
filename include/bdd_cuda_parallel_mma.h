#pragma once

#include "bdd_cuda_base.h"

namespace LPMP {

    template<typename REAL>
    class bdd_cuda_parallel_mma : public bdd_cuda_base<REAL> {
        public:
            // using bdd_cuda_base::bdd_cuda_base; // inherit base constructor

            void init();
            bdd_cuda_parallel_mma() {}
            bdd_cuda_parallel_mma(const BDD::bdd_collection& bdd_col);

            void iteration(const REAL omega = 0.5);

        protected:
            void min_marginals_from_directional_costs(const int hop_index, const REAL omega);
            // Computes min-marginals for hop 'hop_index' and writes at starting from *mm_diff_ptr_with_start_offset (GPU pointer).
            void min_marginals_from_directional_costs(const int hop_index, const REAL omega, thrust::device_ptr<REAL> mm_diff_ptr_with_start_offset);

            void compute_delta();
            void compute_delta(const thrust::device_ptr<const REAL> mm_to_distribute); // uses the provided mm_diff instead of deffered_mm_diff_
            void normalize_delta();

            void flush_mm(thrust::device_ptr<REAL> mm_diff_ptr);
            thrust::device_vector<REAL> hi_cost_out_, lo_cost_out_; // One entry per BDD layer.

            // Deferred min-marginal sums.
            thrust::device_vector<REAL> delta_lo_, delta_hi_; // One entry in each per primal variable.

        private:
            void forward_iteration(const REAL omega);
            void backward_iteration(const REAL omega);

            thrust::device_vector<REAL> mm_lo_local_; // Contains mm_lo for last computed hop. Memory allocated is as per max(cum_nr_layers_per_hop_dist_).
    };
}
