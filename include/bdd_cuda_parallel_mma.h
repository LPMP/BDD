#pragma once

#include "bdd_cuda_base.h"

namespace LPMP {

    template<typename REAL>
    class bdd_cuda_parallel_mma : public bdd_cuda_base<REAL> {
        public:
            // using bdd_cuda_base::bdd_cuda_base; // inherit base constructor

            void init();
            bdd_cuda_parallel_mma() {}
            bdd_cuda_parallel_mma(const BDD::bdd_collection& bdd_col, const double lbfgs_step_size = 0.0);

            void iteration(const REAL omega = 0.5);

            void forward_mm(const REAL omega, thrust::device_vector<REAL>& delta_lo_hi);
            void backward_mm(const REAL omega, thrust::device_vector<REAL>& delta_lo_hi);

            // Normalize delta by num BDDs to distribute isotropically.
            // delta_lo_ -> delta_lo_/#BDDs, delta_hi_ -> delta_hi_/#BDDs
            void normalize_delta(thrust::device_vector<REAL>& delta_lo_hi) const;

            void distribute_delta(thrust::device_ptr<REAL> def_min_marg_diff_ptr);
            void distribute_delta();
        protected:
            void min_marginals_from_directional_costs(const int hop_index, const REAL omega_scalar);

            // Computes min-marginals for hop 'hop_index' and writes at starting from *mm_diff_ptr + layer start offset (GPU pointer). Uses omega_vec instead of omega_scalar if given. 
            void min_marginals_from_directional_costs(const int hop_index, const REAL omega_scalar, thrust::device_ptr<REAL> mm_diff_ptr, const thrust::device_ptr<const REAL> omega_vec = nullptr);

            // compute delta_lo_ and delta_hi_ (per variable) from mm_to_distribute (per bdd node)
            void compute_delta(const thrust::device_ptr<const REAL> mm_to_distribute, thrust::device_ptr<REAL> delta_lo_hi) const;

            // set argument to all infinity
            void flush_mm(thrust::device_ptr<REAL> mm_diff_ptr);

            thrust::device_vector<REAL> hi_cost_out_, lo_cost_out_; // One entry per BDD layer.

            void do_bfgs_update();
            void update_bfgs_states(thrust::device_ptr<REAL> def_mm_diff_ptr);
            bool compute_direction_bfgs(thrust::device_ptr<REAL> grad_f);

            // Deferred min-marginal sums.
            thrust::device_vector<REAL> delta_lo_hi_; // Two entries per primal variable. Even indices contain delta_lo and odd indices contain delta_hi.

        private:
            //void forward_iteration(const REAL omega);
            //void backward_iteration(const REAL omega);

            thrust::device_vector<REAL> mm_lo_local_; // Contains mm_lo for last computed hop. Memory allocated is as per max(cum_nr_layers_per_hop_dist_).
    };
}
