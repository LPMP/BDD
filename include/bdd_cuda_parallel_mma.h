#pragma once

#include "bdd_cuda_base.h"
#include "lbfgs_cuda.h"

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
            void min_marginals_from_directional_costs(const int hop_index, const REAL omega_scalar);

            // Computes min-marginals for hop 'hop_index' and writes at starting from *mm_diff_ptr + layer start offset (GPU pointer). Uses omega_vec instead of omega_scalar if given. 
            void min_marginals_from_directional_costs(const int hop_index, const REAL omega_scalar, thrust::device_ptr<REAL> mm_diff_ptr, const thrust::device_ptr<const REAL> omega_vec = nullptr);

            void compute_delta();
            void compute_delta(const thrust::device_ptr<const REAL> mm_to_distribute); // uses the provided mm_diff instead of deffered_mm_diff_
            void normalize_delta();

            void flush_mm(thrust::device_ptr<REAL> mm_diff_ptr);
            thrust::device_vector<REAL> hi_cost_out_, lo_cost_out_; // One entry per BDD layer.

            void maintain_feasibility_grad(const thrust::device_ptr<const REAL> gradient);
            void update_bfgs_states(lbfgs_cuda<REAL>& lbfgs_solver);
            bool compute_direction_bfgs(lbfgs_cuda<REAL>& lbfgs_solver, thrust::device_ptr<REAL> grad_f);

            // Deferred min-marginal sums.
            thrust::device_vector<REAL> delta_lo_, delta_hi_; // One entry in each per primal variable.
            lbfgs_cuda<REAL> lbfgs_solver_;

        private:
            void forward_iteration(const REAL omega);
            void backward_iteration(const REAL omega);

            thrust::device_vector<REAL> mm_lo_local_; // Contains mm_lo for last computed hop. Memory allocated is as per max(cum_nr_layers_per_hop_dist_).

            int itr_count_ = 0;
            REAL step_size_ = 1e-3;
    };
}
