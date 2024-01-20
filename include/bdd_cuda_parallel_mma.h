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

            void forward_mm(const REAL omega, thrust::device_vector<REAL>& delta_lo_hi);
            void backward_mm(const REAL omega, thrust::device_vector<REAL>& delta_lo_hi);

            // Normalize delta by num BDDs to distribute isotropically.
            // delta_lo_ -> delta_hi_/#BDDs, delta_hi_ -> delta_hi_/#BDDs
            void normalize_delta(thrust::device_vector<REAL>& delta_lo_hi) const;

            thrust::device_vector<REAL> net_solver_costs() const;

            template<typename ITERATOR>
            void gradient_step(ITERATOR grad_begin, ITERATOR grad_end, double step_size);

        protected:
            void min_marginals_from_directional_costs(const int hop_index, const REAL omega_scalar);

            // Computes min-marginals for hop 'hop_index' and writes at starting from *mm_diff_ptr + layer start offset (GPU pointer). Uses omega_vec instead of omega_scalar if given. 
            void min_marginals_from_directional_costs(const int hop_index, const REAL omega_scalar, thrust::device_ptr<REAL> mm_diff_ptr, const thrust::device_ptr<const REAL> omega_vec = nullptr);

            // compute delta_lo_ and delta_hi_ (per variable) from mm_to_distribute (per bdd node)
            void compute_delta(const thrust::device_ptr<const REAL> mm_to_distribute, thrust::device_ptr<REAL> delta_lo_hi) const;

            // set argument to all infinity
            void flush_mm(thrust::device_ptr<REAL> mm_diff_ptr);

            thrust::device_vector<REAL> hi_cost_out_, lo_cost_out_; // One entry per BDD layer.


        private:
            //void forward_iteration(const REAL omega);
            //void backward_iteration(const REAL omega);

            thrust::device_vector<REAL> mm_lo_local_; // Contains mm_lo for last computed hop. Memory allocated is as per max(cum_nr_layers_per_hop_dist_).
    };

    template<typename REAL>
    struct add_scaled_product_func {
        const REAL stepsize;
        __host__ __device__ REAL operator()(const REAL& hi_cost, const REAL& gradient) const {
            return hi_cost + stepsize * gradient;
        }
    };

    template<typename REAL>
    template<typename ITERATOR>
    void bdd_cuda_parallel_mma<REAL>::gradient_step(ITERATOR grad_begin, ITERATOR grad_end, double step_size)
    {
        assert(std::distance(grad_begin, grad_end) == this->hi_cost_.size());
        thrust::transform(this->hi_cost_.begin(), this->hi_cost_.end(), grad_begin, this->hi_cost_.begin(), 
            add_scaled_product_func<REAL>({REAL(step_size)}));
        this->flush_forward_states();
        this->flush_backward_states();
    }

 
}
