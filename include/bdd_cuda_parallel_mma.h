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

            // First sets weights to distribute min-marginal sum across dual variables. The weights should sum up to 1 for each primal variable.
            // Afterwards solve for hop i+1 (i-1) assuming that we have solved for hop i in forward (backward) iteration.
            // template<typename ITERATOR>
            // void solve_with_distribution_coeffs(ITERATOR begin, ITERATOR end); 

            // Computes min-marginal differences for hop i+1 (i-1) assuming that we have solved for hop i in forward (backward) iteration.
            // thrust::device_vector<REAL> compute_next_layer_min_marginals_difference();

        private:
            void forward_iteration(const REAL omega);
            void forward_iteration_layer_based(const REAL omega);

            void backward_iteration(const REAL omega);

            void min_marginals_from_directional_costs(const int hop_index, const REAL omega);

            void normalize_delta();
            void compute_delta();
            void flush_mm();
            void flush_delta_out();

            thrust::device_vector<REAL> delta_lo_, delta_hi_; // One entry in each per primal variable.
            thrust::device_vector<REAL> mm_lo_, mm_diff_, hi_cost_out_, lo_cost_out_; // One entry per BDD layer.
    };
}
