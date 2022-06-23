#pragma once
#include "bdd_branch_instruction.h"
#include "bdd_collection/bdd_collection.h"
#include "bdd_parallel_mma.h"
#include "bdd_cuda_parallel_mma.h"
#include "bdd_parallel_mma_base.h"
#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <limits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace LPMP {

    // return bdd indices for cpu and cuda bases
    std::array<BDD::bdd_collection,2> split_bdd_collection(BDD::bdd_collection& bdd_col, const size_t gpu_th = 0, const size_t cpu_th = std::numeric_limits<size_t>::max());

    template<typename REAL>
    class bdd_multi_parallel_mma_base {
        public:
            // gpu_th: all BDDs shorter than val are going to GPU
            // cpu_th: all BDDs longer than val are going to CPU
            bdd_multi_parallel_mma_base(BDD::bdd_collection& bdd_col);
            bdd_multi_parallel_mma_base(BDD::bdd_collection& cpu_bdd_col, BDD::bdd_collection& gpu_bdd_col);

            size_t nr_bdds() const;
            size_t nr_bdds(const size_t var) const;
            size_t nr_variables() const;
            size_t nr_variables(const size_t bdd_nr) const;
            double lower_bound();
            void add_to_constant(const double c);
            template<typename COST_ITERATOR> 
                void update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end);

            
            void forward_mm( const REAL omega,
                    thrust::device_vector<REAL>& delta_lo, thrust::device_vector<REAL>& delta_hi);
            void backward_mm( const REAL omega,
                    thrust::device_vector<REAL>& delta_lo, thrust::device_vector<REAL>& delta_hi);
            void parallel_mma();

            two_dim_variable_array<std::array<double,2>> min_marginals();
            void fix_variable(const size_t var, const bool val);

        private:

            // TODO: remove
            void normalize_delta(thrust::device_vector<REAL>& delta_lo, thrust::device_vector<REAL>& delta_hi) const;
            void split_delta(
                    const thrust::device_vector<REAL>& total_delta_lo,
                    const thrust::device_vector<REAL>& total_delta_hi,
                    std::vector<std::array<REAL, 2>> &cpu_delta,
                    thrust::device_vector<REAL> &gpu_delta_lo, 
                    thrust::device_vector<REAL> &gpu_delta_hi);
            void average_delta(thrust::device_vector<REAL>& delta);
            std::tuple<thrust::device_vector<REAL>, thrust::device_vector<REAL>> accumulate_delta(
                    const std::vector<std::array<REAL,2>> cpu_delta,
                    const thrust::device_vector<REAL>& gpu_delta_lo,
                    const thrust::device_vector<REAL>& gpu_delta_hi);

            std::vector<std::array<REAL,2>> cpu_delta_in_, cpu_delta_out_;
            thrust::device_vector<REAL> gpu_delta_lo_, gpu_delta_hi_;
            thrust::device_vector<REAL> total_delta_lo_, total_delta_hi_;
            thrust::device_vector<size_t> total_nr_bdds_per_var_;
            thrust::host_vector<size_t> gpu_nr_bdds_per_var_;

            bdd_parallel_mma_base<bdd_branch_instruction<REAL, uint32_t>> cpu_base;
            bdd_cuda_parallel_mma<REAL> cuda_base;
    };

}
