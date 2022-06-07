#pragma once
#include "bdd_branch_instruction.h"
#include "bdd_collection/bdd_collection.h"
#include "bdd_parallel_mma.h"
#include "bdd_cuda_parallel_mma.h"
#include "bdd_parallel_mma_base.h"
#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"
#include <limits>

namespace LPMP {

    template<typename REAL>
    class bdd_multi_parallel_mma_base {
        public:
            bdd_multi_parallel_mma_base(BDD::bdd_collection& bdd_col);

            size_t nr_bdds() const;
            size_t nr_bdds(const size_t var) const;
            size_t nr_variables() const;
            size_t nr_variables(const size_t bdd_nr) const;
            double lower_bound();
            void add_to_constant(const double c);
            template<typename COST_ITERATOR> 
                void update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end);

            void parallel_mma();

            two_dim_variable_array<std::array<double,2>> min_marginals();
            void fix_variable(const size_t var, const bool val);

        private:

            void normalize_delta(thrust::device_vector<REAL>& delta_lo, thrust::device_vector<REAL>& delta_hi) const;
            std::vector<std::array<REAL,2>> cpu_delta_in_, cpu_delta_out_;
            thrust::device_vector<REAL> gpu_delta_lo_, gpu_delta_hi_;
            thrust::device_vector<size_t> total_nr_bdds_per_var_;

            bdd_parallel_mma_base<bdd_branch_instruction<REAL, uint32_t>> cpu_base;
            bdd_cuda_parallel_mma<REAL> cuda_base;
    };

}
