#pragma once

#include <array>
#include "bdd_collection/bdd_collection.h"
#include "two_dimensional_variable_array.hxx"

namespace LPMP {

    class bdd_cuda_base {
        public:
            bdd_cuda_base(bdd_collection& bdd_col);

            double lower_bound();
            void set_cost(const double c, const size_t var);
            two_dim_variable_array<std::array<double,2>> min_marginals();
            size_t nr_variables() const;
            size_t nr_bdds() const;

        protected:
            thrust::device_vector<int> variables_;
            thrust::device_vector<int> lo_offsets_; // = 0
            thrust::device_vector<int> hi_offsets_; // = 1
            thrust::device_vector<float> hi_cost;
            thrust::device_vector<float> cost_from_root;
            thrust::device_vector<float> cost_from_terminal;

    };

}
