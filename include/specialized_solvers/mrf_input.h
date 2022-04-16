#pragma once

#include <vector>
#include <array>
#include "two_dimensional_variable_array.hxx"
#include "ILP_input.h"

namespace LPMP {

    struct mrf_input {

        two_dim_variable_array<double> unaries_;
        two_dim_variable_array<double> pairwise_;
        std::vector<std::array<size_t,2>> pairwise_variables_;

        size_t nr_variables() const;
        size_t nr_labels(const size_t var) const;

        double unary(const size_t var, const size_t label) const;
        double& unary(const size_t pairwise_pot, const size_t label);

        size_t nr_pairwise_potentials() const;
        double pairwise(const size_t pairwise_pot, const std::array<size_t,2> labels) const;
        double& pairwise(const size_t pairwise_pot, const std::array<size_t,2> labels);
        std::array<size_t,2> pairwise_variables(const size_t pairwise_pot) const;

        ILP_input convert_to_ilp() const;
    };


    mrf_input parse_mrf_uai_file(const std::string& filename);
    mrf_input parse_mrf_uai_string(const std::string& string);

}
