#pragma once

#include <vector>
#include <array>
#include "two_dimensional_variable_array.hxx"

namespace LPMP {

    // for storing min mrginals and derived 
    enum class mm_type {
        zero,
        one,
        equal,
        inconsistent
    };

    class mm_primal_decoder {
        public:
            mm_primal_decoder(two_dim_variable_array<std::array<double,2>>&& _mms)
                : mms_(_mms)
            {}

            std::array<double,2> operator()(const size_t var, const size_t bdd_idx) const;
            size_t size() const { return mms_.size(); }
            size_t size(const size_t var) const { return mms_.size(var); }

            mm_type compute_mm_type(const size_t var) const; 
            std::vector<mm_type> mm_types() const;
            std::array<double,2> mm_sum(const size_t var) const;

            size_t nr_one_mms() const;
            size_t nr_zero_mms() const;
            size_t nr_equal_mms() const;
            size_t nr_inconsistent_mms() const;

            bool can_reconstruct_solution() const;
            // #ones, #zeros, #equal, #inconsistent
            std::array<size_t,4> mm_type_statistics() const;
            std::vector<char> solution_from_mms() const;
            char solution(const size_t var) const;

        private:
            two_dim_variable_array<std::array<double,2>> mms_;
    };

}
