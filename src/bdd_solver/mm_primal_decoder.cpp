#include "mm_primal_decoder.h"
#include <cmath>
#include <algorithm>

namespace LPMP {

    std::array<double,2> mm_primal_decoder::operator()(const size_t var, const size_t bdd_idx) const
    {
        return mms_(var,bdd_idx);
    }

    mm_type mm_primal_decoder::compute_mm_type(const size_t var) const
    {
        assert(var < mms_.size());

        const bool all_equal = [&]() { // all min-marginals are equal
            for(size_t j=0; j<mms_.size(var); ++j)
                if(std::abs(mms_(var,j)[1] - mms_(var,j)[0]) > 1e-6)
                    return false;
            return true;
        }();

        const bool all_one = [&]() { // min-marginals indicate one variable should be taken
            for(size_t j=0; j<mms_.size(var); ++j)
                if(!(mms_(var,j)[1] + 1e-6 < mms_(var,j)[0]))
                    return false;
            return true;
        }();

        const bool all_zero = [&]() { // min-marginals indicate zero variable should be taken
            for(size_t j=0; j<mms_.size(var); ++j)
                if(!(mms_(var,j)[0] + 1e-6 < mms_(var,j)[1]))
                    return false;
            return true;
        }();

        assert(int(all_zero) + int(all_one) + int(all_equal) <= 1);

        if(all_zero)
            return mm_type::zero;
        else if(all_one)
            return mm_type::one;
        else if(all_equal)
            return mm_type::equal;
        else
            return mm_type::inconsistent;
    }

    std::vector<mm_type> mm_primal_decoder::mm_types() const
    {
        std::vector<mm_type> mmts;
        mmts.reserve(mms_.size());
        for(size_t var=0; var<mms_.size(); ++var)
            mmts.push_back(compute_mm_type(var));
        return mmts;
    }

    std::array<double,2> mm_primal_decoder::mm_sum(const size_t var) const
    {
        assert(var < mms_.size());

        double sum_0 = 0.0;
        double sum_1 = 0.0;

        for(size_t j=0; j<mms_.size(var); ++j)
        {
            sum_0 += mms_(var,j)[0];
            sum_1 += mms_(var,j)[1];
        }

        return {sum_0, sum_1};
    }

    std::vector<char> mm_primal_decoder::solution_from_mms() const
    {
        std::vector<char> sol(mms_.size(),0);
        for(size_t i=0; i<sol.size(); ++i)
        {
            const mm_type mmt = compute_mm_type(i);
            if(mmt == mm_type::one)
                sol[i] = 1;
            else if(mmt == mm_type::zero)
                sol[i] = 0;
            else 
            {
                const auto sum = mm_sum(i);
                if(sum[0] <= sum[1])
                    sol[i] = 0;
                else
                    sol[i] = 1;
            }
        }
        return sol;
    }

    bool mm_primal_decoder::can_reconstruct_solution() const
    {
        const auto [nr_ones, nr_zeros, nr_equal, nr_inconsistent] = mm_type_statistics();
        return nr_equal == 0 && nr_inconsistent == 0;
    }

    std::array<size_t,4> mm_primal_decoder::mm_type_statistics() const
    {
        const std::vector<mm_type> mmts = mm_types();

        const size_t nr_one_mms = std::count(mmts.begin(), mmts.end(), mm_type::one);
        const size_t nr_zero_mms = std::count(mmts.begin(), mmts.end(), mm_type::zero);
        const size_t nr_equal_mms = std::count(mmts.begin(), mmts.end(), mm_type::equal);
        const size_t nr_inconsistent_mms = std::count(mmts.begin(), mmts.end(), mm_type::inconsistent);

        assert(nr_one_mms + nr_zero_mms + nr_equal_mms + nr_inconsistent_mms == mms_.size());

        return {nr_one_mms, nr_zero_mms, nr_equal_mms, nr_inconsistent_mms};
    }

}

