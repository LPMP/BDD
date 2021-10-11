#pragma once

#include <cstdlib>
#include <array>
#include <vector>
#include <limits>
#include "two_dimensional_variable_array.hxx"

namespace LPMP {

    template<typename REAL>
    std::vector<char> mm_signs(const two_dim_variable_array<std::array<REAL,2>>& mms)
    {
        std::vector<char> signs(mms.size(),-1);

        for(size_t i=0; i<mms.size(); ++i)
        {
            assert(mms.size(i) > 0);
            bool consistent = true;
            auto signum = [](const auto x) -> int {
                if(x == 0)
                    return 0;
                else if(x < 0)
                    return -1;
                else
                    return 1;
            };
            int sign = mms(i,0)[1] - mms(i,0)[0];
            const bool all_equal = [&]() { // all min-marginals are equal
                for(size_t j=0; j<mms.size(i); ++j)
                    if(std::abs(mms(i,j)[1] - mms(i,j)[0]) > 1e-6)
                        return false;
                return true;
            }();
            const bool all_one = [&]() { // min-marginals indicate one variable should be taken
                for(size_t j=0; j<mms.size(i); ++j)
                    if(mms(i,j)[1] + 1e-6 < mms(i,j)[0])
                        return false;
                return true;
            }();
            const bool all_zero = [&]() { // min-marginals indicate zero variable should be taken
                for(size_t j=0; j<mms.size(i); ++j)
                    if(mms(i,j)[0] + 1e-6 < mms(i,j)[1])
                        return false;
                return true;
            }();

            if(all_zero)
                signs[i] = -1;
            else if(all_one)
                signs[i] = 1;
            else if(all_equal)
                signs[i] = 0;
            else // contradictory min-marginals, solvers may have further lower bound improvement if run for longer
                signs[i] = std::numeric_limits<char>::max();
        }

        return signs;
    }

    template<typename SOLVER>
        std::vector<char> incremental_mm_agreement_rounding_iter(SOLVER& s, const double init_delta = 1.0)
        {
            constexpr double delta_growth_factor = 1.2;
            double cur_delta = 1.0/delta_growth_factor * init_delta;

            for(size_t round=0; round<100; ++round)
            {
                std::cout << "[incremental primal rounding] round " << round << ", cost delta " << cur_delta << "\n";
                cur_delta = cur_delta*delta_growth_factor;

                const auto mms = s.min_marginals();
                const auto signs = mm_signs(mms);
                const size_t nr_good_mms = std::count_if(signs.begin(), signs.end(), [](const char v) { return v == -1 || v == 1; });
                std::cout << "#unique min-marginals " << nr_good_mms << "/" << mms.size() << ", % = " << double(nr_good_mms)/double(mms.size()) << "\n";
                if(nr_good_mms == signs.size())
                {
                    std::vector<char> sol(mms.size(),0);
                    for(size_t i=0; i<sol.size(); ++i)
                    {
                        if(signs[i] == -1)
                            sol[i] = 1;
                        else
                        {
                            assert(signs[i] == 1);
                            sol[i] = 0;
                        }
                    }
                    std::cout << "[incremental primal rounding] Found feasible solution\n";
                    return sol;
                }

                std::vector<double> cost_updates(mms.size(), 0.0);
                for(size_t i=0; i<mms.size(); ++i)
                {
                    if(signs[i] == -1)
                        cost_updates[i] = -cur_delta;
                    else if(signs[i] == 1)
                        cost_updates[i] = +cur_delta;
                    else if(signs[i] == 0)
                    {
                        const double r = std::rand()/RAND_MAX;
                        assert(1.0 <= r && r <= 1.0);
                        cost_updates[i] = cur_delta * r;
                    }
                    else
                    {
                        assert(signs[i] == std::numeric_limits<char>::max());
                        cost_updates[i] = 0.0; // there will be lower bound increase in this variable
                    }
                }
                s.update_costs(cost_updates.begin(), cost_updates.end());
                for(size_t solver_iter=0; solver_iter<5; ++solver_iter)
                    s.iteration();
            }

            std::cout << "[incremental primal rounding] No solution found\n";
            return {};
        }

}
