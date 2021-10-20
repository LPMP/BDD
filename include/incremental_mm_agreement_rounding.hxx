#pragma once

#include <random>
#include <array>
#include <vector>
#include <limits>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "two_dimensional_variable_array.hxx"

namespace LPMP {

    enum class mm_type {
        zero,
        one,
        equal,
        inconsistent
    }; 

    template<typename REAL>
    std::vector<mm_type> compute_mm_types(const two_dim_variable_array<std::array<REAL,2>>& mms)
    {
        std::vector<mm_type> diffs(mms.size(),mm_type::inconsistent);

        for(size_t i=0; i<mms.size(); ++i)
        {
            assert(mms.size(i) > 0);

            const bool all_equal = [&]() { // all min-marginals are equal
                for(size_t j=0; j<mms.size(i); ++j)
                    if(std::abs(mms(i,j)[1] - mms(i,j)[0]) > 1e-6)
                        return false;
                return true;
            }();

            const bool all_one = [&]() { // min-marginals indicate one variable should be taken
                for(size_t j=0; j<mms.size(i); ++j)
                    if(!(mms(i,j)[1] + 1e-6 < mms(i,j)[0]))
                        return false;
                return true;
            }();

            const bool all_zero = [&]() { // min-marginals indicate zero variable should be taken
                for(size_t j=0; j<mms.size(i); ++j)
                    if(!(mms(i,j)[0] + 1e-6 < mms(i,j)[1]))
                        return false;
                return true;
            }();

            assert(int(all_zero) + int(all_one) + int(all_equal) <= 1);

            if(all_zero)
                diffs[i] = mm_type::zero;
            else if(all_one)
                diffs[i] = mm_type::one;
            else if(all_equal)
                diffs[i] = mm_type::equal;
            else
                diffs[i] = mm_type::inconsistent;
        }

        return diffs;
    }

    template<typename REAL>
    double compute_initial_delta(const two_dim_variable_array<std::array<REAL,2>>& mms)
    {
        std::vector<double> mm_diffs(mms.size());;
        for(size_t i=0; i<mms.size(); ++i)
        {
            for(size_t j=0; j<mms.size(i); ++j)
                mm_diffs[i] += mms(i,j)[1] - mms(i,j)[0];
            mm_diffs[i] = std::abs(mm_diffs[i])/double(mms.size(i));
        }
        nth_element(mm_diffs.begin(), mm_diffs.begin() + 0.1*mms.size(), mm_diffs.end());
        const double computed_delta = mm_diffs[0.1*mms.size()];
        std::cout << "[incremental primal rounding] computed delta = " << computed_delta << "\n";
        return computed_delta;
    }

    template<typename SOLVER>
        std::vector<char> incremental_mm_agreement_rounding_iter(SOLVER& s, double init_delta = std::numeric_limits<double>::infinity(), const double delta_growth_rate = 1.1)
        {
            assert(init_delta > 0.0);
            assert(delta_growth_rate >= 1.0);

            const auto start_time = std::chrono::steady_clock::now();

            if(init_delta == std::numeric_limits<double>::infinity())
                init_delta = compute_initial_delta(s.min_marginals());

            std::cout << "[incremental primal rounding] initial perturbation delta = " << init_delta << ", growth rate for perturbation " << delta_growth_rate << "\n";

            double cur_delta = 1.0/delta_growth_rate * init_delta;

            //std::random_device rd;
            //std::mt19937 gen(rd);
            std::default_random_engine gen{static_cast<long unsigned int>(0)}; // deterministic seed for repeatable experiments

            for(size_t round=0; round<10000; ++round)
            {
                cur_delta = cur_delta*delta_growth_rate;
                const auto time = std::chrono::steady_clock::now();
                const double time_elapsed = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
                std::cout << "[incremental primal rounding] round " << round << ", cost delta " << cur_delta << ", time elapsed = " << time_elapsed << "\n";

                const auto mms = s.min_marginals();
                const auto mm_types = compute_mm_types(mms);
                const size_t nr_one_mms = std::count(mm_types.begin(), mm_types.end(), mm_type::one);
                const size_t nr_zero_mms = std::count(mm_types.begin(), mm_types.end(), mm_type::zero);
                const size_t nr_equal_mms = std::count(mm_types.begin(), mm_types.end(), mm_type::equal);
                const size_t nr_inconsistent_mms = std::count(mm_types.begin(), mm_types.end(), mm_type::inconsistent);
                assert(nr_one_mms + nr_zero_mms + nr_equal_mms + nr_inconsistent_mms == mms.size());

                const int old_precision = std::cout.precision();
                std::cout << std::setprecision(2);
                std::cout << "[incremental primal rounding] " <<
                    "#one min-marg diffs = " << nr_one_mms << " % " << double(100*nr_one_mms)/double(mms.size()) << ", " <<  
                    "#zero min-marg diffs = " << nr_zero_mms << " % " << double(100*nr_zero_mms)/double(mms.size()) << ", " << 
                    "#equal min-marg diffs = " << nr_equal_mms << " % " << double(100*nr_equal_mms)/double(mms.size()) << ", " << 
                    "#inconsistent min-marg diffs = " << nr_inconsistent_mms << " % " << double(100*nr_inconsistent_mms)/double(mms.size()) << "\n";
                std::cout << std::setprecision(old_precision);
                
                std::uniform_real_distribution<> dis(-cur_delta, cur_delta);

                if(nr_one_mms + nr_zero_mms == mms.size())
                {
                    std::vector<char> sol(mms.size(),0);
                    for(size_t i=0; i<sol.size(); ++i)
                    {
                        if(mm_types[i] == mm_type::one)
                            sol[i] = 1;
                        else
                        {
                            assert(mm_types[i] == mm_type::zero);
                            sol[i] = 0;
                        }
                    }
                    std::cout << "[incremental primal rounding] Found feasible solution\n";
                    return sol;
                }

                std::vector<double> cost_lo_updates(mms.size(), 0.0);
                std::vector<double> cost_hi_updates(mms.size(), 0.0);
                for(size_t i=0; i<mms.size(); ++i)
                {
                    if(mm_types[i] == mm_type::one)
                    {
                        cost_lo_updates[i] = cur_delta;
                        cost_hi_updates[i] = 0.0;
                    }
                    else if(mm_types[i] == mm_type::zero)
                    {
                        cost_lo_updates[i] = 0.0;
                        cost_hi_updates[i] = cur_delta;
                    }
                    else if(mm_types[i] == mm_type::equal)
                    {
                        const double r = dis(gen);
                        assert(-cur_delta <= r && r <= cur_delta);
                        if(r < 0.0)
                        {
                            cost_lo_updates[i] = std::abs(r)*cur_delta;
                            cost_hi_updates[i] = 0.0;
                        }
                        else
                        {
                            cost_lo_updates[i] = 0.0;
                            cost_hi_updates[i] = std::abs(r)*cur_delta;
                        }
                    }
                    else
                    {
                        assert(mm_types[i] == mm_type::inconsistent);
                        const auto mm_sum = [&]() {
                            std::array<double,2> s = {0.0,0.0};
                            for(size_t j=0; j<mms.size(i); ++j)
                            {
                                s[0] += mms(i,j)[0];
                                s[1] += mms(i,j)[1];
                            }
                            return s;
                        }();
                        const double r = dis(gen);
                        if(mm_sum[0] < mm_sum[1])
                        {
                            cost_lo_updates[i] = 0.0;
                            cost_hi_updates[i] = std::abs(r)*cur_delta;
                        }
                        else
                        {
                            cost_lo_updates[i] = std::abs(r)*cur_delta;
                            cost_hi_updates[i] = 0.0;
                        }
                    }
                }
                s.update_costs(cost_lo_updates.begin(), cost_lo_updates.end(), cost_hi_updates.begin(), cost_hi_updates.end());
                for(size_t solver_iter=0; solver_iter<5; ++solver_iter)
                    s.iteration();
                std::cout << "[incremental primal rounding] lower bound = " << s.lower_bound() << "\n";
            }

            std::cout << "[incremental primal rounding] No solution found\n";
            return {};
        }

}
