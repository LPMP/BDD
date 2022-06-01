#pragma once

#include <random>
#include <array>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "mm_primal_decoder.h"
#include "time_measure_util.h"
#include "two_dimensional_variable_array.hxx"
#include "run_solver_util.h"

namespace LPMP {

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

    namespace detail {
        // test whether solver has distribute_delta function
        template <typename S>
            auto distribute_delta(S& solver, int) -> decltype(solver.distribute_delta())
            {
                solver.distribute_delta();
            }

        template <typename S>
            auto distribute_delta(S& solver, double) -> void { } 
    }

    template<typename SOLVER>
        std::vector<char> incremental_mm_agreement_rounding_iter(SOLVER& s, double init_delta = std::numeric_limits<double>::infinity(), const double delta_growth_rate = 1.1, const int num_itr_lb = 100)
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
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

            for(size_t round=0; round<500; ++round)
            {
                cur_delta = std::min(cur_delta*delta_growth_rate, 1e6);
                const auto time = std::chrono::steady_clock::now();
                const double time_elapsed = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
                std::cout << "[incremental primal rounding] round " << round << ", cost delta " << cur_delta << ", time elapsed = " << time_elapsed << "\n";

                // flush stored computations to get best min marginals
                detail::distribute_delta(s, 0);

                const auto mms = mm_primal_decoder(s.min_marginals());
                const auto [nr_one_mms, nr_zero_mms, nr_equal_mms, nr_inconsistent_mms] = mms.mm_type_statistics();
                assert(nr_one_mms + nr_zero_mms + nr_equal_mms + nr_inconsistent_mms == s.nr_variables());

                const int old_precision = std::cout.precision();
                std::cout << std::setprecision(2);
                std::cout << "[incremental primal rounding] " <<
                    "#one min-marg diffs = " << nr_one_mms << " % " << double(100*nr_one_mms)/double(s.nr_variables()) << ", " <<  
                    "#zero min-marg diffs = " << nr_zero_mms << " % " << double(100*nr_zero_mms)/double(s.nr_variables()) << ", " << 
                    "#equal min-marg diffs = " << nr_equal_mms << " % " << double(100*nr_equal_mms)/double(s.nr_variables()) << ", " << 
                    "#inconsistent min-marg diffs = " << nr_inconsistent_mms << " % " << double(100*nr_inconsistent_mms)/double(s.nr_variables()) << "\n";
                std::cout << std::setprecision(old_precision);

                std::uniform_real_distribution<> dis(-cur_delta, cur_delta);

                if(nr_one_mms + nr_zero_mms == s.nr_variables())
                {
                    std::cout << "[incremental primal rounding] Found feasible solution\n";
                    return mms.solution_from_mms();
                }

                std::vector<double> cost_lo_updates(s.nr_variables(), 0.0);
                std::vector<double> cost_hi_updates(s.nr_variables(), 0.0);
                for(size_t i=0; i<s.nr_variables(); ++i)
                {
                    const mm_type mmt = mms.compute_mm_type(i);
                    if(mmt == mm_type::one)
                    {
                        cost_lo_updates[i] = cur_delta;
                        cost_hi_updates[i] = 0.0;
                    }
                    else if(mmt == mm_type::zero)
                    {
                        cost_lo_updates[i] = 0.0;
                        cost_hi_updates[i] = cur_delta;
                    }
                    else if(mmt == mm_type::equal)
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
                        assert(mmt == mm_type::inconsistent);
                        const std::array<double,2> mm_sum = mms.mm_sum(i);
                        //const double r = 5.0*dis(gen);
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
                run_solver(s, num_itr_lb, 1e-7, 0.0001, std::numeric_limits<double>::max(), false);
                std::cout << "[incremental primal rounding] lower bound = " << s.lower_bound() << "\n";
            }

            std::cout << "[incremental primal rounding] No solution found\n";
            return {};
        }
        

    // rounding inspired by Wedelin's algorithm and its refinement in "Learning parameters of the Wedelin heuristic with application to crew and bus driver scheduling"
    // this implementation does not follow the original sequential 
    template<typename SOLVER>
        std::vector<char> wedelin_rounding(
                SOLVER& s, 
                const double theta, // exponential perturbation decay rate
                const double delta, // fixed perturbation strength
                const double kappa_min, const double kappa_max, // proportional perturbation strength w.r.t. min-marginal difference
                const double kappa_step, const double alpha, // adjustment rate for kappa
                const size_t num_itr_lb
                )
        {
            MEASURE_FUNCTION_EXECUTION_TIME;
            constexpr static size_t num_outer_iterations = 500;

            std::cout << "[Wedelin primal rounding] parameters:\n";
            std::cout << "\t\t\ttheta = " << theta << "\n";
            std::cout << "\t\t\tdelta = " << delta << "\n";
            std::cout << "\t\t\tkappa min = " << kappa_min << ", kappa max = " << kappa_max << ", kappa step = " << kappa_step << ", alpha = " << alpha << "\n";
            assert(theta >= 0.0 && theta <= 1.0);
            assert(delta >= 0.0);
            assert(0.0 <= kappa_min && kappa_min < kappa_max && kappa_max < 1.0);
            assert(kappa_step >= 0.0 && kappa_step <= 1.0);
            assert(alpha >= 0.0);

            std::default_random_engine gen{static_cast<long unsigned int>(0)}; // deterministic seed for repeatable experiments
            std::uniform_real_distribution<> dis(-delta, delta);

            const auto mms = s.min_marginals();
            std::vector<size_t> nr_bdds;
            nr_bdds.reserve(s.nr_variables());
            for(size_t i=0; i<s.nr_variables(); ++i)
                nr_bdds.push_back(s.nr_bdds(i));
            two_dim_variable_array<std::array<double,2>> p(nr_bdds.begin(), nr_bdds.end());
            for(size_t i=0; i<p.size(); ++i)
                for(size_t j=0; j<p.size(i); ++j)
                    p(i,j) = {0.0, 0.0};
            two_dim_variable_array<std::array<double,2>> p_delta = p;

            double kappa = kappa_min;
            for(size_t iter=0; iter<num_outer_iterations && kappa <= kappa_max; ++iter)
            {
                // reset perturbation
                // decay perturbtations
                for(size_t i=0; i<p.size(); ++i)
                {
                    for(size_t j=0; j<p.size(i); ++j)
                    {
                        // remove exponential decay
                        p_delta(i,j)[0] = -(1-theta)*p(i,j)[0];
                        p_delta(i,j)[1] = -(1-theta)*p(i,j)[1];
                    }
                }

                mm_primal_decoder mms(s.min_marginals());

                const auto [nr_one_mms, nr_zero_mms, nr_equal_mms, nr_inconsistent_mms] = mms.mm_type_statistics();

                if(mms.can_reconstruct_solution())
                {
                    std::cout << "[Wedelin primal rounding] found primal solution\n";
                    return mms.solution_from_mms();
                }

                std::cout << "[Wedelin primal rounding] iteration " << iter << ", kappa = " << kappa << "\n";
                const int old_precision = std::cout.precision();
                std::cout << std::setprecision(2);
                std::cout << "[Wedelin primal rounding] " <<
                    "#one min-marg diffs = " << nr_one_mms << " % " << double(100*nr_one_mms)/double(s.nr_variables()) << ", " <<  
                    "#zero min-marg diffs = " << nr_zero_mms << " % " << double(100*nr_zero_mms)/double(s.nr_variables()) << ", " << 
                    "#equal min-marg diffs = " << nr_equal_mms << " % " << double(100*nr_equal_mms)/double(s.nr_variables()) << ", " << 
                    "#inconsistent min-marg diffs = " << nr_inconsistent_mms << " % " << double(100*nr_inconsistent_mms)/double(s.nr_variables()) << "\n";
                std::cout << std::setprecision(old_precision);

                double sum_Deltas = 0.0;
                for(size_t i=0; i<p.size(); ++i)
                {
                    const double r = dis(gen);
                    assert(-delta <= r && r <= delta);
                    for(size_t j=0; j<p.size(i); ++j)
                    {
                        const double Delta = kappa / (1-kappa) * std::abs(mms(i,j)[1] - mms(i,j)[0]) + delta;
                        assert(Delta > 0.0);
                        const mm_type mmt = mms.compute_mm_type(i);
                        if(mmt == mm_type::zero)
                            p_delta(i,j)[1] += Delta;
                        else if(mmt == mm_type::zero)
                            p_delta(i,j)[0] -= Delta;
                        else
                        {
                            if(r < 0.0)
                                p_delta(i,j)[0] += Delta;
                            else
                                p_delta(i,j)[1] += Delta;
                        }

                        //if(mms(i,j)[0] < mms(i,j)[1])
                        //    p_delta(i,j)[1] += Delta;
                        //else
                        //    p_delta(i,j)[0] -= Delta;
                        sum_Deltas += Delta;
                    }
                }
                std::cout << "[Wedelin primal rounding] Sum of all Delta perturbations = " << sum_Deltas << "\n";

                s.update_costs(p_delta);
                for(size_t i=0; i<p.size(); ++i)
                {
                    for(size_t j=0; j<p.size(i); ++j)
                    {
                        // this also takes into account the exponential decay
                        p(i,j)[0] += p_delta(i,j)[0];
                        p(i,j)[1] += p_delta(i,j)[1];
                        p_delta(i,j)[0] = 0.0;
                        p_delta(i,j)[1] = 0.0;
                    }
                }

                run_solver(s, num_itr_lb, 1e-7, 0.0001, std::numeric_limits<double>::max(), false);
                std::cout << "[Wedelin primal rounding] lower bound = " << s.lower_bound() << "\n";

                kappa += kappa_step * std::exp( alpha * std::log(double(nr_equal_mms + nr_inconsistent_mms)/double(s.nr_variables())) );
            }

            std::cout << "[Wedelin primal rounding] did not find a primal solution\n";
            return {};
        }

}
