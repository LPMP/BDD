#pragma once

#include "two_dimensional_variable_array.hxx"
#include "mm_primal_decoder.h"
#include "run_solver_util.h"

namespace LPMP {

    template<typename SOLVER>
        std::vector<char> wedelin_rounding(
                SOLVER& s, 
                const ILP_input& ilp,
                const double theta, // exponential perturbation decay rate
                const double delta, // fixed perturbation strength
                const double kappa_min, const double kappa_max, // proportional perturbation strength w.r.t. min-marginal difference
                const double kappa_step, const double alpha, // adjustment rate for kappa
                const size_t num_itr_lb
                )
        {
            std::cout << "[Wedelin primal rounding] parameters:\n";
            std::cout << "\t\t\ttheta = " << theta << "\n";
            std::cout << "\t\t\tdelta = " << delta << "\n";
            std::cout << "\t\t\tkappa min = " << kappa_min << ", kappa max = " << kappa_max << ", kappa step = " << kappa_step << ", alpha = " << alpha << "\n";

            two_dim_variable_array<std::array<double,2>> bdd_cost_updates = s.min_marginals();
            for(size_t i=0; i<bdd_cost_updates.size(); ++i)
                for(size_t j=0; j<bdd_cost_updates.size(i); ++j)
                    bdd_cost_updates(i,j) = {0.0, 0.0};
            two_dim_variable_array<std::array<double,2>> perturbations = bdd_cost_updates;

            double kappa = kappa_min;

            std::vector<char> solution(s.nr_variables());

            for(size_t iter=0; iter<500; ++iter)
            {
                mm_primal_decoder mms(s.min_marginals());

                std::cout << "[Wedelin primal rounding] iteration " << iter << ", kappa = " << kappa << "\n";
                const auto [nr_one_mms, nr_zero_mms, nr_equal_mms, nr_inconsistent_mms] = mms.mm_type_statistics();
                const int old_precision = std::cout.precision();
                std::cout << std::setprecision(2);
                std::cout << "[Wedelin primal rounding] " <<
                    "#one min-marg diffs = " << nr_one_mms << " % " << double(100*nr_one_mms)/double(s.nr_variables()) << ", " <<  
                    "#zero min-marg diffs = " << nr_zero_mms << " % " << double(100*nr_zero_mms)/double(s.nr_variables()) << ", " << 
                    "#equal min-marg diffs = " << nr_equal_mms << " % " << double(100*nr_equal_mms)/double(s.nr_variables()) << ", " << 
                    "#inconsistent min-marg diffs = " << nr_inconsistent_mms << " % " << double(100*nr_inconsistent_mms)/double(s.nr_variables()) << "\n";
                std::cout << std::setprecision(old_precision);

                if(mms.can_reconstruct_solution())
                {
                    std::cout << "[Wedelin primal rounding] found primal solution\n";
                    return mms.solution_from_mms();
                }

                if(iter == 0) // build initial primal
                    solution = mms.solution_from_mms();

                // initialize current bdd_cost_updates and exponentially decay perturbations
                for(size_t i=0; i<bdd_cost_updates.size(); ++i)
                {
                    for(size_t j=0; j<bdd_cost_updates.size(i); ++j)
                    {
                        bdd_cost_updates(i,j)[0] = -(1-theta) * bdd_cost_updates(i,j)[0];
                        bdd_cost_updates(i,j)[1] = -(1-theta) * bdd_cost_updates(i,j)[1];
                        perturbations(i,j)[0] = theta * bdd_cost_updates(i,j)[0];
                        perturbations(i,j)[1] = theta * bdd_cost_updates(i,j)[1];
                    }
                }

                const auto bdd_feasibility = s.bdd_feasibility(solution.begin(), solution.end());

                for(size_t i=0; i<bdd_feasibility.size(); ++i)
                {
                    const bool preferred_sol = [&]() -> bool {
                        std::array<size_t,2> prefs = {0, 0};
                        for(size_t j=0; j<mms.size(i); ++j)
                            if(mms(i,j)[0] <= mms(i,j)[1])
                                ++prefs[0];
                            else
                                ++prefs[1];
                        if(prefs[0] < prefs[1])
                            return 0;
                        else if(prefs[0] > prefs[1])
                            return 1;
                        else
                        {
                            const auto mm_sum = mms.mm_sum(i);
                            return mm_sum[0] <= mm_sum[1];
                        }
                    }();

                    solution[i] = preferred_sol;

                    for(size_t j=0; j<bdd_feasibility.size(i); ++j)
                    {
                        if(!bdd_feasibility(i,j))
                        {
                            const double Delta = kappa / (1-kappa) * std::abs(mms(i,j)[1] - mms(i,j)[0]) + delta;
                            if(mms(i,j)[0] <= mms(i,j)[1])
                            {
                                bdd_cost_updates(i,j)[1] += Delta;
                            }
                            else if(mms(i,j)[0] > mms(i,j)[1])
                            {
                                bdd_cost_updates(i,j)[0] += Delta;
                            }
                        }
                    }
                }

                // TODO: possibly bypass through bdd_feasibility from solver?
                if(ilp.feasible(solution.begin(), solution.end()))
                {
                    std::cout << "[Wedelin primal heuristic] found primal solution\n";
                    return solution;
                }

                s.update_costs(bdd_cost_updates);
                run_solver(s, num_itr_lb, 1e-7, 0.0001, std::numeric_limits<double>::max(), false);

                // add current cost updates to history
                for(size_t i=0; i<perturbations.size(); ++i)
                {
                    for(size_t j=0; j<perturbations.size(i); ++j)
                    {
                        perturbations(i,j)[0] += bdd_cost_updates(i,j)[0];
                        perturbations(i,j)[1] += bdd_cost_updates(i,j)[1];
                    }
                }

                std::cout << "[Wedelin primal rounding] lower bound = " << s.lower_bound() << "\n";
                kappa += kappa_step * std::exp( alpha * std::log(double(nr_equal_mms + nr_inconsistent_mms)/double(s.nr_variables())) );
                if(kappa >= 1.0)
                {
                    std::cout << "[Wedelin primal heuristic] kappa " << kappa << " larger than 1.0, aborting primal search\n";
                    return {};
                }
            }

            std::cout << "[Wedelin primal rounding] could not find solution\n";
            return {};
        }

}
