#pragma once

#include <vector>
#include <random>
#include "ILP_input.h"

namespace LPMP {

    // coefficients, inequality type, right_hand_side
    std::tuple<std::vector<int>, ILP_input::inequality_type, int> generate_random_inequality(const std::size_t nr_vars)
    {
        std::uniform_int_distribution<> d(-10,10);
        std::mt19937 gen;

        std::vector<int> coefficients;
        for(std::size_t i=0; i<nr_vars; ++i)
        {
            int coeff = d(gen);
            if(coeff == 0)
                coeff = 1;
            coefficients.push_back( coeff );
        }

        ILP_input::inequality_type ineq = ILP_input::inequality_type::smaller_equal;
        // otherwise we cannot guarantee that every variable can take both values. Note: this can be reverted once we have bdd preprocessor filtering out fixed variables
        /*
           ILP_input::inequality_type ineq = [&]() {
           const int r = d(gen);
           if(r > 2)
           return ILP_input::inequality_type::smaller_equal;
           else if(r < -2)
           return ILP_input::inequality_type::greater_equal;
           else
           return ILP_input::inequality_type::equal;
           }();
           */

        // make right hand side so that every variable can take both 0 and 1
        int sum_negative = 0;
        for(auto c : coefficients)
            sum_negative += std::min(c,0);
        int max_positive = 0;
        for(auto c : coefficients)
            max_positive = std::max(c, max_positive);

        int rhs = std::max(sum_negative + max_positive, d(gen)); 

        return {coefficients, ineq, rhs}; 
    }

    std::vector<double> generate_random_costs(const std::size_t nr_vars)
    {
        std::uniform_int_distribution<> d(-10,10);
        std::mt19937 gen;

        std::vector<double> coefficients;
        for(std::size_t i=0; i<nr_vars; ++i)
            coefficients.push_back( d(gen) ); 

        return coefficients;
    }

    template<typename LHS_ITERATOR, typename COST_ITERATOR, typename SOL_ITERATOR>
        double min_cost_impl(LHS_ITERATOR lhs_begin, LHS_ITERATOR lhs_end, const ILP_input::inequality_type ineq, const int rhs, COST_ITERATOR cost_begin, COST_ITERATOR cost_end, SOL_ITERATOR sol_begin, const double partial_cost, double& best_current_sol)
        {
            assert(std::distance(lhs_begin, lhs_end) == std::distance(cost_begin, cost_end));

            if(lhs_begin == lhs_end) {
                if(ineq == ILP_input::inequality_type::equal) {
                    return rhs == 0 ? 0.0 : std::numeric_limits<double>::infinity();
                } else if(ineq == ILP_input::inequality_type::smaller_equal) {
                    return rhs >= 0 ? 0.0 : std::numeric_limits<double>::infinity();
                } else if(ineq == ILP_input::inequality_type::greater_equal) {
                    return rhs <= 0 ? 0.0 : std::numeric_limits<double>::infinity();
                } 
            }

            const double zero_cost = min_cost_impl(lhs_begin+1, lhs_end, ineq, rhs, cost_begin+1, cost_end, sol_begin+1, partial_cost, best_current_sol);
            const double one_cost = min_cost_impl(lhs_begin+1, lhs_end, ineq, rhs - *lhs_begin, cost_begin+1, cost_end, sol_begin+1, partial_cost + *cost_begin, best_current_sol) + *cost_begin;

            const double sub_tree_cost = std::min(zero_cost, one_cost);
            const double cur_cost = partial_cost + sub_tree_cost;
            if(cur_cost <= best_current_sol) {
                best_current_sol = cur_cost;
                *sol_begin = zero_cost < one_cost ? 0 : 1; 
            }

            return std::min(zero_cost, one_cost);
        }

    template<typename LHS_ITERATOR, typename COST_ITERATOR>
        std::tuple<double, std::vector<char>> min_cost(LHS_ITERATOR lhs_begin, LHS_ITERATOR lhs_end, const ILP_input::inequality_type ineq, const int rhs, COST_ITERATOR cost_begin, COST_ITERATOR cost_end)
        {
            std::vector<char> sol(std::distance(lhs_begin, lhs_end));

            double opt_val = std::numeric_limits<double>::infinity();
            const double opt_val_2 = min_cost_impl(lhs_begin, lhs_end, ineq, rhs, cost_begin, cost_end, sol.begin(), 0.0, opt_val);
            assert(opt_val == opt_val_2);

            return {opt_val, sol};
        }

    template<typename LHS_ITERATOR, typename COST_ITERATOR>
        double exp_sum_impl(LHS_ITERATOR lhs_begin, LHS_ITERATOR lhs_end, const ILP_input::inequality_type ineq, const int rhs, COST_ITERATOR cost_begin, COST_ITERATOR cost_end, const double partial_sum)
        {
            assert(std::distance(lhs_begin, lhs_end) == std::distance(cost_begin, cost_end));

            if(lhs_begin == lhs_end) {
                if(ineq == ILP_input::inequality_type::equal) {
                    return rhs == 0 ? std::exp(partial_sum) : 0.0;
                } else if(ineq == ILP_input::inequality_type::smaller_equal) {
                    return rhs >= 0 ? std::exp(partial_sum) : 0.0;
                } else if(ineq == ILP_input::inequality_type::greater_equal) {
                    return rhs <= 0 ? std::exp(partial_sum) : 0.0;
                } 
            }

            const double zero_cost = exp_sum_impl(lhs_begin+1, lhs_end, ineq, rhs, cost_begin+1, cost_end, partial_sum);
            const double one_cost = exp_sum_impl(lhs_begin+1, lhs_end, ineq, rhs - *lhs_begin, cost_begin+1, cost_end, partial_sum - *cost_begin);

            return zero_cost + one_cost;
        }

    template<typename LHS_ITERATOR, typename COST_ITERATOR>
        double log_exp(LHS_ITERATOR lhs_begin, LHS_ITERATOR lhs_end, const ILP_input::inequality_type ineq, const int rhs, COST_ITERATOR cost_begin, COST_ITERATOR cost_end)
        {
            const double sum = exp_sum_impl(lhs_begin, lhs_end, ineq, rhs, cost_begin, cost_end, 0.0);
            return -std::log(sum);
        } 

    ILP_input generate_ILP(const std::vector<int>& coefficients, ILP_input::inequality_type ineq, const int rhs)
    {
        for(const auto c : coefficients) {
            std::cout << c << " ";
        }
        if(ineq == ILP_input::inequality_type::equal)
            std::cout << " = ";
        if(ineq == ILP_input::inequality_type::smaller_equal)
            std::cout << " <= ";
        if(ineq == ILP_input::inequality_type::greater_equal)
            std::cout << " >= ";
        std::cout << rhs << "\n";

        ILP_input ilp;
        ilp.begin_new_inequality();
        for(size_t i=0; i<coefficients.size(); ++i)
        {
            ilp.add_new_variable("x" + std::to_string(i));
            ilp.add_to_constraint(coefficients[i], i);
        }
        ilp.set_inequality_type(ineq);
        ilp.set_right_hand_side(rhs);

        const std::vector<double> costs = generate_random_costs(coefficients.size());
        std::cout << "cost: ";
        for(const auto x : costs)
            std::cout << x << " ";
        std::cout << "\n"; 
        for(size_t i=0; i<costs.size(); ++i)
            ilp.add_to_objective(costs[i], i); 

        return ilp;
    }

}
