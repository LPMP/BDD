#include "convert_pb_to_bdd.h"
#include "two_dimensional_variable_array.hxx"
#include <iostream> // TODO: remove

namespace LPMP {

    BDD::node_ref bdd_converter::convert_to_bdd(const std::vector<int>& coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side)
    {
        if(coefficients.size() == 0)
            throw std::runtime_error("Expected non-empty coefficients");
        return convert_to_bdd(coefficients.begin(), coefficients.end(), ineq_type, right_hand_side);
    }

    BDD::node_ref bdd_converter::convert_nonlinear_to_bdd(const std::vector<size_t>& monomial_degrees, const std::vector<int>& coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side)
    {
        assert(monomial_degrees.size() == coefficients.size());

        // first check whether monomial_degrees has entry > 1
        assert(*std::min_element(monomial_degrees.begin(), monomial_degrees.end()) > 0);
        if(*std::max_element(monomial_degrees.begin(), monomial_degrees.end()) == 1)
            return convert_to_bdd(coefficients, ineq_type, right_hand_side);

        // get linear equation
        BDD::node_ref bdd_linear = convert_to_bdd(coefficients, ineq_type, right_hand_side);

        // build node chains for monomials
        // replace nodes with monomials
        auto nodes = bdd_linear.nodes_bfs();
        std::reverse(nodes.begin(), nodes.end());
        std::unordered_map<BDD::node_ref, BDD::node_ref> node_map;
        node_map.insert({bdd_mgr_.botsink(), bdd_mgr_.botsink()});
        node_map.insert({bdd_mgr_.topsink(), bdd_mgr_.topsink()});

        std::vector<size_t> monomial_var_offset;
        // TODO: remove + 1
        monomial_var_offset.reserve(monomial_degrees.size()+1);
        monomial_var_offset.push_back(0);
        for(size_t i=0; i<monomial_degrees.size(); ++i)
            monomial_var_offset.push_back(monomial_var_offset.back() + monomial_degrees[i]);

        for(auto node : nodes)
        {
            assert(!node.is_terminal());
            const size_t var = node.variable();
            assert(var < monomial_degrees.size());

            BDD::node_ref lo = node.low();
            assert(node_map.count(lo) > 0);
            BDD::node_ref lo_nonlinear = node_map.find(lo)->second;

            BDD::node_ref hi = node.high();
            assert(node_map.count(hi) > 0);
            BDD::node_ref hi_nonlinear = node_map.find(hi)->second;

            BDD::node_ref var_node = bdd_mgr_.projection(monomial_var_offset[var] + monomial_degrees[var] - 1);
            BDD::node_ref nonlinear_node = bdd_mgr_.ite_rec(var_node, hi_nonlinear, lo_nonlinear);

            for(std::ptrdiff_t monomial_idx=monomial_degrees[var]-2; monomial_idx>=0; --monomial_idx)
            {
                nonlinear_node = bdd_mgr_.ite_rec(
                        bdd_mgr_.projection(monomial_var_offset[var] + monomial_idx), 
                        nonlinear_node,
                        lo_nonlinear
                        );
            }
            node_map.insert({node, nonlinear_node});
        }

        assert(node_map.count(bdd_linear) > 0);
        return node_map.find(bdd_linear)->second;
    }

    std::tuple<BDD::node_ref, two_dim_variable_array<size_t>> bdd_converter::coefficient_decomposition_convert_to_bdd(const std::vector<int>& coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side)
    {
        assert(coefficients.size() > 0);
        // first divide by largest common denominator
        int gcd = std::abs(right_hand_side);
        if(gcd == 0)
            gcd = std::abs(coefficients[0]);
        for(const int& c : coefficients)
        {
            assert(c != 0);
            gcd = std::gcd(gcd, std::abs(c));
        }

        if(gcd != 1)
        {
            std::vector<int> new_coeffs(coefficients);
            for(auto& c : new_coeffs)
                c /= gcd;
            return coefficient_decomposition_convert_to_bdd(new_coeffs, ineq_type, right_hand_side/gcd);
        }

        std::vector<int> decomposed_coefficients;
        std::vector<size_t> decomposition_multiplicities;
        struct perm_item { size_t pos; size_t orig_coeff_idx; };
        std::vector<perm_item> permutation(decomposed_coefficients.size());

        for(const int& coeff : coefficients)
        {
            assert(coeff != 0);
            std::vector<int> c;
            for(size_t i=1; i<=std::abs(coeff); i*=2)
            {
                if((i & std::abs(coeff)) != 0)
                    c.push_back(i);
            }
            assert(c.size() > 0);
            if(coeff < 0)
                for(int& x : c)
                    x *= -1;
            decomposed_coefficients.insert(decomposed_coefficients.end(), c.begin(), c.end());
            decomposition_multiplicities.push_back(c.size());
            for(size_t i=0; i<c.size(); ++i)
                permutation.push_back({permutation.size(), decomposition_multiplicities.size()-1});
        }
        assert(permutation.size() == decomposed_coefficients.size());

        std::sort(permutation.begin(), permutation.end(), [&](const perm_item i, const perm_item j) {
                return decomposed_coefficients[i.pos] < decomposed_coefficients[j.pos];
                });
        std::vector<int> sorted_decomposed_coefficients(decomposed_coefficients.size());
        for(size_t i=0; i<permutation.size(); ++i)
            sorted_decomposed_coefficients[i] = decomposed_coefficients[permutation[i].pos];
        assert(std::is_sorted(sorted_decomposed_coefficients.begin(), sorted_decomposed_coefficients.end()));

        two_dim_variable_array<size_t> variable_to_coefficient_map(decomposition_multiplicities);
        std::fill(decomposition_multiplicities.begin(), decomposition_multiplicities.end(), 0);
        for(size_t i=0; i<permutation.size(); ++i)
        {
            const size_t orig_coeff_idx = permutation[i].orig_coeff_idx;
            variable_to_coefficient_map(orig_coeff_idx, decomposition_multiplicities[orig_coeff_idx]++) = i;
        }
        for(size_t i=0; i<variable_to_coefficient_map.size(); ++i)
            assert(variable_to_coefficient_map.size(i) == decomposition_multiplicities[i]);

        BDD::node_ref bdd = convert_to_bdd(sorted_decomposed_coefficients, ineq_type, right_hand_side);

        return {bdd, variable_to_coefficient_map};
    }
}
