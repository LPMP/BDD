#include "convert_pb_to_bdd.h"
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

}
