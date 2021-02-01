#include "lineq_bdd.h"

namespace LPMP {

    void lineq_bdd::build_from_inequality(const std::vector<int> coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side)
    {
        build_from_inequality(coefficients.begin(), coefficients.end(), ineq_type, right_hand_side);
    }


    lineq_bdd_node* lineq_bdd::build_bdd_node(const int path_cost, const unsigned int level, const ILP_input::inequality_type ineq_type)
    {
        assert(level < rests.size());
        const int slack = rhs - path_cost;
        const long int rest = rests[level];

        // check sink conditions
        switch (ineq_type)
        {
            case ILP_input::inequality_type::equal:
                if (slack < 0 || slack > rest)
                    return &botsink;
                if (slack == 0 && slack == rest)
                    return &topsink;
                break;
            case ILP_input::inequality_type::smaller_equal:
                if (slack < 0)
                    return &botsink;
                if (slack >= rest)
                    return &topsink;
                break;
            case ILP_input::inequality_type::greater_equal:
                throw std::runtime_error("greater equal constraint not in normal form");
                break;
            default:
                throw std::runtime_error("inequality type not supported");
                break;
        }

        assert(level < levels.size());

        // check for equivalent nodes
        // TODO: implement binary search over nodes at current level
        for (auto it = levels[level].begin(); it != levels[level].end(); it++)
        {
            if (slack >= it->lb_ && slack <= it->ub_)
                return &(*it);
        }

        // otherwise create new node
        lineq_bdd_node node;
        node.ub_ = path_cost;
        levels[level].push_back(node);
        return &levels[level].back();
    }


    BDD::node_ref lineq_bdd::convert_to_lbdd(BDD::bdd_mgr & bdd_mgr_) const
    {
        if (root_node == &topsink)
            return bdd_mgr_.topsink();
        if (root_node == &botsink)
            return bdd_mgr_.botsink();

        std::vector<std::vector<BDD::node_ref>> bdd_nodes(levels.size());
        tsl::robin_map<lineq_bdd_node const*,size_t> node_refs;
        for(std::ptrdiff_t l=levels.size()-1; l>=0; --l)
        {
            for(auto it = levels[l].begin(); it != levels[l].end(); it++)
            {
                auto& lbdd = *it;
                auto get_node = [&](lineq_bdd_node const* ptr) {
                    if(ptr == &botsink)
                        return bdd_mgr_.botsink();
                    else if(ptr == &topsink)
                        return bdd_mgr_.topsink();
                    else
                    {
                        auto ref = node_refs.find(ptr);
                        if (ref != node_refs.end())
                        {
                            assert(ref->second < bdd_nodes[l+1].size());
                            return bdd_nodes[l+1][ref->second];
                        }
                        else
                            throw std::runtime_error("node reference not found");
                    }
                };
                BDD::node_ref zero_bdd_node = get_node(lbdd.zero_kid_);
                BDD::node_ref one_bdd_node = get_node(lbdd.one_kid_);
                if (inverted[l])
                    bdd_nodes[l].push_back(bdd_mgr_.ite_rec(bdd_mgr_.projection(l), one_bdd_node, zero_bdd_node));
                else
                    bdd_nodes[l].push_back(bdd_mgr_.ite_rec(bdd_mgr_.projection(l), zero_bdd_node, one_bdd_node));
                node_refs.insert(std::make_pair(&lbdd, bdd_nodes[l].size()-1));
            }
        }
        assert(bdd_nodes[0].size() == 1);
        return bdd_nodes[0][0];
    }

}
