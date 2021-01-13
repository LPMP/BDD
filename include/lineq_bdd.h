#pragma once

#include <numeric>
#include <list>
#include <vector>
#include <tsl/robin_map.h>
#include "bdd.h"
#include <iostream> // temporary

namespace LPMP {

    struct lineq_bdd_node {

        lineq_bdd_node() {}
        lineq_bdd_node(int lb, int ub, lineq_bdd_node* zero_kid, lineq_bdd_node* one_kid)
        : lb_(lb), ub_(ub), zero_kid_(zero_kid), one_kid_(one_kid)
        {}

        int lb_;
        int ub_;

        lineq_bdd_node* zero_kid_;
        lineq_bdd_node* one_kid_;
    };

    struct lineq_bdd {

        lineq_bdd() {}
        lineq_bdd(lineq_bdd & other) = delete;
        lineq_bdd(const size_t dim) : inverted(dim), levels(dim),
            topsink(0, std::numeric_limits<int>::max(), nullptr, nullptr), 
            botsink(std::numeric_limits<int>::min(), -1, nullptr, nullptr)
        {}

        BDD::node_ref convert_to_bdd(BDD::bdd_mgr & bdd_mgr_) const;

        lineq_bdd_node* root_node;

        std::vector<char> inverted; // flags inverted variables

        lineq_bdd_node topsink;
        lineq_bdd_node botsink;
        std::vector<std::list<lineq_bdd_node>> levels;
    };

    inline BDD::node_ref lineq_bdd::convert_to_bdd(BDD::bdd_mgr & bdd_mgr_) const
    {
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
