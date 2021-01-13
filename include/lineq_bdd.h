#pragma once

#include <numeric>
#include "bdd.h"

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
        lineq_bdd(const size_t dim) : inverted(dim), levels(dim),
            topsink(0, std::numeric_limits<int>::max(), nullptr, nullptr), 
            botsink(std::numeric_limits<int>::min(), -1, nullptr, nullptr)
        {}

        BDD::node_ref convert_to_bdd(BDD::bdd_mgr bdd_mgr_) const; 

        lineq_bdd_node* root_node;

        std::vector<char> inverted; // flags inverted variables

        lineq_bdd_node topsink;
        lineq_bdd_node botsink;
        std::vector<std::vector<lineq_bdd_node>> levels;
    };

    inline BDD::node_ref lineq_bdd::convert_to_bdd(BDD::bdd_mgr bdd_mgr_) const
    {
        std::vector<std::vector<BDD::node_ref>> bdd_nodes(levels.size());
        for(std::ptrdiff_t l=levels.size()-1; l>=0; --l)
        {
            for(size_t i=0; i<levels[l].size(); ++i)
            {
                auto& lbdd = levels[l][i];
                auto get_node = [&](lineq_bdd_node const* ptr) {
                    if(ptr == &botsink)
                    {
                        return bdd_mgr_.botsink();
                    }
                    else if(ptr == &topsink)
                    {
                        return bdd_mgr_.botsink();
                    }
                    else
                    {
                        assert(l+1 < levels.size());
                        const size_t offset = std::distance(&levels[l+1][0], ptr);
                        assert(offset < levels[l+1].size());
                        return bdd_nodes[l+1][offset]; 
                    }
                };
                BDD::node_ref zero_bdd_node = get_node(lbdd.zero_kid_);
                BDD::node_ref one_bdd_node = get_node(lbdd.one_kid_);
                bdd_nodes[l].push_back(bdd_mgr_.ite_rec(bdd_mgr_.projection(l), zero_bdd_node, one_bdd_node));
            }
        }
        return bdd_nodes[0][0];
    }

}
