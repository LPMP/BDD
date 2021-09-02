#pragma once

#include <memory>
#include <array>
#include "bdd_node.h"

namespace BDD {

constexpr static size_t bdd_node_page_size = 4096;

struct bdd_node_page
{
    std::array<node,bdd_node_page_size> data;
    std::unique_ptr<bdd_node_page> next;
};

class bdd_mgr;

class bdd_node_cache
{
    public:
        bdd_node_cache(bdd_mgr* mgr);
        node* reserve_node(void);
        void free_node(node*p);
        size_t nr_nodes() const { return total_nodes; }
        node* botsink() const { return botsink_; }
        node* topsink() const { return topsink_; }

    private:
        void increase_cache(); // double size of node cache

        std::unique_ptr<bdd_node_page> mem_node;
        node* nodeavail; // smallest unused node in mem_node
        node* nodeptr; // stack of nodes available for reuse
        // sink nodes
        node* botsink_;
        node* topsink_; 
        size_t total_nodes = 2; // nr nodes currently in use
        size_t deadnodes = 0; // nr nodes currently having xref < 0
};

}
