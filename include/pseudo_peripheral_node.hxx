#pragma once

#include "union_find.hxx"
#include <queue>
#include <vector>
#include <tuple>
#include <algorithm>
#include "time_measure_util.h"

namespace LPMP {

template <typename ADJACENCY_GRAPH>
std::tuple<size_t, size_t> farthest_node(const ADJACENCY_GRAPH &adjacency, const size_t x)
{
    MEASURE_FUNCTION_EXECUTION_TIME;
    std::vector<char> visited(adjacency.size(), 0);
    return farthest_node(adjacency, x, visited, 0);
}

template <typename ADJACENCY_GRAPH, typename VISITED_VECTOR>
std::tuple<size_t, size_t> farthest_node(const ADJACENCY_GRAPH &adjacency, const size_t x, VISITED_VECTOR &visited, const size_t timestamp)
{
    assert(visited.size() == adjacency.size());
    assert(*std::max_element(visited.begin(), visited.end()) <= timestamp);

    size_t d = 0;
    struct queue_elem
    {
        size_t v;
        size_t d;
    };
    std::queue<queue_elem> Q;
    Q.push({x, 0});
    visited[x] = timestamp + 1;
    size_t farthest_node = x;
    size_t max_distance = 0;
    while (!Q.empty())
    {
        const auto [i, d] = Q.front();
        Q.pop();
        assert(visited[i] == timestamp + 1);
        visited[i] = timestamp + 2;
        if (d > max_distance)
        {
            max_distance = d;
            farthest_node = x;
        }
        for (const auto j : adjacency[i])
        {
            if (visited[j] <= timestamp)
            {
                Q.push({j, d + 1});
                visited[j] = timestamp + 1;
            }
        }
    }

    return {farthest_node, max_distance};
}

    template<typename ADJACENCY_GRAPH>
    size_t find_pseudo_peripheral_node(const ADJACENCY_GRAPH& adjacency)
    {
        size_t min_degree = adjacency[0].size();
        size_t x = 0;
        for(size_t i=0; i<adjacency.size(); ++i) {
            if(adjacency[i].size() < min_degree) {
                min_degree = adjacency[i].size();
                x = i;
            }
        }

        assert(x < adjacency.size());
        auto [y, d_y] = farthest_node(adjacency, x);
        auto [z, d_z] = farthest_node(adjacency, y);
        while(d_z > d_y) {
            std::swap(y,z);
            std::swap(d_z, d_y);
            std::tie(z, d_z) = farthest_node(adjacency,y);
        }
        return y; 
    }

    // return pseudo peripheral node of each connected component
    template<typename ADJACENCY_GRAPH>
    std::vector<size_t> find_pseudo_peripheral_nodes(const ADJACENCY_GRAPH& adjacency)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        // compute connected components of graph and for each connected component determine node of minimum degree
        union_find uf(adjacency.size());
        for(size_t i=0; i<adjacency.size(); ++i)
            for(const size_t j : adjacency[i])
                uf.merge(i, j);

        struct min_degree_elem
        {
            size_t degree = std::numeric_limits<size_t>::max();
            size_t node = std::numeric_limits<size_t>::max();
        };
        std::vector<min_degree_elem> min_degree(adjacency.size());

        for(size_t i=0; i<adjacency.size(); ++i)
        {
            const size_t cc_id = uf.find(i);
            const size_t d = adjacency[i].size();
            if (adjacency[i].size() < min_degree[cc_id].degree)
            {
                min_degree[cc_id].degree = adjacency[i].size();
                min_degree[cc_id].node = i;
            }
        }

        std::vector<size_t> pseudo_peripheral_nodes;
        std::vector<size_t> visited(adjacency.size(), 0);
        size_t iter = 0;

        for (size_t i = 0; i < adjacency.size(); ++i)
        { 
            if(visited[i] != 0)
                continue;

            const size_t x = min_degree[uf.find(i)].node;

            assert(x < adjacency.size());

            auto [y, d_y] = farthest_node(adjacency, x, visited, 2*iter++);
            auto [z, d_z] = farthest_node(adjacency, y, visited, 2*iter++);
            while (d_z > d_y)
            {
                std::swap(y, z);
                std::swap(d_z, d_y);
                std::tie(z, d_z) = farthest_node(adjacency, y, visited, 2*iter++);
            }
            pseudo_peripheral_nodes.push_back(y);
        }

        return pseudo_peripheral_nodes;
    }

    template<typename ADJACENCY_GRAPH, typename NODE_ITERATOR>
    size_t find_pseudo_peripheral_node(const ADJACENCY_GRAPH& adjacency, NODE_ITERATOR node_begin, NODE_ITERATOR node_end)
    {
        //assert(std::distance(node_begin, node_end) > 0);
        size_t min_degree = adjacency[*node_begin].size();
        size_t x = *node_begin;
        for(auto node_it=node_begin; node_it!=node_end; ++node_it) {
            if(adjacency[*node_it].size() < min_degree) {
                min_degree = adjacency[*node_it].size();
                x = *node_it;
            }
        }

        assert(x < adjacency.size());
        auto [y, d_y] = farthest_node(adjacency, x);
        auto [z, d_z] = farthest_node(adjacency, y);
        while(d_z > d_y) {
            std::swap(y,z);
            std::swap(d_z, d_y);
            std::tie(z, d_z) = farthest_node(adjacency,y);
        }
        return y; 
    }

    /*
    struct iterator {
        size_t i;
        size_t operator*() const { return i; }
        void operator++() { ++i; }
        bool operator!=(const iterator o) { return o.i != this->i; }
        size_t operator-(const iterator o) const { return o.i - this->i; }
    }; 

    template<typename ADJACENCY_GRAPH>
    size_t find_pseudo_peripheral_node(const ADJACENCY_GRAPH& adjacency)
    {
        return find_pseudo_peripheral_node(adjacency, iterator({0}), iterator({adjacency.size()}));
    }
    */

} 
