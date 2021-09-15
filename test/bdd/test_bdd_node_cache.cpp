#include "bdd_manager/bdd_node_cache.h"
#include "../test.h"
#include <vector>

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_node_cache cache(nullptr);
    std::vector<node*> v;
    const std::size_t nr_nodes_to_insert = 10000;
    for(std::size_t i=0; i<nr_nodes_to_insert; ++i)
    {
        v.push_back(cache.reserve_node());
        v.back()->lo = cache.topsink();
        v.back()->hi = cache.topsink();
        cache.topsink()->xref++;
        cache.topsink()->xref++;
        test(i+1+2 == cache.nr_nodes(), "node counting error when adding nodes");
    }

    for(std::size_t i=0; i<nr_nodes_to_insert; ++i)
    {
        cache.free_node(v[i]);
        test(nr_nodes_to_insert + 2 - i - 1 == cache.nr_nodes(), "node counting error when freeing nodes");
    } 

    test(2 == cache.nr_nodes());
}
