#include "bdd_manager/bdd_mgr.h"
#include "../test.h"
#include <vector>
#include <unordered_set>

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_mgr mgr;
    for(size_t i=0; i<50; ++i)
        mgr.add_variable();
    std::vector<node_ref> vars;
    for(size_t i=0; i<50; ++i)
        vars.push_back(mgr.projection(i));
    node_ref simplex = mgr.simplex(vars.begin(), vars.end());
    std::vector<node_ref> simplex_nodes = simplex.nodes_postorder();

    std::unordered_set<node_ref> node_map;
    for(node_ref x : simplex_nodes)
    {
        test(node_map.count(x.low()) > 0 || x.low().is_terminal(), "nodes not in postorder");
        test(node_map.count(x.high()) > 0 || x.high().is_terminal(), "nodes not in postorder");
        node_map.insert(x);
    } 
}
