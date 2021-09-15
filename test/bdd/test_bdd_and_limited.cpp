#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_mgr mgr;

    for(size_t i=0; i<10; ++i)
        mgr.add_variable();

    std::vector<node_ref> vars;
    for(size_t i=0; i<10; ++i)
        vars.push_back(mgr.projection(i));

    node_ref simplex_1 = mgr.simplex(vars.begin(), vars.begin()+5);
    test(simplex_1.nr_nodes() == 9, "simplex has wrong nr of nodes");
    node_ref simplex_2 = mgr.simplex(vars.begin()+5, vars.end());
    test(simplex_2.nr_nodes() == 9, "simplex has wrong nr of nodes");

    {
    auto [simplex_and, nr_simplex_and_nodes] = mgr.and_rec_limited(simplex_1, simplex_2, 9 + 9);
    test(simplex_and.address() == nullptr, "simplex intersection too big");
    }

    {
        auto [simplex_and, nr_simplex_and_nodes] = mgr.and_rec_limited(simplex_1, simplex_2, 100);
        test(simplex_and.address() != nullptr, "simplex intersection not constructed big");
        test(simplex_and == mgr.and_rec(simplex_1, simplex_2), "limited simplex intersection computation incorrect");
    } 
}
