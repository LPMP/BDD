#include "bdd_manager/bdd_mgr.h"
#include "bdd_collection/bdd_collection.h"
#include <unordered_set>
#include "../test.h"
#include <iostream> // TODO: remove

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_mgr mgr;
    bdd_collection collection;

    // check simple covering problem
    for(size_t i=0; i<6; ++i)
        mgr.add_variable();

    std::vector<node_ref> vars;
    for(size_t i=0; i<6; ++i)
        vars.push_back(mgr.projection(i));

    auto simplex = mgr.simplex(vars.begin(), vars.end());
    auto relaxed_simplex = mgr.or_rec(simplex, mgr.projection(0), mgr.projection(1));// mgr.negate(mgr.projection(3)), mgr.negate(mgr.projection(4)));

    const size_t simplex_nr = collection.add_bdd(simplex);

    std::unordered_set<size_t> pos_vars;
    pos_vars.insert(0);
    pos_vars.insert(1);
    std::unordered_set<size_t> neg_vars;
    //neg_vars.insert(3);
    //neg_vars.insert(4);

    const size_t relaxed_simplex_nr = collection.bdd_or_var(simplex_nr, pos_vars, neg_vars);

    auto relaxed_simplex_test = mgr.add_bdd(collection, relaxed_simplex_nr);

    std::array<char,6> l;
    for(l[0]=0; l[0]<2; ++l[0])
        for(l[1]=0; l[1]<2; ++l[1])
            for(l[2]=0; l[2]<2; ++l[2])
                for(l[3]=0; l[3]<2; ++l[3])
                    for(l[4]=0; l[4]<2; ++l[4])
                        for(l[5]=0; l[5]<2; ++l[5])
                        {
                            //std::cout << relaxed_simplex.evaluate(l.begin(), l.end()) << " = " << collection.evaluate(relaxed_simplex_nr, l.begin(), l.end()) << "\n";
                            test(simplex.evaluate(l.begin(), l.end()) == collection.evaluate(simplex_nr, l.begin(), l.end()));
                            test(relaxed_simplex.evaluate(l.begin(), l.end()) == collection.evaluate(relaxed_simplex_nr, l.begin(), l.end()));
                            std::cout << relaxed_simplex.evaluate(l.begin(), l.end()) << " = " << relaxed_simplex_test.evaluate(l.begin(), l.end()) << "\n";
                            test(relaxed_simplex.evaluate(l.begin(), l.end()) == relaxed_simplex_test.evaluate(l.begin(), l.end()));
                        }
    test(relaxed_simplex == relaxed_simplex_test, "or_var not giving equivalence");
}
