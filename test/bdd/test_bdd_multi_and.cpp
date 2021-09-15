#include "bdd_collection/bdd_collection.h"
#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_collection collection;
    bdd_mgr mgr;

    for(size_t i=0; i<8; ++i)
        mgr.add_variable();

    std::vector<node_ref> simplex_vars;
    for(size_t i=0; i<8; ++i)
        simplex_vars.push_back(mgr.projection(i));


    node_ref simplex_1 = mgr.simplex(simplex_vars.begin(), simplex_vars.begin()+5); 
    node_ref simplex_2 = mgr.simplex(simplex_vars.begin()+1, simplex_vars.begin()+6); 
    node_ref simplex_3 = mgr.simplex(simplex_vars.begin()+2, simplex_vars.begin()+7); 
    node_ref simplex_and = simplex_1 & simplex_2 & simplex_3;

    collection.add_bdd(simplex_1);
    collection.add_bdd(simplex_2);
    collection.add_bdd(simplex_3);
    std::vector<size_t> bdds = {0,1,2};
    const size_t simplex_and_idx = collection.bdd_and(bdds.begin(), bdds.end());

    for(size_t l0 = 0; l0<1; ++l0)
        for(size_t l1 = 0; l1<1; ++l1)
            for(size_t l2 = 0; l2<1; ++l2)
                for(size_t l3 = 0; l3<1; ++l3)
                    for(size_t l4 = 0; l4<1; ++l4)
                        for(size_t l5 = 0; l5<1; ++l5)
                            for(size_t l6 = 0; l6<1; ++l6)
                                for(size_t l7 = 0; l7<1; ++l7)
                                {
                                    const std::array<size_t,8> labeling = {l0,l1,l2,l3,l4,l5,l5,l7};
                                        test(simplex_and.evaluate(labeling.begin(), labeling.end()) == collection.evaluate(simplex_and_idx, labeling.begin(), labeling.end()), "bdd collection multi and not correct.");
                                }
}
