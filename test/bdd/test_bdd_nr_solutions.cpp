#include "bdd_manager/bdd_mgr.h"
#include "../test.h"
#include <vector>

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_mgr mgr;
    for(size_t i=0; i<5; ++i)
        mgr.add_variable();

    std::vector<node_ref> vars;
    for(size_t i=0; i<5; ++i)
        vars.push_back(mgr.projection(i));

    for(size_t i=2; i<5; ++i)
    {
        node_ref simplex = mgr.simplex(vars.begin(), vars.begin()+i);
        test(simplex.nr_solutions() == i);

        node_ref neg_simplex = mgr.negate(simplex);
        test(neg_simplex.nr_solutions() == std::pow(2,i) - i);

        node_ref at_most_one = mgr.at_most_one(vars.begin(), vars.begin()+i);
        test(at_most_one.nr_solutions() == i+1);

        node_ref neg_at_most_one = mgr.negate(at_most_one);
        test(neg_at_most_one.nr_solutions() == std::pow(2,i) - i - 1);
        
        node_ref all_false = mgr.all_false(vars.begin(), vars.begin()+i);
        test(all_false.nr_solutions() == 1);

        node_ref not_all_false = mgr.negate(all_false);
        test(not_all_false.nr_solutions() == std::pow(2,i) - 1);

        node_ref cardinality_2 = mgr.cardinality(vars.begin(), vars.begin()+i, 2);
        test(cardinality_2.nr_solutions() == (i*(i-1))/2);

        if(i > 2)
        {
            node_ref at_most_2 = mgr.at_most(vars.begin(), vars.begin()+i, 2);
            test(at_most_2.nr_solutions() == 1 + i + (i*(i-1))/2);

            node_ref at_least_2 = mgr.at_least(vars.begin(), vars.begin()+i, 2);
            test(at_least_2.nr_solutions() == std::pow(2,i) - 1 - i);
        }
    }
}
