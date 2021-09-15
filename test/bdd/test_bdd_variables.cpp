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
        std::vector<size_t> var_indices;
        for(size_t x=0; x<i; ++x)
            var_indices.push_back(x);

        node_ref simplex = mgr.simplex(vars.begin(), vars.begin()+i);
        test(simplex.variables() == var_indices);

        node_ref neg_simplex = mgr.negate(simplex);
        test(neg_simplex.variables() == var_indices);

        node_ref at_most_one = mgr.at_most_one(vars.begin(), vars.begin()+i);
        test(at_most_one.variables() == var_indices);

        node_ref neg_at_most_one = mgr.negate(at_most_one);
        test(neg_at_most_one.variables() == var_indices);
        
        node_ref all_false = mgr.all_false(vars.begin(), vars.begin()+i);
        test(all_false.variables() == var_indices);

        node_ref not_all_false = mgr.negate(all_false);
        test(not_all_false.variables() == var_indices);

        node_ref cardinality_2 = mgr.cardinality(vars.begin(), vars.begin()+i, 2);
        test(cardinality_2.variables() == var_indices);

        node_ref at_most_2 = mgr.at_most(vars.begin(), vars.begin()+i, 2);
        if(i > 2)
            test(at_most_2.variables() == var_indices);
        else
            test(at_most_2.variables() == std::vector<size_t>{});

        node_ref at_least_2 = mgr.at_least(vars.begin(), vars.begin()+i, 2);
        test(at_least_2.variables() == var_indices);
    }
}
