#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    auto test_labeling = [&](node_ref& p, const std::array<bool, 3> l, const bool result)
    {
        test(p.evaluate(l.begin(), l.end()) == result, "BDD evaluation error");
    };

    bdd_mgr mgr;
    // add 3 variables
    mgr.add_variable();
    mgr.add_variable();
    mgr.add_variable();

    node_ref or_01 = mgr.or_rec(mgr.projection(0), mgr.projection(1));
    test_labeling(or_01, {0,0}, false);
    test_labeling(or_01, {0,1}, true);
    test_labeling(or_01, {1,0}, true);
    test_labeling(or_01, {1,1}, true);

    node_ref or_012 = mgr.or_rec(or_01, mgr.projection(2));
    test_labeling(or_012, {0,0,0}, false);
    test_labeling(or_012, {1,0,0}, true);
    test_labeling(or_012, {0,1,0}, true);
    test_labeling(or_012, {0,0,1}, true);
    test_labeling(or_012, {1,1,0}, true);
    test_labeling(or_012, {1,0,1}, true);
    test_labeling(or_012, {0,1,1}, true);
    test_labeling(or_012, {1,1,1}, true);

    node_ref or_012_second = mgr.or_rec(mgr.projection(0), mgr.projection(1), mgr.projection(2));
    test_labeling(or_012_second, {0,0,0}, false);
    test_labeling(or_012_second, {1,0,0}, true);
    test_labeling(or_012_second, {0,1,0}, true);
    test_labeling(or_012_second, {0,0,1}, true);
    test_labeling(or_012_second, {1,1,0}, true);
    test_labeling(or_012_second, {1,0,1}, true);
    test_labeling(or_012_second, {0,1,1}, true);
    test_labeling(or_012_second, {1,1,1}, true);
    test(or_012 == or_012_second);

    node_ref and_not01 = mgr.or_rec(mgr.negate(mgr.projection(0)), mgr.negate(mgr.projection(1)));
    test_labeling(and_not01, {0,0}, true);
    test_labeling(and_not01, {0,1}, true);
    test_labeling(and_not01, {1,0}, true);
    test_labeling(and_not01, {1,1}, false);

    node_ref and_not02 = mgr.or_rec(mgr.negate(mgr.projection(0)), mgr.negate(mgr.projection(2)));
    test_labeling(and_not02, {0,0,0}, true);
    test_labeling(and_not02, {0,0,1}, true);
    test_labeling(and_not02, {1,0,0}, true);
    test_labeling(and_not02, {1,0,1}, false);

    node_ref and_not12 = mgr.or_rec(mgr.negate(mgr.projection(1)), mgr.negate(mgr.projection(2)));
    test_labeling(and_not12, {0,0,0}, true);
    test_labeling(and_not12, {0,0,1}, true);
    test_labeling(and_not12, {0,1,0}, true);
    test_labeling(and_not12, {0,1,1}, false);

    node_ref simplex = mgr.and_rec(mgr.and_rec(mgr.and_rec(or_012, and_not01), and_not02), and_not12);
    test(simplex == mgr.and_rec(or_012, and_not01, and_not02, and_not12));

    test_labeling(simplex, {0,0,0}, false);

    test_labeling(simplex, {1,0,0}, true);
    test_labeling(simplex, {0,1,0}, true);
    test_labeling(simplex, {0,0,1}, true);

    test_labeling(simplex, {1,1,0}, false);
    test_labeling(simplex, {0,1,1}, false);
    test_labeling(simplex, {1,0,1}, false);
    test_labeling(simplex, {1,1,1}, false);

    test(simplex.nr_nodes() == 5, "nr of simplex bdd nodes wrong");
    test(simplex.variables() == std::vector<size_t>({0,1,2}), "variables of simplex do not match");

    size_t nr_nodes_max;
    {
    std::unordered_map<size_t,size_t> v_map = {{0,3},{1,4},{2,5}};
    node_ref simplex_345 = mgr.rebase(simplex, v_map);
    test(simplex_345.variables() == std::vector<size_t>({3,4,5}), "variables of rebased simplex do not match");
    std::vector<size_t> v_map_vector = {3,4,5};
    node_ref simplex_345_second = mgr.rebase(simplex, v_map_vector.begin(), v_map_vector.end());
    test(simplex_345 == simplex_345_second, "rebase method error");

    nr_nodes_max = mgr.nr_nodes();
    }
    // TODO: enable garbage collection again
    //mgr.collect_garbage();
    //test(nr_nodes_max > mgr.nr_nodes(), "garbage collection did not remove unused nodes.");
}



