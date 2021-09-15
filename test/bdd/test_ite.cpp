#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{ 
    auto test_labeling = [&](node_ref p, const std::array<bool, 3> l, const bool result)
    {
        test(p.evaluate(l.begin(), l.end()) == result, "BDD evaluation error");
    };

    bdd_mgr mgr;

    mgr.add_variable();
    mgr.add_variable();
    mgr.add_variable();

    node_ref ite_012 = mgr.ite_rec(mgr.projection(0), mgr.projection(1), mgr.projection(2));

    test_labeling(ite_012, {0,0,0}, false);
    test_labeling(ite_012, {0,0,1}, true);
    test_labeling(ite_012, {0,1,0}, false);
    test_labeling(ite_012, {0,1,1}, true);
    test_labeling(ite_012, {1,0,0}, false);
    test_labeling(ite_012, {1,0,1}, false);
    test_labeling(ite_012, {1,1,0}, true);
    test_labeling(ite_012, {1,1,1}, true);

    node_ref ite_210 = mgr.ite_rec(mgr.projection(2), mgr.projection(1), mgr.projection(0));

    test_labeling(ite_210, {0,0,0}, false);
    test_labeling(ite_210, {0,1,0}, false);
    test_labeling(ite_210, {1,0,0}, true);
    test_labeling(ite_210, {1,1,0}, true);
    test_labeling(ite_210, {0,0,1}, false);
    test_labeling(ite_210, {0,1,1}, true);
    test_labeling(ite_210, {1,0,1}, false);
    test_labeling(ite_210, {1,1,1}, true);

    test(ite_012.reference_count() == 1);
    test(ite_210.reference_count() == 1);
}
