#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_mgr mgr;
    mgr.add_variable();
    mgr.add_variable();
    mgr.add_variable();

    node_ref n0 = mgr.negate(mgr.projection(0));
    test(n0.reference_count() == 1);
    
    node_ref n1 = mgr.negate(mgr.projection(1));
    test(n1.reference_count() == 1);
    
    node_ref n2 = mgr.negate(mgr.projection(2));
    test(n2.reference_count() == 1);

    std::array<bool,3> labeling = {0,0,0};
    test(mgr.projection(0).evaluate(labeling.begin(), labeling.end()) == false);
    test(n0.evaluate(labeling.begin(), labeling.end()) == true);
    labeling[0] = 1;
    test(mgr.projection(0).evaluate(labeling.begin(), labeling.end()) == true);
    test(n0.evaluate(labeling.begin(), labeling.end()) == false);

    test(mgr.projection(1).evaluate(labeling.begin(), labeling.end()) == false);
    test(n1.evaluate(labeling.begin(), labeling.end()) == true);
    labeling[1] = 1;
    test(mgr.projection(1).evaluate(labeling.begin(), labeling.end()) == true);
    test(n1.evaluate(labeling.begin(), labeling.end()) == false);

    test(mgr.projection(2).evaluate(labeling.begin(), labeling.end()) == false);
    test(n2.evaluate(labeling.begin(), labeling.end()) == true);
    labeling[2] = 1;
    test(mgr.projection(2).evaluate(labeling.begin(), labeling.end()) == true);
    test(n2.evaluate(labeling.begin(), labeling.end()) == false);

    test(n0.reference_count() == 1);
    test(n1.reference_count() == 1);
    test(n2.reference_count() == 1);

    node_ref x = n0 & n1;
    test(x.reference_count() == 1);
}


