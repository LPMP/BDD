#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace LPMP;
using namespace BDD;

int main(int argc, char** argv)
{
    bdd_mgr mgr;
    // add 3 variables
    for(size_t i=0; i<3; ++i)
        mgr.add_variable();

    std::vector<node_ref> vars;
    for(size_t i=0; i<3; ++i)
        vars.push_back(mgr.projection(i));

    test(mgr.projection(0).exactly_one_solution());

    test((mgr.projection(0) & mgr.projection(1) & mgr.projection(2)).exactly_one_solution());

    test((mgr.projection(0) & mgr.neg_projection(1) & mgr.projection(2)).exactly_one_solution());

    test(mgr.simplex(vars.begin(), vars.end()).exactly_one_solution() == false);

    test(mgr.negate(mgr.simplex(vars.begin(), vars.end())).exactly_one_solution() == false);
}
