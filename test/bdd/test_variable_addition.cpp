#include "bdd_manager/bdd_mgr.h"
#include "../test.h"

using namespace BDD;
using namespace LPMP;

int main(int argc, char** argv)
{
    bdd_mgr mgr;
    const std::size_t n = 10000;
    for(std::size_t i=0; i<n; ++i)
    {
        const std::size_t v = mgr.add_variable();
        test(mgr.nr_variables() == i+1, "wrong number of variables after adding.");
    }
}
