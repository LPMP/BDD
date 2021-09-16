#include "bdd_manager/bdd_mgr.h"
#include "bdd_collection/bdd_collection.h"
#include "../test.h"
#include <fstream>

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

    std::vector<node_ref> covering_ineq;
    std::vector<node_ref> cur_vars = {vars[0], vars[1], vars[3]};
    covering_ineq.push_back(mgr.at_least_one(cur_vars.begin(), cur_vars.end()));

    cur_vars = {vars[0], vars[2], vars[4]};
    covering_ineq.push_back(mgr.at_least_one(cur_vars.begin(), cur_vars.end()));

    cur_vars = {vars[1], vars[2], vars[5]};
    covering_ineq.push_back(mgr.at_least_one(cur_vars.begin(), cur_vars.end()));

    node_ref covering_ineq_intersect = mgr.and_rec(covering_ineq.begin(), covering_ineq.end());

    std::vector<size_t> covering_ineq_coll;
    covering_ineq_coll.push_back(collection.add_bdd(covering_ineq[0]));
    covering_ineq_coll.push_back(collection.add_bdd(covering_ineq[1]));
    covering_ineq_coll.push_back(collection.add_bdd(covering_ineq[2]));
    const size_t covering_ineq_coll_intersect = collection.bdd_and(covering_ineq_coll.begin(), covering_ineq_coll.end());

    node_ref covering_ineq_intersect_transformed = collection.export_bdd(mgr, covering_ineq_coll_intersect); 

    //std::fstream fs;
    //fs.open ("kwas.dot", std::fstream::in | std::fstream::out | std::ofstream::trunc);
    //collection.export_graphviz(covering_ineq_coll_intersect, fs);

    test(covering_ineq_intersect == covering_ineq_intersect_transformed); 
}
