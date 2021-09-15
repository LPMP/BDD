#include "bdd_manager/bdd_mgr.h"
#include "../test.h"
#include <array>
#include <vector>
#include <numeric>

using namespace BDD;
using namespace LPMP;

template<typename LABEL_ITERATOR>
inline void test_labeling(BDD::node_ref p, LABEL_ITERATOR label_begin, LABEL_ITERATOR label_end, const bool result)
{
    test(p.evaluate(label_begin, label_end) == result, "BDD evaluation error");
};

int main(int argc, char** argv)
{
    bdd_mgr mgr;
    for(size_t i=0; i<5; ++i)
        mgr.add_variable();

    std::vector<node_ref> vars;
    for(size_t i=0; i<5; ++i)
        vars.push_back(mgr.projection(i));

    // 2 vars
    {
    node_ref simplex = mgr.simplex(vars.begin(), vars.begin()+2);
    node_ref neg_simplex = mgr.negate(simplex);
    node_ref at_most_one = mgr.at_most_one(vars.begin(), vars.begin()+2);
    node_ref neg_at_most_one = mgr.negate(at_most_one);
    node_ref all_false = mgr.all_false(vars.begin(), vars.begin()+2);
    node_ref not_all_false = mgr.negate(all_false);
    node_ref cardinality_2 = mgr.cardinality(vars.begin(), vars.begin()+2, 2);
    node_ref at_most_2 = mgr.at_most(vars.begin(), vars.begin()+2, 2);
    node_ref at_least_2 = mgr.at_least(vars.begin(), vars.begin()+2, 2);

    for(size_t l0=0; l0<=1; ++l0)
        for(size_t l1=0; l1<=1; ++l1)
        {
            std::array<size_t,2> labeling = {l0,l1};
            const size_t sum = std::accumulate(labeling.begin(), labeling.end(), 0);
            test_labeling(at_most_one, labeling.begin(), labeling.end(), sum <= 1);
            test_labeling(neg_at_most_one, labeling.begin(), labeling.end(), sum > 1);

            test_labeling(simplex, labeling.begin(), labeling.end(), sum == 1);
            test_labeling(neg_simplex, labeling.begin(), labeling.end(), sum != 1);

            test_labeling(all_false, labeling.begin(), labeling.end(), sum == 0);
            test_labeling(not_all_false, labeling.begin(), labeling.end(), sum != 0);
            test_labeling(cardinality_2, labeling.begin(), labeling.end(), sum == 2);
            test_labeling(at_most_2, labeling.begin(), labeling.end(), sum <= 2);
            test_labeling(at_least_2, labeling.begin(), labeling.end(), sum >= 2);
        }
    }

    // 3 vars
    {
    node_ref simplex = mgr.simplex(vars.begin(), vars.begin()+3);
    node_ref neg_simplex = mgr.negate(simplex);
    node_ref at_most_one = mgr.at_most_one(vars.begin(), vars.begin()+3);
    node_ref neg_at_most_one = mgr.negate(at_most_one);
    node_ref all_false = mgr.all_false(vars.begin(), vars.begin()+3);
    node_ref not_all_false = mgr.negate(all_false);
    node_ref cardinality_2 = mgr.cardinality(vars.begin(), vars.begin()+3, 2);
    node_ref at_most_2 = mgr.at_most(vars.begin(), vars.begin()+3, 2);
    node_ref at_least_2 = mgr.at_least(vars.begin(), vars.begin()+3, 2);

    for(size_t l0=0; l0<=1; ++l0)
        for(size_t l1=0; l1<=1; ++l1)
            for(size_t l2=0; l2<=1; ++l2)
            {
                std::array<size_t,3> labeling = {l0,l1,l2};

                const size_t sum = std::accumulate(labeling.begin(), labeling.end(), 0);

                test_labeling(at_most_one, labeling.begin(), labeling.end(), sum <= 1);
                test_labeling(neg_at_most_one, labeling.begin(), labeling.end(), sum > 1);

                test_labeling(simplex, labeling.begin(), labeling.end(), sum == 1);
                test_labeling(neg_simplex, labeling.begin(), labeling.end(), sum != 1);

                test_labeling(all_false, labeling.begin(), labeling.end(), sum == 0);
                test_labeling(not_all_false, labeling.begin(), labeling.end(), sum != 0);
                test_labeling(cardinality_2, labeling.begin(), labeling.end(), sum == 2);
                test_labeling(at_most_2, labeling.begin(), labeling.end(), sum <= 2);
                test_labeling(at_least_2, labeling.begin(), labeling.end(), sum >= 2);
            }
    }

    // 4 vars
    {
    node_ref simplex = mgr.simplex(vars.begin(), vars.begin()+4);
    node_ref neg_simplex = mgr.negate(simplex);
    node_ref at_most_one = mgr.at_most_one(vars.begin(), vars.begin()+4);
    node_ref neg_at_most_one = mgr.negate(at_most_one);
    node_ref all_false = mgr.all_false(vars.begin(), vars.begin()+4);
    node_ref not_all_false = mgr.negate(all_false);
    node_ref cardinality_2 = mgr.cardinality(vars.begin(), vars.begin()+4, 2);
    node_ref at_most_2 = mgr.at_most(vars.begin(), vars.begin()+4, 2);
    node_ref at_least_2 = mgr.at_least(vars.begin(), vars.begin()+4, 2);

    for(size_t l0=0; l0<=1; ++l0)
        for(size_t l1=0; l1<=1; ++l1)
            for(size_t l2=0; l2<=1; ++l2)
                for(size_t l3=0; l3<=1; ++l3)
                    {
                        std::array<size_t,4> labeling = {l0,l1,l2,l3};

                        const size_t sum = std::accumulate(labeling.begin(), labeling.end(), 0);

                        test_labeling(at_most_one, labeling.begin(), labeling.end(), sum <= 1);
                        test_labeling(neg_at_most_one, labeling.begin(), labeling.end(), sum > 1);

                        test_labeling(simplex, labeling.begin(), labeling.end(), sum == 1);
                        test_labeling(neg_simplex, labeling.begin(), labeling.end(), sum != 1);

                        test_labeling(all_false, labeling.begin(), labeling.end(), sum == 0);
                        test_labeling(not_all_false, labeling.begin(), labeling.end(), sum != 0);
            test_labeling(cardinality_2, labeling.begin(), labeling.end(), sum == 2);
            test_labeling(at_most_2, labeling.begin(), labeling.end(), sum <= 2);
            test_labeling(at_least_2, labeling.begin(), labeling.end(), sum >= 2);
                    }
    }

    // 5 vars
    {
    node_ref simplex = mgr.simplex(vars.begin(), vars.begin()+5);
    node_ref neg_simplex = mgr.negate(simplex);
    node_ref at_most_one = mgr.at_most_one(vars.begin(), vars.begin()+5);
    node_ref neg_at_most_one = mgr.negate(at_most_one);
    node_ref all_false = mgr.all_false(vars.begin(), vars.begin()+5);
    node_ref not_all_false = mgr.negate(all_false);
    node_ref cardinality_2 = mgr.cardinality(vars.begin(), vars.begin()+5, 2);
    node_ref at_most_2 = mgr.at_most(vars.begin(), vars.begin()+5, 2);
    node_ref at_least_2 = mgr.at_least(vars.begin(), vars.begin()+5, 2);

    for(size_t l0=0; l0<=1; ++l0)
        for(size_t l1=0; l1<=1; ++l1)
            for(size_t l2=0; l2<=1; ++l2)
                for(size_t l3=0; l3<=1; ++l3)
                    for(size_t l4=0; l4<=1; ++l4)
                    {
                        std::array<size_t,5> labeling = {l0,l1,l2,l3,l4};

                        const size_t sum = std::accumulate(labeling.begin(), labeling.end(), 0);

                        test_labeling(at_most_one, labeling.begin(), labeling.end(), sum <= 1);
                        test_labeling(neg_at_most_one, labeling.begin(), labeling.end(), sum > 1);

                        test_labeling(simplex, labeling.begin(), labeling.end(), sum == 1);
                        test_labeling(neg_simplex, labeling.begin(), labeling.end(), sum != 1);

                        test_labeling(all_false, labeling.begin(), labeling.end(), sum == 0);
                        test_labeling(not_all_false, labeling.begin(), labeling.end(), sum != 0);
                        test_labeling(cardinality_2, labeling.begin(), labeling.end(), sum == 2);
                        test_labeling(at_most_2, labeling.begin(), labeling.end(), sum <= 2);
                        test_labeling(at_least_2, labeling.begin(), labeling.end(), sum >= 2);
                    }
    }

    for(size_t n=2; n<=5; ++n)
    {
        test(mgr.simplex(vars.begin(), vars.begin()+n) == mgr.cardinality(vars.begin(), vars.begin()+n, 1));
        test(mgr.at_least_one(vars.begin(), vars.begin()+n) == mgr.at_least(vars.begin(), vars.begin()+n, 1));
        test(mgr.at_most_one(vars.begin(), vars.begin()+n) == mgr.at_most(vars.begin(), vars.begin()+n, 1));
    }
    

}
