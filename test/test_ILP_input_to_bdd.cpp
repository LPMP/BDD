#include "test.h"
#include "convert_pb_to_bdd.h"
#include "lineq_bdd.h"
#include <vector>

using namespace LPMP;

std::vector<int> create_vector(const std::vector<std::size_t> one_indices, const std::vector<std::size_t> minus_one_indices)
{
    std::vector<int> vec;
    for(auto x : one_indices) {
        if(x >= vec.size())
            vec.resize(x+1, 0.0);
        assert(vec[x] == 0.0);
        vec[x] = 1.0;
    }
    for(auto x : minus_one_indices) {
        if(x >= vec.size())
            vec.resize(x+1, 0.0);
        assert(vec[x] == 0.0);
        vec[x] = -1.0;
    }

    return vec;
}

void test_mrf(bdd_converter & converter)
{
    auto simplex_1 = converter.convert_to_bdd(create_vector({0,1,2,3,4},{}), ILP_input::inequality_type::equal, 1);
    auto simplex_2 = converter.convert_to_bdd(create_vector({5,6,7,8,9},{}), ILP_input::inequality_type::equal, 1);
    auto simplex_3 = converter.convert_to_bdd(create_vector({10,11,12,13,14},{}), ILP_input::inequality_type::equal, 1);

    auto potts_12_1 = converter.convert_to_bdd(create_vector({0},{5,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_12_2 = converter.convert_to_bdd(create_vector({1},{6,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_12_3 = converter.convert_to_bdd(create_vector({2},{7,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_12_4 = converter.convert_to_bdd(create_vector({3},{8,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_12_5 = converter.convert_to_bdd(create_vector({4},{9,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_21_1 = converter.convert_to_bdd(create_vector({5},{0,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_21_2 = converter.convert_to_bdd(create_vector({6},{1,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_21_3 = converter.convert_to_bdd(create_vector({7},{2,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_21_4 = converter.convert_to_bdd(create_vector({8},{3,15}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_21_5 = converter.convert_to_bdd(create_vector({9},{4,15}), ILP_input::inequality_type::smaller_equal , 0);

    auto potts_13_1 = converter.convert_to_bdd(create_vector({0},{10,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_13_2 = converter.convert_to_bdd(create_vector({1},{11,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_13_3 = converter.convert_to_bdd(create_vector({2},{12,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_13_4 = converter.convert_to_bdd(create_vector({3},{13,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_13_5 = converter.convert_to_bdd(create_vector({4},{14,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_31_1 = converter.convert_to_bdd(create_vector({10},{0,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_31_2 = converter.convert_to_bdd(create_vector({11},{1,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_31_3 = converter.convert_to_bdd(create_vector({12},{2,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_31_4 = converter.convert_to_bdd(create_vector({13},{3,16}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_31_5 = converter.convert_to_bdd(create_vector({14},{4,16}), ILP_input::inequality_type::smaller_equal , 0);

    auto potts_23_1 = converter.convert_to_bdd(create_vector({5},{10,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_23_2 = converter.convert_to_bdd(create_vector({6},{11,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_23_3 = converter.convert_to_bdd(create_vector({7},{12,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_23_4 = converter.convert_to_bdd(create_vector({8},{13,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_23_5 = converter.convert_to_bdd(create_vector({9},{14,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_32_1 = converter.convert_to_bdd(create_vector({10},{5,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_32_2 = converter.convert_to_bdd(create_vector({11},{6,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_32_3 = converter.convert_to_bdd(create_vector({12},{7,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_32_4 = converter.convert_to_bdd(create_vector({13},{8,17}), ILP_input::inequality_type::smaller_equal , 0);
    auto potts_32_5 = converter.convert_to_bdd(create_vector({14},{9,17}), ILP_input::inequality_type::smaller_equal , 0);

    //auto all_simplex = simplex_1.And(simplex_2.And( simplex_3));
    //auto bdd_potts_12 = potts_12_1.And(potts_12_2.And(potts_12_3.And(potts_12_4.And(potts_12_5.And(potts_21_1.And(potts_21_2.And(potts_21_3.And(potts_21_4.And(potts_21_5)))))))));
    //auto bdd_potts_13 = potts_13_1.And(potts_13_2.And(potts_13_3.And(potts_13_4.And(potts_13_5.And(potts_31_1.And(potts_31_2.And(potts_31_3.And(potts_31_4.And(potts_31_5)))))))));
    //auto bdd_potts_23 = potts_23_1.And(potts_23_2.And(potts_23_3.And(potts_23_4.And(potts_23_5.And(potts_32_1.And(potts_32_2.And(potts_32_3.And(potts_32_4.And(potts_32_5)))))))));

    //auto all = all_simplex.and_rec(bdd_potts_12.And(bdd_potts_13.And(bdd_potts_23)));
}

void test_simplex(bdd_converter & converter)
{
    std::vector<int> simplex_weights;
    size_t nr_vars = 10000;
    for(size_t i=0; i<nr_vars; ++i)
        simplex_weights.push_back(1);
    auto bdd = converter.convert_to_bdd(simplex_weights.begin(), simplex_weights.end(), ILP_input::inequality_type::equal, 1);
    std::cout << "# bdd nodes of simplex with " << nr_vars << " vars = " << bdd.nodes_postorder().size() << "\n";   
}

void test_cardinality(bdd_converter & converter)
{
    std::vector<int> cardinality_weights;
    size_t nr_vars = 10000;
    int rhs = 20;
    for(size_t i=0; i<nr_vars; ++i)
        cardinality_weights.push_back(1);
    auto bdd = converter.convert_to_bdd(cardinality_weights.begin(), cardinality_weights.end(), ILP_input::inequality_type::equal, rhs);
    std::cout << "# bdd nodes of cardinality constraint with " << nr_vars << " vars " << " and right-and side " << rhs << " : " << bdd.nodes_postorder().size() << "\n";   
}

void test_miplib(bdd_converter & converter)
{
    std::vector<int> weights = {+ 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, - 138, - 148, - 158, - 168, - 178, - 138, - 148, - 158, - 168, - 178, - 128, - 138, - 148, - 158, - 168, - 178, - 128, - 138, - 148, - 158, - 168, - 178, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 88, - 98, - 108, - 118, - 128, - 138, - 88, - 98, - 108, - 118, - 128, - 138};
    std::cout << "nr coefficients = " << weights.size() << "\n";
    auto bdd = converter.convert_to_bdd(weights.begin(), weights.end(), ILP_input::inequality_type::smaller_equal, 0);
    std::cout << "# bdd nodes = " << bdd.nodes_postorder().size() << "\n";
}

void test_subset_sum(bdd_converter & converter)
{
    std::vector<int> weights = {+ 1, + 1, + 2};
    std::cout << "nr coefficients = " << weights.size() << "\n";
    auto bdd = converter.convert_to_bdd(weights.begin(), weights.end(), ILP_input::inequality_type::equal, 2);
    std::cout << "# bdd nodes = " << bdd.nodes_postorder().size() << "\n";
}

int main(int argc, char** argv)
{
    BDD::bdd_mgr bdd_mgr;
    bdd_converter converter(bdd_mgr);

    test_mrf(converter);
    // test_simplex(converter);
    test_cardinality(converter);
    test_miplib(converter);
    test_subset_sum(converter);
}
