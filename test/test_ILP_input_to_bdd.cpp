#include "test.h"
#include "convert_pb_to_bdd.h"
#include "lineq_bdd.h"
#include "hard_ineqs.h"
#include <vector>

using namespace LPMP;

void test_against_utility_functions(bdd_converter& converter)
{
    BDD::bdd_mgr& bdd_mgr = converter.bdd_mgr();

    std::vector<BDD::node_ref> vars;
    for(size_t i=0; i<10; ++i)
        vars.push_back(bdd_mgr.projection(i));

    for(size_t i=3; i<10; ++i)
    {
        std::vector<int> coeffs;
        for(size_t x=0; x<i; ++x)
            coeffs.push_back(1);
        BDD::node_ref simplex_converted = converter.convert_to_bdd(coeffs, ILP_input::inequality_type::equal, 1);
        BDD::node_ref simplex = bdd_mgr.simplex(vars.begin(), vars.begin()+i);
        test(simplex == simplex_converted);
    }

    for(size_t i=3; i<10; ++i)
    {
        std::vector<int> coeffs;
        for(size_t x=0; x<i; ++x)
            coeffs.push_back(1);
        BDD::node_ref at_most_one_converted = converter.convert_to_bdd(coeffs, ILP_input::inequality_type::smaller_equal, 1);
        BDD::node_ref at_most_one = bdd_mgr.at_most_one(vars.begin(), vars.begin()+i);
        test(at_most_one == at_most_one_converted);
    }

    for(size_t i=3; i<10; ++i)
    {
        BDD::node_ref all_false = bdd_mgr.all_false(vars.begin(), vars.begin()+i);
        std::vector<int> coeffs;
        for(size_t x=0; x<i; ++x)
            coeffs.push_back(1);
        BDD::node_ref all_false_converted = converter.convert_to_bdd(coeffs, ILP_input::inequality_type::smaller_equal, 0);
        test(all_false == all_false_converted);
    }

    for(size_t i=3; i<10; ++i)
    {
        BDD::node_ref not_all_false = bdd_mgr.negate(bdd_mgr.all_false(vars.begin(), vars.begin()+i));
        std::vector<int> coeffs;
        for(size_t x=0; x<i; ++x)
            coeffs.push_back(1);
        BDD::node_ref not_all_false_converted = converter.convert_to_bdd(coeffs, ILP_input::inequality_type::greater_equal, 1);
        test(not_all_false == not_all_false_converted);
    }

    for(size_t i=3; i<10; ++i)
    {
        BDD::node_ref cardinality_2 = bdd_mgr.cardinality(vars.begin(), vars.begin()+i, 2);
        std::vector<int> coeffs;
        for(size_t x=0; x<i; ++x)
            coeffs.push_back(1);
        BDD::node_ref cardinality_2_converted = converter.convert_to_bdd(coeffs, ILP_input::inequality_type::equal, 2);
        test(cardinality_2 == cardinality_2_converted);
    }

    for(size_t i=3; i<10; ++i)
    {
        BDD::node_ref at_most_2 = bdd_mgr.at_most(vars.begin(), vars.begin()+i, 2);
        std::vector<int> coeffs;
        for(size_t x=0; x<i; ++x)
            coeffs.push_back(1);
        BDD::node_ref at_most_2_converted = converter.convert_to_bdd(coeffs, ILP_input::inequality_type::smaller_equal, 2);
        test(at_most_2 == at_most_2_converted);
    }

    for(size_t i=3; i<10; ++i)
    {
        BDD::node_ref at_least_2 = bdd_mgr.at_least(vars.begin(), vars.begin()+i, 2);
        std::vector<int> coeffs;
        for(size_t x=0; x<i; ++x)
            coeffs.push_back(1);
        BDD::node_ref at_least_2_converted = converter.convert_to_bdd(coeffs, ILP_input::inequality_type::greater_equal, 2);
        test(at_least_2 == at_least_2_converted);
    }
}

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

void test_mrf(bdd_converter& converter)
{
    auto& bdd_mgr = converter.bdd_mgr();

    auto simplex_1 = converter.convert_to_bdd(create_vector({0,1,2,3,4},{}), ILP_input::inequality_type::equal, 1);
    auto simplex_2 = converter.convert_to_bdd(create_vector({5,6,7,8,9},{}), ILP_input::inequality_type::equal, 1);
    auto simplex_3 = converter.convert_to_bdd(create_vector({10,11,12,13,14},{}), ILP_input::inequality_type::equal, 1);

    auto potts_12_1 = converter.convert_to_bdd(create_vector({0},{5,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_12_2 = converter.convert_to_bdd(create_vector({1},{6,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_12_3 = converter.convert_to_bdd(create_vector({2},{7,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_12_4 = converter.convert_to_bdd(create_vector({3},{8,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_12_5 = converter.convert_to_bdd(create_vector({4},{9,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_21_1 = converter.convert_to_bdd(create_vector({5},{0,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_21_2 = converter.convert_to_bdd(create_vector({6},{1,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_21_3 = converter.convert_to_bdd(create_vector({7},{2,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_21_4 = converter.convert_to_bdd(create_vector({8},{3,15}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_21_5 = converter.convert_to_bdd(create_vector({9},{4,15}), ILP_input::inequality_type::smaller_equal, 0);

    auto potts_13_1 = converter.convert_to_bdd(create_vector({0},{10,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_13_2 = converter.convert_to_bdd(create_vector({1},{11,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_13_3 = converter.convert_to_bdd(create_vector({2},{12,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_13_4 = converter.convert_to_bdd(create_vector({3},{13,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_13_5 = converter.convert_to_bdd(create_vector({4},{14,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_31_1 = converter.convert_to_bdd(create_vector({10},{0,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_31_2 = converter.convert_to_bdd(create_vector({11},{1,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_31_3 = converter.convert_to_bdd(create_vector({12},{2,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_31_4 = converter.convert_to_bdd(create_vector({13},{3,16}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_31_5 = converter.convert_to_bdd(create_vector({14},{4,16}), ILP_input::inequality_type::smaller_equal, 0);

    auto potts_23_1 = converter.convert_to_bdd(create_vector({5},{10,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_23_2 = converter.convert_to_bdd(create_vector({6},{11,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_23_3 = converter.convert_to_bdd(create_vector({7},{12,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_23_4 = converter.convert_to_bdd(create_vector({8},{13,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_23_5 = converter.convert_to_bdd(create_vector({9},{14,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_32_1 = converter.convert_to_bdd(create_vector({10},{5,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_32_2 = converter.convert_to_bdd(create_vector({11},{6,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_32_3 = converter.convert_to_bdd(create_vector({12},{7,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_32_4 = converter.convert_to_bdd(create_vector({13},{8,17}), ILP_input::inequality_type::smaller_equal, 0);
    auto potts_32_5 = converter.convert_to_bdd(create_vector({14},{9,17}), ILP_input::inequality_type::smaller_equal, 0);

    auto all_simplex = bdd_mgr.and_rec(simplex_1, simplex_2, simplex_3);
    auto bdd_potts_12 = bdd_mgr.and_rec(potts_12_1,potts_12_2,potts_12_3,potts_12_4,potts_12_5,potts_21_1,potts_21_2,potts_21_3,potts_21_4,potts_21_5);
    auto bdd_potts_13 = bdd_mgr.and_rec(potts_13_1,potts_13_2,potts_13_3,potts_13_4,potts_13_5,potts_31_1,potts_31_2,potts_31_3,potts_31_4,potts_31_5);
    auto bdd_potts_23 = bdd_mgr.and_rec(potts_23_1,potts_23_2,potts_23_3,potts_23_4,potts_23_5,potts_32_1,potts_32_2,potts_32_3,potts_32_4,potts_32_5);

    auto all = bdd_mgr.and_rec(all_simplex,bdd_potts_12,bdd_potts_13,bdd_potts_23);

    test(simplex_1.nr_solutions() == 5);
    test(simplex_2.nr_solutions() == 5);
    test(simplex_3.nr_solutions() == 5);
    test(all_simplex.nr_solutions() == 5*5*5);
    test(all.nr_solutions() >= 5*5*5); // pairwise variables have additional states
}

void test_simplex(bdd_converter & converter)
{
    std::vector<int> simplex_weights;
    size_t nr_vars = 10000;
    for(size_t i=0; i<nr_vars; ++i)
        simplex_weights.push_back(1);
    auto bdd = converter.convert_to_bdd(simplex_weights.begin(), simplex_weights.end(), ILP_input::inequality_type::equal, 1);
    test(bdd.nr_solutions() == nr_vars);
    test(bdd.nr_nodes() == nr_vars*2 - 1);
    std::cout << "# bdd nodes of simplex with " << nr_vars << " vars = " << bdd.nodes_postorder().size() << "\n";   
}

void test_cardinality(bdd_converter & converter)
{
    size_t nr_vars = 100;
    for(size_t rhs=1; rhs<8; ++rhs)
    {
        std::vector<int> cardinality_weights;
        for(size_t i=0; i<nr_vars; ++i)
            cardinality_weights.push_back(1);
        auto bdd = converter.convert_to_bdd(cardinality_weights.begin(), cardinality_weights.end(), ILP_input::inequality_type::equal, rhs);
        auto n_choose_k = [](size_t n, size_t k) -> size_t {
            assert(n >= k);
            if (k * 2 > n) k = n-k;
            if (k == 0) return 1;

            size_t result = n;
            for(size_t i=2; i<=k; ++i)
            {
                result *= (n-i+1);
                result /= i;
            }
            return result;
        };
        test(bdd.nr_solutions() == n_choose_k(nr_vars, rhs));
        std::cout << "# bdd nodes of cardinality constraint with " << nr_vars << " vars " << " and right-and side " << rhs << " : " << bdd.nodes_postorder().size() << "\n";   
    }
}

void test_miplib(bdd_converter & converter)
{
    std::vector<int> weights = {+ 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, + 88, - 138, - 148, - 158, - 168, - 178, - 138, - 148, - 158, - 168, - 178, - 128, - 138, - 148, - 158, - 168, - 178, - 128, - 138, - 148, - 158, - 168, - 178, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 178, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 168, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 158, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 88, - 98, - 108, - 118, - 128, - 138, - 148, - 88, - 98, - 108, - 118, - 128, - 138, - 88, - 98, - 108, - 118, - 128, - 138};
    std::cout << "nr coefficients = " << weights.size() << "\n";
    auto bdd = converter.convert_to_bdd(weights.begin(), weights.end(), ILP_input::inequality_type::smaller_equal, 0);
    std::cout << "# bdd nodes = " << bdd.nodes_postorder().size() << "\n";

    {
        std::cout << "app2_2 ct_0 nr coefficients = " << app2_2_weights_ct_0.size() << "\n";
        auto [bdd,multiplicities] = converter.coefficient_decomposition_convert_to_bdd(app2_2_weights_ct_0, app2_2_ineq_type_ct_0, app2_2_rhs_ct_0);
        std::cout << "app2_2 ct_0 # bdd nodes = " << bdd.nodes_postorder().size() << "\n";
    }

    {
        std::cout << "app2_2 ct_1 nr coefficients = " << app2_2_weights_ct_1.size() << "\n";
        auto [bdd,multiplicities] = converter.coefficient_decomposition_convert_to_bdd(app2_2_weights_ct_1, app2_2_ineq_type_ct_1, app2_2_rhs_ct_1);
        std::cout << "app2_2 ct_1 # bdd nodes = " << bdd.nodes_postorder().size() << "\n";
    }

    // currently too large!
    {
        //std::cout << "cap6000 c1 nr coefficients = " << cap6000_weights_c1.size() << "\n";
        //auto [bdd,multiplicities] = converter.coefficient_decomposition_convert_to_bdd(cap6000_weights_c1, cap6000_ineq_type_c1, cap6000_rhs_c1);
        //std::cout << "cap6000 c1 # bdd nodes = " << bdd.nodes_postorder().size() << "\n";
    }

    {
        //std::cout << "cap6000 c2 nr coefficients = " << cap6000_weights_c2.size() << "\n";
        //auto [bdd,multiplicities] = converter.coefficient_decomposition_convert_to_bdd(cap6000_weights_c2, cap6000_ineq_type_c2, cap6000_rhs_c2);
        //std::cout << "cap6000 c2 # bdd nodes = " << bdd.nodes_postorder().size() << "\n";
    }

}

void test_coefficient_decomposition_conversion(bdd_converter& converter)
{
    // trivial examples
    for(size_t n=1; n<=15; ++n)
    {
        std::vector<int> weights(n, 1);
        auto [bdd, decomposed_coefficient_map] = converter.coefficient_decomposition_convert_to_bdd(weights, ILP_input::inequality_type::equal, 1);
        test(bdd == converter.bdd_mgr().simplex(n));
        test(decomposed_coefficient_map.size() == n);
        for(size_t i=0; i<n; ++i)
            test(decomposed_coefficient_map.size(i) == 1);
    }

    // check if gcd works
    for(size_t n=1; n<=15; ++n)
    {
        std::vector<int> weights(n, -3);
        auto [bdd, decomposed_coefficient_map] = converter.coefficient_decomposition_convert_to_bdd(weights, ILP_input::inequality_type::equal, -3);
        test(bdd == converter.bdd_mgr().simplex(n));
        test(decomposed_coefficient_map.size() == n);
        for(size_t i=0; i<n; ++i)
            test(decomposed_coefficient_map.size(i) == 1);
    }

    // check decomposed coefficients correctly given back.
    {
        std::vector<int> weights = {16+1, 8+2, 4};
        auto [bdd, decomposed_coefficient_map] = converter.coefficient_decomposition_convert_to_bdd(weights, ILP_input::inequality_type::smaller_equal, 17);
        test(decomposed_coefficient_map.size() == 3);
        test(decomposed_coefficient_map.size(0) == 2);
        test(decomposed_coefficient_map.size(1) == 2);
        test(decomposed_coefficient_map.size(2) == 1);

        test(decomposed_coefficient_map(0,0) == 0);
        test(decomposed_coefficient_map(0,1) == 4);
        test(decomposed_coefficient_map(1,0) == 1);
        test(decomposed_coefficient_map(1,1) == 3);
        test(decomposed_coefficient_map(2,0) == 2);
    }
}

void test_subset_sum(bdd_converter & converter)
{
    std::vector<int> weights = {+ 1, + 1, + 2};
    std::cout << "nr coefficients = " << weights.size() << "\n";
    auto bdd = converter.convert_to_bdd(weights.begin(), weights.end(), ILP_input::inequality_type::equal, 2);
    std::cout << "# bdd nodes = " << bdd.nodes_postorder().size() << "\n";
}

void test_covering(bdd_converter& converter)
{
    auto bdd_1 = converter.convert_to_bdd({1,1,1}, ILP_input::inequality_type::greater_equal, 1);
    auto bdd_2 = converter.convert_to_bdd({-1,-1,-1}, ILP_input::inequality_type::smaller_equal, -1);
    auto bdd_3 = converter.convert_to_bdd({3,3,3}, ILP_input::inequality_type::greater_equal, 2);
    for(size_t l1=0; l1<=1; ++l1)
        for(size_t l2=0; l2<=1; ++l2)
            for(size_t l3=0; l3<=1; ++l3)
            {
                std::array<size_t,3> l{l1,l2,l3};
                const bool eval = l1 != 0 || l2 != 0 || l3 != 0;
                std::cout << "[covering inequality] label (" << l1 << "," << l2 << "," << l3 << "), predicted evaluation " << eval << ", computed evaluation " << bdd_1.evaluate(l.begin(), l.end()) << "\n";
                //test(bdd_1.evaluate(l.begin(), l.end()) == eval);
            }
    test(bdd_1 == bdd_2);
    test(bdd_1 == bdd_3);
    test(bdd_1.nr_solutions() == 8-1);
}

void test_nonlinear(bdd_converter& converter)
{
    for(size_t nr_monomials=2; nr_monomials<5; ++nr_monomials)
    {
        for(size_t degree=2; degree<4; ++degree)
        {
            std::vector<size_t> monomial_degrees(nr_monomials,degree);
            std::vector<int> coefficients(nr_monomials,1);
            BDD::node_ref bdd = converter.convert_nonlinear_to_bdd(monomial_degrees, coefficients, ILP_input::inequality_type::equal, 1);
            test(bdd.variables().size() == nr_monomials * degree);

            std::vector<char> labeling(nr_monomials * degree, 0);
            test(bdd.evaluate(labeling.begin(), labeling.end()) == false);
            for(size_t i=0; i<nr_monomials; ++i)
            {
                std::fill(labeling.begin(), labeling.end(), 0);
                for(size_t d=0; d<degree; ++d)
                {
                    test(bdd.evaluate(labeling.begin(), labeling.end()) == false);
                    labeling[i*degree + d] = 1;
                }
                test(bdd.evaluate(labeling.begin(), labeling.end()) == true);
            }
        }
    }
}

int main(int argc, char** argv)
{
    BDD::bdd_mgr bdd_mgr;
    bdd_converter converter(bdd_mgr);

    test_against_utility_functions(converter);
    test_mrf(converter);
    test_simplex(converter);
    test_miplib(converter);
    test_coefficient_decomposition_conversion(converter);
    test_subset_sum(converter);
    test_covering(converter);
    test_cardinality(converter);
    test_nonlinear(converter);
}
