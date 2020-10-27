#pragma once

#include "bdd.h"
#include "ILP_input.h"
#include "hash_helper.hxx"
#include <tsl/robin_map.h>
#include <numeric>
#include <tuple>

namespace LPMP {

    class bdd_converter {
        public:
            bdd_converter(BDD::bdd_mgr& bdd_mgr) : bdd_mgr_(bdd_mgr) 
        {}

            template<typename LEFT_HAND_SIDE_ITERATOR>
                BDD::node_ref convert_to_bdd(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side);

            BDD::node_ref convert_to_bdd(const std::vector<int> coefficients, const ILP_input::inequality_type ineq, const int right_hand_side); 

        private:
            // returned vector has as its first element the right hand side, then follow the coefficients
            template<typename COEFF_ITERATOR>
                static std::tuple< std::vector<int>, ILP_input::inequality_type >
                normal_form(COEFF_ITERATOR begin, COEFF_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side);

            BDD::node_ref convert_to_bdd_impl(std::vector<int>& nf, const ILP_input::inequality_type ineq);

            BDD::bdd_mgr& bdd_mgr_;
            //using constraint_cache_type = std::unordered_map<std::vector<int>,BDD::node_ref>;
            using constraint_cache_type = tsl::robin_map<std::vector<int>,BDD::node_ref>;
            constraint_cache_type equality_cache;
            constraint_cache_type lower_equal_cache;
    };

    template<typename COEFF_ITERATOR>
        std::tuple< std::vector<int>, ILP_input::inequality_type >
        bdd_converter::normal_form(COEFF_ITERATOR begin, COEFF_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            assert(std::distance(begin,end) >= 1);
            int d = std::gcd(right_hand_side, *begin);
            for(auto it = begin+1; it != end; ++it)
                d = std::gcd(d, *it);

            std::vector<int> c;
            c.reserve(std::distance(begin, end) + 1);
            c.push_back(right_hand_side/d);
            for(auto it = begin; it != end; ++it)
                c.push_back(*it/d);

            if(ineq == ILP_input::inequality_type::greater_equal)
                for(auto& x : c)
                    x *= -1.0;

            return {c, ineq != ILP_input::inequality_type::greater_equal ? ineq : ILP_input::inequality_type::smaller_equal};
        }

    template<typename LEFT_HAND_SIDE_ITERATOR>
        BDD::node_ref bdd_converter::convert_to_bdd(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            auto [nf, ineq_nf] = normal_form(begin, end, ineq, right_hand_side);
            return convert_to_bdd_impl(nf, ineq_nf); 
        }

}
