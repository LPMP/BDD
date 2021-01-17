#pragma once

#include "bdd.h"
#include "ILP_input.h"
#include "hash_helper.hxx"
#include <tsl/robin_map.h>
#include "lineq_bdd.h"
#include <stack>
#include <numeric>
#include <tuple>
#include <iostream> // TODO: remove

namespace LPMP {
    

    class bdd_converter {
        public:
            bdd_converter(BDD::bdd_mgr& bdd_mgr) : bdd_mgr_(bdd_mgr) 
        {}

            template<typename LEFT_HAND_SIDE_ITERATOR>
                BDD::node_ref convert_to_bdd(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side);

            BDD::node_ref convert_to_bdd(const std::vector<int> coefficients, const ILP_input::inequality_type ineq, const int right_hand_side);

        private:
            mutable size_t tmp_rec_calls = 0;
            // returned vector has as its first element the right hand side, then follow the coefficients

            // implemenation with global subinequality cache
            bool is_always_true(const int min_val, const int max_val, const std::vector<int>& nf, const ILP_input::inequality_type ineq) const;
            bool is_always_false(const int min_val, const int max_val, const std::vector<int>& nf, const ILP_input::inequality_type ineq) const;
            BDD::node_ref convert_to_bdd_impl(std::vector<int>& nf, const ILP_input::inequality_type ineq, const int min_val, const int max_val);
   
            BDD::bdd_mgr& bdd_mgr_;
            //using constraint_cache_type = std::unordered_map<std::vector<int>,BDD::node_ref>;
            using constraint_cache_type = tsl::robin_map<std::vector<int>,BDD::node_ref>;
            constraint_cache_type equality_cache;
            constraint_cache_type lower_equal_cache;

            lineq_bdd bdd_;
    };

    template<typename LEFT_HAND_SIDE_ITERATOR>
        BDD::node_ref bdd_converter::convert_to_bdd(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            auto [nf, ineq_nf] = bdd_.normal_form(begin, end, ineq, right_hand_side);

            int min_val = 0;
            int max_val = 0;
            for(size_t i=1; i<nf.size(); ++i)
            {
                min_val += std::min(0, nf[i]);
                max_val += std::max(0, nf[i]);
            }

            tmp_rec_calls = 0;
            return convert_to_bdd_impl(nf, ineq_nf, min_val, max_val); 
        }
}
