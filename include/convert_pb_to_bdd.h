#pragma once

#include "bdd_manager/bdd.h"
#include "ILP_input.h"
#include "hash_helper.hxx"
#include <tsl/robin_map.h>
#include "lineq_bdd.h"
#include "two_dimensional_variable_array.hxx"
#include <stack>
#include <numeric>
#include <tuple>
#include <iostream> // TODO: remove

namespace LPMP {
    
    // convert linear inequalities to BDDs
    class bdd_converter {
        public:
            bdd_converter(BDD::bdd_mgr& bdd_mgr) : bdd_mgr_(bdd_mgr) 
            {
                bdd_ = lineq_bdd();
            }

            template<typename LEFT_HAND_SIDE_ITERATOR>
                BDD::node_ref convert_to_bdd(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq_type, const int right_hand_side);

            BDD::node_ref convert_to_bdd(const std::vector<int>& coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side);

            BDD::node_ref convert_nonlinear_to_bdd(const std::vector<size_t>& monomial_degrees, const std::vector<int>& coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side);

            // use the method from "A new look at BDDs for pseudo-Boolean constraints" from Abio et al for linear inequality conversion.
            // The BDD encodes the inequality for the decomposed coefficients.
            // The vector of integers gives the number of copies of each variable.
            std::tuple<BDD::node_ref, two_dim_variable_array<size_t>> coefficient_decomposition_convert_to_bdd(const std::vector<int>& coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side);
            
            const lineq_bdd& get_lineq_bdd() const { return bdd_; }
            lineq_bdd& get_lineq_bdd() { return bdd_; }

            BDD::bdd_mgr& bdd_mgr() { return bdd_mgr_; }

        private:
   
            BDD::bdd_mgr& bdd_mgr_;
            using constraint_cache_type = tsl::robin_map<std::vector<int>,BDD::node_ref>;
            constraint_cache_type equality_cache;
            constraint_cache_type lower_equal_cache;

            lineq_bdd bdd_;
    };

    template<typename LEFT_HAND_SIDE_ITERATOR>
        BDD::node_ref bdd_converter::convert_to_bdd(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            auto [nf, ineq_type] = bdd_.normal_form(begin, end, ineq, right_hand_side);

            // check cache for inequality type
            switch(ineq_type) {
                case ILP_input::inequality_type::equal: 
                    {
                        auto cached = equality_cache.find(nf);
                        if(cached != equality_cache.end())
                            return cached->second;
                    }
                    break;
                case ILP_input::inequality_type::smaller_equal:
                    {
                        auto cached = lower_equal_cache.find(nf);
                        if(cached != lower_equal_cache.end())
                            return cached->second;
                    }
                    break;
                case ILP_input::inequality_type::greater_equal:
                    throw std::runtime_error("greater equal constraint not in normal form");
                    break;
                default:
                    throw std::runtime_error("inequality type not supported");
                    break;
            }

            // otherwise build BDD
            bdd_.build_from_inequality(nf, ineq_type);
            BDD::node_ref bdd_ref = bdd_.convert_to_lbdd(bdd_mgr_);

            // store in cache
            switch(ineq_type) {
                case ILP_input::inequality_type::equal: 
                    equality_cache.insert(std::make_pair(nf,bdd_ref));
                    break;
                case ILP_input::inequality_type::smaller_equal:
                    lower_equal_cache.insert(std::make_pair(nf,bdd_ref));
                    break;
                case ILP_input::inequality_type::greater_equal:
                    throw std::runtime_error("greater equal constraint not in normal form");
                    break;
                default:
                    throw std::runtime_error("inequality type not supported");
                    break;
            }

            return bdd_ref;
        }
}
