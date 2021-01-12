#pragma once

#include "bdd.h"
#include "ILP_input.h"
#include "hash_helper.hxx"
#include <tsl/robin_map.h>
#include "lineq_bdd.h"
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

            template<typename LEFT_HAND_SIDE_ITERATOR>
                lineq_bdd & build_bdd(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side);
            lineq_bdd & build_bdd(const std::vector<int> coefficients, const ILP_input::inequality_type ineq, const int right_hand_side);
            lineq_bdd & get_bdd() { return bdd_; };

        private:
            mutable size_t tmp_rec_calls = 0;
            // returned vector has as its first element the right hand side, then follow the coefficients
            template<typename COEFF_ITERATOR>
                static std::tuple< std::vector<int>, ILP_input::inequality_type >
                normal_form(COEFF_ITERATOR begin, COEFF_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side);

            // implemenation with global subinequality cache
            bool is_always_true(const int min_val, const int max_val, const std::vector<int>& nf, const ILP_input::inequality_type ineq) const;
            bool is_always_false(const int min_val, const int max_val, const std::vector<int>& nf, const ILP_input::inequality_type ineq) const;
            BDD::node_ref convert_to_bdd_impl(std::vector<int>& nf, const ILP_input::inequality_type ineq, const int min_val, const int max_val);

            // implementation with equivalent node detection per inequality (Behle, 2007)
            lineq_bdd_node* build_bdd_node(const int slack, const int level, const int rest, const std::vector<int> & ineq, const ILP_input::inequality_type ineq_type);

            BDD::bdd_mgr& bdd_mgr_;
            //using constraint_cache_type = std::unordered_map<std::vector<int>,BDD::node_ref>;
            using constraint_cache_type = tsl::robin_map<std::vector<int>,BDD::node_ref>;
            constraint_cache_type equality_cache;
            constraint_cache_type lower_equal_cache;

            lineq_bdd bdd_;
    };

    template<typename COEFF_ITERATOR>
        std::tuple< std::vector<int>, ILP_input::inequality_type >
        bdd_converter::normal_form(COEFF_ITERATOR begin, COEFF_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            assert(std::distance(begin,end) >= 1);
            int d = std::gcd(right_hand_side, *begin);
            for(auto it = begin+1; it != end; ++it)
            {
                assert(*it != 0);
                d = std::gcd(d, *it);
            }

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

    template<typename LEFT_HAND_SIDE_ITERATOR>
        lineq_bdd & bdd_converter::build_bdd(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            auto [nf, ineq_nf] = normal_form(begin, end, ineq, right_hand_side);

            const size_t dim = nf.size() - 1;
            bdd_ = lineq_bdd(dim);

            // transform to nonnegative coefficients
            for (size_t i = 1; i < dim + 1; i++)
            {
                if (nf[i] < 0)
                {
                    nf[0] -= nf[i];
                    nf[i] = -nf[i];
                    bdd_.inverted[i] = 1;
                }
            }

            const int rest = std::accumulate(nf.begin()+1, nf.end(), 0);
            const int level = 0;
            const int slack = nf[0];

            tmp_rec_calls = 0;
            bdd_.root_node = build_bdd_node(slack, level, rest, nf, ineq_nf);
            return bdd_;
        }
}
