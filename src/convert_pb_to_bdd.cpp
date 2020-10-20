#include "convert_pb_to_bdd.h"

namespace LPMP {

    BDD::node_ref bdd_converter::convert_to_bdd(const std::vector<int> coefficients, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            return convert_to_bdd(coefficients.begin(), coefficients.end(), ineq, right_hand_side);
        }

    BDD::node_ref bdd_converter::convert_to_bdd_impl(std::vector<int>& nf, const ILP_input::inequality_type ineq)
        {
            assert(nf.size() > 0);
            const int right_hand_side = nf[0];
            if(nf.size() == 1) {
                switch(ineq) {
                    case ILP_input::inequality_type::equal: 
                        return right_hand_side == 0 ? bdd_mgr_.topsink() : bdd_mgr_.botsink();
                        //return right_hand_side == 0 ? bdd_mgr_.bddOne() : bdd_mgr_.bddZero();
                        break;
                    case ILP_input::inequality_type::smaller_equal:
                        return right_hand_side >= 0 ? bdd_mgr_.topsink() : bdd_mgr_.botsink();
                        break;
                    case ILP_input::inequality_type::greater_equal:
                        return right_hand_side <= 0 ? bdd_mgr_.topsink() : bdd_mgr_.botsink();
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                        break;
                }
            }

            // check, if constraint has already been seen. If so, retrieve
            switch(ineq) {
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

            // otherwise, recurse and build up constraint
            const int cur_coefficient = nf.back();
            nf.resize(nf.size()-1);

            BDD::node_ref bdd_0 = convert_to_bdd_impl(nf, ineq);
            // set first var to 1
            nf[0] -= cur_coefficient;
            BDD::node_ref bdd_1 = convert_to_bdd_impl(nf, ineq);
            nf[0] += cur_coefficient;

            BDD::node_ref cur_var = bdd_mgr_.projection(nf.size()-1);
            //auto bdd = cur_var.Ite(bdd_0, bdd_1);
            auto bdd = bdd_mgr_.ite_rec(cur_var, bdd_0, bdd_1);

            nf.push_back(cur_coefficient);
            // record bdd in cache
            switch(ineq) {
                case ILP_input::inequality_type::equal: 
                    equality_cache.insert(std::make_pair(nf,bdd));
                    break;
                case ILP_input::inequality_type::smaller_equal:
                    lower_equal_cache.insert(std::make_pair(nf,bdd));
                    break;
                case ILP_input::inequality_type::greater_equal:
                    throw std::runtime_error("greater equal constraint not in normal form");
                    break;
                default:
                    throw std::runtime_error("inequality type not supported");
                    break;
            } 

            return bdd;
        }


}
