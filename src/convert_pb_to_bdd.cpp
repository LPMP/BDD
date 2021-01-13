#include "convert_pb_to_bdd.h"
#include <iostream> // TODO: remove

namespace LPMP {

    BDD::node_ref bdd_converter::convert_to_bdd(const std::vector<int> coefficients, const ILP_input::inequality_type ineq, const int right_hand_side)
    {
        return convert_to_bdd(coefficients.begin(), coefficients.end(), ineq, right_hand_side);
    }

    bool bdd_converter::is_always_true(const int min_val, const int max_val, const std::vector<int>& nf, const ILP_input::inequality_type ineq) const
    {
        assert(nf.size() > 0);
        assert(min_val == std::accumulate(nf.begin()+1, nf.end(), 0, [](const int a, const int b) { return a + std::min(0,b); })); 
        assert(max_val == std::accumulate(nf.begin()+1, nf.end(), 0, [](const int a, const int b) { return a + std::max(0,b); })); 
        const int right_hand_side = nf[0];
        switch(ineq) {
            case ILP_input::inequality_type::equal: 
                return nf.size() == 1 && right_hand_side == 0; 
                break;
            case ILP_input::inequality_type::smaller_equal:
                {
                    // compute maximum value attainable
                    return max_val <= right_hand_side;
                    break;
                }
            case ILP_input::inequality_type::greater_equal:
                throw std::runtime_error("not implemented yet");
                break;
        } 
    }

    bool bdd_converter::is_always_false(const int min_val, const int max_val, const std::vector<int>& nf, const ILP_input::inequality_type ineq) const
    {
        assert(nf.size() > 0);
        assert(min_val == std::accumulate(nf.begin()+1, nf.end(), 0, [](const int a, const int b) { return a + std::min(0,b); })); 
        assert(max_val == std::accumulate(nf.begin()+1, nf.end(), 0, [](const int a, const int b) { return a + std::max(0,b); })); 
        const int right_hand_side = nf[0];
        switch(ineq) {
            case ILP_input::inequality_type::equal: 
                {
                    if(nf.size() == 1 && right_hand_side != 0)
                        return true;
                    if(min_val > right_hand_side)
                        return true;
                    if(max_val < right_hand_side)
                        return true;
                    return false;
                    break;
                }
            case ILP_input::inequality_type::smaller_equal:
                {
                    // compute maximum value attainable
                    return min_val > right_hand_side;
                    break;
                }
            case ILP_input::inequality_type::greater_equal:
                throw std::runtime_error("not implemented yet");
                break;
        } 
    }

    BDD::node_ref bdd_converter::convert_to_bdd_impl(std::vector<int>& nf, const ILP_input::inequality_type ineq, const int min_val, const int max_val)
    {
        tmp_rec_calls++;
        if(is_always_true(min_val, max_val, nf, ineq))
            return bdd_mgr_.topsink();
        if(is_always_false(min_val, max_val, nf, ineq))
            return bdd_mgr_.botsink();

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

        BDD::node_ref bdd_0 = convert_to_bdd_impl(nf, ineq, min_val - std::min(0, cur_coefficient), max_val - std::max(0, cur_coefficient));
        // set first var to 1
        nf[0] -= cur_coefficient;
        BDD::node_ref bdd_1 = convert_to_bdd_impl(nf, ineq, min_val - std::min(0, cur_coefficient), max_val - std::max(0, cur_coefficient));
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


    lineq_bdd & bdd_converter::build_bdd(const std::vector<int> coefficients, const ILP_input::inequality_type ineq, const int right_hand_side)
    {
        return build_bdd(coefficients.begin(), coefficients.end(), ineq, right_hand_side);
    }

    lineq_bdd_node* bdd_converter::build_bdd_node(const int slack, const int level, const int rest, const std::vector<int> & ineq, const ILP_input::inequality_type ineq_type)
    {
        assert(rest == std::accumulate(ineq.begin()+level+1, ineq.end(), 0)); 
        tmp_rec_calls++;

        // anchor
        switch (ineq_type)
        {
            case ILP_input::inequality_type::equal:
                if (slack < 0 || slack > rest)
                    return &bdd_.botsink;
                if (slack == 0 && slack == rest)
                    return &bdd_.topsink;
                break;
            case ILP_input::inequality_type::smaller_equal:
                if (slack < 0)
                    return &bdd_.botsink;
                if (slack >= rest)
                    return &bdd_.topsink;
                break;
            case ILP_input::inequality_type::greater_equal:
                throw std::runtime_error("greater equal constraint not in normal form");
                break;
            default:
                throw std::runtime_error("inequality type not supported");
                break;
        }

        assert(level < bdd_.levels.size());
        assert(level < ineq.size() - 1);

        // check for equivalent nodes
        // TODO: implement binary search over nodes at current level
        for (auto it = bdd_.levels[level].begin(); it != bdd_.levels[level].end(); it++)
        {
            if (slack >= it->lb_ && slack <= it->ub_)
                return &(*it);
        }

        // otherwise build children recursively
        const int coeff = ineq[level+1]; // first entry is right hand side
        auto* bdd_0 = build_bdd_node(slack, level+1, rest - coeff, ineq, ineq_type);
        auto* bdd_1 = build_bdd_node(slack - coeff, level+1, rest - coeff, ineq, ineq_type);
        assert(bdd_0 != nullptr);
        assert(bdd_1 != nullptr);

        const int lb = std::max(bdd_0->lb_, bdd_1->lb_ + coeff);
        const int ub = std::min(bdd_0->ub_, bdd_1->ub_ + coeff);

        lineq_bdd_node node(lb, ub, bdd_0, bdd_1);
        bdd_.levels[level].push_back(node);

        return &bdd_.levels[level].back();
    }

}
