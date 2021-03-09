#pragma once

#include <numeric>
#include <list>
#include <stack>
#include <vector>
#include <tsl/robin_map.h>
#include "bdd.h"
#include "ILP_input.h"
#include "avl_tree.hxx"
#include <iostream> // temporary

namespace LPMP {

    struct lineq_bdd_node {

        lineq_bdd_node() {}
        lineq_bdd_node(int lb, int ub, lineq_bdd_node* zero_kid, lineq_bdd_node* one_kid)
        : lb_(lb), ub_(ub), zero_kid_(zero_kid), one_kid_(one_kid)
        {}

        int lb_;
        int ub_; // initially also serves as cost of path from root

        lineq_bdd_node* zero_kid_;
        lineq_bdd_node* one_kid_;
        avl_node<lineq_bdd_node>* wrapper_; // wrapper node in the AVL tree
    };

    // Implementation of BDD construction from a linear inequality/equation (cf. Behle, 2007)
    class lineq_bdd {
        public:

            lineq_bdd() : topsink(0, std::numeric_limits<int>::max(), nullptr, nullptr), 
                botsink(std::numeric_limits<int>::min(), -1, nullptr, nullptr)
            {}
            lineq_bdd(lineq_bdd & other) = delete;

            void build_from_inequality(const std::vector<int> coefficients, const ILP_input::inequality_type ineq_type, const int right_hand_side);
            template<typename LEFT_HAND_SIDE_ITERATOR>
                void build_from_inequality(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side);
            BDD::node_ref convert_to_lbdd(BDD::bdd_mgr & bdd_mgr_) const;

            template<typename COEFF_ITERATOR>
                static std::tuple< std::vector<int>, ILP_input::inequality_type >
                normal_form(COEFF_ITERATOR begin, COEFF_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side);

        private:

            bool build_bdd_node(lineq_bdd_node* &node_ptr, const int path_cost, const unsigned int level, const ILP_input::inequality_type ineq_type);

            std::vector<char> inverted; // flags inverted variables
            std::vector<int> coefficients;
            std::vector<long int> rests;
            int rhs;

            lineq_bdd_node* root_node;
            // std::vector<avl_tree<lineq_bdd_node>> levels;
            std::vector<std::list<lineq_bdd_node>> levels;
            lineq_bdd_node topsink;
            lineq_bdd_node botsink;
    };


    template<typename COEFF_ITERATOR>
        std::tuple< std::vector<int>, ILP_input::inequality_type >
        lineq_bdd::normal_form(COEFF_ITERATOR begin, COEFF_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            assert(std::distance(begin,end) >= 1);
            int d = std::gcd(right_hand_side, *begin);
            for(auto it = begin+1; it != end; ++it)
                if(*it != 0)
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
        void lineq_bdd::build_from_inequality(LEFT_HAND_SIDE_ITERATOR begin, LEFT_HAND_SIDE_ITERATOR end, const ILP_input::inequality_type ineq, const int right_hand_side)
        {
            auto [nf, ineq_nf] = normal_form(begin, end, ineq, right_hand_side);

            const size_t dim = nf.size() - 1;
            inverted = std::vector<char>(dim);
            // levels = std::vector<avl_tree<lineq_bdd_node>>(dim);
            levels = std::vector<std::list<lineq_bdd_node>>(dim);

            rhs = nf[0];
            coefficients = std::vector<int>(nf.begin()+1, nf.end());

            // transform to nonnegative coefficients
            for (size_t i = 0; i < dim; i++)
            {
                if (coefficients[i] < 0)
                {
                    rhs -= coefficients[i];
                    coefficients[i] = -coefficients[i];
                    inverted[i] = 1;
                }
            }

            rests = std::vector<long int>(dim+1);
            rests[0] = std::accumulate(coefficients.begin(), coefficients.end(), 0);
            for (size_t i = 0; i < coefficients.size(); i++)
                rests[i+1] = rests[i] - coefficients[i];

            unsigned int level = 0;
            build_bdd_node(root_node, 0, level, ineq_nf);
            if (root_node == &topsink || root_node == &botsink)
                return;
            
            std::stack<lineq_bdd_node*> node_stack;
            node_stack.push(root_node);

            while (!node_stack.empty())
            {
                lineq_bdd_node* current_node = node_stack.top();
                assert(level < dim);
                const int coeff = coefficients[level];

                if (current_node->zero_kid_ == nullptr) // build zero child
                {
                    bool is_new = build_bdd_node(current_node->zero_kid_, current_node->ub_ + 0, level+1, ineq_nf);
                    if (!is_new)
                        continue;
                    node_stack.push(current_node->zero_kid_);
                    level++;
                }
                else if (current_node->one_kid_ == nullptr) // build one child
                {
                    bool is_new = build_bdd_node(current_node->one_kid_, current_node->ub_ + coeff, level+1, ineq_nf);
                    if (!is_new)
                        continue;
                    node_stack.push(current_node->one_kid_);
                    level++;
                }
                else // set bounds and go to parent
                {
                    auto* bdd_0 = current_node->zero_kid_;
                    auto* bdd_1 = current_node->one_kid_;
                    // lower bound of topsink needs to be changed if it is a shortcut
                    const int lb_0 = (bdd_0 == &topsink) ? rests[level+1] : bdd_0->lb_;
                    const int lb_1 = (bdd_1 == &topsink) ? rests[level+1] + coeff : bdd_1->lb_ + coeff;
                    const int lb = std::max(lb_0, lb_1);
                    // prevent integer overflow (coefficient is non-negative)
                    const int ub_1 = (std::numeric_limits<int>::max() - coeff < bdd_1->ub_) ? std::numeric_limits<int>::max() : bdd_1->ub_ + coeff;
                    const int ub = std::max(std::min(bdd_0->ub_, ub_1), lb); // ensure that bound-interval is non-empty
                    current_node->lb_ = lb;
                    current_node->ub_ = ub;
                    // std::cout << "Insert into AVL tree: level = " << level << ", range = [" << lb << "," << ub << "]" << std::endl;
                    // levels[level].insert(current_node->wrapper_); // when bounds are determined, insert into AVL tree
                    node_stack.pop();
                    level--;
                }
            }
        }


}
