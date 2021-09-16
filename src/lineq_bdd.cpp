#include "lineq_bdd.h"
#include <fstream>
#include <filesystem>

namespace LPMP {

    bool lineq_bdd::build_bdd_node(lineq_bdd_node * &node_ptr, const integer path_cost, const unsigned int level, const ILP_input::inequality_type ineq_type)
    {
        assert(level < rests.size());
        const integer slack = rhs - path_cost;
        const integer rest = rests[level];

        // check sink conditions
        switch (ineq_type)
        {
            case ILP_input::inequality_type::equal:
                if (slack < 0 || slack > rest)
                {
                    node_ptr = &botsink;
                    return false;
                }
                if (slack == 0 && slack == rest)
                {
                    node_ptr = &topsink;
                    return false;
                }
                break;
            case ILP_input::inequality_type::smaller_equal:
                if (slack < 0)
                {
                    node_ptr = &botsink;
                    return false;
                }
                if (slack >= rest)
                {
                    node_ptr = &topsink;
                    return false;
                }
                break;
            case ILP_input::inequality_type::greater_equal:
                throw std::runtime_error("greater equal constraint not in normal form");
                break;
            default:
                throw std::runtime_error("inequality type not supported");
                break;
        }

        assert(level < levels.size());

        // check for equivalent nodes
        avl_node<lineq_bdd_node> * ptr = levels[level].find(slack);
        if (ptr != nullptr)
        {
            if (ptr->wraps_botsink) // check if node is equivalent to botsink (only applicable for equations)
                node_ptr = &botsink;
            else
                node_ptr = &(ptr->data);
            return false;
        }

        // otherwise create new node
        lineq_bdd_node node;
        node.ub_ = path_cost;
        node_ptr = levels[level].create_node(node);
        assert(node_ptr != nullptr);
        return true;
    }


    void lineq_bdd::build_from_inequality(const std::vector<int>& nf, const ILP_input::inequality_type ineq_type)
    {
        const size_t dim = nf.size() - 1;
        inverted = std::vector<char>(dim);
        levels = std::vector<avl_tree<lineq_bdd_node>>(dim);

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

        rests = std::vector<integer>(dim+1);
        rests[0] = std::accumulate(coefficients.begin(), coefficients.end(), 0);
        for (size_t i = 0; i < coefficients.size(); i++)
            rests[i+1] = rests[i] - coefficients[i];

        unsigned int level = 0;
        build_bdd_node(root_node, 0, level, ineq_type);
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
                bool is_new = build_bdd_node(current_node->zero_kid_, current_node->ub_ + 0, level+1, ineq_type);
                if (!is_new)
                    continue;
                node_stack.push(current_node->zero_kid_);
                level++;
            }
            else if (current_node->one_kid_ == nullptr) // build one child
            {
                bool is_new = build_bdd_node(current_node->one_kid_, current_node->ub_ + coeff, level+1, ineq_type);
                if (!is_new)
                    continue;
                node_stack.push(current_node->one_kid_);
                level++;
            }
            else // set bounds and go to parent
            {
                auto* bdd_0 = current_node->zero_kid_;
                auto* bdd_1 = current_node->one_kid_;
                switch (ineq_type)
                {
                    case ILP_input::inequality_type::equal:
                    {
                        // replace children by botsink if they are equivalent
                        if ((bdd_0->wrapper_ != nullptr) && bdd_0->wrapper_->wraps_botsink)
                            bdd_0 = &botsink;
                        if ((bdd_1->wrapper_ != nullptr) && bdd_1->wrapper_->wraps_botsink)
                            bdd_1 = &botsink;
                        if (bdd_0 == &botsink && bdd_1 == &botsink) // label node equivalent to botsink
                        {
                            assert(current_node->wrapper_ != nullptr);
                            current_node->wrapper_->wraps_botsink = true;
                        }
                        // set lower bound and upper bound to match slack
                        current_node->lb_ = rhs - current_node->ub_;
                        current_node->ub_ = rhs - current_node->ub_;
                        break;
                    }
                    case ILP_input::inequality_type::smaller_equal:
                    {
                        // lower bound of topsink needs to be adjusted if it is a shortcut
                        const integer lb_0 = (bdd_0 == &topsink) ? rests[level+1] : bdd_0->lb_;
                        const integer lb_1 = (bdd_1 == &topsink) ? rests[level+1] + coeff : bdd_1->lb_ + coeff;
                        const integer lb = std::max(lb_0, lb_1);
                        const integer ub = std::max(std::min(bdd_0->ub_, bdd_1->ub_ + coeff), lb); // ensure that bound-interval is non-empty
                        current_node->lb_ = lb;
                        current_node->ub_ = ub;
                        break;
                    }
                    case ILP_input::inequality_type::greater_equal:
                        throw std::runtime_error("greater equal constraint not in normal form");
                        break;
                    default:
                        throw std::runtime_error("inequality type not supported");
                        break;
                }
                levels[level].insert(current_node->wrapper_); // when bounds are determined, insert into AVL tree
                node_stack.pop();
                level--;
            }
        }
    }


    BDD::node_ref lineq_bdd::convert_to_lbdd(BDD::bdd_mgr& bdd_mgr_) const
    {
        if (root_node == &topsink)
            return bdd_mgr_.topsink();
        if (root_node == &botsink)
            return bdd_mgr_.botsink();

        std::vector<std::vector<BDD::node_ref>> bdd_nodes(levels.size());
        tsl::robin_map<lineq_bdd_node const*,size_t> node_refs;
        for(std::ptrdiff_t l=levels.size()-1; l>=0; --l)
        {
            auto& nodes = levels[l].get_avl_nodes();
            for(auto it = nodes.begin(); it != nodes.end(); it++)
            {
                auto& lbdd = it->data;
                auto get_node = [&](lineq_bdd_node const* ptr) {
                    if(ptr == &botsink)
                        return bdd_mgr_.botsink();
                    else if(ptr == &topsink)
                        return bdd_mgr_.topsink();
                    else
                    {
                        auto ref = node_refs.find(ptr);
                        if (ref != node_refs.end())
                        {
                            assert(ref->second < bdd_nodes[l+1].size());
                            return bdd_nodes[l+1][ref->second];
                        }
                        else
                            throw std::runtime_error("node reference not found");
                    }
                };
                BDD::node_ref zero_bdd_node = get_node(lbdd.zero_kid_);
                BDD::node_ref one_bdd_node = get_node(lbdd.one_kid_);
                if(inverted[l])
                    bdd_nodes[l].push_back(bdd_mgr_.ite_rec(bdd_mgr_.projection(l), zero_bdd_node, one_bdd_node));
                else
                    bdd_nodes[l].push_back(bdd_mgr_.ite_rec(bdd_mgr_.projection(l), one_bdd_node, zero_bdd_node));
                node_refs.insert(std::make_pair(&lbdd, bdd_nodes[l].size()-1));
            }
        }
        assert(bdd_nodes[0].size() == 1);
        return bdd_nodes[0][0];
    }

    void lineq_bdd::export_graphviz(const std::string& filename)
    {
        const std::string dot_file = std::filesystem::path(filename).replace_extension("dot");
        std::fstream f;
        f.open(dot_file, std::ios::out | std::ios::trunc);
        export_graphviz(f);
        f.close();

        const std::string png_file = std::filesystem::path(filename).replace_extension("png");
        const std::string convert_command = "dot -Tpng " + dot_file + " > " + png_file;
        std::system(convert_command.c_str());
    }

}
