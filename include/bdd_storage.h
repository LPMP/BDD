#pragma once

#include "bdd.h"
#include "bdd_preprocessor.h"
#include "bdd_collection.h"
#include "hash_helper.hxx"
#include <vector>
#include <stack>
#include <numeric>
#include <tsl/robin_map.h>

namespace LPMP {

    // used for storing all participating BDDs temporarily before exporting them to the bdd solver.
    class bdd_storage {
        public:

        struct bdd_node {
            constexpr static std::size_t terminal_0 = std::numeric_limits<std::size_t>::max()-1;
            constexpr static std::size_t terminal_1 = std::numeric_limits<std::size_t>::max();
            bool low_is_terminal() const { return low == terminal_0 || low == terminal_1; }
            bool high_is_terminal() const { return high == terminal_0 || high == terminal_1; }

            std::size_t low;
            std::size_t high;
            std::size_t variable;
        };

        bdd_storage(bdd_preprocessor& bdd_pre);
        bdd_storage();

        template <typename BDD_VARIABLES_ITERATOR>
        void add_bdd(BDD::bdd_mgr& bdd_mgr, BDD::node_ref bdd, BDD_VARIABLES_ITERATOR bdd_vars_begin, BDD_VARIABLES_ITERATOR bdd_vars_end);

        void add_bdd(BDD::bdd_collection_entry bdd);

        template <typename STREAM>
        void export_dot(STREAM &s) const;

        std::size_t nr_bdds() const { return bdd_delimiters().size()-1; }
        std::size_t nr_variables() const { return nr_variables_; }
        std::size_t nr_bdd_nodes(const std::size_t bdd_nr) const { assert(bdd_nr < nr_bdds()); return bdd_delimiters_[bdd_nr+1] - bdd_delimiters_[bdd_nr]; }

        const std::vector<bdd_node>& bdd_nodes() const { return bdd_nodes_; }
        const std::vector<std::size_t>& bdd_delimiters() const { return bdd_delimiters_; }

        // TODO: rename to ..._variable
        std::size_t first_bdd_node(const std::size_t bdd_nr) const;
        std::size_t last_bdd_node(const std::size_t bdd_nr) const;

        // return all edges with endpoints being variables that are consecutive in some BDD
        std::vector<std::array<size_t,2>> dependency_graph() const;

    private:
        void check_node_valid(const bdd_node bdd) const;

        template<typename BDD_NODE_TYPE, typename BDD_GETTER, typename VARIABLE_GETTER, typename NEXT_BDD_NODE, typename BDD_VARIABLE_ITERATOR>
            void add_bdd_impl(
                    const size_t nr_bdds,
                    BDD_GETTER bdd_getter,
                    VARIABLE_GETTER& get_var,
                    NEXT_BDD_NODE& next_bdd_node,
                    BDD_VARIABLE_ITERATOR bdd_vars_begin, BDD_VARIABLE_ITERATOR bdd_vars_end
                    );

        void check_bdd_node(const bdd_node bdd) const;

        std::vector<bdd_node> bdd_nodes_;
        std::vector<std::size_t> bdd_delimiters_ = {0};
        std::size_t nr_variables_ = 0;

    public:
        // for BDD decomposition
        struct intervals {
            std::vector<size_t> variable_interval; // in which interval does a variable fall
            std::vector<size_t> interval_boundaries; // variables at which new interval starts
            size_t interval(const size_t var) const;
            size_t nr_intervals() const; 
        };
        intervals compute_intervals(const size_t nr_intervals);

    public:
        // TODO: add hash function
        struct duplicate_variable {
            size_t interval_1;
            size_t bdd_index_1;
            size_t interval_2;
            size_t bdd_index_2; 

            bool operator==(const duplicate_variable& o) const
            {
return interval_1 == o.interval_1 && bdd_index_1 == o.bdd_index_1 && interval_2 == o.interval_2 && bdd_index_2 == o.bdd_index_2;
            }
        };

        struct duplicate_variable_hash {
            size_t operator()(const duplicate_variable& n) const
            {
                return hash::hash_array(std::array<size_t,4>{n.interval_1, n.bdd_index_1, n.interval_2, n.bdd_index_2});
            }
        };

        std::tuple<std::vector<bdd_storage>, tsl::robin_set<duplicate_variable, duplicate_variable_hash>> split_bdd_nodes(const size_t nr_intervals);
    };


    template<typename BDD_VARIABLES_ITERATOR>
        void bdd_storage::add_bdd(BDD::bdd_mgr& bdd_mgr, BDD::node_ref bdd, BDD_VARIABLES_ITERATOR bdd_vars_begin, BDD_VARIABLES_ITERATOR bdd_vars_end)
        {
            assert(std::is_sorted(bdd_vars_begin, bdd_vars_end));
            assert(std::distance(bdd_vars_begin, bdd_vars_end) > 0);

            std::vector<BDD::node_ref> bdd_nodes = bdd.nodes_postorder();

            auto get_bdd = [&](const size_t bdd_nr) {
                assert(bdd_nr < bdd_nodes.size());
                return bdd_nodes[bdd_nr];
            };

            auto get_variable = [&](BDD::node_ref& bdd) { 
                const size_t i = bdd.variable();
                return i;
                assert(i < std::distance(bdd_vars_begin, bdd_vars_end));
                return *(bdd_vars_begin + i); 
            };

            auto get_next_node = [&](BDD::node_ref& bdd) -> BDD::node_ref { 
                const ptrdiff_t p = std::distance(&bdd_nodes[0], &bdd);
                assert(p >= 0 && p < bdd_nodes.size());
                return bdd_nodes[p+1];
            };

            add_bdd_impl<BDD::node_ref>(
                    bdd_nodes.size(),
                    get_bdd,
                    get_variable,
                    get_next_node,
                    bdd_vars_begin, bdd_vars_end
                    );

            return;
        }

    template<typename BDD_NODE_TYPE, typename BDD_GETTER, typename VARIABLE_GETTER, typename NEXT_BDD_NODE, typename BDD_VARIABLE_ITERATOR>
        void bdd_storage::add_bdd_impl(
                const size_t nr_bdds,
                BDD_GETTER bdd_getter,
                VARIABLE_GETTER& get_var, 
                NEXT_BDD_NODE& next_bdd_node,
                BDD_VARIABLE_ITERATOR bdd_vars_begin, BDD_VARIABLE_ITERATOR bdd_vars_end)
        {
            //std::unordered_map<BDD_NODE_TYPE, size_t> node_to_index;
            tsl::robin_map<BDD_NODE_TYPE, size_t> node_to_index;

            auto get_node_index = [&](BDD_NODE_TYPE node) -> std::size_t {
                if(node.is_botsink()) {
                    return bdd_node::terminal_0;
                } else if(node.is_topsink()) {
                    return bdd_node::terminal_1;
                } else {
                    assert(node_to_index.count(node) > 0);
                    return node_to_index.find(node)->second;
                }
            };

            // node indices of chain pointing to terminal_1
            constexpr static std::size_t pointer_to_terminal_1_not_set = std::numeric_limits<std::size_t>::max()-2;
            std::vector<std::size_t> var_to_bdd_node_terminal_1(std::distance(bdd_vars_begin, bdd_vars_end), pointer_to_terminal_1_not_set);
            var_to_bdd_node_terminal_1.back() = bdd_node::terminal_1;

            auto add_intermediate_nodes = [&](BDD_NODE_TYPE start, BDD_NODE_TYPE end) -> std::size_t {

                const std::size_t start_var = get_var(start);

                if(!end.is_terminal()) {
                    const size_t end_var = get_var(end);
                    size_t last_index = get_node_index(end);
                    for(std::size_t i = end_var-1; i != start_var; --i) {
                        assert(i>0);
                        const std::size_t v_intermed = *(bdd_vars_begin + i);
                        bdd_nodes_.push_back({last_index, last_index, v_intermed});
                        last_index = bdd_nodes_.size()-1;
                    }
                    return last_index; 

                } else if(get_node_index(end) == bdd_node::terminal_1) {

                    if(var_to_bdd_node_terminal_1[start_var] == pointer_to_terminal_1_not_set) {
                        for(std::ptrdiff_t i = std::ptrdiff_t(std::distance(bdd_vars_begin, bdd_vars_end))-2; i >= std::ptrdiff_t(start_var); --i) {
                            assert(i >= 0 && i < var_to_bdd_node_terminal_1.size());
                            if(var_to_bdd_node_terminal_1[i] == pointer_to_terminal_1_not_set) {
                                const std::size_t v_intermed = *(bdd_vars_begin + i+1);
                                bdd_nodes_.push_back({var_to_bdd_node_terminal_1[i+1], var_to_bdd_node_terminal_1[i+1], v_intermed}); 
                                check_node_valid(bdd_nodes_.back());
                                var_to_bdd_node_terminal_1[i] = bdd_nodes_.size()-1;
                            }
                        }
                    }
                    return var_to_bdd_node_terminal_1[start_var];
                    
                } else if(get_node_index(end) == bdd_node::terminal_0) {
                    return get_node_index(end);
                } else {
                    assert(false);
                    throw std::runtime_error("invalid node");
                }
            };

            const size_t nr_bdd_nodes_begin = bdd_nodes_.size();

            for(size_t i=0; i<nr_bdds; ++i)
            {
                auto node = bdd_getter(i);
                const size_t variable = get_var(node);
                nr_variables_ = std::max(*(bdd_vars_begin+variable)+1, nr_variables_);

                const size_t low_index = add_intermediate_nodes(node, node.low());
                const size_t high_index = add_intermediate_nodes(node, node.high());

                assert(node_to_index.count(node) == 0);
                node_to_index.insert({node, bdd_nodes_.size()});
                bdd_nodes_.push_back(bdd_node{high_index, low_index, *(bdd_vars_begin+variable)});
                check_node_valid(bdd_nodes_.back()); 
            } 

            const size_t nr_bdd_nodes_end = bdd_nodes_.size();
            bdd_delimiters_.push_back(bdd_delimiters_.back() + nr_bdd_nodes_end - nr_bdd_nodes_begin);
        }


    template<typename STREAM>
        void bdd_storage::export_dot(STREAM& s) const
        {
            s << "digraph bdd_min_marginal_averaging {\n";

            auto get_node_string = [&](const std::size_t i) -> std::string {
                if(i == bdd_node::terminal_0)
                    return "false";
                if(i == bdd_node::terminal_1)
                    return "true";
                return std::to_string(i);
            };

            for(std::size_t i=0; i<bdd_nodes_.size(); ++i) {
                s << i << " -> " << get_node_string(bdd_nodes_[i].low) << " [label=\"0\"];\n";
                s << i << " -> " << get_node_string(bdd_nodes_[i].high) << " [label=\"1\"];\n";
            }
            s << "}\n"; 
        }


} // namespace LPMP
