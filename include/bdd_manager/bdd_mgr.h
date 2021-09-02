#pragma once

#include "bdd_node.h"
#include "bdd_node_cache.h"
#include "bdd_var.h"
#include "bdd_memo_cache.h"
#include <vector>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <cassert>

namespace BDD {

    class bdd_collection; // forward declaration for enabling import from bdds

    class bdd_mgr {
        public:
            bdd_mgr();
            ~bdd_mgr();
            size_t add_variable();
            size_t nr_variables() const { return vars.size(); }
            size_t nr_nodes() const { return node_cache_.nr_nodes(); }
            node_ref projection(const size_t var);
            node_ref neg_projection(const size_t var);
            node_ref negate(node_ref p);
            node_ref topsink() const { return node_ref(node_cache_.topsink()); }
            node_ref botsink() const { return node_ref(node_cache_.botsink()); }

            node_ref unique_find(const size_t var, node_ref lo, node_ref hi);

            template<class... NODES>
                node_ref and_rec(node_ref p, NODES...);
            template<class ITERATOR>
                node_ref and_rec(ITERATOR nodes_begin, ITERATOR nodes_end);
            node_ref and_rec(node_ref f, node_ref g);
            // return and conjunction if it has fewer than node_limit nodes. Otherwise return null reference and abort early
            std::tuple<node_ref,size_t> and_rec_limited(node_ref f, node_ref g, const size_t node_limit);

            template<class... NODES>
                node_ref or_rec(node_ref p, NODES...);
            template<class ITERATOR>
                node_ref or_rec(ITERATOR nodes_begin, ITERATOR nodes_end);
            node_ref or_rec(node_ref f, node_ref g);

            template<class... NODES>
                node_ref xor_rec(node_ref p, NODES... tail);
            template<class ITERATOR>
                node_ref xor_rec(ITERATOR nodes_begin, ITERATOR nodes_end);
            node_ref xor_rec(node_ref f, node_ref g);

            // f is if-condition, g is for 1-outcome, h is for lo outcome
            node_ref ite_rec(node_ref f, node_ref g, node_ref h);
            //node_ref ite_non_rec(node_ref f, node_ref g, node_ref h, std::stack<>& stack);

            // make a copy of bdd rooted at node to variables given
            // assume variable map is given by hash
            template<typename VAR_MAP>
            node_ref rebase(node_ref p, const VAR_MAP& var_map);
            // assume variable map is given by vector
            template<typename ITERATOR>
            node_ref rebase(node_ref p, ITERATOR var_map_begin, ITERATOR var_map_end);

            // make private and add friend classes
            bdd_node_cache& get_node_cache() { return node_cache_; }
            unique_table_page_caches& get_unique_table_page_cache() { return page_cache_; }

            void collect_garbage();

            // utility functions for computing common functions
            template<typename BDD_ITERATOR>
                node_ref all_false(BDD_ITERATOR begin, BDD_ITERATOR end);
            template<typename BDD_ITERATOR>
                node_ref simplex(BDD_ITERATOR begin, BDD_ITERATOR end);
            template<typename BDD_ITERATOR>
                node_ref at_most_one(BDD_ITERATOR begin, BDD_ITERATOR end);
            template<typename BDD_ITERATOR>
                node_ref at_least_one(BDD_ITERATOR begin, BDD_ITERATOR end);

            template<typename BDD_ITERATOR>
                node_ref at_least(BDD_ITERATOR begin, BDD_ITERATOR end, const size_t b);
            template<typename BDD_ITERATOR>
                node_ref at_most(BDD_ITERATOR begin, BDD_ITERATOR end, const size_t b);
            template<typename BDD_ITERATOR>
                node_ref cardinality(BDD_ITERATOR begin, BDD_ITERATOR end, const size_t b);

            node_ref transform_to_base();
            node_ref add_bdd(bdd_collection& bdd_col, const size_t bdd_nr);

        private:

            bdd_node_cache node_cache_;
            unique_table_page_caches page_cache_;
            memo_cache memo_;
            std::vector<var_struct> vars; // vars must be after node cache und page cache for correct destructor calling order

    }; 

    template<typename VAR_MAP>
    node_ref bdd_mgr::rebase(node_ref p, const VAR_MAP& var_map)
    {
        size_t last_var = 0;
        for(const auto [x,y] : var_map)
            last_var = std::max(y, last_var);
        for(size_t i=nr_variables(); i<=last_var; ++i)
            add_variable();

        // copy nodes one by one in postordering
        const auto postorder = p.address()->nodes_postorder();
        std::unordered_map<node*, node*> node_map;
        node_map.insert({node_cache_.botsink(), node_cache_.botsink()});
        node_map.insert({node_cache_.topsink(), node_cache_.topsink()});
        for(node* p : postorder) {
            const size_t v_orig = p->index;
            assert(var_map.count(v_orig) > 0);
            const size_t v_new = var_map.find(v_orig)->second;

            assert(node_map.count(p->lo) > 0);
            assert(node_map.count(p->hi) > 0);
            node* lo_mapped = node_map.find(p->lo)->second;
            node* hi_mapped = node_map.find(p->hi)->second;
            node_map.insert({p, vars[v_new].unique_find(lo_mapped, hi_mapped)});
        }

        return node_ref(node_map.find(p.address())->second);
    }

    template<typename ITERATOR>
    node_ref bdd_mgr::rebase(node_ref p, ITERATOR var_map_begin, ITERATOR var_map_end)
    {
        const size_t nr_vars = std::distance(var_map_begin, var_map_end);
        assert(p.variables().back() <= nr_vars);
        const size_t last_var = *std::max_element(var_map_begin, var_map_end);
        for(size_t i=nr_variables(); i<=last_var; ++i)
            add_variable();

        const auto postorder = p.address()->nodes_postorder();
        std::unordered_map<node*, node*> node_map;
        node_map.insert({node_cache_.botsink(), node_cache_.botsink()});
        node_map.insert({node_cache_.topsink(), node_cache_.topsink()});
        for(node* p : postorder) {
            const size_t v_orig = p->index;
            assert(v_orig < nr_vars);
            const size_t v_new = *(var_map_begin + v_orig);

            assert(node_map.count(p->lo) > 0);
            assert(node_map.count(p->hi) > 0);
            node* lo_mapped = node_map.find(p->lo)->second;
            node* hi_mapped = node_map.find(p->hi)->second;
            node_map.insert({p, vars[v_new].unique_find(lo_mapped, hi_mapped)});
        }

        return node_ref(node_map.find(p.address())->second);
    }

    //remplate<class... NODES, class = std::conjunction<std::is_same<node*, NODES>...>
    template<class... NODES>
        node_ref bdd_mgr::and_rec(node_ref p, NODES... tail)
        {
            node_ref and_tail(and_rec(tail...));
            return and_rec(p, and_tail); 
        }

    template<class ITERATOR>
        node_ref bdd_mgr::and_rec(ITERATOR nodes_begin, ITERATOR nodes_end)
        {
            const size_t n = std::distance(nodes_begin, nodes_end);
            assert(n >= 2);
            if(n == 2)
                return and_rec(*nodes_begin, *(nodes_begin+1));
            else if(n == 3)
                return and_rec(*nodes_begin, and_rec(*(nodes_begin+1), *(nodes_begin+2))); 

            node_ref a1(and_rec(nodes_begin, nodes_begin+n/2));
            node_ref a2(and_rec(nodes_begin+n/2, nodes_end));
            return and_rec(a1, a2);
        }

    template<class... NODES>
        node_ref bdd_mgr::or_rec(node_ref p, NODES... tail)
        {
            node_ref or_tail(or_rec(tail...));
            return or_rec(p, or_tail); 
        }
    template<class ITERATOR>
        node_ref bdd_mgr::or_rec(ITERATOR nodes_begin, ITERATOR nodes_end)
        {
            const size_t n = std::distance(nodes_begin, nodes_end);
            assert(n >= 2);
            if(n == 2)
                return or_rec(*nodes_begin, *(nodes_begin+1));
            else if(n == 3)
                return or_rec(*nodes_begin, or_rec(*(nodes_begin+1), *(nodes_begin+2))); 

            node_ref o1(or_rec(nodes_begin, nodes_begin+n/2));
            node_ref o2(or_rec(nodes_begin+n/2, nodes_end));
            return or_rec(o1, o2);
        }

    template<class... NODES>
        node_ref bdd_mgr::xor_rec(node_ref p, NODES... tail)
        {
            node_ref xor_tail(xor_rec(tail...));
            return xor_rec(p, xor_tail); 
        }
    template<class ITERATOR>
        node_ref bdd_mgr::xor_rec(ITERATOR nodes_begin, ITERATOR nodes_end)
        {
            const size_t n = std::distance(nodes_begin, nodes_end);
            assert(n >= 2);
            if(n == 2)
                return xor_rec(*nodes_begin, *(nodes_begin+1));
            else if(n == 3)
                return xor_rec(*nodes_begin, xor_rec(*(nodes_begin+1), *(nodes_begin+2))); 

            node_ref o1(xor_rec(nodes_begin, nodes_begin+n/2));
            node_ref o2(xor_rec(nodes_begin+n/2, nodes_end));
            return xor_rec(o1, o2);
        } 

    inline node_ref operator&(node_ref a, node_ref b)
    {
        bdd_mgr* mgr = a.find_bdd_mgr();
        return mgr->and_rec(a, b);

    }

    inline node_ref operator||(node_ref a, node_ref b)
    {
        bdd_mgr* mgr = a.find_bdd_mgr();
        return mgr->or_rec(a, b);

    }

    inline node_ref operator^(node_ref a, node_ref b)
    {
        bdd_mgr* mgr = a.find_bdd_mgr();
        return mgr->xor_rec(a, b); 
    }

    inline node_ref operator!(node_ref a)
    {
        bdd_mgr* mgr = a.find_bdd_mgr();
        return mgr->negate(a);
    }

    template<typename BDD_ITERATOR>
        node_ref bdd_mgr::all_false(BDD_ITERATOR begin, BDD_ITERATOR end)
        {
            const size_t n = std::distance(begin, end);
            assert(n > 0);
            if(n == 1)
                return negate(*begin);
            if(n == 2)
                return and_rec(negate(*begin), negate(*(begin+1)));
            return and_rec(
                    all_false(begin, begin+n/2),
                    all_false(begin+n/2, end)
                    );
        }

    template<typename BDD_ITERATOR>
        node_ref bdd_mgr::simplex(BDD_ITERATOR begin, BDD_ITERATOR end)
        {
            assert(std::distance(begin, end) > 0);
            if(std::distance(begin,end) == 1)
                return *begin;

            // at least one is active
            node_ref alo = or_rec(begin,end);

            // at most one is active
            node_ref amo = at_most_one(begin, end);

            return and_rec(alo, amo); 
        }

    template<typename BDD_ITERATOR>
        node_ref bdd_mgr::at_most_one(BDD_ITERATOR begin, BDD_ITERATOR end)
        {
            assert(std::distance(begin, end) > 0);
            const size_t n = std::distance(begin, end);
            if(n == 1)
                return node_ref(node_cache_.topsink());
            if(n == 2)
                return or_rec(negate(*begin), negate(*(begin+1)));
            if(n == 3)
                return and_rec( 
                        or_rec(negate(*begin), negate(*(begin+1))),
                        or_rec(negate(*begin), negate(*(begin+2))),
                        or_rec(negate(*(begin+1)), negate(*(begin+2)))
                        );

            node_ref at_most_one_1 = at_most_one(begin, begin + n/2);
            node_ref all_false_1 = all_false(begin, begin + n/2);
            node_ref at_most_one_2 = at_most_one(begin + n/2, end);
            node_ref all_false_2 = all_false(begin + n/2, end);
            return or_rec(
                    and_rec(at_most_one_1, all_false_2),
                    and_rec(at_most_one_2, all_false_1)
                    ); 
        }

    template<typename BDD_ITERATOR>
        node_ref bdd_mgr::at_least_one(BDD_ITERATOR begin, BDD_ITERATOR end)
        {
            return negate(all_false(begin, end));
        }

    template<typename BDD_ITERATOR>
        node_ref bdd_mgr::at_least(BDD_ITERATOR begin, BDD_ITERATOR end, const size_t b)
        {
            assert(std::distance(begin, end) > 0);
            const size_t n = std::distance(begin, end);

            if(n < b)
                return node_cache_.botsink();

            if(n == 1 && b == 1)
                return *begin;
            if(b == 0)
                return node_cache_.topsink();

            std::vector<node_ref> left;
            left.reserve(b+1);
            for(size_t b_left=0; b_left<=b; ++b_left)
                left.push_back(at_least(begin, begin + n/2, b_left));

            std::vector<node_ref> right;
            right.reserve(b+1);
            for(size_t b_right=0; b_right<=b; ++b_right)
                right.push_back(at_least(begin + n/2, end, b_right));

            std::vector<node_ref> combine;
            combine.reserve(b+1);
            for(size_t b_left=0; b_left<=b; ++b_left)
                combine.push_back(and_rec(left[b_left], right[b-b_left]));

            return or_rec(combine.begin(), combine.end()); 
        }

    template<typename BDD_ITERATOR>
        node_ref bdd_mgr::at_most(BDD_ITERATOR begin, BDD_ITERATOR end, const size_t b)
        {
            assert(std::distance(begin, end) > 0);
            const size_t n = std::distance(begin, end);

            if(n < b)
                return node_cache_.botsink();

            if(n == 1 && b == 1)
                return node_cache_.topsink();
            if(n == 1 && b == 0)
                return negate(*begin);
            if(b == 0)
                return all_false(begin, end);

            std::vector<node_ref> left;
            left.reserve(b+1);
            for(size_t b_left=0; b_left<=b; ++b_left)
                left.push_back(at_most(begin, begin + n/2, b_left));

            std::vector<node_ref> right;
            right.reserve(b+1);
            for(size_t b_right=0; b_right<=b; ++b_right)
                right.push_back(at_most(begin + n/2, end, b_right));

            std::vector<node_ref> combine;
            combine.reserve(b+1);
            for(size_t b_left=0; b_left<=b; ++b_left)
                combine.push_back(and_rec(left[b_left], right[b-b_left]));

            return or_rec(combine.begin(), combine.end()); 
        }

    template<typename BDD_ITERATOR>
        node_ref bdd_mgr::cardinality(BDD_ITERATOR begin, BDD_ITERATOR end, const size_t b)
        {
            assert(std::distance(begin, end) > 0);
            const size_t n = std::distance(begin, end);

            if(n < b)
                return node_cache_.botsink();

            if(n == 1 && b == 1)
                return *begin;
            if(n == 1 && b == 0)
                return negate(*begin);
            if(b == 0)
                return all_false(begin, end);

            std::vector<node_ref> left;
            left.reserve(b+1);
            for(size_t b_left=0; b_left<=b; ++b_left)
                left.push_back(cardinality(begin, begin + n/2, b_left));

            std::vector<node_ref> right;
            right.reserve(b+1);
            for(size_t b_right=0; b_right<=b; ++b_right)
                right.push_back(cardinality(begin + n/2, end, b_right));

            std::vector<node_ref> combine;
            combine.reserve(b+1);
            for(size_t b_left=0; b_left<=b; ++b_left)
                combine.push_back(and_rec(left[b_left], right[b-b_left]));

            return or_rec(combine.begin(), combine.end()); 
        }
}
