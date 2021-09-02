#pragma once

#include "bdd_node.h"
#include "bdd_node_cache.h"
#include <vector>
#include <unordered_map>
#include <array>

namespace BDD {


    struct memo_struct {
        node* f = nullptr;
        node* g = nullptr;
        node* h = nullptr;
        node* r = nullptr;

        template<typename T>
        constexpr static node* and_symb_impl() { return static_cast<T*>(nullptr) + 1; }
        constexpr static node* and_symb() { return and_symb_impl<node>(); }

        template<typename T>
        constexpr static node* or_symb_impl() { return static_cast<T*>(nullptr) + 2; }
        constexpr static node* or_symb() { return or_symb_impl<node>(); }

        template<typename T>
        constexpr static node* xor_symb_impl() { return static_cast<T*>(nullptr) + 3; }
        constexpr static node* xor_symb() { return or_symb_impl<node>(); }

        bool operator==(const memo_struct& m) const;
        bool operator!=(const memo_struct& m) const;

        bool can_be_purged() const;
    };

    using memo = memo_struct;

    class memo_cache {
        public:
            memo_cache(bdd_node_cache& _node_cache);
            node* cache_lookup(node* f, node* g, node* h);
            void cache_insert(node* f, node* g, node* h, node* r);
            memo_struct& get_memo(const size_t slot);

            void purge();

        private:
            size_t cache_hash(node* f, node* g, node* h);
            void init_cache();
            void double_cache();
            size_t choose_cache_size(const size_t items) const;
            size_t nr_occupied_slots() const;

            std::vector<memo_struct> memos;
            size_t memos_mask = 0;
            size_t cache_inserts = 0; // nr of times we have inserted into cache
            size_t threshold = 0; // nr of inserts that triggers cache doubling

            struct array_hasher {
                std::size_t operator()(const std::array<node*, 3>& a) const {
                    size_t h = 0;
                    for (auto e : a) {
                        h ^= std::hash<node*>{}(e)  + 0x9e3779b9 + (h << 6) + (h >> 2); 
                    }
                    return h;
                }   
            };
            std::unordered_map<std::array<node*,3>, std::tuple<node*,size_t>, array_hasher> memos2;
            bdd_node_cache& node_cache;
            size_t insert_time_stamp = 0;

    };

}
