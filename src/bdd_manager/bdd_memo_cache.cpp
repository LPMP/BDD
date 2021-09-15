#include "bdd_manager/bdd_memo_cache.h"
#include <cassert>
#include <algorithm>

namespace BDD {

    bool memo_struct::operator==(const memo_struct& m) const
    {
        return f == m.f && g == m.g && h == m.h && r == m.r; 
    }

    bool memo_struct::operator!=(const memo_struct& m) const
    {
        return !(*this == m);
    }

    memo_cache::memo_cache(bdd_node_cache& _node_cache)
        : node_cache(_node_cache)
    {
        init_cache();
    }

    size_t memo_cache::cache_hash(node* f, node* g, node* h)
    {
        //const std::size_t f_hash = f->index;
        //const std::size_t g_hash = (h == nullptr) ? (g->index << 1) : (reinterpret_cast<size_t>(g) << 1);
        //const std::size_t h_hash = reinterpret_cast<size_t>(h) << 2;

        const size_t f_hash = f->hash_key;
        assert(g != nullptr);
        const size_t g_hash = g->hash_key << 1;
        const size_t h_hash = reinterpret_cast<size_t>(h) << 2;

        size_t hash = f_hash;
        hash ^= g_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= h_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        //return f_hash ^ g_hash ^ h_hash; 
        return hash;
    }

    memo_struct& memo_cache::get_memo(const size_t slot)
    {
        assert(memos_mask + 1 == memos.size());
        const size_t n = slot & memos_mask;
        assert(n < memos.size());
        //std::cout << "memo slot = " << n << ", memos size = " << memos.size() << ", cache mask = " << memos_mask << "\n";
        return memos[n];
    }

    node* memo_cache::cache_lookup(node* f, node* g, node* h)
    {
        /*
        if(memos2.count({f,g,h}) == 0)
            return nullptr;

        auto it = memos2.find({f,g,h})->second;
        std::get<1>(it->second)++;
        node* r = std::get<0>(it->second);
        if(cache_inserts++ > threshold)
        {
            cache_inserts = 0;
            for(auto [k, v] : memos)
            {
                const size_t ts = std::get<1>(v);

            }

        }
        return r;
        */
        ////////

        const size_t slot = cache_hash(f,g,h);
        const memo_struct& m = get_memo(slot); 
        if(m.r == nullptr)
            return nullptr;
        if(m.f == f && m.g == g && m.h == h) 
        {
            assert(m.r != nullptr);
            if(m.r->xref < 0)
            {
                assert(false);
                m.r->recursively_revive();
                return m.r;
            }
            //m.r->xref++;
            return m.r;
        }
        return nullptr;
    }

    void memo_cache::cache_insert(node* f, node* g, node* h, node* r)
    {
        //memos2.insert(std::make_pair(std::array<node*,3>({f,g,h}),{r,insert_time_stamp++}));
        //return;
        ////////

        const size_t slot = cache_hash(f,g,h);
        if(++cache_inserts >= threshold)
            double_cache();
        memo_struct& m = get_memo(cache_hash(f,g,h));
        m.f = f;
        m.g = g;
        m.h = h;
        m.r = r;
    }

    void memo_cache::init_cache()
    {
        assert(memos.size() == 0);
        memos.resize(1);
        memos_mask = 0;
    }

    void memo_cache::double_cache()
    {
        //std::cout << "double cache size\n";
        if(memos.size() >= 2<<21)
            return;
        cache_inserts = 0;
        threshold = 1 + memos.size();
        assert(memos.size() > 0);
        memos.resize(2 * memos.size());
        memos_mask = memos.size()-1;
        //std::cout << "new cache size = " << memos.size() << "\n";

        // rehash items on bottom half
        size_t occupied_slots = 0;
        for(size_t k = 0; k < memos.size()/2; ++k)
        {
            memo& m = memos[k];
            if(m.r != nullptr)
            {
                ++occupied_slots;
                memo& mm = get_memo(cache_hash(m.f, m.g, m.h));
                if(mm != m)
                {
                    mm = m;
                    m.r = nullptr;
                } 
            } 
        } 
        cache_inserts = occupied_slots;
    }

    bool memo_struct::can_be_purged() const
    {
        if(r == nullptr)
            return true; // TODO: or false?
        if(r->xref < 0)
            return true;
        if(f->xref < 0)
            return true;
        if(h != nullptr && g->xref < 0)
            return true;
        return false;
    }

    size_t memo_cache::choose_cache_size(const size_t items) const
    {
        // size shall be power of 2 such that not more than a quarter of slots are taken
        size_t count = 0;
        size_t c = items;
        while(c != 0)
        {
            c >>= 1;  
            count += 1;  
        }  
        size_t n = 1 << (count+2);  
        assert(4*items >= n);
        assert(8*items <= n);
        return n;
    }

    size_t memo_cache::nr_occupied_slots() const
    {
        return std::count_if(memos.begin(), memos.end(), [](const auto& m) { return !m.can_be_purged(); });
    }

    void memo_cache::purge()
    {
        size_t items = 0;
        for(size_t k = 0; k < memos.size(); ++k)
            if(memos[k].can_be_purged())
                memos[k].r = nullptr;
            else
                ++items;

        return;
        // TODO: resize memo cache? A resize will often not physically deallocate freed memory.
        cache_inserts = items;
        const size_t new_cache_size = choose_cache_size(items);

        if(new_cache_size < memos.size())
        {
            size_t memos_mask = new_cache_size - 1;
            for(size_t k = 0; k < memos.size(); ++k)
            {
                if(memos[k].r != nullptr && memos[k].h != nullptr)
                {
                    const size_t new_slot = cache_hash(memos[k].f, memos[k].g, memos[k].h) & memos_mask;
                    if(new_slot != k)
                        memos[new_slot] = memos[k];
                }
            }
            memos.resize(new_cache_size);
        }
    }
}
