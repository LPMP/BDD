#include "bdd_manager/bdd_var.h"
#include "bdd_manager/bdd_mgr.h"
#include <cassert>
#include <algorithm>
#include <stdexcept>

namespace BDD {

    size_t var_struct::base_index(const size_t k) const
    {
        assert(mask >= 63);
        return k & mask;
    }

    node* var_struct::fetch_node(const size_t k) const
    {
        assert(k <= mask);
        assert(base != nullptr);
        return base[k];
    }

    size_t var_struct::next_free_slot(const size_t k) const
    {
        for(size_t l = k;; ++l)
            if(fetch_node(base_index(l)) == nullptr)
                return base_index(l);
        assert(false); // in this case there is no free slot anymore
        return std::numeric_limits<size_t>::max();
    }

    void var_struct::store_node(const size_t k, node* p)
    {
        assert(k <= mask);
        base[k] = p;
    }

    node** var_struct::new_page(const size_t new_mask)
    {
        unique_table_page_caches& cache = bdd_mgr_.get_unique_table_page_cache();
        switch(new_mask) {
            case 63: return reinterpret_cast<node**>(cache.cache_128.reserve_page()); 
            case 127: return reinterpret_cast<node**>(cache.cache_256.reserve_page());
            case 255: return reinterpret_cast<node**>(cache.cache_512.reserve_page());
            case 511: return reinterpret_cast<node**>(cache.cache_1024.reserve_page());
            case 1023: return reinterpret_cast<node**>(cache.cache_2048.reserve_page());
            case 2047: return reinterpret_cast<node**>(cache.cache_4096.reserve_page());
            case 4095: return reinterpret_cast<node**>(cache.cache_8192.reserve_page());
            case 8191: return reinterpret_cast<node**>(cache.cache_16384.reserve_page());
            case 16383: return reinterpret_cast<node**>(cache.cache_32768.reserve_page());
            case 32767: return reinterpret_cast<node**>(cache.cache_65536.reserve_page());
            case 65535: return reinterpret_cast<node**>(cache.cache_131072.reserve_page());
            case 131071:return reinterpret_cast<node**>(cache.cache_262144.reserve_page());
            case 262143:return reinterpret_cast<node**>(cache.cache_262144.reserve_page());
            case 524287:return reinterpret_cast<node**>(cache.cache_524288.reserve_page());
            case 1048575: throw std::runtime_error("Cannot increase unique table page cache size.");;
            default: assert(false);
        } 
    }

    void var_struct::double_cache()
    {
        // if maximum size is already reached, do not double
        if(mask == 1048575)
            return;

        assert(free == nr_free_slots_debug());
        node** old_page = base;
        const size_t old_mask = mask;

        mask = mask + mask + 1;
        base = new_page(mask); 

        for(size_t i=0; i<=mask; ++i)
            assert(base[i] == nullptr);

        // rehash existing entries from old page 
        for(std::size_t k = 0; k <= old_mask; ++k)
        {
            node* p = old_page[k];
            if(p != nullptr)
                store_node(next_free_slot(base_index(hash_code(p))), p);
        }

        free += old_mask+1;

        free_page(old_page, old_mask);
    }

    void var_struct::free_page(node** p, const size_t p_mask)
    {
        if(p == nullptr)
            return;

        auto& cache = bdd_mgr_.get_unique_table_page_cache();
        switch(p_mask) {
            case 63: return cache.cache_64.free_page(reinterpret_cast<unique_table_page<64>*>(p));
            case 127: return cache.cache_128.free_page(reinterpret_cast<unique_table_page<128>*>(p));
            case 255: return cache.cache_256.free_page(reinterpret_cast<unique_table_page<256>*>(p));
            case 511: return cache.cache_512.free_page(reinterpret_cast<unique_table_page<512>*>(p));
            case 1023: return cache.cache_1024.free_page(reinterpret_cast<unique_table_page<1024>*>(p));
            case 2047: return cache.cache_2048.free_page(reinterpret_cast<unique_table_page<2048>*>(p));
            case 4095: return cache.cache_4096.free_page(reinterpret_cast<unique_table_page<4096>*>(p));
            case 8191: return cache.cache_8192.free_page(reinterpret_cast<unique_table_page<8192>*>(p));
            case 16383: return cache.cache_16384.free_page(reinterpret_cast<unique_table_page<16384>*>(p));
            case 32767: return cache.cache_32768.free_page(reinterpret_cast<unique_table_page<32768>*>(p));
            case 65535: return cache.cache_65536.free_page(reinterpret_cast<unique_table_page<65536>*>(p));
            case 131071: return cache.cache_131072.free_page(reinterpret_cast<unique_table_page<131072>*>(p));
            case 262143: return cache.cache_262144.free_page(reinterpret_cast<unique_table_page<262144>*>(p));
            case 524287: return cache.cache_524288.free_page(reinterpret_cast<unique_table_page<524288>*>(p));
            case 1048575: return cache.cache_1048576.free_page(reinterpret_cast<unique_table_page<1048576>*>(p));
            default: assert(false);
        }

    }

    var_struct::var_struct(const std::size_t index, bdd_mgr& _bdd_mgr)
        : var(index),
        bdd_mgr_(_bdd_mgr)
    {
        base_64 = _bdd_mgr.get_unique_table_page_cache().cache_64.reserve_page();
        mask = 64-1;
        free = 64;
    }

    var_struct::var_struct(var_struct&& o)
        : var(o.var),
        timer(o.timer),
        name(o.name),
        timestamp(o.timestamp),
        aux(o.aux),
        up(o.up),
        down(o.down),
        bdd_mgr_(o.bdd_mgr_) 
    {
        std::swap(base, o.base);
        std::swap(mask, o.mask);
        std::swap(free, o.free);
    } 

    void var_struct::release_nodes()
    {
        for(std::size_t i=0; i<=mask; ++i)
        {
            node* p = fetch_node(i);
            if(p != nullptr)
                bdd_mgr_.get_node_cache().free_node(p); 
            store_node(i, nullptr);
        }
    }

    var_struct::~var_struct()
    {
        if(mask >= 63)
            for(std::size_t i=0; i<=mask; ++i)
                assert(fetch_node(i) == nullptr);

        free_page();
    }

    std::size_t var_struct::hash_code(node* p) const
    {
        assert(p != nullptr);
        return hash_code(p->lo, p->hi);
    }

    std::size_t var_struct::hash_code(node* l, node* r) const
    {
        return l->hash_key ^ (2*r->hash_key);
    }

    size_t var_struct::nr_free_slots_debug() const
    {
        size_t n = 0;
        for(size_t i=0; i<=mask; ++i)
            if(base[i] == nullptr)
                n++;
        return n;

    }

    node* var_struct::unique_table_lookup(node* l, node* h)
    {
        assert(l != h);
        for(std::size_t hash = hash_code(l,h);; hash++)
        {
            node* p = fetch_node(base_index(hash));
            if(p == nullptr)
                return nullptr;
            if(p->lo == l && p->hi == h)
                return p;
        }
        throw std::runtime_error("unique table corrupted."); 
    }

    double var_struct::occupied_rate() const
    {
        return double(hash_table_size() - free) / double(hash_table_size());
    }

    double var_struct::occupied_rate(const size_t new_page_size) const
    {
        return double(new_page_size - free) / double(new_page_size);
    }

    void var_struct::remove_dead_nodes()
    {
        for(std::size_t k = 0; k < hash_table_size(); ++k)
        {
            node* p = fetch_node(k);
            if(p != nullptr && p->dead())
            {
                bdd_mgr_.get_node_cache().free_node(p);
                store_node(k, nullptr);
                free++;
                // move nodes that follow this one back if hash value indicates so
                size_t first_free_slot = k;
                for(size_t j = k+1;; ++j)
                {
                    const size_t jj = j & mask;
                    node* p = fetch_node(jj);
                    if(p == nullptr)
                    {
                        k = j;
                        break;
                    }
                    if(p->dead())
                    {
                        store_node(jj, nullptr);
                        free++;
                    }
                    else
                    {
                        // shift one place up if hash code says so
                        // TODO: can be made faster.
                        store_node(next_free_slot(hash_code(p)), p);
                    }
                }
            }
        }

        // reduce nr of pages if unique table too sparsely populated
        if(mask > 63 && occupied_rate() <= min_unique_table_fill)
        {
            const size_t new_mask = [&]() {
                size_t n = (mask+1)/2 - 1;
                while(n >= 63 && occupied_rate(n) >= min_unique_table_fill)
                    n /= (mask+1)/2 - 1;
                return n;
            }();

            node** new_base = new_page(new_mask);

            for(std::size_t k = 0; k <= mask; ++k)
            {
                node* p = fetch_node(k);
                if(p != nullptr)
                {
                    const size_t next_free_slot = [&]() {
                        for(size_t l=k;; l++)
                        {
                            l = l & new_mask;
                            if(new_base[k] == nullptr)
                                return l;
                        }
                        assert(false);
                        return size_t(0);
                    }();

                    new_base[next_free_slot] = p;
                } 
            }

            free_page(base, mask);
            base = new_base;
            mask = new_mask;
        }
    }

    node* var_struct::unique_find(const size_t index, node* l, node* h)
    {
        //assert(index < l->find_bdd_mgr()->nr_variables()); 
        if(l==h)
            return l;

        node* p = unique_table_lookup(l, h);

        if(p != nullptr) // node present
        {
            return p;
            if(p->xref < 0)
            {
                //dead_nodes--;
                //p->xref= 0;
                return p;
            }
            else 
            {
                //l->xref--;
                //h->xref--;
                //p->xref++;
                return p;
            }
        }
        else // node not present
        {
            // garbage collection
            //if((++timer % timerinterval) == 0 && (dead_nodes > unique_table_size()/dead_fraction))
            if((++timer % timerinterval) == 0)
            {
                // TODO: implement
                //remove_dead_nodes();
                return unique_find(l, h);
            }
        }

        // allocate free node and add it to unique table
        if(occupied_rate() > max_unique_table_fill) // double number of base pages for unique table
            double_cache();

        // allocate new node and insert it into unique table
        p = bdd_mgr_.get_node_cache().reserve_node();
        assert(p != nullptr);
        p->init_new_node(index,l,h);
        assert(free > 0);
        --free;
        store_node(next_free_slot(hash_code(p)), p);
        return p;
    }

    node* var_struct::unique_find(node* l, node* h)
    {
        return unique_find(var, l, h);
    }
}
