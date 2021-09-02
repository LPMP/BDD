#pragma once

#include <memory>
#include <array>
#include <vector>
#include "bdd_node.h"
#include "bdd_node_cache.h"

namespace BDD {

    constexpr static std::size_t log_nr_unique_table_pages = 10;
    constexpr static std::size_t nr_unique_table_pages = static_cast<std::size_t>(1) << log_nr_unique_table_pages; // 1024
    constexpr static std::size_t unique_table_page_mask = (nr_unique_table_pages-1);

    constexpr static std::size_t log_nr_unique_table_slots_per_page = 10;
    constexpr static std::size_t nr_unique_table_slots_per_page = static_cast<std::size_t>(1) << log_nr_unique_table_slots_per_page; // 1024;
    constexpr static std::size_t unique_table_slots_mask = (nr_unique_table_slots_per_page-1);

    constexpr static std::size_t log_max_hash_size = log_nr_unique_table_pages + log_nr_unique_table_slots_per_page;

    constexpr static double min_unique_table_fill = 1.0/8.0;
    constexpr static double max_unique_table_fill = 1.0/2.0;
    //constexpr static std::size_t max_hash_pages = (((static_cast<std::size_t>(1) << log_max_hash_size) + slots_per_page - 1) / slots_per_page);
    constexpr static std::size_t initial_page_mem_size = 512;

template<size_t PAGE_SIZE>
class unique_table_page {
    public:
    // TODO: make union of below two variables
    union {
    std::array<node*, PAGE_SIZE> data;
    unique_table_page* next_available = nullptr; 
    };
};

template<size_t PAGE_SIZE, size_t NR_SIMUL_ALLOC>
class unique_table_page_cache {
    public:
        constexpr static size_t nr_pages_simultaneous_allocation = NR_SIMUL_ALLOC;

        unique_table_page_cache();
        ~unique_table_page_cache();
        unique_table_page<PAGE_SIZE>* reserve_page();
        void free_page(unique_table_page<PAGE_SIZE>* p); 
        std::size_t nr_pages() const { return pages.size() * nr_pages_simultaneous_allocation; }

    private: 
        void increase_cache();
        unique_table_page<PAGE_SIZE>* page_avail; // stack of pages for reuse
        std::vector<unique_table_page<PAGE_SIZE>*> pages;
}; 

class unique_table_page_caches {
    public:
        unique_table_page_cache<64,16384> cache_64;
        unique_table_page_cache<128,8192> cache_128;
        unique_table_page_cache<256,4096> cache_256;
        unique_table_page_cache<512,2048> cache_512;
        unique_table_page_cache<1024,1024> cache_1024;
        unique_table_page_cache<2048,512> cache_2048;
        unique_table_page_cache<4096,256> cache_4096;
        unique_table_page_cache<8192,128> cache_8192;
        unique_table_page_cache<16384,64> cache_16384;
        unique_table_page_cache<32768,32> cache_32768;
        unique_table_page_cache<65536,16> cache_65536;
        unique_table_page_cache<131072,8> cache_131072;
        unique_table_page_cache<262144,4> cache_262144;
        unique_table_page_cache<524288,2> cache_524288;
        unique_table_page_cache<1048576,1> cache_1048576;
};

class bdd_mgr; // forward declaration

class var_struct {
    public:
        var_struct(const std::size_t index, bdd_mgr& _bdd_mgr);
        ~var_struct();
        void release_nodes();
        var_struct(var_struct&&);
        var_struct(const var_struct&) = delete;

        std::size_t hash_code(node* p) const;
        std::size_t hash_code(node* l, node* r) const;
        size_t nr_free_slots_debug() const;
        node* unique_table_lookup(node* l, node* h);
        node* unique_find(const std::size_t index, node* l,node* h); 
        node* unique_find(node* l,node* h); 
        //node* projection() const;
        void remove_dead_nodes();

    private:
        const size_t var;
        double occupied_rate() const;
        double occupied_rate(const size_t new_nr_pages) const;
        void initialize_unique_table();

        node** new_page(const size_t new_mask);
        void free_page(node** p, const size_t mask);
        void free_page() { return free_page(base, mask); };

        union {
        unique_table_page<64>* base_64; // 2^6
        unique_table_page<128>* base_128; // 2^7
        unique_table_page<256>* base_256; // 2^8
        unique_table_page<512>* base_512; // 2^9
        unique_table_page<1024>* base_1024; // 2^10
        unique_table_page<2048>* base_2048; // 2^11
        unique_table_page<4096>* base_4096; // 2^12
        unique_table_page<8192>* base_8192; // 2^13
        unique_table_page<16384>* base_16384; // 2^14
        unique_table_page<32768>* base_32768; // 2^15
        unique_table_page<65536>* base_65536; // 2^16
        unique_table_page<131072>* base_131072; // 2^17
        unique_table_page<262144>* base_262144; // 2^18
        unique_table_page<524288>* base_524288; // 2^19
        unique_table_page<1048576>* base_1048576; // 2^20
        node** base = nullptr;
        };

        std::size_t hash_table_size() const { return mask+1; }
        size_t base_index(const size_t k) const;
        node* fetch_node(const std::size_t k) const;
        std::size_t next_free_slot(const std::size_t hash) const;
        void store_node(const size_t k, node* p);
        void double_cache();

        size_t mask = 0; // number of pages for the unique table minus 1 
        size_t free = 0; // number of unused slots in the unique table for v
        //std::size_t dead_nodes = 0;
        std::size_t timer = 0;
        constexpr static std::size_t timerinterval = 1024;
        constexpr static double dead_fraction = 1.0;
        //std::array<unique_table_page*, nr_unique_table_pages> base = {}; // base addresses for its pages
        std::size_t name; // user's name (subscript) for this variable
        unsigned int timestamp; // time stamp for composition
        int aux; // flag used by sifting algorithm
        struct var_struct *up, *down; // the neighboring active variables

        bdd_mgr& bdd_mgr_;
};

using var = var_struct;

    template<size_t PAGE_SIZE, size_t NR_SIMUL_ALLOC>
unique_table_page_cache<PAGE_SIZE, NR_SIMUL_ALLOC>::unique_table_page_cache()
{
    page_avail = nullptr; 
}

    template<size_t PAGE_SIZE, size_t NR_SIMUL_ALLOC>
unique_table_page_cache<PAGE_SIZE, NR_SIMUL_ALLOC>::~unique_table_page_cache()
{
    for(unique_table_page<PAGE_SIZE>* p : pages)
        delete[] p;
}

    template<size_t PAGE_SIZE, size_t NR_SIMUL_ALLOC>
void unique_table_page_cache<PAGE_SIZE, NR_SIMUL_ALLOC>::increase_cache()
{
    assert(page_avail == nullptr);
    pages.push_back(new unique_table_page<PAGE_SIZE>[nr_pages_simultaneous_allocation]);
    for(std::size_t i=0; i+1 < nr_pages_simultaneous_allocation; ++i)
        pages.back()[i].next_available = &(pages.back()[i+1]);
    page_avail = &(pages.back()[0]);
}

    template<size_t PAGE_SIZE, size_t NR_SIMUL_ALLOC>
unique_table_page<PAGE_SIZE>* unique_table_page_cache<PAGE_SIZE, NR_SIMUL_ALLOC>::reserve_page()
{
    unique_table_page<PAGE_SIZE>* r = page_avail;
    if(r != nullptr)
    {
        page_avail = page_avail->next_available;
        std::fill(r->data.begin(), r->data.end(), nullptr);
        return r;
    }
    else
    {
        increase_cache();
        return reserve_page();
    }
}

    template<size_t PAGE_SIZE, size_t NR_SIMUL_ALLOC>
void unique_table_page_cache<PAGE_SIZE, NR_SIMUL_ALLOC>::free_page(unique_table_page<PAGE_SIZE>* p)
{
    p->next_available = page_avail;
    page_avail= p;
}
}
