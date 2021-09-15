#include "bdd_manager/bdd_var.h"
#include "../test.h"
#include <vector>

using namespace BDD;
using namespace LPMP;

template<size_t PAGE_SIZE, size_t NR_SIMUL_ALLOC>
void test_unique_table_page_cache()
{
    unique_table_page_cache<PAGE_SIZE, NR_SIMUL_ALLOC> cache;
    std::vector<unique_table_page<PAGE_SIZE>*> v;
    const std::size_t n = 100;
    for(std::size_t i=0; i<n; ++i)
    {
        v.push_back(cache.reserve_page());
        test(i + unique_table_page_cache<PAGE_SIZE, NR_SIMUL_ALLOC>::nr_pages_simultaneous_allocation >= cache.nr_pages() && i <= cache.nr_pages() , "page counting error when requesting pages");
    }

    for(std::size_t i=0; i<n; ++i)
    {
        cache.free_page(v[i]);
    } 
}

int main(int argc, char** argv)
{
    test_unique_table_page_cache<64,16384>();
    test_unique_table_page_cache<128,8192>();
    test_unique_table_page_cache<256,4096>();
    test_unique_table_page_cache<512,2048>();
    test_unique_table_page_cache<1024,1024>();
    test_unique_table_page_cache<2048,512>();
    test_unique_table_page_cache<4096,256>();
    test_unique_table_page_cache<8192,128>();
    test_unique_table_page_cache<16384,64>();
    test_unique_table_page_cache<32768,32>();
    test_unique_table_page_cache<65536,16>();
}

