add_library(bdd_node bdd_node.cpp)
target_link_libraries(bdd_node LPMP-BDD)

add_library(bdd_node_cache bdd_node_cache.cpp) 
target_link_libraries(bdd_node_cache bdd_node LPMP-BDD)

add_library(bdd_var bdd_var.cpp)
target_link_libraries(bdd_var bdd_node_cache LPMP-BDD)

add_library(bdd_memo_cache bdd_memo_cache.cpp)
target_link_libraries(bdd_memo_cache bdd_node bdd_node_cache LPMP-BDD)

add_library(bdd_mgr bdd_mgr.cpp)
target_link_libraries(bdd_mgr bdd_node_cache bdd_var bdd_memo_cache LPMP-BDD)

#target_link_libraries(LPMP-BDD INTERFACE bdd_node)
#target_link_libraries(LPMP-BDD INTERFACE bdd_node_cache)
#target_link_libraries(LPMP-BDD INTERFACE bdd_var)
#target_link_libraries(LPMP-BDD INTERFACE bdd_memo_cache)
#target_link_libraries(LPMP-BDD INTERFACE bdd_mgr)
