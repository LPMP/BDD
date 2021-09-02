add_library(bdd_collection bdd_collection.cpp)
target_link_libraries(bdd_collection bdd_node_cache bdd_var bdd_memo_cache bdd_mgr LPMP-BDD)
target_link_libraries(LPMP-BDD INTERFACE bdd_collection)
