set(BDD_MGR_SOURCES bdd_node.cpp bdd_node_cache.cpp bdd_var.cpp bdd_memo_cache.cpp bdd_mgr.cpp)
add_library(bdd_mgr ${BDD_MGR_SOURCES})
target_link_libraries(bdd_mgr LPMP-BDD)