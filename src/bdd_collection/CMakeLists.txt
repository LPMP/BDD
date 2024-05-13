set(BDD_COLLECTION_SOURCES bdd_collection.cpp transitive_closure_dag.cpp)
add_library(bdd_collection ${BDD_COLLECTION_SOURCES})
target_link_libraries(bdd_collection bdd_mgr LPMP-BDD)
