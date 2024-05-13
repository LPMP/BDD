set(BDD_CONVERSION_SOURCES convert_pb_to_bdd.cpp lineq_bdd.cpp bdd_preprocessor.cpp)
add_library(bdd_conversion ${BDD_CONVERSION_SOURCES})
target_link_libraries(bdd_conversion LPMP-BDD ILP)