add_subdirectory(eval)

add_library(ILP_input ILP_input.cpp)
target_link_libraries(ILP_input LPMP-BDD)

add_library(ILP_parser ILP_parser.cpp)
target_link_libraries(ILP_parser ILP_input LPMP-BDD)

add_library(convert_pb_to_bdd convert_pb_to_bdd.cpp)
target_link_libraries(convert_pb_to_bdd ILP_input LPMP-BDD LBDD)

add_library(bdd_preprocessor bdd_preprocessor.cpp)
target_link_libraries(bdd_preprocessor ILP_input convert_pb_to_bdd LPMP-BDD LBDD)

add_library(bdd_storage bdd_storage.cpp)
target_link_libraries(bdd_storage bdd_preprocessor LPMP-BDD LBDD)

add_library(bdd_mma bdd_mma.cpp)
target_link_libraries(bdd_mma LPMP-BDD LBDD)

add_library(decomposition_bdd_mma decomposition_bdd_mma.cpp)
target_link_libraries(decomposition_bdd_mma LPMP-BDD LBDD)

add_executable(bdd_solver bdd_solver.cpp)
target_link_libraries(bdd_solver bdd_mma decomposition_bdd_mma bdd_preprocessor ILP_parser bdd_storage ILP_input LPMP-BDD LBDD pthread)
