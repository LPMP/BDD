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

add_library(bdd_mma_srmp bdd_mma_srmp.cpp)
target_link_libraries(bdd_mma_srmp LPMP-BDD LBDD)

add_library(bdd_mma_agg bdd_mma_agg.cpp)
target_link_libraries(bdd_mma_agg LPMP-BDD LBDD)

add_library(bdd_mma_anisotropic bdd_mma_anisotropic.cpp)
target_link_libraries(bdd_mma_anisotropic LPMP-BDD LBDD)

add_library(decomposition_bdd_mma_base decomposition_bdd_mma_base.cpp)
target_link_libraries(decomposition_bdd_mma_base LPMP-BDD LBDD)

add_library(decomposition_bdd_mma decomposition_bdd_mma.cpp)
target_link_libraries(decomposition_bdd_mma decomposition_bdd_mma_base LPMP-BDD LBDD)

add_library(bdd_mma_vec bdd_mma_vec.cpp)
target_link_libraries(bdd_mma_vec LPMP-BDD LBDD) 

add_library(bdd_fix bdd_fix.cpp)
target_link_libraries(bdd_fix LPMP-BDD LBDD) 

add_library(bdd_solver bdd_solver.cpp)
target_link_libraries(bdd_solver bdd_mma decomposition_bdd_mma bdd_mma_srmp bdd_mma_agg bdd_mma_anisotropic bdd_mma_vec bdd_fix bdd_preprocessor ILP_parser bdd_storage ILP_input LPMP-BDD LBDD pthread)

target_link_libraries(LPMP-BDD INTERFACE bdd_solver)

add_executable(bdd_solver_cl bdd_solver_cl.cpp)
target_link_libraries(bdd_solver_cl LPMP-BDD)

