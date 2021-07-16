add_subdirectory(bdd_manager)
add_subdirectory(bdd_collection)
add_subdirectory(eval)

add_library(ILP_input ILP_input.cpp)
target_link_libraries(ILP_input LPMP-BDD)

add_library(ILP_parser ILP_parser.cpp)
target_link_libraries(ILP_parser ILP_input LPMP-BDD)

add_library(OPB_parser OPB_parser.cpp)
target_link_libraries(OPB_parser ILP_input LPMP-BDD)

add_library(lineq_bdd lineq_bdd.cpp)
target_link_libraries(lineq_bdd ILP_input LPMP-BDD)

add_library(convert_pb_to_bdd convert_pb_to_bdd.cpp)
target_link_libraries(convert_pb_to_bdd ILP_input lineq_bdd LPMP-BDD)

add_library(bdd_preprocessor bdd_preprocessor.cpp)
target_link_libraries(bdd_preprocessor ILP_input convert_pb_to_bdd lineq_bdd LPMP-BDD)

add_library(bdd_storage bdd_storage.cpp)
target_link_libraries(bdd_storage bdd_preprocessor LPMP-BDD)

add_library(bdd_mma bdd_mma.cpp)
target_link_libraries(bdd_mma LPMP-BDD)

add_library(bdd_mma_srmp bdd_mma_srmp.cpp)
target_link_libraries(bdd_mma_srmp LPMP-BDD)

add_library(bdd_mma_agg bdd_mma_agg.cpp)
target_link_libraries(bdd_mma_agg LPMP-BDD)

add_library(bdd_mma_anisotropic bdd_mma_anisotropic.cpp)
target_link_libraries(bdd_mma_anisotropic LPMP-BDD)

add_library(decomposition_bdd_mma_base decomposition_bdd_mma_base.cpp)
target_link_libraries(decomposition_bdd_mma_base LPMP-BDD)

add_library(decomposition_bdd_mma decomposition_bdd_mma.cpp)
target_link_libraries(decomposition_bdd_mma decomposition_bdd_mma_base LPMP-BDD)

add_library(bdd_mma_vec bdd_mma_vec.cpp)
target_link_libraries(bdd_mma_vec LPMP-BDD) 

add_library(bdd_fix bdd_fix.cpp)
target_link_libraries(bdd_fix LPMP-BDD) 

add_library(bdd_solver bdd_solver.cpp)
target_link_libraries(bdd_solver bdd_mma decomposition_bdd_mma bdd_mma_srmp bdd_mma_agg bdd_mma_anisotropic bdd_mma_vec bdd_fix bdd_preprocessor ILP_parser OPB_parser bdd_storage ILP_input LPMP-BDD pthread)

target_link_libraries(LPMP-BDD INTERFACE bdd_solver)

add_executable(bdd_solver_cl bdd_solver_cl.cpp)
target_link_libraries(bdd_solver_cl LPMP-BDD)

add_executable(convert_lp_to_opb convert_lp_to_opb.cpp)
target_link_libraries(convert_lp_to_opb LPMP-BDD)

add_executable(convert_opb_to_lp convert_opb_to_lp.cpp)
target_link_libraries(convert_opb_to_lp OPB_parser LPMP-BDD) 

pybind11_add_module(ILP_instance_py ILP_instance_py.cpp)
target_link_libraries(ILP_instance_py PRIVATE LPMP-BDD ILP_input ILP_parser OPB_parser)

pybind11_add_module(bdd_mp_py bdd_mp_py.cpp)
target_link_libraries(bdd_mp_py PRIVATE LPMP-BDD ILP_input bdd_storage bdd_preprocessor)
