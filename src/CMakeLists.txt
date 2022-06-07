add_subdirectory(bdd_manager)
add_subdirectory(bdd_collection)

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

add_library(mm_primal_decoder mm_primal_decoder.cpp)
target_link_libraries(mm_primal_decoder LPMP-BDD)

add_library(bdd_mma bdd_mma.cpp)
target_link_libraries(bdd_mma LPMP-BDD) 

add_library(bdd_mma_smooth bdd_mma_smooth.cpp)
target_link_libraries(bdd_mma_smooth LPMP-BDD) 

add_library(bdd_parallel_mma bdd_parallel_mma.cpp)
target_link_libraries(bdd_parallel_mma LPMP-BDD) 

add_library(bdd_parallel_mma_smooth bdd_parallel_mma_smooth.cpp)
target_link_libraries(bdd_parallel_mma_smooth LPMP-BDD) 

if(WITH_CUDA)
    add_library(incremental_mm_agreement_rounding_cuda incremental_mm_agreement_rounding_cuda.cu)
    target_link_libraries(incremental_mm_agreement_rounding_cuda LPMP-BDD)

    add_library(bdd_cuda_base bdd_cuda_base.cu)
    target_link_libraries(bdd_cuda_base LPMP-BDD)

    add_library(bdd_cuda_parallel_mma bdd_cuda_parallel_mma.cu)
    target_link_libraries(bdd_cuda_parallel_mma bdd_cuda_base incremental_mm_agreement_rounding_cuda LPMP-BDD) 

    add_library(bdd_multi_parallel_mma_base bdd_multi_parallel_mma_base.cu)
    target_link_libraries(bdd_multi_parallel_mma_base bdd_cuda_base bdd_cuda_parallel_mma LPMP-BDD)

    add_library(bdd_cuda_learned_mma bdd_cuda_learned_mma.cu)
    target_link_libraries(bdd_cuda_learned_mma bdd_cuda_parallel_mma LPMP-BDD) 
endif()

add_library(bdd_cuda bdd_cuda.cpp)
target_link_libraries(bdd_cuda LPMP-BDD)

if(WITH_CUDA)
    target_link_libraries(bdd_cuda bdd_cuda_parallel_mma)
endif()

add_library(bdd_multi_parallel_mma bdd_multi_parallel_mma.cpp)
target_link_libraries(bdd_multi_parallel_mma LPMP-BDD)

if(WITH_CUDA)
    target_link_libraries(bdd_multi_parallel_mma bdd_multi_parallel_mma_base)
endif()

add_library(bdd_solver bdd_solver.cpp)
target_link_libraries(bdd_solver bdd_mma bdd_mma_smooth bdd_parallel_mma bdd_parallel_mma_smooth bdd_cuda bdd_multi_parallel_mma bdd_preprocessor ILP_parser OPB_parser ILP_input mm_primal_decoder LPMP-BDD pthread)
if(WITH_CUDA)
    target_link_libraries(bdd_solver bdd_cuda_base bdd_cuda_parallel_mma bdd_multi_parallel_mma_base incremental_mm_agreement_rounding_cuda)
    target_compile_options(bdd_solver PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>")
    
    pybind11_add_module(bdd_cuda_learned_mma_py bdd_cuda_learned_mma_py.cu)
    target_link_libraries(bdd_cuda_learned_mma_py PRIVATE LPMP-BDD ILP_input bdd_preprocessor bdd_cuda_learned_mma)
endif()

target_link_libraries(LPMP-BDD INTERFACE bdd_solver)

add_executable(bdd_solver_cl bdd_solver_cl.cpp)
target_link_libraries(bdd_solver_cl LPMP-BDD)

add_executable(convert_lp_to_opb convert_lp_to_opb.cpp)
target_link_libraries(convert_lp_to_opb LPMP-BDD)

add_executable(convert_opb_to_lp convert_opb_to_lp.cpp)
target_link_libraries(convert_opb_to_lp OPB_parser LPMP-BDD) 

pybind11_add_module(bdd_mp_py bdd_mp_py.cpp)
target_link_libraries(bdd_mp_py PRIVATE LPMP-BDD ILP_input bdd_preprocessor)

add_subdirectory(specialized_solvers)

pybind11_add_module(ILP_instance_py ILP_instance_py.cpp)
target_link_libraries(ILP_instance_py PRIVATE LPMP-BDD mrf_uai_parser mrf_input)
