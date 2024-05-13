add_library(bdd_parallel_mma_base bdd_parallel_mma_base.cpp) # must be in own library since both cpu bdd solver as well as bdd_mutli_parallel_mma_base depend on it
target_link_libraries(bdd_parallel_mma_base LPMP-BDD bdd_conversion bdd_mgr bdd_collection)

if(WITH_CUDA)
    set(CUDA_BDD_SOLVER_SOURCES bdd_cuda_base.cu bdd_cuda_parallel_mma.cu bdd_multi_parallel_mma_base.cu lbfgs.cu incremental_mm_agreement_rounding_cuda.cu)
    add_library(cuda_bdd_solver ${CUDA_BDD_SOLVER_SOURCES})
    target_link_libraries(cuda_bdd_solver LPMP-BDD bdd_parallel_mma_base)

    add_library(bdd_cuda_learned_mma bdd_cuda_learned_mma.cu)
    target_link_libraries(bdd_cuda_learned_mma cuda_bdd_solver LPMP-BDD) 
endif()

set(BDD_SOLVER_SOURCES bdd_solver.cpp bdd_mma_base.cpp lbfgs.cpp mm_primal_decoder.cpp)
add_library(bdd_solver ${BDD_SOLVER_SOURCES})
target_link_libraries(bdd_solver LPMP-BDD ILP bdd_conversion bdd_mgr bdd_collection bdd_parallel_mma_base)
if(WITH_CUDA)
    target_link_libraries(bdd_solver cuda_bdd_solver)
endif()

add_executable(bdd_solver_cl bdd_solver_cl.cpp)
target_link_libraries(bdd_solver_cl LPMP-BDD bdd_solver)
