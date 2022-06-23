add_subdirectory(bdd)

add_executable(test_avl_tree test_avl_tree.cpp)
target_link_libraries(test_avl_tree lineq_bdd LPMP-BDD)
add_test(test_avl_tree test_avl_tree)

add_executable(test_exp_sum test_exp_sum.cpp)
target_link_libraries(test_exp_sum lineq_bdd LPMP-BDD)
add_test(test_exp_sum test_exp_sum)

add_executable(test_ILP_parser test_ILP_parser.cpp)
target_link_libraries(test_ILP_parser ILP_parser LPMP-BDD)
add_test(test_ILP_parser test_ILP_parser)

add_executable(test_ILP_input_to_bdd test_ILP_input_to_bdd.cpp)
target_link_libraries(test_ILP_input_to_bdd ILP_parser LPMP-BDD)
add_test(test_ILP_input_to_bdd test_ILP_input_to_bdd)

add_executable(test_ILP_input test_ILP_input.cpp)
target_link_libraries(test_ILP_input LPMP-BDD)
add_test(test_ILP_input test_ILP_input)

add_executable(test_ILP_input_variable_fixation test_ILP_input_variable_fixation.cpp)
target_link_libraries(test_ILP_input_variable_fixation LPMP-BDD)
add_test(test_ILP_input_variable_fixation test_ILP_input_variable_fixation)

#add_executable(test_single_bdd_inference test_single_bdd_inference.cpp)
#target_link_libraries(test_single_bdd_inference ILP_parser LPMP-BDD)
#add_test(test_single_bdd_inference test_single_bdd_inference)

#add_executable(test_two_bdd_inference test_two_bdd_inference.cpp)
#target_link_libraries(test_two_bdd_inference ILP_parser LPMP bdd)
#add_test(test_two_bdd_inference test_two_bdd_inference)

add_executable(test_bdd_min_marginals test_bdd_min_marginals.cpp)
target_link_libraries(test_bdd_min_marginals LPMP-BDD)
add_test(test_bdd_min_marginals test_bdd_min_marginals)

add_executable(test_bdd_simplex_sum_marginals test_bdd_simplex_sum_marginals.cpp)
target_link_libraries(test_bdd_simplex_sum_marginals LPMP-BDD)
add_test(test_bdd_simplex_sum_marginals test_bdd_simplex_sum_marginals)

add_executable(test_bdd_cardinality_sum_marginals test_bdd_cardinality_sum_marginals.cpp)
target_link_libraries(test_bdd_cardinality_sum_marginals LPMP-BDD)
add_test(test_bdd_cardinality_sum_marginals test_bdd_cardinality_sum_marginals)

add_executable(test_bdd_smooth_mma test_bdd_smooth_mma.cpp)
target_link_libraries(test_bdd_smooth_mma LPMP-BDD)
add_test(test_bdd_smooth_mma test_bdd_smooth_mma)

add_executable(test_bdd_bipartite_matching_problem test_bdd_bipartite_matching_problem.cpp)
target_link_libraries(test_bdd_bipartite_matching_problem LPMP-BDD)
add_test(test_bdd_bipartite_matching_problem test_bdd_bipartite_matching_problem)

add_executable(test_bdd_small_binary_mrfs test_bdd_small_binary_mrfs.cpp)
target_link_libraries(test_bdd_small_binary_mrfs LPMP-BDD)
add_test(test_bdd_small_binary_mrfs test_bdd_small_binary_mrfs)

#add_executable(test_bdd_grid_graph test_bdd_grid_graph.cpp)
#target_link_libraries(test_bdd_grid_graph LPMP bdd ILP_parser)
#add_test(test_bdd_grid_graph test_bdd_grid_graph)

add_executable(test_ILP_input_export test_ILP_input_export.cpp)
target_link_libraries(test_ILP_input_export LPMP-BDD ILP_input ILP_parser OPB_parser)
add_test(test_ILP_input_export test_ILP_input_export)

#add_executable(test_random_inequality_to_bdd test_random_inequality_to_bdd.cpp)
#target_link_libraries(test_random_inequality_to_bdd LPMP-BDD)
#add_test(test_random_inequality_to_bdd test_random_inequality_to_bdd)

#add_executable(test_random_chains_bdd test_random_chains_bdd.cpp)
#target_link_libraries(test_random_chains_bdd ILP_parser LPMP bdd)
#add_test(test_random_chains_bdd test_random_chains_bdd)

#add_executable(test_ILP_input_reordering test_ILP_input_reordering.cpp)
#target_link_libraries(test_ILP_input_reordering ILP_parser LPMP)
#add_test(test_ILP_input_reordering test_ILP_input_reordering)

#add_executable(test_bdd_preprocessor test_bdd_preprocessor.cpp)
#target_link_libraries(test_bdd_preprocessor ILP_parser LPMP bdd)
#add_test(test_bdd_preprocessor test_bdd_preprocessor)

add_executable(test_loose_gm_tightening test_loose_gm_tightening.cpp)
target_link_libraries(test_loose_gm_tightening LPMP-BDD)
add_test(test_loose_gm_tightening test_loose_gm_tightening)

add_executable(test_loose_covering_problem test_loose_covering_problem.cpp)
target_link_libraries(test_loose_covering_problem LPMP-BDD)
add_test(test_loose_covering_problem test_loose_covering_problem)

add_executable(test_bdd_parallel_mma_base test_bdd_parallel_mma_base.cpp)
target_link_libraries(test_bdd_parallel_mma_base LPMP-BDD)
add_test(test_bdd_parallel_mma_base test_bdd_parallel_mma_base)

add_executable(test_bdd_parallel_mma test_bdd_parallel_mma.cpp)
target_link_libraries(test_bdd_parallel_mma LPMP-BDD)
add_test(test_bdd_parallel_mma test_bdd_parallel_mma)

add_executable(test_bdd_smooth_parallel_mma test_bdd_smooth_parallel_mma.cpp)
target_link_libraries(test_bdd_smooth_parallel_mma LPMP-BDD)
add_test(test_bdd_smooth_parallel_mma test_bdd_smooth_parallel_mma)

if(WITH_CUDA)
    add_executable(test_bdd_cuda_base test_bdd_cuda_base.cpp)
    target_link_libraries(test_bdd_cuda_base LPMP-BDD bdd_cuda_base)
    add_test(test_bdd_cuda_base test_bdd_cuda_base)

    add_executable(test_bdd_cuda_base_serialization test_bdd_cuda_base_serialization.cpp)
    target_link_libraries(test_bdd_cuda_base_serialization LPMP-BDD bdd_cuda_base)
    add_test(test_bdd_cuda_base_serialization test_bdd_cuda_base_serialization)

    add_executable(test_bdd_cuda_parallel_mma test_bdd_cuda_parallel_mma.cu)
    target_link_libraries(test_bdd_cuda_parallel_mma LPMP-BDD bdd_cuda_parallel_mma)
    add_test(test_bdd_cuda_parallel_mma test_bdd_cuda_parallel_mma)

    add_executable(test_bdd_cuda_min_marginals test_bdd_cuda_min_marginals.cpp)
    target_link_libraries(test_bdd_cuda_min_marginals LPMP-BDD bdd_cuda_base)
    add_test(test_bdd_cuda_min_marginals test_bdd_cuda_min_marginals)

    add_executable(test_bdd_cuda_base_sol test_bdd_cuda_base_sol.cpp)
    target_link_libraries(test_bdd_cuda_base_sol LPMP-BDD bdd_cuda_base)
    add_test(test_bdd_cuda_base_sol test_bdd_cuda_base_sol)

    add_executable(test_bdd_cuda_learned_mma test_bdd_cuda_learned_mma.cu)
    target_link_libraries(test_bdd_cuda_learned_mma LPMP-BDD bdd_cuda_learned_mma)
    add_test(test_bdd_cuda_learned_mma test_bdd_cuda_learned_mma)

    add_executable(test_bdd_cuda_grad_learned_mma test_bdd_cuda_grad_learned_mma.cu)
    target_link_libraries(test_bdd_cuda_grad_learned_mma LPMP-BDD bdd_cuda_learned_mma)
    add_test(test_bdd_cuda_grad_learned_mma test_bdd_cuda_grad_learned_mma)

    add_executable(test_bdd_cuda_grad_learned_pert test_bdd_cuda_grad_learned_pert.cu)
    target_link_libraries(test_bdd_cuda_grad_learned_pert LPMP-BDD bdd_cuda_learned_mma)
    add_test(test_bdd_cuda_grad_learned_pert test_bdd_cuda_grad_learned_pert)

    add_executable(test_bdd_cuda_learned_omega test_bdd_cuda_learned_omega.cu)
    target_link_libraries(test_bdd_cuda_learned_omega LPMP-BDD bdd_cuda_learned_mma)
    add_test(test_bdd_cuda_learned_omega test_bdd_cuda_learned_omega)

    add_executable(test_cuda_parallel_mma test_cuda_parallel_mma.cu)
    target_link_libraries(test_cuda_parallel_mma LPMP-BDD)
    add_test(test_cuda_parallel_mma test_cuda_parallel_mma)

    add_executable(test_hybrid_parallel_mma_base test_hybrid_parallel_mma_base.cu)
    target_link_libraries(test_hybrid_parallel_mma_base LPMP-BDD bdd_multi_parallel_mma_base)
    add_test(test_hybrid_parallel_mma_base test_hybrid_parallel_mma_base)
endif()

add_executable(test_bdd_solver_fix_variable test_bdd_solver_fix_variable.cpp)
target_link_libraries(test_bdd_solver_fix_variable LPMP-BDD)
add_test(test_bdd_solver_fix_variable test_bdd_solver_fix_variable)

add_executable(test_variable_reordering_invariance test_variable_reordering_invariance.cpp)
target_link_libraries(test_variable_reordering_invariance LPMP-BDD)
add_test(test_variable_reordering_invariance test_variable_reordering_invariance)

add_executable(test_bdd_infeasible_problem test_bdd_infeasible_problem.cpp)
target_link_libraries(test_bdd_infeasible_problem LPMP-BDD)
add_test(test_bdd_infeasible_problem test_bdd_infeasible_problem)

add_executable(test_bdd_preprocessor test_bdd_preprocessor.cpp)
target_link_libraries(test_bdd_preprocessor LPMP-BDD)
add_test(test_bdd_preprocessor test_bdd_preprocessor)

add_executable(test_bdd_solvers_update_costs test_bdd_solvers_update_costs.cpp)
target_link_libraries(test_bdd_solvers_update_costs LPMP-BDD)
add_test(test_bdd_solvers_update_costs test_bdd_solvers_update_costs)

add_executable(test_bdd_solver_feasibility test_bdd_solver_feasibility.cpp)
target_link_libraries(test_bdd_solver_feasibility LPMP-BDD)
add_test(test_bdd_solver_feasibility test_bdd_solver_feasibility)

add_subdirectory(specialized_solvers)

# regression test
if(WITH_REGRESSION_TEST)
    # download instances
    set(BDD_REGRESSION_TEST_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/test/bdd-solver-regression-test-instances.zip")
    if(NOT EXISTS ${BDD_REGRESSION_TEST_ARCHIVE})
        message("Download regression test instances for bdd solver to ${BDD_REGRESSION_TEST_ARCHIVE}")
        file(DOWNLOAD https://keeper.mpdl.mpg.de/f/0f01761d59a745e0a893/?dl=1 ${BDD_REGRESSION_TEST_ARCHIVE} SHOW_PROGRESS)
    else()
        message("bdd regression test instances already downloaded to ${BDD_REGRESSION_TEST_ARCHIVE}")
    endif()

    # unpack instance archive
    set(BDD_REGRESSION_TEST_INSTANCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/bdd-regression-test-instances/)
    if(NOT EXISTS ${BDD_REGRESSION_TEST_INSTANCE_DIR})
        message("Unpack regression test instances to ${BDD_REGRESSION_TEST_INSTANCE_DIR}")
        file(ARCHIVE_EXTRACT INPUT ${BDD_REGRESSION_TEST_ARCHIVE} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/test/ VERBOSE)
    else()
        message("bdd regression test instances already unpacked to ${BDD_REGRESSION_TEST_INSTANCE_DIR}")
    endif()

    add_executable(bdd_solver_regression_test bdd_solver_regression_test.cpp)
    target_compile_definitions(bdd_solver_regression_test PUBLIC BDD_SOLVER_REGRESSION_TEXT_DIR="${BDD_REGRESSION_TEST_INSTANCE_DIR}")
    target_link_libraries(bdd_solver_regression_test LPMP-BDD)
    add_test(bdd_solver_regression_test bdd_solver_regression_test)

    add_executable(bdd_preprocessor_regression_test bdd_preprocessor_regression_test.cpp)
    target_compile_definitions(bdd_preprocessor_regression_test PUBLIC BDD_SOLVER_REGRESSION_TEXT_DIR="${BDD_REGRESSION_TEST_INSTANCE_DIR}")
    target_link_libraries(bdd_preprocessor_regression_test LPMP-BDD)
    add_test(bdd_preprocessor_regression_test bdd_preprocessor_regression_test)
endif()

