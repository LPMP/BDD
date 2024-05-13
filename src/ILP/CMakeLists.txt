set(ILP_SOURCES ILP_input.cpp ILP_parser.cpp OPB_parser.cpp)
add_library(ILP ${ILP_SOURCES})
target_link_libraries(ILP LPMP-BDD)

#pybind11_add_module(ILP_instance_py ILP_instance_py.cpp)
#target_link_libraries(ILP_instance_py PRIVATE LPMP-BDD mrf_uai_parser mrf_input)

add_executable(convert_lp_to_opb convert_lp_to_opb.cpp)
target_link_libraries(convert_lp_to_opb LPMP-BDD ILP bdd_mgr)

add_executable(convert_opb_to_lp convert_opb_to_lp.cpp)
target_link_libraries(convert_opb_to_lp LPMP-BDD ILP bdd_mgr) 

