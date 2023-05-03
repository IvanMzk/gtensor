add_executable(Test)
target_link_libraries(Test PRIVATE expression_template)
target_compile_features(Test PRIVATE cxx_std_17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(Test PRIVATE /W4 /arch:AVX2 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(Test PRIVATE -Wall "$<$<CONFIG:RELEASE>:-O3>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(Test PRIVATE -stdlib=libc++)
    else()
    endif()
endif()


target_sources(Test PRIVATE
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_expression_template.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_mapping_view.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_integral_type.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_helpers_for_testing.cpp
#
    #${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_libdivide_helper.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_init_list.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_slice.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_descriptor.cpp
##
    #${CMAKE_CURRENT_SOURCE_DIR}/test_data_accessor_indexer.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_data_accessor_walker.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_data_accessor_traverser.cpp
##
    #${CMAKE_CURRENT_SOURCE_DIR}/test_iterator.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_implementation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_expression_template_engine.cpp

    #${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_view_factory.cpp
#
    #${CMAKE_CURRENT_SOURCE_DIR}/test_operator.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_reduce.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_combine.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)