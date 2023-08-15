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
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_random.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_sort_search.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_algorithm.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_mapping_view.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_helpers_for_testing.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_reduce.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_tensor.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_traverser.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_reduce_binary.cpp

    #${CMAKE_CURRENT_SOURCE_DIR}/test_tmp.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_iterator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_copy.cpp

    #${CMAKE_CURRENT_SOURCE_DIR}/test_storage.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_libdivide_helper.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_init_list_helper.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_slice.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_descriptor.cpp
#
    #${CMAKE_CURRENT_SOURCE_DIR}/test_data_accessor_indexer.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_data_accessor_walker.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_data_accessor_traverser.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_iterator.cpp
#
    #${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_core.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_implementation.cpp
#
    #${CMAKE_CURRENT_SOURCE_DIR}/test_view_factory.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_expression_template_engine.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_operators.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp
#
    #${CMAKE_CURRENT_SOURCE_DIR}/test_indexing.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_reduce.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_manipulation.cpp
#
    #${CMAKE_CURRENT_SOURCE_DIR}/test_math_elementwise.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_math_along_axes.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_statistic.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_sort_search.cpp
#
    #${CMAKE_CURRENT_SOURCE_DIR}/test_builder.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/test_random.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)