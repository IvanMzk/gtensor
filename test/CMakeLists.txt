# GTensor - computation library
# Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
#
# Distributed under the Boost Software License, Version 1.0.
# The full license is in the file LICENSE.txt, distributed with this software.

add_executable(Test)
target_link_libraries(Test PRIVATE gtensor)
target_include_directories(Test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../qa_common)
target_compile_features(Test PRIVATE cxx_std_17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(Test PRIVATE /W4 /Zc:__cplusplus /bigobj "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(Test PRIVATE -Wall "$<$<CONFIG:RELEASE>:-O2>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(Test PRIVATE -stdlib=libc++)
    else()
    endif()
endif()


target_sources(Test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/test_tmp.cpp

    #${CMAKE_CURRENT_LIST_DIR}/test_helpers_for_testing.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_multithreading.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_storage.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_common.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_libdivide_helper.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_init_list_helper.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_slice.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_descriptor.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_data_accessor_indexer.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_data_accessor_walker.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_data_accessor_traverser.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_iterator.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_core.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_implementation.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_expression_template_engine.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_view_factory_detail.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_view_factory_reshape_transpose_view.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_view_factory_slice_view.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_view_factory_mapping_view.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_operators.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_std_complex.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_construction.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_assign.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_equality.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_data_accessor.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_view.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_reduce_slide_transform.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_tensor_application_specific.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_indexing.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_manipulation_detail.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_manipulation_stack_concatenate.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_manipulation_vstack_hstack.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_manipulation_block.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_manipulation_split.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_reduce_detail.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_reduce_reduce_range.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_reduce_reduce_binary.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_reduce_slide_transform.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_math_elementwise.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_math_diff_gradient.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_math_sum_prod.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_math_cumsum_cumprod.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_math_all_any.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_math_min_max.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_math_matmul.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_statistic_ptp_mean.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_statistic_var_std.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_statistic_quantile_median.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_statistic_average_moving_average.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_statistic_moving_mean_histogram.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_statistic_big.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_sort_search_sort_argsort.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_sort_search_partition_argpartition.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_sort_search_argmin_argmax.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_sort_search_nonzero_argwhere_searchsorted.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_sort_search_unique.cpp
#
    #${CMAKE_CURRENT_LIST_DIR}/test_builder.cpp
    #${CMAKE_CURRENT_LIST_DIR}/test_random.cpp

    ${CMAKE_CURRENT_LIST_DIR}/test.cpp
)