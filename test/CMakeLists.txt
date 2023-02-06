add_executable(Test)
add_executable(Test_selected)
target_include_directories(Test PRIVATE ${${project_name}_SOURCE_DIR}\\src)
target_compile_options(Test PRIVATE -W1 -std:c++17 /Zc:__cplusplus)
target_include_directories(Test_selected PRIVATE ${${project_name}_SOURCE_DIR}\\src ${${project_name}_SOURCE_DIR}\\src\\expression_template_engine)
target_compile_options(Test_selected PRIVATE -W1 -std:c++17 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:-O2>" /bigobj /MP)
target_link_options(Test_selected PRIVATE /STACK:104857600)

target_sources(Test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp


    ${CMAKE_CURRENT_SOURCE_DIR}/test_libdivide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dispatcher.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_walker.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test_shareable_storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_slice.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test_multiindex_iterator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_evaluating_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_storage_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_walker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_helpers_for_testing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_deep_tree_compilation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_experimental_expression_template.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_init_list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_variadic_container.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_experimental_expression_template.cpp


    ${CMAKE_CURRENT_SOURCE_DIR}/test_libdivide_helper.cpp
)

target_sources(Test_selected PRIVATE
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_expression_template.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test_expression_template_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_view.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_view_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)