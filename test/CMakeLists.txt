add_executable(Test)
target_compile_options(Test PRIVATE -W1 -std:c++17 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:-O2>" /bigobj /MP)
target_link_options(Test PRIVATE /STACK:104857600)
target_link_libraries(Test PRIVATE expression_template)

target_sources(Test PRIVATE
    #${CMAKE_CURRENT_SOURCE_DIR}/benchmark_expression_template.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_init_list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_helpers_for_testing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_libdivide_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_expression_template_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_view.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_view_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)