add_executable(Test)
add_executable(Test_selected)
target_include_directories(Test PRIVATE ${${project_name}_SOURCE_DIR}\\src)
target_compile_options(Test PRIVATE -W1 -std:c++17 /Zc:__cplusplus)
target_include_directories(Test_selected PRIVATE ${${project_name}_SOURCE_DIR}\\src)
target_compile_options(Test_selected PRIVATE -W1 -std:c++17 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:-O2>" /bigobj)

target_sources(Test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
    
    ${CMAKE_CURRENT_SOURCE_DIR}/test_shareable_storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_view_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_libdivide_helper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stensor_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor_init_list.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_slice.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test_view.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_libdivide.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multiindex_iterator.cpp
)

target_sources(Test_selected PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_walker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_expression_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stensor_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_view_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)