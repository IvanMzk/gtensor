add_library(expression_template INTERFACE)
target_include_directories(expression_template INTERFACE
    ${${project_name}_SOURCE_DIR}\\src
    ${${project_name}_SOURCE_DIR}\\src\\expression_template_engine
)
target_compile_features(expression_template INTERFACE cxx_std_17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(expression_template INTERFACE /W4 /arch:AVX2 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(expression_template INTERFACE -Wall -mavx2 "$<$<CONFIG:RELEASE>:-O3>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(expression_template INTERFACE -stdlib=libc++)
    else()
    endif()
endif()


