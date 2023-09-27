add_library(gtensor INTERFACE)
target_include_directories(gtensor INTERFACE
    ${${project_name}_SOURCE_DIR}\\src
    ${${project_name}_SOURCE_DIR}\\src\\expression_template_engine
)

target_compile_features(gtensor INTERFACE cxx_std_17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(gtensor INTERFACE /W4 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(gtensor INTERFACE -Wall "$<$<CONFIG:RELEASE>:-O2>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(gtensor INTERFACE -stdlib=libc++)
    else()
    endif()
endif()


