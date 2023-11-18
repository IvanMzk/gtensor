# GTensor - computation library
# Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
#
# Distributed under the Boost Software License, Version 1.0.
# The full license is in the file LICENSE.txt, distributed with this software.

add_executable(Benchmark)

target_link_libraries(Benchmark PRIVATE gtensor)
target_include_directories(Benchmark PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../qa_common)
target_compile_features(Benchmark PRIVATE cxx_std_17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(Benchmark PRIVATE /W4 /Zc:__cplusplus /bigobj "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(Benchmark PRIVATE -Wall -march=native "$<$<CONFIG:RELEASE>:-O2>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(Benchmark PRIVATE -stdlib=libc++)
    else()
    endif()
endif()

target_sources(Benchmark PRIVATE
    #${CMAKE_CURRENT_LIST_DIR}/benchmark_copy_eval.cpp
    #${CMAKE_CURRENT_LIST_DIR}/benchmark_iterator.cpp
    #${CMAKE_CURRENT_LIST_DIR}/benchmark_statistic.cpp
    ${CMAKE_CURRENT_LIST_DIR}/benchmark_matmul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/benchmark.cpp
)