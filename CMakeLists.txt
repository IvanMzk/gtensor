# GTensor - computation library
# Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
#
# Distributed under the Boost Software License, Version 1.0.
# The full license is in the file LICENSE.txt, distributed with this software.


cmake_minimum_required(VERSION 3.2)
project(gtensor)

set(include_dir ${CMAKE_CURRENT_LIST_DIR}/include)

add_library(gtensor INTERFACE)
target_include_directories(gtensor INTERFACE
    $<BUILD_INTERFACE:${include_dir}>
    $<BUILD_INTERFACE:${include_dir}/expression_template_engine>
    $<BUILD_INTERFACE:${include_dir}/libdivide>
)
target_compile_features(gtensor INTERFACE cxx_std_17)
add_library(gtensor::gtensor ALIAS gtensor)

option(BUILD_TEST "build gtensor tests" OFF)
option(BUILD_BENCHMARK "build gtensor benchmark" OFF)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(BUILD_TEST)
        add_subdirectory(test)
    endif()
    if(BUILD_BENCHMARK)
        add_subdirectory(benchmark)
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS gtensor
    EXPORT gtensor_targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtensor
        ${CMAKE_INSTALL_INCLUDEDIR}/gtensor/expression_template_engine
        ${CMAKE_INSTALL_INCLUDEDIR}/gtensor/libdivide
)
install(EXPORT gtensor_targets
    FILE gtensor_targets.cmake
    NAMESPACE gtensor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gtensor
)
install(DIRECTORY ${include_dir}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gtensor)

export(EXPORT gtensor_targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/gtensor_targets.cmake
    NAMESPACE gtensor::
)