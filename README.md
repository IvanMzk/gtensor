![gtensor_logo](/docs/gtensor_logo.png)

## What is GTensor?

GTensor is C++ library for multidimensional array computations, its name stands for G(eneric)Tensor.
Library provides `basic_tensor` class template that represents multidimensional array and various routines
to operate on it, including mathematical,logical,shape manipulation,sorting,selecting,basic statistical operations,
random simulation.

Library features are:

- it is header only library
- API is similar to numpy
- support lazy computations, thanks to expression templates
- support all standart integral, floating-point data types and std::complex
- can be easily configurable to support user provided data types
- designed to be easily extensible

## Requirements

GTensor uses c++17 in its code and thus requires appropriate compiler.
It is tested with clang 15.0.7, gcc 12.2.0, msvc 19.26 compilers.
It has no other dependencies apart from standart library.
Cmake build system is used to build tests, benchmarks and to automate installation.

## Documentation

1. [Getting started guide](/docs/getting_started.md)
2. [Tutorial](/docs/tutorial.md)
3. [Reference](/docs/reference.md)
4. [Extending GTensor](/docs/extending.md)

## Including into project

GTensor is header only library so to use it in project you can include needed header files and everything will work. GTensor doesn't provide any binaries to
link to, all library code is defined in headers.

To install `gtensor package` on system:

```cmake
cmake -B build_dir -DCMAKE_INSTALL_PREFIX=your_install_prefix
cmake --install build_dir
```

To add installed package to your project:

```cmake
cmake_minimum_required(VERSION 3.5)
project(my_project)
list(APPEND CMAKE_PREFIX_PATH your_install_prefix)
find_package(gtensor)
add_executable(my_target)
target_link_libraries(my_target PRIVATE gtensor::gtensor)
...
```

To use GTensor without installation you can utilize `add_subdirectory(...)` CMake command, e.g.:

```cmake
cmake_minimum_required(VERSION 3.5)
project(my_project)
add_subdirectory(path_to_gtensor_dir gtensor)
add_executable(my_target)
target_link_libraries(my_target PRIVATE gtensor::gtensor)
...
```

## Build tests and benchmarks

GTensor uses [Catch](https://github.com/catchorg/Catch2) framework for testing.

To build and run tests:

```cmake
cmake -B build_dir -DBUILD_TEST=ON
cmake --build build_dir
build_dir/test/Test
```

To build and run benchmarks:

```cmake
cmake -B build_dir -DBUILD_BENCHMARK=ON
cmake --build build_dir
build_dir/benchmark/Benchmark
```

## License
GTensor is licensed under the [BSL 1.0](LICENSE.txt).