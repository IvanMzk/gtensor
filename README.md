![gtensor_logo](/docs/gtensor_logo.png)

## What is GTensor?

GTensor is C++ library for multidimensional array computation, its name stands for G(eneric)Tensor.
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

## Including into project

GTensor is header only library so to use it in project you can include needed header files and everything will work. GTensor doesn't provide any binaries to
link to, all library code is defined in headers.

If you use Cmake build system, you can utilize `add_subdirectory(...)` CMake command, e.g.:

```cmake
cmake_minimum_required(VERSION 3.2)
project(my_project)
add_subdirectory(path_to_gtensor_dir)
add_executable(my_target)
target_link_libraries(my_target PRIVATE gtensor::gtensor)
...
```

CMake install is also supported:

```cmake
cmake -B build_dir -DCMAKE_INSTALL_PREFIX=your_install_prefix
cmake --install build_dir
```

After that you can include header files from `your_install_prefix` directory or use `include(...)` CMake command, e.g.:

```cmake
cmake_minimum_required(VERSION 3.2)
project(my_project)
include(your_install_prefix/lib/cmake/gtensor/gtensor_targets.cmake)
add_executable(my_target)
target_link_libraries(my_target PRIVATE gtensor::gtensor)
...
```

## Build test and benchmark