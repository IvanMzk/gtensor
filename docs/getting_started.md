# Getting started

This section show some examples of simple usage of GTensor library.

## First example

#### **`first_example.cpp`**
```cpp
#include <iostream>
#include "tensor.hpp"

int main(int argc, const char*argv[]){

    auto t = gtensor::tensor<double>{{1,2,3},{4,5,6},{7,8,9}};
    auto res = (t+1).transpose()(1);
    std::cout<<std::endl<<res;

    return 0;
}
```

This simple example creates 2-dim tensor of doubles, adds 1 to its elements, transposes result and takes second row of transposed.

### Compiling first example

GTensor is header only library, and to use it in project compiler must know path to gtensor include files.

```bash
g++ --std=c++17 -Ipath/to/gtensor/include first_example.cpp -o first_example
```

When you run `first_example` it produces the following output:

```bash
[(3){3,6,9}]
```

### Building first example using CMake

Assuming the following folders structure:

```bash
first_example
    |--src
    |   |--first_example.cpp
    |--CMakeLists.txt
```

#### **`CMakeLists.txt`**
```cmake
cmake_minimum_required(VERSION 3.2)
project(first_example)
add_subdirectory(path/to/gtensor gtensor)
add_executable(first_example)
target_link_libraries(first_example PRIVATE gtensor::gtensor)
target_sources(first_example PRIVATE src/first_example.cpp)
target_compile_features(first_example PRIVATE cxx_std_17)
```

The following commands create build_dir folder inside first_example folder, build and run first_example project:

```bash
cd path/to/first_example
cmake -B build_dir -GNinja
cmake --build build_dir
./build_dir/first_example
```

## Second example: fancy indexing


