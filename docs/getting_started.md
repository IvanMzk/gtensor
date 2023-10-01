# Getting started

This section shows some examples of simple usage of GTensor library.

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

## Second example: fancy indexing and assignment

#### **`second_example.cpp`**
```cpp
#include <iostream>
#include "tensor.hpp"

int main(int argc, const char*argv[]){

    auto t = gtensor::tensor<double>{{7,3,4,6},{1,5,6,2},{1,8,3,5},{0,2,6,2}};
    t(t>3 && t.not_equal(6))+=1;
    std::cout<<std::endl<<t;

    return 0;
}
```

Second example creates 2-d tensor and assign-add 1 to elements that greater than 3 and not equal 6.

When build and run this produces the following output:

```bash
[(4,4){{8,3,5,6},{1,6,6,2},{1,9,3,6},{0,2,6,2}}]
```

## Third example: broadcast and lazy evaluation

#### **`third_example.cpp`**
```cpp
#include <iostream>
#include <numeric>
#include "tensor_math.hpp"
#include "tensor.hpp"

int main(int argc, const char*argv[]){

    auto t = gtensor::tensor<double>(5,0);
    std::iota(t.begin(),t.end(),0);

    auto dist = hypot(t.reshape(1,-1),t.reshape(-1,1));

    std::cout<<std::endl<<dist;
    return 0;
}
```

Third example computes distance from origin to each point of 5x5 grid.

Statement
```cpp
auto dist = hypot(t.reshape(1,-1),t.reshape(-1,1));
```
doesn't perform any computations, it returns special kind of tensor called **expession view**. Actual evaluation take place when `dist` is printed to std::cout.

When build and run this produces the following output:

```bash
[(5,5){{0,1,2,3,4},{1,1.41,2.24,3.16,4.12},{2,2.24,2.83,3.61,4.47},{3,3.16,3.61,4.24,5},{4,4.12,4.47,5,5.66}}]
```

## Forth example: random numbers

#### **`forth_example.cpp`**
```cpp
#include <iostream>
#include "statistic.hpp"
#include "random.hpp"
#include "tensor.hpp"

int main(int argc, const char*argv[]){

    const auto seed = 123;
    auto rng = gtensor::default_rng(seed);

    auto rnd_sum = gtensor::tensor<double>(1000,0);
    for (int n=30; n!=0; --n){
        rnd_sum += rng.random(1000);
    }

    auto hist = gtensor::histogram(rnd_sum);
    std::cout<<std::endl<<"hist bins"<<hist.first;
    std::cout<<std::endl<<"hist edges"<<hist.second;

    return 0;
}
```

In Forth example we construct random number generator and make elementwise sum of 30 uniformly distributed in range [0,1) random tensors.

We expected that elements in result tensor will have close to Gauss distribution:

```bash
hist bins[(10){6,34,102,188,231,225,129,64,18,3}]
hist edges[(11){10.3,11.2,12.2,13.2,14.2,15.2,16.2,17.2,18.2,19.1,20.1}]
```

Results look like expected.