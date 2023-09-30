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
No other dependencies apart from standart library.

## Installation

