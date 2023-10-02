# Tutorial

This tutorial describes the main points of using GTensor library, its structure and functions.

## 1. Multidimensional array abstraction, `data` and `meta-data`

GTensor library is meant for computing over multidimensional arrays. Such an array is abstraction which mainly consists of two parts: meta-data and data as an analogy to form and matter in philosophy.

In most practical implementations data and meta-data are implemented using flat arrays of elements but with different meanings.
- data elements can be of any type, suitable for goals of computation, e.g. integral, floating-point, complex or even some user-defined type
- meta-data elements ussually of integral type due to its purpose: describe multidimensional structure of data elements, index data elements

To be useful, array abstraction generally should provide interface to access its data and meta-data elements and hide other implementation details.
It is common practice to use `iterator interface` for data and member functions like `shape()`, `strides()`, `dim()`, `size()` for meta-data.

## 2. `tensor` and `basic_tensor` class templates



