# `tensor` and `basic_tensor`

**Tensor** module resides in **tensor.hpp** header.
It defines `tensor` and `basic_tensor` class templates and also includes tensor operators and routines to make views.

`basic_tensor` class template represents multidimensional array abstraction, which may have different implementations.
`basic_tensor` objects should be never constructed directly.
`tensor` class template is intended to construct `basic_tensor` object with storage implementation.
In fact it just defines constructors suitable to initialize storage implementation and nothing more.

Tensor object is instance of `basic_tensor` class template specialization, regardless of used implementation type.

## Template parameters

```cpp
gtensor::tensor<double> //tensor with double value_type
gtensor::tensor<double, gtensor::config::c_order> //elements layout explicitly specified to be row major
gtensor::tensor<double, gtensor::config::f_order> //elements layout explicitly specified to be column major
```

## Initialization

Tensor can be initialized from scalar, initializer_list, shape, shape and scalar, shape and iterators range.
Default constructor also provided.

```cpp
using gtensor::tensor;
//default
tensor<double> t1{};    //1d tensor with shape (0), size is 0
//from scalar
tensor<double> t2(3);    //0d tensor (aka tensor-scalar), shape is empty, size is 1
//initializer_list
tensor<double> t3{{1,2,3},{4,5,6}}; //2d tensor with shape (2,3), initialized using values from nested initializer list
//shape
tensor<double> t4(std::vector<int>{2,3});   //2d tensor with shape (2,3), whether elements are initialized depends on underlaying storage type
//shape and scalar
tensor<double> t5({2,3},4);   //2d tensor with shape (2,3), all elements are initialized to have value 4
//shape and iterators range
std::vector<double> elems{1,2,3,4,5,6}
tensor<double> t6({2,3},elems.begin(),elemes.end());   //2d tensor with shape (2,3), elements are initialized using values from range
```

## Copy and move constructors

Tensor has reference copy semantic.

```cpp
gtensor::tensor<double> a{1,2,3,4,5};
auto b = a; //b refers to same elements as a, mutating a causes mutating b
auto c = std::move(a);  //c refers to same elements as b, a is empty
```

## Assignment

Tensor exposes two kinds of assign semantic: value assign semantic and broadcast assign semantic.
Compaund assignment always has broadcast sementic.

```cpp
gtensor::tensor<double> a{7,8,9};
gtensor::tensor<double> b{{1,2,3},{4,5,6}};
b+=a;   //broadcast compaund assign, b is [(2,3){{8,10,12},{11,13,15}}]
b*=2;   //b is [(2,3){{16,20,24},{22,26,30}}]
b.assign(a);    //broadcast assign, b is [(2,3){{7,8,9},{7,8,9}}]
b.assign(5);    //b is [(2,3){{5,5,5},{5,5,5}}]
b = a;  //value assign, b is [(3){7,8,9}]
b = 5;  //b is tensor-scalar [(){5}]
```

Assignment to rvalue always has broadcast semantic. Motivation is assignment to view.

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
a(1) = 7;   //assign to rvalue view slice, a is [(2,3){{1,2,3},{7,7,7}}]
a(1,1) = 8; //a is [(2,3){{1,2,3},{7,8,7}}]
a({{},{0,1}}) = 9;  //a is [(2,3){{9,2,3},{9,8,7}}]
std::move(a) = gtensor::tensor<double>{1,2,3}; //equivalent to a.assign(...), a is [(2,3){{1,2,3},{1,2,3}}]
```

## copy, eval

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a.copy();  //b is deep copy of a
auto c = a.copy(gtensor::config::f_order{});  //c is deep copy of a, c elements are column major
auto d = a.eval();  //a is not view (nothing to evaluate), d is shallow copy of a,  mutating d causes mutating a
```

`copy()` and `eval()` have the same effect when called on view

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = (a+a).eval(multithreading::exec_pol<4>{});  //b is tensor constructed from shape and elements of expression view a+a; parallel evaluation requested
auto c = (a+a).copy(multithreading::exec_pol<4>{});  //the same as above
```

## swap

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
gtensor::tensor<double> b{7,8,9};
a.swap(b);  //swap implementations, no data copy
```

## resize

`resize()` preserves existing elements.

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
a.resize(std::vector<int>{3,4});
std::cout<<std::endl<<a;    //[(3,4){{1,2,3,4},{5,6,2.23e-306,1.6e-306},{1.78e-306,1.69e-306,1.25e-306,0}}]
a.resize({2,2});
std::cout<<std::endl<<a;    //[(2,2){{1,2},{3,4}}]
```

## flatten

`flatten()` always returns copy, not view.

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a.flatten(gtensor::config::c_order{});
auto c = a.flatten(gtensor::config::f_order{});
auto d = a.flatten();   //use c_order by default
std::cout<<std::endl<<b;    //[(6){1,2,3,4,5,6}]
std::cout<<std::endl<<c;    //[(6){1,4,2,5,3,6}]
std::cout<<std::endl<<d;    //[(6){1,2,3,4,5,6}]
```

## dim, shape, size

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto dim = t.dim();   //2
auto size = t.size();   //6
auto is_empty = t.empty();   //0
const auto& shape = t.shape();  //{2,3}
const auto& strides = t.strides();  //{2,1}
```

## Iterators

Tensor provides itertor, reverse iterator and their const counterparts.

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
std::copy(t.begin(),t.end(),std::ostream_iterator<double>(std::cout,","));  //1,2,3,4,5,6,
std::copy(t.rbegin(),t.rend(),std::ostream_iterator<double>(std::cout,","));    //6,5,4,3,2,1,
```

To change traverse order explicitly `traverse_order_adapder` helper is proveded.

```cpp
using gtensor::config::c_order;
using gtensor::config::f_order;
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto tr_adapt_c = t.traverse_order_adapter(c_order{});  //creates traverse_order_adapter to make iterators with row major traverse order
std::copy(tr_adapt_c.begin(),tr_adapt_c.end(),std::ostream_iterator<double>(std::cout,","));    //1,2,3,4,5,6,
std::copy(tr_adapt_c.rbegin(),tr_adapt_c.rend(),std::ostream_iterator<double>(std::cout,","));  //6,5,4,3,2,1,
auto tr_adapt_f = t.traverse_order_adapter(f_order{});  ////creates traverse_order_adapter to make iterators with column major traverse order
std::copy(tr_adapt_f.begin(),tr_adapt_f.end(),std::ostream_iterator<double>(std::cout,","));    //1,4,2,5,3,6,
std::copy(tr_adapt_f.rbegin(),tr_adapt_f.rend(),std::ostream_iterator<double>(std::cout,","));  //6,3,5,2,4,1,
```

## data()

The underlaying data buffer can be accessed using `data()` member function, not implemented for views.

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
t.data()[3] = 0;
std::cout<<std::endl<<t;    //[(2,3){{1,2,3},{0,5,6}}]
//(t+t).data()  //will not compile, t+t is expression view
```

## Reshape view

```cpp
using gtensor::config::c_order;
using gtensor::config::f_order;
gtensor::tensor<double> t{{1,2,3,4,5,6},{7,8,9,10,11,12}};
auto v1 = t.reshape(std::vector<int>{-1,3});
auto v2 = t.reshape({3,4});
auto v3 = t.reshape({2,-1,3},c_order{});
auto v4 = t.reshape({6,-1},f_order{});
std::cout<<std::endl<<v1;   //[(4,3){{1,2,3},{4,5,6},{7,8,9},{10,11,12}}]
std::cout<<std::endl<<v2;   //[(3,4){{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
std::cout<<std::endl<<v3;   //[(2,2,3){{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}]
std::cout<<std::endl<<v4;   //[(6,2){{1,4},{7,10},{2,5},{8,11},{3,6},{9,12}}]
```

## ravel

`ravel(order)` is equivalent to `reshape({-1},order)`, result is always view.

```cpp
gtensor::tensor<double> t{{1,2,3,4,5,6},{7,8,9,10,11,12}};
auto v1 = t.ravel();    //use c_order by default
auto v2 = t.ravel(gtensor::config::c_order{});
auto v3 = t.ravel(gtensor::config::f_order{});
std::cout<<std::endl<<v1;   //[(12){1,2,3,4,5,6,7,8,9,10,11,12}]
std::cout<<std::endl<<v2;   //[(12){1,2,3,4,5,6,7,8,9,10,11,12}]
std::cout<<std::endl<<v3;   //[(12){1,7,2,8,3,9,4,10,5,11,6,12}]
```

## Slice view

`slice_type` object is constructed using three parameters: `start`,`stop`,`step`. Negative values are supported and interpreted as counting from the end.
Any of three parameters can be missed. To select all from axis, `slice_type` object should be constructed with no arguments.

```cpp
using tensor_type = gtensor::tensor<double>;
using slice_type = typename tensor_type::slice_type;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto v1 = t(slice_type{{},-1}); //select all but last row and all columns
auto v2 = t(slice_type{},slice_type{{},{},2});  //select all rows and columns with step 2
auto v3 = t(slice_type{{},{},-1},slice_type{1,3});  //select all rows in inverse order and columns 1,2
auto v4 = t(slice_type{5},slice_type{});    //out of bounds, empty result
auto v5 = t(slice_type{{},5},slice_type{}); //select all
std::cout<<std::endl<<v1;   //[(2,4){{1,2,3,4},{5,6,7,8}}]
std::cout<<std::endl<<v2;   //[(3,2){{1,3},{5,7},{9,11}}]
std::cout<<std::endl<<v3;   //[(3,2){{10,11},{6,7},{2,3}}]
std::cout<<std::endl<<v4;   //[(0,4){}]
std::cout<<std::endl<<v5;   //[(3,4){{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
```

There is **shortcut syntax**, without using `slice_type` explicitly, it has the same effect as above.

```cpp
using tensor_type = gtensor::tensor<double>;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto v1 = t({{{},{-1}}});
auto v2 = t({{},{{},{},2}});
auto v3 = t({{{},{},-1},{1,3}});
auto v4 = t({{5},{}});
auto v5 = t({{{},5},{}});
std::cout<<std::endl<<v1;   //[(2,4){{1,2,3,4},{5,6,7,8}}]
std::cout<<std::endl<<v2;   //[(3,2){{1,3},{5,7},{9,11}}]
std::cout<<std::endl<<v3;   //[(3,2){{10,11},{6,7},{2,3}}]
std::cout<<std::endl<<v4;   //[(0,4){}]
std::cout<<std::endl<<v5;   //[(3,4){{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
```

`slice_type` object can be mixed with scalar indexes to achieve dimension reduction.

```cpp
using tensor_type = gtensor::tensor<double>;
using slice_type = typename tensor_type::slice_type;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto v1 = t(1); //select second row, result is 1d
auto v2 = t(2,1);   //select second element from third row, result is 0d (tensor-scalar)
auto v3 = t(slice_type{},0);    //select first column, result id 1d
auto v4 = t(slice_type{1},2);   //select third column, starting from second row, result is 1d
auto v5 = t(1,slice_type{1,-1});    //select second row, columns 1,2, result is 1d
std::cout<<std::endl<<v1;   //[(4){5,6,7,8}]
std::cout<<std::endl<<v2;   //[(){10}]
std::cout<<std::endl<<v3;   //[(3){1,5,9}]
std::cout<<std::endl<<v4;   //[(2){7,11}]
std::cout<<std::endl<<v5;   //[(2){6,7}]
```

It is also possible to use container of `slice_type` objects.

```cpp
using tensor_type = gtensor::tensor<double>;
using slice_type = typename tensor_type::slice_type;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
std::vector<slice_type> subscripts{};
subscripts.push_back(slice_type{0,-1});
subscripts.push_back(slice_type{1,-1});
auto v = t(subscripts);
std::cout<<std::endl<<v;    //[(2,2){{2,3},{6,7}}]
```

## Transpose view

```cpp
gtensor::tensor<double> t{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}};
auto v1 = t.transpose();    //by default rearrange axes in reverse order
auto v2 = t.transpose(2,1,0);
auto v3 = t.transpose(std::vector<int>{1,0,2});
auto v4 = t.transpose(0,1,2);
std::cout<<std::endl<<v1;   //[(2,2,3){{{1,5,9},{3,7,11}},{{2,6,10},{4,8,12}}}]
std::cout<<std::endl<<v2;   //[(2,2,3){{{1,5,9},{3,7,11}},{{2,6,10},{4,8,12}}}]
std::cout<<std::endl<<v3;   //[(2,3,2){{{1,2},{5,6},{9,10}},{{3,4},{7,8},{11,12}}}]
std::cout<<std::endl<<v4;   //[(3,2,2){{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}]
```

## Mapping view

Selecting using tensors of indeces and bools is performed according to rules in **numpy**.

```cpp
using gtensor::tensor;
tensor<double> t{1,2,3,4,5,6,7,8,9,10,11,12};
auto v1 = t(tensor<int>{2,1,0,8,9,0});
auto v2 = t(tensor<int>{{2,1,0},{8,9,0}});
auto v3 = t.reshape(3,4)(tensor<int>{1,0,1,2});
auto v4 = t.reshape(3,4)(tensor<int>{0,0,2,2},tensor<int>{0,3,0,3});
std::cout<<std::endl<<v1;   //[(6){3,2,1,9,10,1}]
std::cout<<std::endl<<v2;   //[(2,3){{3,2,1},{9,10,1}}]
std::cout<<std::endl<<v3;   //[(4,4){{5,6,7,8},{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
std::cout<<std::endl<<v4;   //[(4){1,4,9,12}]
```

```cpp
gtensor::tensor<double> t{{7,3,4,6},{1,5,6,2},{1,8,3,5},{0,2,6,2}};
auto v1 = t(tensor<bool>{false,true,false,true});
auto v2 = t(tensor<bool>{{false,true,false,true},{true,false,true,false},{false,true,true,false},{true,false,false,true}});
auto v3 = t(t>3 && t.not_equal(6));
std::cout<<std::endl<<v1;   //[(2,4){{1,5,6,2},{0,2,6,2}}]
std::cout<<std::endl<<v2;   //[(8){3,6,1,6,8,3,0,2}]
std::cout<<std::endl<<v3;   //[(5){7,4,5,8,5}]
```

## reduce_range, reduce_binary, slide, transform

These member function are shortcuts to routines defined in [reduce module](./reduce_ref.md).

## all, any, min, max, sum, prod, cumsum, cumprod

These member function are shortcuts to routines defined in [tensor_math module](./tensor_math_ref.md).

## mean, var, stdev, median, ptp

These member function are shortcuts to routines defined in [statistic module](./statistic_ref.md).

## sort, argsort, argmin, argmax, nonzero

These member function are shortcuts to routines defined in [sort_search module](./sort_search_ref.md).

`sort` member function sorts tensor in-place, whereas module routine makes sorted copy.
