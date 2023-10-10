# Sort, search

**sort_search** module is defined in **sort_search.hpp** header.

## sort

Returns a sorted copy of a tensor.

```cpp
template<typename...Ts, typename DimT=int, typename Comparator=std::less<void>>
auto sort(const basic_tensor<Ts...>& t, const DimT& axis=-1, const Comparator& comparator=Comparator{});
//parallel version
template<typename Policy, typename...Ts, typename DimT=int, typename Comparator=std::less<void>>
auto sort(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis=-1, const Comparator& comparator=Comparator{});
```

`axis` along which to sort. By default sort along last axis.

`comparator` is binary predicate, like `std::less<void>` or `std::greater<void>`

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3}};
auto res1 = gtensor::sort(t);
auto res2 = gtensor::sort(t,0,std::greater<void>{});
auto res3 = gtensor::sort(multithreading::exec_pol<4>{},t,-1,std::greater<void>{});
std::cout<<std::endl<<res1; //[(4,5){{0,1,2,3,4},{1,1,4,5,7},{1,2,3,6,6},{2,3,4,4,7}}]
std::cout<<std::endl<<res2; //[(4,5){{4,6,7,7,5},{4,2,4,6,3},{3,1,1,4,3},{1,1,0,2,2}}]
std::cout<<std::endl<<res3; //[(4,5){{4,3,2,1,0},{7,5,4,1,1},{6,6,3,2,1},{7,4,4,3,2}}]
```

## argsort

Returns the indices that would sort a tensor. Has the same parameters as `sort`.

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3}};
auto res1 = gtensor::argsort(t);
auto res2 = gtensor::argsort(t,0,std::greater<void>{});
auto res3 = gtensor::argsort(multithreading::exec_pol<4>{},t,-1,std::greater<void>{});
std::cout<<std::endl<<res1; //[(4,5){{2,1,3,4,0},{0,1,2,4,3},{2,4,0,1,3},{1,4,0,3,2}}]
std::cout<<std::endl<<res2; //[(4,5){{0,2,3,1,1},{3,3,1,2,0},{2,0,2,3,3},{1,1,0,0,2}}]
std::cout<<std::endl<<res3; //[(4,5){{0,4,3,1,2},{3,4,2,0,1},{1,3,0,4,2},{2,0,3,4,1}}]
```

## partition

Returns a partially sorted copy of a tensor.

```cpp
template<typename...Ts, typename Nth, typename DimT=int, typename Comparator=std::less<void>>
auto partition(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis=-1, const Comparator& comparator=Comparator{});
template<typename Policy, typename...Ts, typename Nth, typename DimT=int, typename Comparator=std::less<void>>
auto partition(Policy policy, const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis=-1, const Comparator& comparator=Comparator{});
```

`nth` contains index of partition point, can be container or scalar.

`axis` along which to partially sort. By default sort along last axis.

`comparator` is binary predicate, like `std::less<void>` or `std::greater<void>`

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3}};
auto res1 = gtensor::partition(t,2);
auto res2 = gtensor::partition(t,3,0,std::greater<void>{});
auto res3 = gtensor::partition(multithreading::exec_pol<4>{},t,2,-1,std::greater<void>{});
std::cout<<std::endl<<res1; //[(4,5){{0,1,2,3,4},{1,1,4,5,7},{1,2,3,6,6},{2,3,4,4,7}}]
std::cout<<std::endl<<res2; //[(4,5){{4,6,7,7,5},{4,2,4,6,3},{3,1,1,4,3},{1,1,0,2,2}}]
std::cout<<std::endl<<res3; //[(4,5){{4,3,2,1,0},{7,5,4,1,1},{6,6,3,2,1},{7,4,4,3,2}}]
```

## argpartition

Returns indeces that partially sort a tensor. Has the same parameters as `partition`.

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3}};
auto res1 = gtensor::argpartition(t,2);
auto res2 = gtensor::argpartition(t,3,0,std::greater<void>{});
auto res3 = gtensor::argpartition(multithreading::exec_pol<4>{},t,2,-1,std::greater<void>{});
std::cout<<std::endl<<res1; //[(4,5){{2,1,3,4,0},{0,1,2,4,3},{2,4,0,3,1},{1,4,3,0,2}}]
std::cout<<std::endl<<res2; //[(4,5){{0,2,3,1,1},{3,3,1,2,0},{2,0,2,3,3},{1,1,0,0,2}}]
std::cout<<std::endl<<res3; //[(4,5){{0,4,3,1,2},{3,4,2,0,1},{1,3,0,4,2},{2,0,3,4,1}}]
```

## argmin

Returns the indices of the minimum values along an axes.

```cpp
template<typename...Ts, typename Axes>
auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false);
template<typename...Ts, typename DimT>
auto NAME(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false);
template<typename...Ts>
auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false);
template<typename Policy, typename...Ts, typename Axes>
//parallel version
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false);
template<typename Policy, typename...Ts, typename DimT>
auto NAME(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false);
template<typename Policy, typename...Ts>
auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false);
```

`axes` along which to find minimum. Can be container or scalar. Negative values are allowed and mean counting from the end.
If not specified minimum is found like along flatten input tensor.

`keep_dims` is `false` by default. If `true` is passed, the axes which are reduced are left in the result as dimensions with size one.

`policy` type is specialization of `multithreading::exec_pol`.

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3}};
auto res1 = gtensor::argmin(t);
auto res2 = gtensor::argmin(t,1);
auto res3 = gtensor::argmin(multithreading::exec_pol<4>{},t,0);
std::cout<<std::endl<<res1; //[(){2}]
std::cout<<std::endl<<res2; //[(4){2,0,2,1}]
std::cout<<std::endl<<res3; //[(5){1,0,0,0,2}]
```

## argmax

Returns the indices of the maximum values along an axes. Has the same parameters as `argmin`.

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3}};
auto res1 = gtensor::argmax(t);
auto res2 = gtensor::argmax(t,1);
auto res3 = gtensor::argmax(multithreading::exec_pol<4>{},t,0);
std::cout<<std::endl<<res1; //[(){8}]
std::cout<<std::endl<<res2; //[(4){0,3,1,2}]
std::cout<<std::endl<<res3; //[(5){0,2,3,1,1}]
```

## nanargmin, nanargmax

Nan ignoring counterparts of `argmin` and `argmax` routines described above.

## count_nonzero

Counts the number of **non-zero** elements in the given tensor.
Element is **non-zero** if `static_cast<const bool&>(element)` evaluates to `true`.
Has the same parameters as `argmin`.

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,0,0,6,2},{4,2,7,0,3}};
auto res1 = gtensor::count_nonzero(t);
auto res2 = gtensor::count_nonzero(t,1);
auto res3 = gtensor::count_nonzero(multithreading::exec_pol<4>{},t,0);
std::cout<<std::endl<<res1; //[(){16}]
std::cout<<std::endl<<res2; //[(4){4,5,3,4}]
std::cout<<std::endl<<res3; //[(5){4,3,2,3,4}]
```

## nonzero

Returns container of tensors, one for each dimension of input tensor, containing indeces of non-zero elements.

```cpp
template<typename...Ts>
auto nonzero(const basic_tensor<Ts...>& t);
```

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,0,0,6,2},{4,2,7,0,3}};
auto res = gtensor::nonzero(t);
std::cout<<std::endl<<res[0];   //[(16){0,0,0,0,1,1,1,1,1,2,2,2,3,3,3,3}]
std::cout<<std::endl<<res[1];   //[(16){0,1,3,4,0,1,2,3,4,0,3,4,0,1,2,4}]
```

## argwhere

Returns tensor of indeces of non-zero elements, result shape is `(N,t.dim())` where N is number of non-zero elements, `t` - input tensor.

```cpp
template<typename...Ts>
auto argwhere(const basic_tensor<Ts...>& t);
```

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,0,0,6,2},{4,2,7,0,3}};
auto res = gtensor::argwhere(t);
std::cout<<std::endl<<res;  //[(16,2){{0,0},{0,1},{0,3},...,{3,1},{3,2},{3,4}}]
```

## searchsorted

Finds the indices into a sorted tensor `t` such that, if the corresponding elements in `v` were inserted before the indices, the order of `t` would be preserved.

```cpp
template<typename...Ts, typename V, typename Side=std::false_type, typename Sorter=detail::no_value>
auto searchsorted(const basic_tensor<Ts...>& t, const V& v, Side side=Side{}, const Sorter& sorter=Sorter{});
```

`t` must be 1d tensor.

`v` may be tensor or scalar.

`side` should be `std::false_type` for left side (lower bound) or `std::true_type` for right side (upper bound).

`sorter` may be 1d tensor of indexes to sort `t` or no_value, if no_value then `t` is considered sorted in ascending order.

```cpp
gtensor::tensor<double> t{0,1,1,3,5,6,7,8,8,9,13,14,20};
auto res1 = searchsorted(t,8.0);
auto res2 = searchsorted(t,8.0,std::true_type{});
auto res3 = searchsorted(t,gtensor::tensor<double>{3.3,5.6,11.9});
std::cout<<std::endl<<res1; //7
std::cout<<std::endl<<res2; //9
std::cout<<std::endl<<res3; //[(3){4,5,10}]
```

## unique

Finds the unique elements of a tensor. There are three optional outputs in addition to the unique elements:
- the indices of the input tensor that give the unique values
- the indices of the unique tensor that reconstruct the input tensor
- the number of times each unique value comes up in the input tensor

```cpp
template<typename...Ts, typename ReturnIndex=std::false_type, typename ReturnInverse=std::false_type, typename ReturnCounts=std::false_type, typename Axis=detail::no_value>
auto unique(const basic_tensor<Ts...>& t, ReturnIndex return_index=ReturnIndex{}, ReturnInverse return_inverse=ReturnInverse{}, ReturnCounts return_counts=ReturnCounts{}, const Axis& axis=Axis{});
```

`return_index`, `return_inverse`, `return_counts` - controls whether corresponding optional output will be returned.
To add to return `std::true_type` object should be passed.

`axis` along which to find unique elements. Can be scalar. If not specified then routine operates like over flatten input tensor.

If no optional outputs then routine returns tensor of unique values.
If there are optional outputs then routine returns tuple of tensors.

```cpp
gtensor::tensor<double> t{{1,2,1},{0,1,3},{1,2,1},{5,2,6},{3,7,4},{0,1,3},{1,2,1}};
auto res1 = gtensor::unique(t);
auto res2 = gtensor::unique(t,std::true_type{},std::true_type{},std::true_type{});
auto res3 = gtensor::unique(t,std::true_type{},std::true_type{},std::true_type{},0);
std::cout<<std::endl<<"unique values "<<res1; //unique values [(8){0,1,2,3,4,5,6,7}]
std::cout<<std::endl<<"unique values "<<std::get<0>(res2);  //unique values [(8){0,1,2,3,4,5,6,7}]
std::cout<<std::endl<<"unique indeces "<<std::get<1>(res2); //unique indeces [(8){3,0,1,5,14,9,11,13}]
std::cout<<std::endl<<"reconstruct indecses "<<std::get<2>(res2);   //reconstruct indecses [(21){1,2,1,0,1,3,1,2,1,5,2,6,3,7,4,0,1,3,1,2,1}]
std::cout<<std::endl<<"unique counts "<<std::get<3>(res2);  //unique counts [(8){2,8,4,3,1,1,1,1}]
std::cout<<std::endl<<"unique values "<<std::get<0>(res3);  //unique values [(4,3){{0,1,3},{1,2,1},{3,7,4},{5,2,6}}]
std::cout<<std::endl<<"unique indeces "<<std::get<1>(res3); //unique indeces [(4){1,0,4,3}]
std::cout<<std::endl<<"reconstruct indecses "<<std::get<2>(res3);   //reconstruct indecses [(7){1,0,1,3,2,0,1}]
std::cout<<std::endl<<"unique counts "<<std::get<3>(res3);  //unique counts [(4){2,3,1,1}]
```
