# Indexing

`indexing` module is defined in `indexing.hpp` header.

## take

Take elements from a tensor along an `axis`. The effect is same as in **numpy**.

```cpp
template<typename DimT, typename...Ts, typename...Us>
auto take(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes, const DimT& axis);
//take like over flatten
template<typename...Ts, typename...Us>
auto take(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes);
```

`indexes` the indixes of the values to extract.

`axis` the axis over which to select values. If not specified, the flattened input tensor is used.

```cpp
using gtensor::tensor;
tensor<double> t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto res1 = take(t,tensor<int>{{2,0},{3,1}});
auto res2 = take(t,tensor<int>{{2,0},{1,2}},0);
std::cout<<std::endl<<res1; //[(2,2){{3,1},{4,2}}]
std::cout<<std::endl<<res2; //[(2,2,4){{{9,10,11,12},{1,2,3,4}},{{5,6,7,8},{9,10,11,12}}}]
```

## take_along_axis

Takes values from the input tensor by matching 1d index and data slices.
Functions returning an index along an axis, like `argsort` and `argpartition`, produce suitable indices for this function.
The effect is same as in **numpy**.

```cpp
template<typename DimT, typename...Ts, typename...Us>
auto take_along_axis(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes, const DimT& axis);
//take_along_axis like over flatten
template<typename...Ts, typename...Us>
auto take_along_axis(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes);
```

```cpp
gtensor::tensor<double> t{{1,0,2},{5,4,3}};
auto res1 = take_along_axis(t,argsort(t,1),1);
auto res2 = take_along_axis(t,argsort(t.flatten()));
std::cout<<std::endl<<res1; //[(2,3){{0,1,2},{3,4,5}}]
std::cout<<std::endl<<res2; //[(6){0,1,2,3,4,5}]
```
