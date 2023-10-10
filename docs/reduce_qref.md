# Reduce

**reduce** module resides in **reduce.hpp** header. It contains routines to make tensor reductions using custom functors.
Almost all reduce routines in other modules use **reduce** module in their implementations.

## reduce_binary

Returns tensor that is result of reduction of input tensor using **binary** operation.

```cpp
template<typename F, typename Axes, typename...Ts, typename Initial=detail::no_value>
auto reduce_binary(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, const Initial& initial=Initial{});
//parallel version
template<typename Policy, typename F, typename Axes, typename...Ts, typename Initial=detail::no_value>
auto reduce_binary(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, const Initial& initial=Initial{});
```

Reduction is performed along axes, `axes` can be scalar or container.

`f` is **binary** reduce function object that operates on input tensor's elements.

If `keep_dims` is `true`, the axes which are reduced are left in the result as dimensions with size one.

Optional `initial` value can be specified.

`policy` is specialization of `multithreading::exec_pol`.

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto res1 = reduce_binary(t,0,std::plus<void>{},false);
auto res2 = reduce_binary(t,1,std::multiplies<void>{},true,2.0);
auto res3 = reduce_binary(multithreading::exec_pol<4>{},t,std::vector<int>{0,1},std::multiplies<void>{});
std::cout<<std::endl<<res1; //[(3){5,7,9}]
std::cout<<std::endl<<res2; //[(2,1){{12},{240}}]
std::cout<<std::endl<<res3; //[(){720}]
```

## reduce_range

Returns tensor that is result of reduction of input tensor using operation that takes iterators range to be reduced and return scalar - reduction result.

```cpp
template<typename F, typename Axes, typename...Ts, typename...Args>
auto reduce_range(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, bool any_order, const Args&...args);
//parallel version
template<typename Policy, typename F, typename Axes, typename...Ts, typename...Args>
auto reduce_range(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, bool any_order, const Args&...args);
```

Reduction is performed along axes, `axes` can be scalar or container.

`f` is reduce function object with signature equivalent to:

```cpp
template<typename It, typename...Args>
auto reduce_f(It first, It last, const Args&...args);
```

`args` are optional parameters.

If `any_order` is true iterators traverse order unspecified, `c_order` otherwise.

`policy` is specialization of `multithreading::exec_pol`.

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto sum = [](auto first, auto last, auto init){
    if (first==last){return init;}
    return std::accumulate(first,last,init,std::plus<void>{});
};
auto res1 = reduce_range(t,0,sum,false,true,0.0);
auto res2 = reduce_range(t,std::vector<int>{0,1},sum,false,true,1.0);
auto res3 = reduce_range(multithreading::exec_pol<4>{},t,1,sum,false,true,0.0);
std::cout<<std::endl<<res1; //[(3){5,7,9}]
std::cout<<std::endl<<res2; //[(){22}]
std::cout<<std::endl<<res3; //[(2){6,15}]
```

## slide

Returns tensor that is result of transformation of input tensor along axis.
Such transformations like **moving mean** and **cumulative sum** may be implemented using `slide`.

```cpp
template<typename ResultT, typename Axis, typename...Ts, typename F, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, const Axis& axis, F f, const IdxT& window_size, const IdxT& window_step, const Args&...args);
//parallel version
template<typename ResultT, typename Policy, typename Axis, typename...Ts, typename F, typename IdxT, typename...Args>
auto slide(Policy policy, const basic_tensor<Ts...>& t, const Axis& axis, F f, const IdxT& window_size, const IdxT& window_step, const Args&...args);
```

`axis` is scalar.

`f` is slide function object with signature equivalent to:

```cpp
template<typename It, typename DstIt, typename...Args>
void reduce_f(It first, It last, DstIt dfirst, DstIt dlast, const Args&...args);
```


`window_size` and `window_step` specify width of sliding window and its step along input axis.
As result transformed axis size equals to `(input_axis_size - window_size)/window_step + 1`.

`dlast-dfirst` is guaranteed to equal to transformed axis size.

`args` are optional parameters.

Result tensor's `value_type` should be spicified explicitly using template parameter `ResultT`.

Next example shows possible **first differences** implementation using `slide`.

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3},{2,2,0,1,2}};
auto diff = [](auto first, auto last, auto dfirst, auto dlast){
    (void)last;
    for (;dfirst!=dlast;++dfirst){
        auto prev = *first;
        *dfirst = *(++first) - prev;
    }
};
auto res1 = gtensor::slide<double>(t,0,diff,2,1);
auto res2 = gtensor::slide<double>(multithreading::exec_pol<4>{},t,1,diff,2,1);
std::cout<<std::endl<<res1; //[(4,5){{-3,0,4,5,2},{2,5,-3,-1,-3},{1,-4,6,-2,1},{-2,0,-7,-3,-1}}]
std::cout<<std::endl<<res2; //[(5,4){{-3,-1,2,1},{0,3,3,-2},{3,-5,5,-4},{-2,5,-3,-1},{0,-2,1,1}}]
```

## transform

Transform tensor inplace along specified axis. Axis is scalar. For example inplace `sort` along axis may be implemented using `transform`.

```cpp
template<typename...Ts, typename DimT, typename F, typename...Args>
void transform(basic_tensor<Ts...>& t, const DimT& axis, F f, const Args&...args);
//parallel version
template<typename Policy, typename...Ts, typename DimT, typename F, typename...Args>
void transform(Policy policy, basic_tensor<Ts...>& t, const DimT& axis, F f, const Args&...args);
```

`axis` is scalar.

`f` is transform function object with signature equivalent to:

```cpp
template<typename It, typename...Args>
void transform_f(It first, It last, const Args&...args);
```

`args` are optional parameters.

Next example shows possible **inplace sort** implementation using `transform`.

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3},{2,2,0,1,2}};
auto sort = [](auto first, auto last){
    std::sort(first,last);
};
gtensor::transform(t,1,sort);
std::cout<<std::endl<<t;    //[(5,5){{0,1,2,3,4},{1,1,4,5,7},{1,2,3,6,6},{2,3,4,4,7},{0,1,2,2,2}}]
gtensor::transform(multithreading::exec_pol<4>{},t,0,sort);
std::cout<<std::endl<<t;    //[(5,5){{0,1,2,2,2},{0,1,2,3,4},{1,1,3,4,6},{1,2,4,5,7},{2,3,4,6,7}}]
```
