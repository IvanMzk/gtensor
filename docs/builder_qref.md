# Tensor builders

GTensor **builder** module resides in **builder.hpp** header.
It provides routines to make tensors of specified structure and values.

## empty

`empty()` routine makes tensor of specified shape and value_type.
Whether tensor's elements will be initialized depends on underlaying storage type.
By default elements are not initialized for trivially-copyable data type, and initialized to default otherwise.

```cpp
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename ShT>
auto empty(ShT&& shape);

template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename U>
auto empty(std::initializer_list<U> shape);
```

```cpp
auto a = gtensor::empty<int>({2,3});
auto b = gtensor::empty<double, gtensor::config::f_order>(std::vector<int>{3,2});
std::cout<<std::endl<<a;    //[(2,3){{0,0,-1029898752},{459,1,0}}]
std::cout<<std::endl<<b;    //[(3,2){{-3.72e-103,0},{9.76e-312,9.76e-312},{1.48e-323,4.94e-324}}]
```

## full

Makes tensor of given shape, filled with value.

```cpp
auto a = gtensor::full<int>({2,3},0);
auto b = gtensor::full<double, gtensor::config::f_order>(std::vector<int>{3,2},2);
std::cout<<std::endl<<a;    //[(2,3){{0,0,0},{0,0,0}}]
std::cout<<std::endl<<b;    //[(3,2){{2,2},{2,2},{2,2}}]
```

## zeros

```cpp
auto a = gtensor::zeros<int>({2,3});
auto b = gtensor::zeros<double, gtensor::config::f_order>(std::vector<int>{3,2});
std::cout<<std::endl<<a;    //[(2,3){{0,0,0},{0,0,0}}]
std::cout<<std::endl<<b;    //[(3,2){{0,0},{0,0},{0,0}}]
```

## ones

```cpp
auto a = gtensor::ones<int>({2,3});
auto b = gtensor::ones<double, gtensor::config::f_order>(std::vector<int>{3,2});
std::cout<<std::endl<<a;    //[(2,3){{1,1,1},{1,1,1}}]
std::cout<<std::endl<<b;    //[(3,2){{1,1},{1,1},{1,1}}]
```

## identity

Makes tensor of shape (n,n) with ones on main diagonal

```cpp
auto a = gtensor::identity<int>(2);
auto b = gtensor::identity<double, gtensor::config::f_order>(4);
std::cout<<std::endl<<a;    //[(2,2){{1,0},{0,1}}]
std::cout<<std::endl<<b;    //[(4,4){{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}]
```

## eye

Makes tensor of shape (n,m) with ones on kth diagonal.
k=0 refers to the main diagonal, k>0 refers to an upper diagonal, k<0 to lower diagonal.

```cpp
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename IdxT = int>
auto eye(const IdxT& n, const IdxT& m, const IdxT& k=0);
```

```cpp
auto a = gtensor::eye<double>(3,3);
auto b = gtensor::eye<double, gtensor::config::f_order>(4,3,-1);
auto c = gtensor::eye<double, gtensor::config::c_order>(4,3,1);
std::cout<<std::endl<<a;    //[(3,3){{1,0,0},{0,1,0},{0,0,1}}]
std::cout<<std::endl<<b;    //[(4,3){{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]
std::cout<<std::endl<<c;    //[(4,3){{0,1,0},{0,0,1},{0,0,0},{0,0,0}}]
```

## empty_like, full_like, zeros_like, ones_like

Makes tensor of the same layout, value_type and config as input tensor.
If no shape specified result has input tensor's shape

```cpp
template<typename ShT, typename...Ts> auto
empty_like(const basic_tensor<Ts...>& t, ShT&& shape);

template<typename...Ts>
auto empty_like(const basic_tensor<Ts...>& t);
```

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto empty = empty_like(t);
auto full = full_like(t,7, std::vector<int>{3,3});
auto zeros = zeros_like(t);
auto ones = ones_like(t);
std::cout<<std::endl<<empty;    //[(2,3){{6.46e+286,2.12e+285,2.25e+142},{0,0,0}}]
std::cout<<std::endl<<full;     //[(3,3){{7,7,7},{7,7,7},{7,7,7}}]
std::cout<<std::endl<<zeros;    //[(2,3){{0,0,0},{0,0,0}}]
std::cout<<std::endl<<ones;     //[(2,3){{1,1,1},{1,1,1}}]
```

## arange

Makes 1d tensor of evenly spaced values whithin a given interval.
Result's value_type, layout and config may be specified by explicit specialization of T,Order,Config template parameters.
If T is not specialized explicitly result value_type is infered from Start,Stop,Step types.

```cpp
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename Step=int>
auto arange(const Start& start, const Stop& stop, const Step& step=Step{1});

template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Stop>
auto arange(const Stop& stop);
```

```cpp
auto res1 = gtensor::arange(12);
auto res2 = gtensor::arange<std::size_t>(15);
auto res3 = gtensor::arange(1,20,1.5);
std::cout<<std::endl<<res1; //[(12){0,1,2,3,4,5,6,7,8,9,10,11}]
std::cout<<std::endl<<res2; //[(15){0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
std::cout<<std::endl<<res3; //[(13){1,2.5,4,5.5,7,8.5,10,11.5,13,14.5,16,17.5,19}]
```

## linspace

Makes tensor of num evenly spaced samples, calculated over the interval start, stop.
Start, stop may be scalar or tensor, if either is tensor samples will be along axis.
If start and stop are both tensors, they must be broadcastable.
Result's value_type, layout and config may be specified by explicit specialization of T,Order,Config template parameters.
If T is not specialized explicitly result value_type is infered from Start,Stop,Num types.
Result value_type is always floating-point type.

```cpp
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename Num=int, typename DimT=int>
auto linspace(const Start& start, const Stop& stop, const Num& num=50, bool end_point=true, const DimT& axis=0);
```

```cpp
auto res1 = gtensor::linspace(0,1,15);
auto res2 = gtensor::linspace(0,gtensor::tensor<double>{1,10,100},6);
auto res3 = gtensor::linspace(gtensor::tensor<double>{0,5,10},gtensor::tensor<double>{1,10,100},8,true,1);
std::cout<<std::endl<<res1; //[(15){0,0.0714,0.143,0.214,0.286,0.357,0.429,0.5,0.571,0.643,0.714,0.786,0.857,0.929,1}]
std::cout<<std::endl<<res2; //[(6,3){{0,0,0},{0.2,2,20},{0.4,4,40},{0.6,6,60},{0.8,8,80},{1,10,100}}]
std::cout<<std::endl<<res3; //[(3,8){{0,0.143,0.286,0.429,0.571,0.714,0.857,1},{5,5.71,6.43,7.14,7.86,8.57,9.29,10},{10,22.9,35.7,48.6,61.4,74.3,87.1,100}}]
```

## logspace

Makes tensor of numbers spaced evenly on a log scale.

```cpp
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename Num=int, typename Base=double,typename DimT=int>
auto logspace(const Start& start, const Stop& stop, const Num& num=50, bool end_point=true, const Base& base=10.0, const DimT& axis=0);
```

```cpp
auto res1 = gtensor::logspace(2,3,4);
auto res2 = gtensor::logspace(2,3,4,true,2);
auto res3 = gtensor::logspace(1,gtensor::tensor<double>{2,3,4},6);
std::cout<<std::endl<<str(res1,6);  //[(4){100,215.443,464.159,1000}]
std::cout<<std::endl<<str(res2,6);  //[(4){4,5.03968,6.3496,8}]
std::cout<<std::endl<<str(res3,6);  //[(6,3){{10,10,10},{15.8489,25.1189,39.8107},{25.1189,63.0957,158.489},{39.8107,158.489,630.957},{63.0957,398.107,2511.89},{100,1000,10000}}]
```

## geomspace

Makes tensor of numbers spaced evenly on a log scale with endpoints specified directly

```cpp
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename Num=int, typename DimT=int>
auto geomspace(const Start& start, const Stop& stop, const Num& num=50, bool end_point=true, const DimT& axis=0);
```

```cpp
auto res1 = gtensor::geomspace(1,1000,4);
auto res2 = gtensor::geomspace(1,gtensor::tensor<double>{2,10,100},6);
std::cout<<std::endl<<res1;  //[(4){1,10,100,1e+03}]
std::cout<<std::endl<<res2;  //[(6,3){{1,1,1},{1.15,1.58,2.51},{1.32,2.51,6.31},{1.52,3.98,15.8},{1.74,6.31,39.8},{2,10,100}}]
```

## diag

If t is 2d tensor returns 1d tensor that is t's kth diagonal.
If t is 1d tensor returns 2d square tensor with t on its kth diagonal.
k=0 refers to the main diagonal, k>0 refers to an upper diagonal, k<0 to lower diagonal.

```cpp
template<typename IdxT=int, typename...Ts>
auto diag(const basic_tensor<Ts...>& t, const IdxT& k=0);
```

```cpp
auto res1 = diag(gtensor::tensor<double>{{1,2,3,4},{5,6,7,8},{9,10,11,12}});
auto res2 = diag(gtensor::tensor<double>{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1);
auto res3 = diag(gtensor::tensor<double>{{1,2,3,4},{5,6,7,8},{9,10,11,12}},-1);
auto res4 = diag(gtensor::tensor<double>{1,2,3});
auto res5 = diag(gtensor::tensor<double>{1,2,3},1);
auto res6 = diag(gtensor::tensor<double>{1,2,3},-1);
std::cout<<std::endl<<res1; //[(3){1,6,11}]
std::cout<<std::endl<<res2; //[(3){2,7,12}]
std::cout<<std::endl<<res3; //[(2){5,10}]
std::cout<<std::endl<<res4; //[(3,3){{1,0,0},{0,2,0},{0,0,3}}]
std::cout<<std::endl<<res5; //[(4,4){{0,1,0,0},{0,0,2,0},{0,0,0,3},{0,0,0,0}}]
std::cout<<std::endl<<res6; //[(4,4){{0,0,0,0},{1,0,0,0},{0,2,0,0},{0,0,3,0}}]
```