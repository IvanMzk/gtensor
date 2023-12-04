# Tensor math

**tensor_math** module resides in **tensor_math.hpp** header. It contains two kinds of routines:
- **math broadcast routines** - their bihaviour similar to tensor operators: they are lazy, return **expression view** object,
arguments can be tensor or scalar and must be broadcastable.
- **math reduce routines** like sum,max,prod, which perform reduction along given axis or axes, they are not lazy.

## Math broadcast routines

Most of routines in this section implemented using appropriate function from Standart Library to operate on tensors elements.

### basic

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<double> t2{3,2,1};
auto res1 = abs(t1);
auto res2 = fmod(t1,t2);
auto res3 = remainder(t1,t2);
auto res4 = fma(t1,t2,1);
auto res5 = fmax(t1,t2);
auto res6 = fmin(t1,t2);
auto res7 = fdim(t1,t2);
auto res8 = clip(t1,t2,5);  //clip (limit) the values in tensor
auto res9 = divmod(t1,t2);  //returns tensor of pairs of quotient and remainder
```

### exponential

```cpp
gtensor::tensor<double> t{1,2,3,4,5};
auto res1 = exp(t);
auto res2 = exp2(t);
auto res3 = expm1(t);
auto res4 = log(t);
auto res5 = log10(t);
auto res6 = log2(t);
auto res7 = log1p(t);
```

### power

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<double> t2{3,2,1};
auto res1 = pow(t1,t2);
auto res2 = sqrt(t1);
auto res3 = cbrt(t1);
auto res4 = hypot(t1,t2);
```

### trigonometric

```cpp
gtensor::tensor<double> t{0.1,0.2,0.3,0.4,0.5};
auto res1 = sin(t);
auto res2 = cos(t);
auto res3 = tan(t);
auto res4 = asin(t);
auto res5 = acos(t);
auto res6 = atan(t);
auto res7 = atan2(t);
```

### hyperbolic

```cpp
gtensor::tensor<double> t{0.1,0.2,0.3,0.4,0.5};
auto res1 = sinh(t);
auto res2 = cosh(t);
auto res3 = tanh(t);
auto res4 = asinh(t);
auto res5 = acosh(t);
auto res6 = atanh(t);
```

### nearest

```cpp
gtensor::tensor<double> t{1.1,2.2,3.3,4.4,5.5};
auto res1 = ceil(t);
auto res2 = floor(t);
auto res3 = trunc(t);
auto res4 = round(t);
auto res5 = nearbyint(t);
auto res6 = rint(t);
```

### floating point manipulation

```cpp
gtensor::tensor<double> t1{{1.1,2.2,3.3},{4.4,5.5,6.6}};
gtensor::tensor<double> t2{0.1,0.2,0.3};
auto res1 = frexp(t);   //decomposes a number into significand and a power of 2, returns tensor of pairs
auto res2 = ldexp(t1,t2);
auto res3 = modf(); //decomposes a number into integer and fractional parts, return tensor of pairs
auto res4 = nextafter(t1,t2);
auto res5 = copysign(t1);
auto res6 = nan_to_num(t1);   //by default replaces nan with zero, pos_infinity with large finite positive number and neg_infinity with small finite negative number
auto res7 = nan_to_num(t1,0,1E50,-1E50);   //numbers to replace nan, pos_infinity and neg_infinity can be specified explicitly
```

### classification

Result is tensor of bools.

```cpp
gtensor::tensor<double> t{1.1,2.2,3.3,4.4,5.5};
auto res1 = isfinite(t);
auto res2 = isinf(t);
auto res3 = isnan(t);
auto res4 = isnormal(t);
```
### comparison

Result is tensor of bools.

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<double> t2{3,2,1};
auto res1 = isgreater(t1,t2);
auto res2 = isgreaterequal(t1,t2);
auto res3 = isless(t1,t2);
auto res4 = islessequal(t1,t2);
auto res5 = islessgreater(t1,t2);
```

### functions in rational domain

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<double> t2{3,2,1};
auto res1 = gcd(t1,t2);
auto res2 = lcm(t1,t2);
```

### complex numbers

```cpp
using namespace std::complex_literals;
gtensor::tensor<std::complex<double>> t{{1.1+2.2i,2.2+3.3i},{3.3+4.4i,4.4+1.1i}};
auto res1 = real(t);
auto res2 = imag(t);
auto res3 = conj(t);
auto res4 = angle(t);
```

## Math reduce routines

Routines described in this section are not lazy. They perform reduction along axes.
**Axes** argument can be container or scalar, negative axes supported and mean counting from end.
There is version of each routine that doesn't take `axes` argument and make reduction like over flatten tensor.

Each routine has optional bool parameter `keep_dims`.
It is `false` by default. If `true` is passed, the axes which are reduced are left in the result as dimensions with size one.

There is parallel version of each routine that takes specialization of `multithreading::exec_pol` class template as its first parameter.
Integral constant in specialization means number of tasks original task is diveded into.

```cpp
gtensor::tensor<double> t(1000000,1.0);
auto res = sum(multithreading::exec_pol<8>{},t);    //will try to sum tensor elements using 8 threads
```

### sum

Sum of elements along given axes, axes may be scalar or container.

```cpp
template<typename...Ts, typename Axes, typename Initial = gtensor::detail::no_value>
auto sum(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false, const Initial& initial = Initial{});
template<typename...Ts, typename DimT, typename Initial = gtensor::detail::no_value>
auto sum(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false, const Initial& initial = Initial{});
//sum like over flatten
template<typename...Ts, typename Initial = gtensor::detail::no_value>
auto sum(const basic_tensor<Ts...>& t, bool keep_dims = false, const Initial& initial = Initial{});

//parallell versions
template<typename Policy, typename...Ts, typename Axes, typename Initial = gtensor::detail::no_value>
auto sum(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false, const Initial& initial = Initial{});
template<typename Policy, typename...Ts, typename DimT, typename Initial = gtensor::detail::no_value>
auto sum(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false, const Initial& initial = Initial{});
template<typename Policy, typename...Ts, typename Initial = gtensor::detail::no_value>
auto sum(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false, const Initial& initial = Initial{});
```

```cpp
gtensor::tensor<double> t{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}};
auto res1 = sum(t); //sum like over flatten
auto res2 = sum(t,0);   //sum along axis 0
auto res3 = sum(t,{1,2});   //sum along axes {1,2}
auto res4 = sum(t,{1,2},true,-1);   //sum along axes {1,2}, with keep_dims=true and initial=-1
auto res5 = sum(multithreading::exec_pol<4>{},t);   //sum like over flatten, execution in parallel requested
std::cout<<std::endl<<res1; //[(){171}]
std::cout<<std::endl<<res2; //[(2,3){{21,24,27},{30,33,36}}]
std::cout<<std::endl<<res3; //[(3){21,57,93}]
std::cout<<std::endl<<res4; //[(3,1,1){{{20}},{{56}},{{92}}}]
std::cout<<std::endl<<res5; //[(){171}]
```

### prod

Product of elements along given axes, axes may be scalar or container.
Has the same parameters as `sum`.

```cpp
gtensor::tensor<double> t{{{0.1,0.2,0.3},{0.4,0.5,0.6}},{{0.7,0.8,0.9},{1.0,1.1,1.2}},{{1.3,1.4,1.5},{1.6,1.7,1.8}}};
auto res1 = prod(t); //prod like over flatten
auto res2 = prod(t,0);   //prod along axis 0
auto res3 = prod(t,{1,2});   //prod along axes {1,2}
auto res4 = prod(t,{1,2},true,-1.0);   //prod along axes {1,2}, with keep_dims=true and initial=-1
auto res5 = prod(multithreading::exec_pol<4>{},t);   //prod like over flatten, execution in parallel requested
std::cout<<std::endl<<res1; //[(){0.0064}]
std::cout<<std::endl<<res2; //[(2,3){{0.091,0.224,0.405},{0.64,0.935,1.3}}]
std::cout<<std::endl<<res3; //[(3){0.00072,0.665,13.4}]
std::cout<<std::endl<<res4; //[(3,1,1){{{-0.00072}},{{-0.665}},{{-13.4}}}]
std::cout<<std::endl<<res5; //[(){0.0064}]
```

### min, amin

Minimum of elements along given axes, axes may be scalar or container.
Has the same parameters as `sum`.

```cpp
gtensor::tensor<double> t{{{0.1,0.2,0.3},{0.4,0.5,0.6}},{{0.7,0.8,0.9},{1.0,1.1,1.2}},{{1.3,1.4,1.5},{1.6,1.7,1.8}}};
auto res1 = gtensor::min(t); //min like over flatten
auto res2 = gtensor::min(t,0);   //min along axis 0
auto res3 = gtensor::min(t,{1,2});   //min along axes {1,2}
auto res4 = gtensor::min(t,{1,2},true,1.0);   //min along axes {1,2}, with keep_dims=true and initial=1.0
auto res5 = gtensor::min(multithreading::exec_pol<4>{},t);   //min like over flatten, execution in parallel requested
std::cout<<std::endl<<res1; //[(){0.1}]
std::cout<<std::endl<<res2; //[(2,3){{0.1,0.2,0.3},{0.4,0.5,0.6}}]
std::cout<<std::endl<<res3; //[(3){0.1,0.7,1.3}]
std::cout<<std::endl<<res4; //[(3,1,1){{{0.1}},{{0.7}},{{1}}}]
std::cout<<std::endl<<res5; //[(){0.1}]
```

### max, amax

Maximum of elements along given axes, axes may be scalar or container.
Has the same parameters as `sum`.

```cpp
gtensor::tensor<double> t{{{0.1,0.2,0.3},{0.4,0.5,0.6}},{{0.7,0.8,0.9},{1.0,1.1,1.2}},{{1.3,1.4,1.5},{1.6,1.7,1.8}}};
auto res1 = gtensor::max(t); //max like over flatten
auto res2 = gtensor::max(t,0);   //max along axis 0
auto res3 = gtensor::max(t,{1,2});   //max along axes {1,2}
auto res4 = gtensor::max(t,{1,2},true,1.0);   //max along axes {1,2}, with keep_dims=true and initial=1.0
auto res5 = gtensor::max(multithreading::exec_pol<4>{},t);   //max like over flatten, execution in parallel requested
std::cout<<std::endl<<res1; //[(){1.8}]
std::cout<<std::endl<<res2; //[(2,3){{1.3,1.4,1.5},{1.6,1.7,1.8}}]
std::cout<<std::endl<<res3; //[(3){0.6,1.2,1.8}]
std::cout<<std::endl<<res4; //[(3,1,1){{{1}},{{1.2}},{{1.8}}}]
std::cout<<std::endl<<res5; //[(){1.8}]
```

### all

Test if all elements along given axes evaluate to true, axes may be scalar or container.
Result is tensor of bools.
Has the same parameters as `sum` except initial;

```cpp
gtensor::tensor<double> t{{{1,2,3},{4,5,6}},{{7,8,0},{10,11,12}},{{13,0,15},{16,17,0}}};
auto res1 = all(t); //all like over flatten
auto res2 = all(t,0);   //all along axis 0
auto res3 = all(t,{1,2});   //all along axes {1,2}
auto res4 = all(multithreading::exec_pol<4>{},t);   //all like over flatten, execution in parallel requested
std::cout<<std::endl<<res1; //[(){0}]
std::cout<<std::endl<<res2; //[(2,3){{1,0,0},{1,1,0}}]
std::cout<<std::endl<<res3; //[(3){1,0,0}]
std::cout<<std::endl<<res4; //[(){0}]
```

### any

Test if any elements along given axes evaluate to true, axes may be scalar or container.
Result is tensor of bools.
Has the same parameters as `sum` except initial;

```cpp
gtensor::tensor<double> t{{{1,2,3},{4,5,6}},{{0,0,0},{0,0,0}},{{13,0,15},{16,17,0}}};
auto res1 = any(t); //any like over flatten
auto res2 = any(t,0);   //any along axis 0
auto res3 = any(t,{1,2});   //any along axes {1,2}
auto res4 = any(multithreading::exec_pol<4>{},t);   //any like over flatten, execution in parallel requested
std::cout<<std::endl<<res1; //[(){1}]
std::cout<<std::endl<<res2; //[(2,3){{1,1,1},{1,1,1}}]
std::cout<<std::endl<<res3; //[(3){1,0,1}]
std::cout<<std::endl<<res4; //[(){1}]
```

### nansum, nanprod, nanmin, nanmax

Nan versions of routines differ from their non-nan counterparts in a way nan elements are processed:
- `nansum` treats nans as 0
- `nanprod` treats nans as 1
- `nanmin` and `nanmax` ignoring nan elements

## Math cumulate routines

Cumulative routines not lazy. Axis along which cumulative operation is performed is scalar.
Negative axis allowed. If no axis provided operation performed like over flatten tensor.

### cumsum

Cumulative sum of elements along given axis, axis is scalar.

```cpp
template<typename...Ts, typename DimT>
auto cumsum(const basic_tensor<Ts...>& t, const DimT& axis);
//cumsum like over flatten
template<typename...Ts>
auto cumsum(const basic_tensor<Ts...>& t);

//parallel versions
template<typename Policy, typename...Ts, typename DimT>
auto cumsum(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis);
template<typename Policy, typename...Ts>
auto cumsum(Policy policy, const basic_tensor<Ts...>& t);
```

```cpp
gtensor::tensor<double> t{{{0.1,0.2,0.3},{0.4,0.5,0.6}},{{0.7,0.8,0.9},{1.0,1.1,1.2}},{{1.3,1.4,1.5},{1.6,1.7,1.8}}};
auto res1 = cumsum(t); //cumsum like over flatten
auto res2 = cumsum(t,0);   //cumsum along axis 0
auto res3 = cumsum(multithreading::exec_pol<4>{},t,2);   //cumsum along axis 2, execution in parallel requested
std::cout<<std::endl<<res1; //[(18){0.1,0.3,0.6,1,1.5,2.1,2.8,3.6,4.5,5.5,6.6,7.8,9.1,10.5,12,13.6,15.3,17.1}]
std::cout<<std::endl<<res2; //[(3,2,3){{{0.1,0.2,0.3},{0.4,0.5,0.6}},{{0.8,1,1.2},{1.4,1.6,1.8}},{{2.1,2.4,2.7},{3,3.3,3.6}}}]
std::cout<<std::endl<<res3; //[(3,2,3){{{0.1,0.3,0.6},{0.4,0.9,1.5}},{{0.7,1.5,2.4},{1,2.1,3.3}},{{1.3,2.7,4.2},{1.6,3.3,5.1}}}]
```

### cumprod

Cumulative product of elements along given axis, axis is scalar.

```cpp
gtensor::tensor<double> t{{{0.1,0.2,0.3},{0.4,0.5,0.6}},{{0.7,0.8,0.9},{1.0,1.1,1.2}},{{1.3,1.4,1.5},{1.6,1.7,1.8}}};
auto res1 = cumprod(t); //cumprod like over flatten
auto res2 = cumprod(t,0);   //cumprod along axis 0
auto res3 = cumprod(multithreading::exec_pol<4>{},t,2);   //cumprod along axis 2, execution in parallel requested
std::cout<<std::endl<<res1; //[(18){0.1,0.02,0.006,0.0024,0.0012,0.00072,0.000504,0.000403,0.000363,0.000363,0.000399,0.000479,0.000623,0.000872,0.00131,0.00209,0.00356,0.0064}]
std::cout<<std::endl<<res2; //[(3,2,3){{{0.1,0.2,0.3},{0.4,0.5,0.6}},{{0.07,0.16,0.27},{0.4,0.55,0.72}},{{0.091,0.224,0.405},{0.64,0.935,1.3}}}]
std::cout<<std::endl<<res3; //[(3,2,3){{{0.1,0.02,0.006},{0.4,0.2,0.12}},{{0.7,0.56,0.504},{1,1.1,1.32}},{{1.3,1.82,2.73},{1.6,2.72,4.9}}}]
```

### nancumsum, nancumprod

Nan versions of routines differ from their non-nan counterparts in a way nan elements are processed:
- `nancumsum` treats nans as 0
- `nancumprod` treats nans as 1


## Other routines

### diff, diff2

`diff` computes n-th difference along given axis.
Axis is scalar, default is last axis.

`diff2` is none recursive implementation of second differences, more efficient than `diff` with n=2.

```cpp
template<typename...Ts, typename DimT=int>
auto diff(const basic_tensor<Ts...>& t, std::size_t n = 1, const DimT& axis = -1);
//none recursive implementation of second differences, more efficient than diff with n=2
template<typename Policy, typename...Ts, typename DimT=int>
auto diff2(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis = -1);

//parallel version
template<typename Policy, typename...Ts, typename DimT=int>
auto diff(Policy policy, const basic_tensor<Ts...>& t, std::size_t n = 1, const DimT& axis = -1);
template<typename Policy, typename...Ts, typename DimT=int>
auto diff2(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis = -1);
```

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3},{2,2,0,1,2}};
auto res1 = diff(t);    //first differences along axis 1
auto res2 = diff(t,1,0);    //first differences along axis 0
auto res3 = diff2(multithreading::exec_pol<4>{}, t);    //second differences, parallel execution requested
std::cout<<std::endl<<res1; //[(5,4){{-3,-1,2,1},{0,3,3,-2},{3,-5,5,-4},{-2,5,-3,-1},{0,-2,1,1}}]
std::cout<<std::endl<<res2; //[(4,5){{-3,0,4,5,2},{2,5,-3,-1,-3},{1,-4,6,-2,1},{-2,0,-7,-3,-1}}]
std::cout<<std::endl<<res3; //[(5,3){{2,3,-1},{3,0,-5},{-8,10,-9},{7,-8,2},{-2,3,0}}]
```

### gradient

Gradient along given axis, interior points has 2-nd order accuracy approximation using central difference, boundary points has 1-st order accuracy approximation.
Axis is scalar, spacing is scalar or container, scalar means uniform sample distance, container specifies coordinates along dimension.
If spacing is container it must be the same size as size along axis. Spacing is 1 by default.

```cpp
template<typename...Ts, typename DimT, typename Spacing>
auto gradient(const basic_tensor<Ts...>& t, const DimT& axis, const Spacing& spacing);
template<typename...Ts, typename DimT>
auto gradient(const basic_tensor<Ts...>& t, const DimT& axis);


template<typename Policy, typename...Ts, typename DimT, typename Spacing>
auto gradient(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const Spacing& spacing);
template<typename Policy, typename...Ts, typename DimT>
auto gradient(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis);
```

```cpp
gtensor::tensor<double> t{{4,1,0,2,3},{1,1,4,7,5},{3,6,1,6,2},{4,2,7,4,3},{2,2,0,1,2}};
auto res1 = gradient(t,1);  //gradient along axis 1, uniform sample distance 1
auto res2 = gradient(t,0,std::vector<double>{1,3,4,5,7});  //gradient along axis 0, points coordinates in container
auto res3 = gradient(multithreading::exec_pol<4>{},t,1);  //parallel execution requested
std::cout<<std::endl<<res1; //[(5,5){{-3,-2,0.5,1.5,1},{0,1.5,3,0.5,-2},{3,-1,0,0.5,-4},{-2,1.5,1,-2,-1},{0,-1,-0.5,1,1}}]
std::cout<<std::endl<<res2; //[(5,5){{-1.5,0,2,2.5,1},{0.833,3.33,-1.33,0.167,-1.67},{1.5,0.5,1.5,-1.5,-1},{0.333,-2.67,2.83,-1.83,0.5},{-1,0,-3.5,-1.5,-0.5}}]
std::cout<<std::endl<<res3; //[(5,5){{-3,-2,0.5,1.5,1},{0,1.5,3,0.5,-2},{3,-1,0,0.5,-4},{-2,1.5,1,-2,-1},{0,-1,-0.5,1,1}}]
```

### matmul

Matrix product of two tensors. Implementation is optimized for single and double precision floating point data type as well as `std::complex` specialization
for such types.

```cpp
template<typename...Ts,typename...Us>
auto matmul(const basic_tensor<Ts...>& a, const basic_tensor<Us...>& b);
//parallel version
template<typename Policy, typename...Ts,typename...Us>
auto matmul(Policy policy, const basic_tensor<Ts...>& a, const basic_tensor<Us...>& b);
```
The behavior depends on the arguments in the following way:
- if both arguments are 2d they are multiplied like conventional matrices.
- if either argument is nd, n>2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
- if the first argument is 1d, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
- if the second argument is 1d, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
gtensor::tensor<double> b{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}};
auto res1 = matmul(a,b);
auto res2 = matmul(a,b(1));
auto res3 = matmul(a,b.flatten()({{0,3}}));
std::cout<<std::endl<<res1; //[(2,2,2){{{22,28},{49,64}},{{58,64},{139,154}}}]
std::cout<<std::endl<<res2; //[(2,2){{58,64},{139,154}}]
std::cout<<std::endl<<res3; //[(2){14,32}]
```
