# Statistic

**statistic** module resides in **statistic.hpp** header.

## mean

Computes the arithmetic mean along the specified axes.

```cpp
template<typename...Ts, typename Axes>
auto mean(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false);
template<typename...Ts, typename DimT>
auto mean(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false);
template<typename...Ts>
auto mean(const basic_tensor<Ts...>& t, bool keep_dims = false);
//parallel version
template<typename Policy, typename...Ts, typename Axes>
auto mean(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false);
template<typename Policy, typename...Ts, typename DimT>
auto mean(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false);
template<typename Policy, typename...Ts>
auto mean(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false);
```

`axes` can be scalar or container. If no axes specified computations is performed like over flatten input.
Negative values are allowed and mean counting from the end.

`keep_dims` is `false` by default. If `true` is passed, the axes which are reduced are left in the result as dimensions with size one.

`policy` type is specialization of `multithreading::exec_pol`.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = mean(t);
auto res2 = mean(t,0);
auto res3 = mean(t,{0,1},true);
auto res4 = mean(multithreading::exec_pol<4>{},t,2);
std::cout<<std::endl<<res1; //[(){3.3}]
std::cout<<std::endl<<res2; //[(2,5){{3.5,3.5,0.5,4,2.5},{2.5,1.5,5.5,5.5,4}}]
std::cout<<std::endl<<res3; //[(1,1,5){{{3,2.5,3,4.75,3.25}}}]
std::cout<<std::endl<<res4; //[(2,2){{2,3.6},{3.6,4}}]
```

## var

Computes the variance along the specified axes. Parameters are the same as for `mean`.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = var(t);
auto res2 = var(t,0);
auto res3 = var(t,{0,1},true);
auto res4 = var(multithreading::exec_pol<4>{},t,2);
std::cout<<std::endl<<res1; //[(){4.21}]
std::cout<<std::endl<<res2; //[(2,5){{0.25,6.25,0.25,4,0.25},{2.25,0.25,2.25,2.25,1}}]
std::cout<<std::endl<<res3; //[(1,1,5){{{1.5,4.25,7.5,3.69,1.19}}}]
std::cout<<std::endl<<res4; //[(2,2){{2,5.44},{4.24,2.8}}]
```

## stdev

Computes the standart deviation along the specified axes. Parameters are the same as for `mean`.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = stdev(t);
auto res2 = stdev(t,0);
auto res3 = stdev(t,{0,1},true);
auto res4 = stdev(multithreading::exec_pol<4>{},t,2);
std::cout<<std::endl<<res1; //[(){2.05}]
std::cout<<std::endl<<res2; //[(2,5){{0.5,2.5,0.5,2,0.5},{1.5,0.5,1.5,1.5,1}}]
std::cout<<std::endl<<res3; //[(1,1,5){{{1.22,2.06,2.74,1.92,1.09}}}]
std::cout<<std::endl<<res4; //[(2,2){{1.41,2.33},{2.06,1.67}}]
```

## ptp

Computes the range of values (maximum-minimum) along the specified axes. Parameters are the same as for `mean`.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = ptp(t);
auto res2 = ptp(t,0);
auto res3 = ptp(t,{0,1},true);
auto res4 = ptp(multithreading::exec_pol<4>{},t,2);
std::cout<<std::endl<<res1; //[(){7}]
std::cout<<std::endl<<res2; //[(2,5){{1,5,1,4,1},{3,1,3,3,2}}]
std::cout<<std::endl<<res3; //[(1,1,5){{{3,5,7,5,3}}}]
std::cout<<std::endl<<res4; //[(2,2){{4,6},{5,5}}]
```

## median

Computes the median along the specified axes. Parameters are the same as for `mean`.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = median(t);
auto res2 = median(t,0);
auto res3 = median(t,{0,1},true);
auto res4 = median(multithreading::exec_pol<4>{},t,2);
std::cout<<std::endl<<res1; //[(){3}]
std::cout<<std::endl<<res2; //[(2,5){{3.5,3.5,0.5,4,2.5},{2.5,1.5,5.5,5.5,4}}]
std::cout<<std::endl<<res3; //[(1,1,5){{{3.5,1.5,2.5,5,3}}}]
std::cout<<std::endl<<res4; //[(2,2){{2,4},{3,4}}]
```

## quantile

Computes the q-th quantile along the specified axes. Parameters are the same as for `mean` except additional `q` parameter, that is floating-point quantile to compute.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = quantile(t,0.5);
auto res2 = quantile(t,0,0.5);
auto res3 = quantile(t,{0,1},0.5,true);
auto res4 = quantile(multithreading::exec_pol<4>{},t,2,0.5);
std::cout<<std::endl<<res1; //[(){3}]
std::cout<<std::endl<<res2; //[(2,5){{3.5,3.5,0.5,4,2.5},{2.5,1.5,5.5,5.5,4}}]
std::cout<<std::endl<<res3; //[(1,1,5){{{3.5,1.5,2.5,5,3}}}]
std::cout<<std::endl<<res4; //[(2,2){{2,4},{3,4}}]
```

## nanmean, nanvar, nanstdev, nanmedian, nanquantile

Nan ignoring counterparts of statistic routines described above.

## average

Computes the weighted average along the specified axes. Axes may be scalar or container.

```cpp
template<typename...Ts, typename Axes, typename Container>
auto average(const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false);
template<typename...Ts, typename DimT, typename Container>
auto average(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Container& weights, bool keep_dims=false);
template<typename...Ts, typename Container>
auto average(const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false);
//parallel version
template<typename Policy, typename...Ts, typename Axes, typename Container>
auto average(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false);
template<typename Policy, typename...Ts, typename DimT, typename Container>
auto average(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Container& weights, bool keep_dims=false);
template<typename Policy, typename...Ts, typename Container>
auto average(Policy policy, const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false);
```

`weights` is container, size of weights must be equal to size along given axes, weights must not sum to zero.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = average(t,std::vector<double>{1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1});
auto res2 = average(t,2,std::vector<double>{1,2,3,2,1});
std::cout<<std::endl<<res1; //[(){3.4}]
std::cout<<std::endl<<res2; //[(2,2){{1.44,3.78},{3.56,4.44}}]
```

## moving average

Computes the weighted moving average along the specified axis. Axis is scalar.

```cpp
template<typename...Ts, typename DimT, typename Container, typename IdxT>
auto moving_average(const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step);
template<typename...Ts, typename Container, typename IdxT>
auto moving_average(const basic_tensor<Ts...>& t, const Container& weights, const IdxT& step);
//parallel version
template<typename Policy, typename...Ts, typename DimT, typename Container, typename IdxT>
auto moving_average(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step);
template<typename Policy, typename...Ts, typename Container, typename IdxT>
auto moving_average(Policy policy, const basic_tensor<Ts...>& t, const Container& weights, const IdxT& step);
```

`weights` is container, moving window size is weights size, weights must not sum to zero.

`step` is moving window step along axis.

Result axis size will be (n - window_size)/step + 1, where n is input's axis size.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = moving_average(t,std::vector<double>{1,3,1},1);
auto res2 = moving_average(t,2,std::vector<double>{1,3,1},1);
std::cout<<std::endl<<res1; //[(18){1.4,0.6,1.8,2.4,1.4,1.6,4,6,5,4,4.4,3,4.2,3.2,3.2,3.4,5.4,4.4}]
std::cout<<std::endl<<res2; //[(2,2,3){{{1.4,0.6,1.8},{1.6,4,6}},{{4.4,3,4.2},{3.4,5.4,4.4}}}]
```

## moving mean

Computes the moving arithmetic mean along the specified axis. Axis is scalar.

```cpp
template<typename...Ts, typename DimT, typename IdxT>
auto moving_mean(const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& window_size, const IdxT& step);
template<typename...Ts, typename IdxT>
auto moving_mean(const basic_tensor<Ts...>& t, const IdxT& window_size, const IdxT& step);
//parallel version
template<typename Policy, typename...Ts, typename DimT, typename IdxT>
auto moving_mean(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& window_size, const IdxT& step);
template<typename Policy, typename...Ts, typename IdxT>
auto moving_mean(Policy policy, const basic_tensor<Ts...>& t, const IdxT& window_size, const IdxT& step);
```

`window_size` must be greater than zero and less or equal than axis size.

`step` is moving window step along axis.

Result axis size will be (n - window_size)/step + 1, where n is input's axis size.

```cpp
gtensor::tensor<double> t{{{4,1,0,2,3},{1,1,4,7,5}},{{3,6,1,6,2},{4,2,7,4,3}}};
auto res1 = moving_mean(t,3,1);
auto res2 = moving_mean(t,2,3,1);
std::cout<<std::endl<<res1; //[(18){1.67,1,1.67,2,1.67,2,4,5.33,5,4.67,3.33,4.33,3,4,2.67,4.33,4.33,4.67}]
std::cout<<std::endl<<res2; //[(2,2,3){{{1.67,1,1.67},{2,4,5.33}},{{3.33,4.33,3},{4.33,4.33,4.67}}}]
```

## histogram

Computes the histogram of given tensor elements. The histogram is computed over the flattened tensor.

```cpp
enum class histogram_algorithm : std::size_t {automatic,fd,scott,rice,sturges,sqrt};

template<typename...Ts, typename Bins, typename Range, typename Weights>
auto histogram(const basic_tensor<Ts...>& t, const Bins& bins, const Range& range, bool density, const Weights& weights);
template<typename...Ts, typename Bins=int>
auto histogram(const basic_tensor<Ts...>& t, const Bins& bins=10, bool density=false);
```

`bins` can be integral type or histogram_algorithm enum or container with at least bidirectional iterator.
- when `bins` is of integral type it means equal width bins number, `bins` must be > 0
- when bins is of `histogram_algorithm` type equal width bins number is calculated according to algorithm
- when bins is container its elements mean bins edges and must increase monotonically

`range` is the lower and upper range of the bins. Must be std::pair like type, `range.first` must be less or equal than `range.second`.

`density`
- when `false` the result will contain the number of samples in each bin
- when `true`, the result is the value of the probability density function at the bin, normalized such that the integral over the `range` is 1.

`weights` is tensor of same shape as `t`. Each value in `weights` only contributes its associated weight towards the bin count (instead of 1).

```cpp
gtensor::tensor<double> gauss_100{-1.98,0.71,2.6,-0.02,0.03,0.18,-1.86,0.43,-1.61,-0.43,1.24,-0.74,0.5,1.01,0.28,-1.37,
    -0.33,1.96,-2.03,-0.28,-0.55,0.12,0.75,1.61,-0.27,0.81,0.5,0.47,-0.56,-1.0,-1.1,-0.76,0.32,0.76,0.32,-0.55,1.81,1.52,
    -0.35,-0.82,0.13,1.27,0.33,0.56,-0.21,0.46,1.54,-0.24,0.14,0.25,0.28,-1.41,-1.88,-1.02,0.17,0.55,-0.53,1.38,-0.14,0.02,
    -0.19,0.13,0.7,0.67,-0.9,1.52,-1.1,0.08,-0.27,-1.05,-0.08,-0.74,0.07,0.4,1.47,0.31,-0.61,-0.39,0.14,0.09,1.46,1.4,-0.36,
    -0.55,-2.56,-0.55,-0.98,-0.35,0.39,0.18,-0.03,0.2,-0.13,0.2,-3.23,-0.27,-0.11,-0.34,-0.22,0.7
};
auto res1 = histogram(gauss_100);
auto res2 = histogram(gauss_100,gtensor::histogram_algorithm::automatic,true);
std::cout<<std::endl<<"bins1 "<<res1.first;     //bins1 [(10){1,1,5,9,18,30,22,5,8,1}]
std::cout<<std::endl<<"edges1 "<<res1.second;   //edges1 [(11){-3.23,-2.65,-2.06,-1.48,-0.898,-0.315,0.268,0.851,1.43,2.02,2.6}]
std::cout<<std::endl<<"bins2 "<<res2.first;     //bins2 [(14){0.024,0.024,0.024,0.0961,0.048,0.264,0.336,0.456,0.576,0.216,0.072,0.192,0.048,0.024}]
std::cout<<std::endl<<"edges2 "<<res2.second;   //edges2 [(15){-3.23,-2.81,-2.4,-1.98,-1.56,-1.15,-0.731,-0.315,0.101,0.518,0.934,1.35,1.77,2.18,2.6}]
```

## cov

Estimate a covariance matrix.

```cpp
template<typename...Ts>
auto cov(const basic_tensor<Ts...>& t, bool rowvar = true);
//parallel version
template<typename Policy, typename...Ts>
auto cov(Policy policy, const basic_tensor<Ts...>& t, bool rowvar = true);
```

`t` should be 1d or 2d tensor containing variables and observations.

If `rowvar` is `true`, then each row of `t` represents a variable with observations in the columns.
If `rowvar` is `false`, then each column of `t` represents a variable with observations in the rows.

```cpp
gtensor::tensor<double> a{{1,2,3},{3,2,1},{3,0,1}};
std::cout<<std::endl<<cov(a);   //[(3,3){{1,-1,-1},{-1,1,1},{-1,1,2.33}}]
std::cout<<std::endl<<cov(a.transpose(),false); //[(3,3){{1,-1,-1},{-1,1,1},{-1,1,2.33}}]
```

## corrcoef

Estimate Pearson product-moment correlation coefficients.

```cpp
template<typename...Ts>
auto corrcoef(const basic_tensor<Ts...>& t, bool rowvar = true);
//parallel version
template<typename Policy, typename...Ts>
auto corrcoef(Policy policy, const basic_tensor<Ts...>& t, bool rowvar = true);
```

The meaning of parameters is the same as for `cov`.

```cpp
gtensor::tensor<double> a{{1,2,3},{3,2,1},{3,0,1}};
std::cout<<std::endl<<corrcoef(a);  //[(3,3){{1,-1,-0.655},{-1,1,0.655},{-0.655,0.655,1}}]
std::cout<<std::endl<<corrcoef(a.transpose(),false);    //[(3,3){{1,-1,-0.655},{-1,1,0.655},{-0.655,0.655,1}}]
```