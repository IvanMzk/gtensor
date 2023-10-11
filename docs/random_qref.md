# Random

`random` module resides in `random.hpp` header. It defines **pseudo random number generator (PRNG)** class template which incapsulates random bits generator object
and member functions to draw samples from probability distributions as well as random permutations member functions.

GTensor library provides routines to create **PRNG** object and initialize its **random bits gerator** with specified **seed** values as well as random seed values.

## rng

Returns PRNG object. Type of **random bits generator** must be specified explicitly.

```cpp
template<typename BitGenerator, typename Config=config::default_config, typename...Seeds>
auto rng(const Seeds&...seeds);

//returns random number generator object initialized with random seeds, new for each call
template<typename BitGenerator, typename Config=config::default_config>
auto rng();
```

`BitGenerator` type template parameter should satisfy **UniformRandomBitGenerator** and **RandomNumberEngine** named requirements and must be specified explicitly.
It is type of **random bits generator** used by **PRNG**.

`seeds` should be integral numbers to initialize `BitGenerator` state.

```cpp
auto prng1 = gtensor::rng<std::minstd_rand>(1,2,3); //create PRNG using std::minstd_rand random bits generator and seeds {1,2,3}
auto prng2 = gtensor::rng<std::ranlux48>(); //create PRNG using std::ranlux48 random bits generator and random seeds
std::cout<<std::endl<<prng1.integers(0,5,10);   //[(10){2,0,0,4,4,0,2,4,0,2}]
std::cout<<std::endl<<prng2.integers(0,5,10);   //[(10){1,1,3,2,3,2,3,0,1,2}]
```

## default_rng

The same as `rng` but with random bits generator type predefined to `std::mt19937_64`.

```cpp
auto prng1 = gtensor::default_rng(1,2,3); //create PRNG using predefined std::mt19937_64 random bits generator and seeds {1,2,3}
auto prng2 = gtensor::default_rng(); //create PRNG using predefined std::mt19937_64 random bits generator and random seeds
std::cout<<std::endl<<prng1.integers(0,5,10);   //[(10){2,3,2,2,1,3,0,3,0,0}]
std::cout<<std::endl<<prng2.integers(0,5,10);   //[(10){0,0,0,2,2,3,1,1,3,2}]
```

## PRNG random permutations member functions

### shuffle

Do shuffle in-place along `axis`, the order of sub-tensors is changed but their contents remains the same.

```cpp
template<typename DimT=int, typename...Ts>
void shuffle(basic_tensor<Ts...>& t, const DimT& axis=0);
```

```cpp
auto prng = gtensor::default_rng(1,2,3);
gtensor::tensor<double> a{{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
prng.shuffle(a);
std::cout<<std::endl<<a;    //[(4,3){{4,5,6},{1,2,3},{10,11,12},{7,8,9}}]
prng.shuffle(a,1);
std::cout<<std::endl<<a;    //[(4,3){{4,6,5},{1,3,2},{10,12,11},{7,9,8}}]
auto b = a.flatten();
prng.shuffle(b);
std::cout<<std::endl<<b;    //[(12){7,10,3,9,2,12,11,4,1,6,8,5}]
```

### permutation

If `t` is tensor, returns its shuffled copy.
If `t` is integral, returns shuffled `arange(t)`.

```cpp
template<typename T, typename DimT=int>
auto permutation(const T& t, const DimT& axis=0);
```

```cpp
auto prng = gtensor::default_rng(1,2,3);
gtensor::tensor<double> a{{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
auto res1 = prng.permutation(a,1);
auto res2 = prng.permutation(7);
std::cout<<std::endl<<res1; //[(4,3){{2,1,3},{5,4,6},{8,7,9},{11,10,12}}]
std::cout<<std::endl<<res2; //[(7){0,1,4,2,3,5,6}]
```

### permuted

Returns copy, permuted along given `axis`.
If no `axis` specified flatten input tensor is shuffled.
Each slice along the given `axis` is shuffled independently of the others (which is not the case for `shuffle` and `permutation`).

```cpp
template<typename Axis=detail::no_value, typename...Ts>
auto permuted(const basic_tensor<Ts...>& t, const Axis& axis=Axis{});
```

```cpp
auto prng = gtensor::default_rng(1,2,3);
gtensor::tensor<double> a{{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
auto res1 = prng.permuted(a);
auto res2 = prng.permuted(a,1);
std::cout<<std::endl<<res1; //[(12){11,5,8,6,1,10,2,9,12,4,7,3}]
std::cout<<std::endl<<res2; //[(4,3){{2,3,1},{5,6,4},{7,8,9},{12,11,10}}]
```

### choice

If `t` is tensor, returns random sample taken from given tensor.
If `t` is integral, returns random sample taken from `arange(t)`.

```cpp
template<typename T, typename Size, typename DimT, typename Probabilities=detail::no_value>
auto choice(const T& t, Size&& size, bool replace=true, const Probabilities& p=Probabilities{}, const DimT& axis=0, bool shuffle=true);
```

`size` specifies result shape, can be scalar or container.

`replace` whether the sample is with or without replacement. Default is `true`, meaning that a value of `t` can be selected multiple times.

`p` The probabilities associated with each element in `t`. If not given, the sample assumes a uniform distribution over all elements in `t`.
Must be 1d tensor or container.

`axis` is axis along which selection is performed. Default is 0.

`shuffle` whether the sample is shuffled when sampling without replacement. Default is `true`.

```cpp
auto prng = gtensor::default_rng(3,4,5);
gtensor::tensor<double> a{{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
auto res1 = prng.choice(a,6);
auto res2 = prng.choice(a,4,false);
auto res3 = prng.choice(9,std::vector<int>{3,3},true,std::vector<double>{0.05,0.6,0.05,0.05,0.05,0.05,0.05,0.05,0.05});
std::cout<<std::endl<<res1; //[(6,3){{7,8,9},{10,11,12},{7,8,9},{7,8,9},{1,2,3},{4,5,6}}]
std::cout<<std::endl<<res2; //[(4,3){{10,11,12},{4,5,6},{7,8,9},{1,2,3}}]
std::cout<<std::endl<<res3; //[(3,3){{4,2,2},{1,1,1},{2,1,1}}]
```

## PRNG member functions to draw from probability distributions

All drawing member functions return tensor of random values with specified `value_type`, `layout` and `shape`.
- result's `value_type` and `layout` can be specified using appropriate template parameters.
By default `value_type` for functions in **integral domain** is `int` and in **floating-point domain** is `double`.
Default `layout` is row major.
- shape of tensor is specified with `size` function parameter which can be scalar or container.
In case of scalar 1d tensor is returned.

### integers

Returns tensor of samples of integral type drawn from uniform distribution.

```cpp
template<typename T=int, typename Order=config::c_order, typename U, typename Size>
auto integers(const U& low_, const U& high_, Size&& size, bool end_point=false);
```

Samples drawn from range `[low, high)` if `end_point` is `false`.
Samples drawn from range `[low, high]` if `end_point` is `true`.

```cpp
auto prng = gtensor::default_rng(1,2,3);
auto res1 = prng.integers(3,10,12);
auto res2 = prng.integers<std::int64_t, gtensor::config::f_order>(3,10,std::vector<int>{3,4});
std::cout<<std::endl<<res1; //[(12){5,9,9,8,6,5,5,4,6,3,6,9}]
std::cout<<std::endl<<res2; //[(3,4){{3,7,8,6},{3,3,3,5},{5,9,4,9}}]
```

### uniform

Returns tensor of samples of floating point type drawn from uniform distribution. Samples drawn from range `[low,high)`.

```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename Size>
auto uniform(const U& low_, const U& high_, Size&& size);
```

### random

Returns tensor of samples of floating point type drawn from uniform distribution. Samples drawn from range `[0,1)`.

```cpp
template<typename T=double, typename Order=config::c_order, typename Size>
auto random(Size&& size);
```

### binomial

Returns tensor of samples drawn from a binomial distribution.
Drawn sample represents number of successes in sequence of `n` experiments, each of which succeeds with probability `p`.

```cpp
template<typename T=int, typename Order=config::c_order, typename U, typename V, typename Size>
auto binomial(const U& n, const V& p, Size&& size);
```
`n` trials number, must be `>=0`.

`p` probability of success, in range `[0,1]`.

### negative_binomial

Returns tensor of samples drawn from a negative binomial distribution.
Drawn sample represents number of failures in a sequence of experiments, each succeeds with probability `p`, before exactly `k` successes occur.

```cpp
template<typename T=int, typename Order=config::c_order, typename U, typename V, typename Size>
auto negative_binomial(const U& k, const V& p, Size&& size);
```

`k` successes number, must be `>0`.

`p` probability of success, in range `(0,1]`.

### poisson

Returns tensor of samples drawn from a poisson distribution.
Drawn sample represents number of occurrences of random event, if the expected, number of its occurrence under the same conditions (on the same time/space interval) is `mean`.

```cpp
template<typename T=int, typename Order=config::c_order, typename V, typename Size>
auto poisson(const V& mean, Size&& size);
```

`mean` expected number of events occurring in a fixed-time/space interval, must be `>0`.

### exponential

Returns tensor of samples drawn from a exponential distribution.
Drawn sample represents the time/distance until the next random event if random events occur at constant rate `lambda` per unit of time/distance.

```cpp
template<typename T=double, typename Order=config::c_order, typename V, typename Size>
auto exponential(const V& lambda_, Size&& size);
```

`lambda` - rate of event occurrence.

### gamma

Returns tensor of samples drawn from a gamma distribution.

```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
auto gamma(const U& shape, const V& scale, Size&& size);
```

`shape` - gamma distribution parameter, sometimes designated **"k"**.
`scale` - gamma distribution parameter, sometimes designated **"theta"**.

### weibull

Returns tensor of samples drawn from a weibull distribution.

```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
auto weibull(const U& shape, const V& scale, Size&& size);
```

`shape` - weibull distribution parameter, sometimes designated **"k"**.

`scale` - weibull distribution parameter, sometimes designated **"lambda"**.

### normal

Returns tensor of samples drawn from a normal distribution.
```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
auto normal(const U& mean_, const V& stdev_, Size&& size);
```

`mean` - normal distribution parameter.

`stdev` - normal distribution parameter, must be `>0`.

### lognormal

Returns tensor of samples drawn from a lognormal distribution.

```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
auto lognormal(const U& mean_, const V& stdev_, Size&& size);
```

`mean` - lognormal distribution parameter.

`stdev` - lognormal distribution parameter, must be `>0`.

### chisquare

Returns tensor of samples drawn from a chisquare distribution.

```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename Size>
auto chisquare(const U& df_, Size&& size);
```

`df` - chisquare distribution parameter - degrees of freeedom, must be `>0`.

### cauchy

Returns tensor of samples drawn from a cauchy distribution.

```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
auto cauchy(const U& location_, const V& scale_, Size&& size);
```

`location` - cauchy distribution parameter.

`scale` - cauchy distribution parameter, must be `>0`.

### f

Returns tensor of samples drawn from a fisher distribution.

```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename V, typename Size>
auto f(const U& dfnum_, const V& dfden_, Size&& size);
```

`dfnum` - fisher distribution parameter - degrees of freedom in numerator, must be `>0`.

`dfden` - fisher distribution parameter - Degrees of freedom in denominator, must be `>0`.

### t

Returns tensor of samples drawn from a student distribution.

```cpp
template<typename T=double, typename Order=config::c_order, typename U, typename Size>
auto t(const U& df_, Size&& size);
```

`df` - student distribution parameter - degrees of freeedom, must be `>0`.