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
Drawn sample represents number of successes in sequence of `n` experiments, each of which succeeds with probability `p`

```cpp
template<typename T=int, typename Order=config::c_order, typename U, typename V, typename Size>
auto binomial(const U& n, const V& p, Size&& size);
```
`n` trials number, must be `>=0`.

`p` probability of success, in range `[0,1]`.

### negative_binomial

Returns tensor of samples drawn from a negative binomial distribution.
Drawn sample represents number of failures in a sequence of experiments, each succeeds with probability `p`, before exactly `k` successes occur

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