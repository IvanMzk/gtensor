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
