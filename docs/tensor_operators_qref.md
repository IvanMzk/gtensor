# Tensor operators quick reference

Almost all tensor operators produce **expression views** which are specializations of `basic_tensor` class template.
As `tensor` class template intended to construct `basic_tensor` objects with storage implementation,
tensor operators (and many other library functions) produce `basic_tensor` objects with expression view implementation.

## n_operator()

`n_operator()` free function is generalization of any lazy computing function.

```cpp
template<typename F, typename...Operands> inline auto n_operator(F&& f, Operands&&...operands);
```

It takes function object and expression operands as arguments and returns expression view object.
Arity of function object must equal to number of operands.
Operands can be tensors or scalars, and must be broadcastable. At least one operand must be tensor.
In fact all library functions that perform lazy computations (tensor operators, math functions) use `n_operator()` in their implementation.

For example possible implementation of `operator+()`:

```cpp
template<typename Impl1, typename Impl2>
auto operator+(const basic_tensor<Impl1>& op1, const basic_tensor<Impl2>& op2){
    return n_operator(std::plus<void>{},op1,op2);
}
```

## a_operator()

`a_operator()` free function is generalization of any function that perform broadcast assign or broadcast compaund assign.

```cpp
template<typename F, typename Tensor, typename Rhs> inline std::decay_t<Tensor>& a_operator(F&& f, Tensor&& lhs, Rhs&& rhs);
```

It takes function object, lhs and rhs operands of assign expression.
Lhs operand must be tensor, rhs tensor or scalar.
Operands must be broadcastable.
In fact all library functions that perform broadcast assign or broadcast compaund assign use `a_operator()` in their implementation.
Function is not lazy, after call assign is completed.

For example possible implementation of `operator+=()`:

```cpp
template<typename Impl1, typename Impl2>
auto& operator+=(basic_tensor<Impl1>& lhs, const basic_tensor<Impl2>& rhs){
    auto plus_assign = [](auto& lhs_element, const auto& rhs_element){lhs_element += rhs_element;};
    a_operator(plus_assign,lhs,rhs);
    return lhs;
}
```

## Tensor equality

### Strict equality

### Close equality


## Tensor broadcast assign

All assign operators and functions are not lazy and return reference to lhs.
Lhs must be tensor. Rhs may be tensor or scalar.
Lhs and rhs must be broadcastable.

```cpp
gtensor::tensor<int> t1{{1,2,3},{4,5,6}};
gtensor::tensor<int> t2{2,3,4};
assign(t1,t2);  //t2 is assigned to t1
t1.assign(t2);  //the same as above
t1+=t2;
t1-=t2;
t1*=t2;
t1/=t2;
t1%=t2;
t1&=t2;
t1|=t2;
t1^=t2;
t1<<=t2;
t1>>=t2;
```

## Tensor broadcast operators and related functions

All operators and functions described below are lazy and return **expression view** object.
Operands may be tensor or scalar, but at least one operand must be tensor.
Operands must be broadcastable.

### Cast

### Where (broadcast ternary operator)

### Arithmetic operators

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<double> t2{2,3,4};
auto res1 = -t1;
auto res2 = +t1;
auto res3 = t1+t2;
auto res4 = t1-t2;
auto res5 = t1*t2;
auto res6 = t1/t2;
auto res7 = t1%t2;
```

### Bitwise operators

```cpp
gtensor::tensor<int> t1{{1,2,3},{4,5,6}};
gtensor::tensor<int> t2{2,3,4};
auto res1 = ~t2;
auto res2 = t1&t2;
auto res3 = t1|t2;
auto res4 = t1^t2;
auto res5 = t1<<t2;
auto res6 = t1>>t2;
```

### Strict comparison

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<double> t2{2,3,4};
auto res1 = equal(t1,t2);
auto res2 = not_equal(t1,t2);
auto res3 = t1>t2;
auto res4 = t1>=t2;
auto res5 = t1<t2;
auto res6 = t1<=t2;
```

### Close comparison

### Logical

```cpp
gtensor::tensor<bool> t1{{true,false,true},{false,true,false}};
gtensor::tensor<bool> t2{true,false,true};
auto res1 = !t1;
auto res2 = t1&&t2;
auto res3 = t1||t2;
```