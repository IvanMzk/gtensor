# Tensor operators

Almost all tensor operators produce **expression views** which are specializations of `basic_tensor` class template.
As `tensor` class template intended to construct `basic_tensor` objects with storage implementation,
tensor operators produce `basic_tensor` objects with expression view implementation.

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

## str and output

```cpp
//returns tensor string representation as std::string object
template<typename P=int, typename...Ts> auto str(const basic_tensor<Ts...>& t, const P& precision=3);
```

`operator<<()` uses `str` with default precision in its implementation.

```cpp
gtensor::tensor<double> t{1.12355,2.12345,3.12348,4.12345,5.13345};
std::cout<<std::endl<<str(t);   //[(5){1.12,2.12,3.12,4.12,5.13}]
std::cout<<std::endl<<str(t,5); //[(5){1.1236,2.1235,3.1235,4.1235,5.1334}]
std::cout<<std::endl<<t;    //[(5){1.12,2.12,3.12,4.12,5.13}]
```

## Tensor equality

### Strict equality

Two tensors are considered **equal** if their shapes and elements are equal.

```cpp
template<typename...Us, typename...Vs>
bool tensor_equal(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false);
```

`operator==` and `operator!=` use `tensor_equal` in their implementation.

```cpp
using tensor_type = tensor<double>;
tensor_type a{{1,2,3},{4,5,6}};
tensor_type b{1,2,3,4,5,6};
tensor_type c{{1,2,3},{3,2,1}};
std::cout<<std::endl<<(a==b);   //0
std::cout<<std::endl<<tensor_equal(a,b);   //0
std::cout<<std::endl<<(a==b.reshape(2,3));  //1
std::cout<<std::endl<<tensor_equal(a,b.reshape(2,3));  //1
std::cout<<std::endl<<(a!=c);   //1
```

### Close equality

```cpp
//return true if two tensors have same shape and close elements within specified tolerance
template<typename...Us, typename...Vs, typename Tol>
bool tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false);

//use machine epsilon as tolerance
template<typename...Us, typename...Vs>
bool tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false);

//return true if two tensors have close elements within specified tolerance, shapes may not be equal, but must broadcast
template<typename...Us, typename...Vs, typename Tol>
bool allclose(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false);

//use machine epsilon as tolerance
template<typename...Us, typename...Vs>
bool allclose(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false)
```

```cpp
using tensor_type = tensor<double>;
tensor_type a{{1.12345,2.12345,3.12345},{4.12345,5.12345,6.12345}};
tensor_type b{{1.12345,2.12345,3.12355},{4.12325,5.12345,6.12375}};
std::cout<<std::endl<<tensor_close(a,b);    //0
std::cout<<std::endl<<tensor_close(a,b,1E-6,1E-6);  //0
std::cout<<std::endl<<tensor_close(a,b,1E-3,1E-3);  //1
```

```cpp
using tensor_type = tensor<double>;
tensor_type a{{1.12345,2.12345,3.12345},{1.12345,2.12345,3.12345}};
tensor_type b{1.12345,2.12345,3.12355};
std::cout<<std::endl<<allclose(a,b);    //0
std::cout<<std::endl<<allclose(a,b,1E-6,1E-6);  //0
std::cout<<std::endl<<allclose(a,b,1E-3,1E-3);  //1
```

## Tensor broadcast operators

All operators and functions described below are lazy and return **expression view** object.
Operands may be tensor or scalar, but at least one operand must be tensor.
Operands must be broadcastable.

### cast

`static_cast` is used underneath.

```cpp
gtensor::tensor<int> t{1,2,3,4,5};
auto t_double = gtensor::cast<double>(t);
std::cout<<std::endl<<t_double; //[(5){1,2,3,4,5}]
std::cout<<std::endl<<std::is_same_v<typename decltype(t_double)::value_type,double>;   //1
```

### where (broadcast ternary operator)

```cpp
gtensor::tensor<int> t{{1,2,3},{4,5,6}};
auto res = where(t((t%2).not_equal(0)),t,0);    //select odd elements from t and select zero for even
std::cout<<std::endl<<res;  //[(2,3){{1,0,3},{0,5,0}}]
```

### Arithmetic operators

```cpp
gtensor::tensor<int> t1{{1,2,3},{4,5,6}};
gtensor::tensor<int> t2{2,3,4};
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
auto res2 = t1.equal(t2);   //the same as above
auto res3 = not_equal(t1,t2);
auto res4 = t1.not_equal(t2);   //the same as above
auto res5 = t1>t2;
auto res6 = t1>=t2;
auto res7 = t1<t2;
auto res8 = t1<=t2;
```

### Close comparison

`isclose` is similar to `equal`, except using tolerance to say if elements are close

```cpp
template<typename T, typename U, typename Tol, typename EqualNan = std::false_type>
inline auto isclose(T&& t, U&& u, Tol relative_tolerance, Tol absolute_tolerance, EqualNan equal_nan = EqualNan{});

//by default tolerance equal to machine epsilon
template<typename T, typename U, typename EqualNan = std::false_type>
inline auto isclose(T&& t, U&& u, EqualNan equal_nan = EqualNan{});
```

```cpp
gtensor::tensor<double> a{1.12345,2.12345,3.12345,4.12345,5.12345};
gtensor::tensor<double> b{1.12355,2.12345,3.12348,4.12345,5.13345};
std::cout<<std::endl<<isclose(a,b); //[(5){0,1,0,1,0}]
std::cout<<std::endl<<isclose(a,b,1E-2,1E-2);   //[(5){1,1,1,1,1}]
std::cout<<std::endl<<isclose(a,b,1E-4,1E-4);   //[(5){1,1,1,1,0}]
```

### Logical

```cpp
gtensor::tensor<bool> t1{{true,false,true},{false,true,false}};
gtensor::tensor<bool> t2{true,false,true};
auto res1 = !t1;
auto res2 = t1&&t2;
auto res3 = t1||t2;
```

## Tensor broadcast assign operators

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