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

For example possible implementation of `operator+=()`:

```cpp
template<typename Impl1, typename Impl2>
auto operator+=(basic_tensor<Impl1>& lhs, const basic_tensor<Impl2>& rhs){
    auto plus_assign = [](auto& lhs_element, const auto& rhs_element){lhs_element += rhs_element;};
    return a_operator(plus_assign,lhs,rhs);
}
```

