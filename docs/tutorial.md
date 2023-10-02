# Tutorial

This tutorial describes the main points of using GTensor library, its structure and functions.

## 1. Multidimensional array abstraction, `data` and `meta-data`

GTensor library is meant for computing over multidimensional arrays. Such an array is abstraction which mainly consists of two parts: meta-data and data as an analogy to form and matter in philosophy.

In most practical implementations data and meta-data are implemented using flat arrays of elements but with different meanings.
- data elements can be of any type, suitable for goals of computation, e.g. integral, floating-point, complex or even some user-defined type
- meta-data elements usually of integral type due to its purpose: describe multidimensional structure of data elements, index data elements

To be useful, array abstraction generally should provide interface to access its data and meta-data elements and hide other implementation details.
It is common practice to use `iterator interface` for data and member functions like `shape()`, `strides()`, `dim()`, `size()` for meta-data.

## 2. `tensor` and `basic_tensor` class templates

`basic_tensor` class template represent multidimensional array abstraction, its declaration:

```cpp
template<typename Impl> class basic_tensor;
```

It takes single type template parameter `Impl` that is type of array implementation. You should never create `basic_tensor` objects directly.

`tensor` class template is intended to make `basic_tensor` with storage implementation, its declaration:

```cpp
template<typename T, typename Layout = config::f_order, typename Config = config::extend_config_t<config::default_config,T>>
class tensor : public basic_tensor<typename tensor_factory_selector_t<Config,T,Layout>::result_type>
```

As we see `tensor` is `basic_tensor` and it directly specifies its implementation type using trait.

`tensor` class template takes three type template parameters:
- T is type of data element
- Layout can be of type gtensor::config::c_order or gtensor::config::f_order and defines storage scheme of data elements
- Config is struct that contain `tensor` implementation details: alias templates of containers for data and meta-data elements, default traverse order for iterators and other.
It will be covered in more details further.

Consider example:

```cpp
gtensos::tensor<int> t1{{5,5,5,5},{5,5,5,5},{5,5,5,5}};
gtensos::tensor<int> t2({3,4},5);
std::vector<int> v(12,5);
gtensos::tensor<int> t3({3,4},v.begin(),v.end());
```

Here we create three tensors with `int` data element type, default `Layout` and default `Config`.
All tensors have shape (3,4) and filled with value 5.

Now investigate `gtensos::tensor<int>` base type:

```cpp
template<typename Impl>
auto as_basic_tensor(const basic_tensor<Impl>& t){
    return t;
}
```

What is `decltype(as_basic_tensor(t1))`?

It looks something like this: `gtensor::basic_tensor<gtensor::tensor_implementation<gtensor::storage_core<...>>>`, where `<...>` may be `<T,Layout,Config>`.

We see that `tensor<int>` is `basic_tensor` parameterized with storage implementation, which is parameterized with `<T,Layout,Config>`.

In fact `tensor` class template just defines constructors suitable to initialize storage implementation and nothing more. All of member functions are defined in `basic_tensor`.

## 3. `tensor` constructors




















Now consider sum of tensors:

```cpp
auto sum = t1+t2+t3;
```

What is `decltype(sum)` ? It looks like: `gtensor::basic_tensor<gtensor::tensor_implementation<gtensor::expression_template_core<...>>>`.

We see that sum of tensors is also tensor, but with different implementation type. We call such tensors `expression view`.



Where `<...>` dependes on