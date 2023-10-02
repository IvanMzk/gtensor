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

It takes single type template parameter `Impl` that is type of array implementation.

You should never create `basic_tensor` objects directly. To construct `basic_tensor` object from value you should use `tensor` class template.

`tensor` class template is intended to make `basic_tensor` with storage implementation, its definition:

```cpp
template<typename T, typename Layout = config::f_order, typename Config = config::extend_config_t<config::default_config,T>>
class tensor : public basic_tensor<typename tensor_factory_selector_t<Config,T,Layout>::result_type>
{
...
};
```

As we see `tensor` is `basic_tensor` and it directly specifies its implementation type using trait.

`tensor` class template takes three type template parameters:
- T is type of data element
- Layout can be of type gtensor::config::c_order or gtensor::config::f_order and defines storage scheme of data elements
- Config is struct that contain `tensor` implementation details: alias templates of containers for data and meta-data elements, default traverse order for iterators and other.
It will be covered in more details further.

Consider example:

```cpp
gtensos::tensor<int> t{{5,5,5,5},{5,5,5,5},{5,5,5,5}};
```

Here we create tensor with `int` data element type, default `Layout` and default `Config`.
Tensors has shape (3,4) and all its elements initialized with value 5.

Now investigate `gtensos::tensor<int>` base type:

```cpp
template<typename Impl>
auto as_basic_tensor(const basic_tensor<Impl>& t){
    return t;
}
```

What is `decltype(as_basic_tensor(t))`?

It looks like: `gtensor::basic_tensor<gtensor::tensor_implementation<gtensor::storage_core<...>>>`, where `<...>` may be `<T,Layout,Config>`.

We see that `tensor<int>` is `basic_tensor` parameterized with storage implementation, which is parameterized with `<T,Layout,Config>`.

In fact `tensor` class template just defines constructors suitable to initialize storage implementation and nothing more. All of member functions are defined in `basic_tensor`.

## 3. `basic_tensor` construction and copy-move semantic

As mentioned above we should use `tensor` class template to construct `basic_tensor` object from value.

Next examples shows possible ways to do this:

```cpp
using gtensor::tensor;
using gtensor::config::c_order;
using gtensor::config::f_order;
//initializer list constructor
tensor<double> t1{1,2,3,4,5};
tensor<double,c_order> t2{{1,2,3},{4,5,6},{7,8,9}};
tensor<double,f_order> t3{{{1,2},{3,4}},{{5,6},{7,8}}};
std::cout<<std::endl<<t1;   //[(5){1,2,3,4,5}]
std::cout<<std::endl<<t2;   //[(3,3){{1,2,3},{4,5,6},{7,8,9}}]
std::cout<<std::endl<<t3;   //[(2,2,2){{{1,2},{3,4}},{{5,6},{7,8}}}]
```

We use initializer_list constructor to make three tensors, regardless of tensor's layout elements in initializer_list are always considered to be in c_order.

```cpp
//shape constructor
using gtensor::tensor;
tensor<double> t4(std::vector<int>{3,4});
tensor<double> t5(std::list<int>{3,4});
std::cout<<std::endl<<t4;   //[(3,4){{1.1e-311,6.95e-310,6.95e-310,6.95e-310},{1.1e-311,0,6.95e-310,0},{0,0,6.95e-310,2.07e-236}}]
std::cout<<std::endl<<t5;   //[(3,4){{1.1e-311,1.1e-311,1.1e-311,1.1e-311},{1.1e-311,1.1e-311,1.1e-311,1.1e-311},{1.1e-311,1.1e-311,4.94e-324,1.1e-311}}]
```

We use shape constructor to make two tensors of shape (3,4). Shape argument can be any container.
Are tensor's elements initialized dependes on `storage` alias specified in Config template parameter.
By default elements are not initialized for trivially-copyable data type, and initialized to default value otherwise.

```cpp
//default construtor
gtensor::tensor<double> t6{};
std::cout<<std::endl<<t6;           //[(0){}]
std::cout<<std::endl<<t6.dim();     //1
std::cout<<std::endl<<t6.size();    //0
```

Default constructor makes 1d empty tensor. It is equivalent to call shape constructor `tensor<double>(std::vector<int>{0})`.


```cpp
//0dim tensor (tensor-scalar) constructor
gtensor::tensor<double> t7(5);
std::cout<<std::endl<<t7;           //[(){5}]
std::cout<<std::endl<<t7.dim();     //0
std::cout<<std::endl<<t7.size();    //1
```

0Dim tensor constructor makes tensor with empty shape and unit size.


```cpp
using gtensor::tensor;
//shape and value constructor
tensor<double> t8(10,5);
tensor<double> t9(std::array<int,2>{3,4},5);
tensor<double> t10({3,4},5);
tensor<double> t11(std::vector<int>{},5);
std::cout<<std::endl<<t8;   //[(10){5,5,5,5,5,5,5,5,5,5}]
std::cout<<std::endl<<t9;   //[(3,4){{5,5,5,5},{5,5,5,5},{5,5,5,5}}]
std::cout<<std::endl<<t10;  //[(3,4){{5,5,5,5},{5,5,5,5},{5,5,5,5}}]
std::cout<<std::endl<<t11;  //[(){5}]
```

Shape and value constructor makes tensor of specified shape and initialized its elements with value.
Shape argument can be scalar, container or std::initializer_list.
In case of scalar 1d tensor is constructed. In case of empty container 0d tensor (tensor-scalar) is constructed.

```cpp
using gtensor::tensor;
using gtensor::config::c_order;
using gtensor::config::f_order;
//shape and iterators range constructor
std::vector<double> data{1,2,3,4,5,6,7,8,9,10,11,12};
tensor<double,c_order> t12(12,data.begin(),data.end());
tensor<double,c_order> t13(std::vector<int>{},data.begin(),data.end());
tensor<double,c_order> t14(std::vector<int>{3,3},data.begin(),data.end());
tensor<double,c_order> t15(std::vector<int>{4,4},data.begin(),data.end());
tensor<double,f_order> t16({3,4},data.begin(),data.end());
std::cout<<std::endl<<t12;  //[(12){1,2,3,4,5,6,7,8,9,10,11,12}]
std::cout<<std::endl<<t13;  //[(){1}]
std::cout<<std::endl<<t14;  //[(3,3){{1,2,3},{4,5,6},{7,8,9}}]
std::cout<<std::endl<<t15;  //[(4,4){{1,2,3,4},{5,6,7,8},{9,10,11,12},{1.36e-311,1.36e-311,5,5}}]
std::cout<<std::endl<<t16;  //[(3,4){{1,4,7,10},{2,5,8,11},{3,6,9,12}}]
```

Shape and iterators range constructor makes tensor of specified shape and fills it with elements from range.
Shape argument can be scalar, container or std::initializer_list.
In case of scalar 1d tensor is constructed. In case of empty container 0d tensor (tensor-scalar) is constructed.
There are two points here:
- if tensor size n is smaller or equal than range - tensor initialized with first n range elements
if tensor size is greater than range - first tensor elements initialized with range, are rest tensor elements initialized dependes on underlaying storage
- tensor layout matters

Explicit call construtor of `tensor` class template is not only way to construct `basic_tensor` object. Another way is to operate on already constructed objects.

Consider example:

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto sum = t+t;
```

What is `decltype(sum)`? It is not of type `tensor<double>` as you might think.
It looks like: `gtensor::basic_tensor<gtensor::tensor_implementation<gtensor::expression_template_core<...>>>`. It is also `basic_tensor`, but parameterized with special implementation type.
We call such tensors **expression view**. Almost all operators on tensor produce expression views. More detailed this topic will be discussed in next sections.