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

It takes single type template parameter `Impl` that is type of implementation.

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

## 3. `basic_tensor` construction

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
Whether tensor's elements will be initialized depends on `storage` alias specified in `Config` template parameter.
By default elements are not initialized for trivially-copyable data type, and initialized to default otherwise.

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
if tensor size is greater than range - first tensor elements initialized with range, whether rest tensor elements will be initialized depends on underlaying storage
- tensor layout matters

Explicit call construtor of `tensor` class template is not the only way to construct `basic_tensor` object. Another way is to operate on already constructed objects.

Consider example:

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto sum = t+t;
```

What is `decltype(sum)`? It is not of type `tensor<double>` as you might think.
It looks like: `gtensor::basic_tensor<gtensor::tensor_implementation<gtensor::expression_template_core<...>>>`. It is also `basic_tensor` specialization, but parameterized with special implementation type. We call such tensors **expression view**. Almost all operators on tensor produce expression views. More detailed this topic will be discussed in next sections.

## 4. `basic_tensor` copy and move construction semantic

`basic_tensor` has reference copy-construction semantic. Possible implementation of `basic_tensor` class template:

```cpp
template<typename Impl>
class basic_tensor
{
    std::shared_ptr<Impl> impl_;
public:

    basic_tensor(const basic_tensor&) = default;
    basic_tensor(basic_tensor&&) = default;
    ...
};
```

Consider example:

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a;
std::cout<<std::endl<<a;    //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<b;    //[(2,3){{1,2,3},{4,5,6}}]
a+=1;
std::cout<<std::endl<<a;    //[(2,3){{2,3,4},{5,6,7}}]
std::cout<<std::endl<<b;    //[(2,3){{2,3,4},{5,6,7}}]
```

Here we first construct tensor `a`, than copy construct `b` from `a` and mutate `a`.
Due to reference semantic `a` and `b` share the same implementation, that is mutating `a` causes mutating `b`.

To make deep copy:

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a.copy();
a+=1;
std::cout<<std::endl<<a;    //[(2,3){{2,3,4},{5,6,7}}]
std::cout<<std::endl<<b;    //[(2,3){{1,2,3},{4,5,6}}]
```

To test if tensors share the same implementation:

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a;
auto c = a.copy();
std::cout<<std::endl<<a.is_same(b); //1
std::cout<<std::endl<<a.is_same(c); //0
```

After move tensor refers to no implementation, in this case it is guaranteed `empty()` returns true.

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = std::move(a);
std::cout<<std::endl<<a.empty(); //1
std::cout<<std::endl<<b.empty(); //0
```

Having reference semantic we can return tensor objects from functions by value without data copying:

```cpp
auto make_squares(std::size_t n){
    gtensor::tensor<double> t(n,0);
    std::iota(t.begin(),t.end(),0);
    return t*t;
}
auto squares = make_squares(7);
std::cout<<std::endl<<squares;  //[(7){0,1,4,9,16,25,36}]
```

## 5 Expression view and lazy evaluation

In general, given multidimensional array object, view of this array is another object which provides us with some different way to look at original array's data.
And we can operate on this view object as it were true array. And no data copy required.

GTensor provides two kinds of views:
- View based on some index transformations, it can be slice of original tensor, reshape, transpose or selecting elements using tensor of indeces.
This kind of view always refers to original tensor's elements, using some indexing scheem.
- **Expression view** is fundamentally different. It can produce new elements from original tensor or tensors by applying function object to its elements.
What is important that we can apply function object only when we refers to view element - **computation of values of expression view elements is lazy**.

Going back to last example from section 3 we extend it to show nature of lazy evaluation:

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<double> t2{{7,8,9},{10,11,12}};
auto sum = t1+t2; //no addition is performed here
double sum_sum{0};
for (auto it=sum.begin(),last=sum.end(); it!=last; ++it){
    sum_sum+=*it;   //each dereference of it performs addition of corresponding elements of t1 and t2
}
std::cout<<std::endl<<sum_sum;  //78
```

`sum` here is **expression view**, its type is specialization of `basic_tensor` class template.
In fact such specialization type holds shallow copies of `t1` and `t2` (thanks to reference copy semantic) and binary functor like std::plus\<void\> as data members.
It also provide logic to make such lazy evaluation possible.

Operands of expression not necessary to have equal shapes, **broadcasting** is supported:

```cpp
gtensor::tensor<double> t1{1,2,3};
gtensor::tensor<double> t2{{4},{5},{6},{7}};
std::cout<<std::endl<<(t1*t2);  //[(4,3){{4,8,12},{5,10,15},{6,12,18},{7,14,21}}]
```

Operands of expression view can be of any tensor type i.e. any specialization of `basic_tensor` class template and scalars.
Expressions can have any complexity, it is not limited by implementation. Very **deep** expressions may increase compilation time.
Almost all operators on tensor and math functions produce expression views.

As we see **expression view** tensor doesn't perform any computations when constructed. There are two member functions to force evaluation: `copy()` and `eval()`.

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto res = (1/(t+1)).eval();
std::cout<<std::endl<<res;  //[(2,3){{0.5,0.333,0.25},{0.2,0.167,0.143}}]
```

In this example `res` is not view, it is `basic_tensor` with storage implementation i.e. the same kind of tensor we construct using `tensor` class template.

The distinction between `copy()` and `eval()` is that when `eval()` is called on tensor with storage implementation (nothing to evaluate) it returns its shallow copy.
`copy` on other hand always returns deep copy. When applied to any kind of view `copy()` and `eval()` have the same effect.

Next example shows the difference:

```cpp
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
std::cout<<std::endl<<t.is_same(t.copy());  //0
std::cout<<std::endl<<t.is_same(t.eval());  //1
auto v = t+1;
std::cout<<std::endl<<v.is_same(v.copy());  //0
std::cout<<std::endl<<v.is_same(v.eval());  //0
```

There are several important points regarding expression views:
- there is no any caching, each time you refer to element of view computation is performed
- view holds shallow copies of its operands, so if you mutate operands it will affect view values
- no temporary copies created, evaluation is elementwise
- evaluation can be easily parallelized

Evaluation can be easily parallelized using overloaded versions of `copy()` and `eval()`:

```cpp
gtensor::tensor<double> t(1000000,0);
std::iota(t.begin(),t.end(),0);
auto res = sin((1/(t+1))).eval(multithreading::exec_pol<4>{});
std::cout<<std::endl<<res;  //[(1000000){0.841,0.479,0.327,...,1e-06,1e-06,1e-06}]
```

GTensor provides its own parallel execution subsystem defined in namespace `multithreading`.
To control level of parallelizm it uses type tags which is specializations of `multithreading::exec_pol` class template.
Integral constant in specialization means number of tasks original task is diveded to.
For example `multithreading::exec_pol<1>` means execute whole task in single thread.
`multithreading::exec_pol<4>` means divide task into four roughly equal parts and try to run they in parallel.
Whether these tasks will actually be executed in parallel mostly depends on the target system.




GTensor provides means to parallelize eva

each deref cause eval

broadcasting

arbitrary expression comlexity, n_operator

assignment, a_operator

composition with any tensor