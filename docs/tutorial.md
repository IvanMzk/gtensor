# Tutorial

**Contents**
1. [Multidimensional array abstraction](#section_1)
2. [`tensor` and `basic_tensor` class templates](#section_2)
3. [`basic_tensor` construction](#section_3)
4. [`basic_tensor` copy and move construction semantic](#section_4)
5. [Expression view and lazy evaluation](#section_5)
6. [Slice, reshape, transpose and mapping view](#section_6)
7. [`basic_tensor` assign semantic](#section_7)
8. [`basic_tensor` equality](#section_8)
9. [`basic_tensor` data and meta-data interface](#section_9)
10. [GTensor config](#section_10)

## 1. Multidimensional array abstraction <a id=section_1></a>

GTensor library is meant for computing over multidimensional arrays. Such an array is abstraction which mainly consists of two parts: **meta-data** and **data** as an analogy to form and matter in philosophy.

In most practical implementations both data and meta-data are implemented using flat arrays of elements but with different meanings.
- data elements can be of any type, suitable for goals of computation, e.g. integral, floating-point, complex or even some user-defined type
- meta-data elements usually of integral type due to its purpose: describe multidimensional structure of data elements, index data elements

To be useful, array abstraction generally should provide interface to access its data and meta-data elements and hide other implementation details.
It is common practice to use `iterator interface` for data and member functions like `shape()`, `strides()`, `dim()`, `size()` for meta-data.

## 2. `tensor` and `basic_tensor` class templates <a id=section_2></a>

`basic_tensor` class template represents multidimensional array abstraction, its declaration:

```cpp
template<typename Impl> class basic_tensor;
```

It takes single type template parameter `Impl` that is type of implementation.

You should never create `basic_tensor` objects directly. To construct `basic_tensor` object you may use `tensor` class template.

`tensor` class template is intended to construct `basic_tensor` object with storage implementation, its definition:

```cpp
template<typename T, typename Layout = config::c_order, typename Config = config::extend_config_t<config::default_config,T>>
class tensor : public basic_tensor<typename tensor_factory_selector_t<Config,T,Layout>::result_type>
{
...
};
```

As we see `tensor` is `basic_tensor` and it directly specifies its implementation type using trait.

`tensor` class template takes three type template parameters:
- **T** is type of data element
- **Layout** can be of type `gtensor::config::c_order` or `gtensor::config::f_order` and defines storage scheme of data elements
- **Config** is struct that contain `tensor` settings: alias templates of containers for data and meta-data elements, default traverse order for iterators and other.
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

## 3. `basic_tensor` construction <a id=section_3></a>

As mentioned above we should use `tensor` class template to construct `basic_tensor` object.

Next examples show possible ways to do this:

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

We use initializer_list constructor to make three tensors.
Regardless of tensor's layout, elements in initializer_list are always considered to be row major, i.e.
`{{1,2,3},{4,5,6}}` means two rows and three columns.

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

## 4. `basic_tensor` copy and move construction semantics <a id=section_4></a>

`basic_tensor` can have **deep copy-construction semantics** or **shallow copy-construction semantics**.

Possible implementation of `basic_tensor` class template:

```cpp
template<typename Impl>
class basic_tensor
{
    std::shared_ptr<Impl> impl_;
    const config::cloning_semantics semantics_;
    ...
};
```

Having **deep copy-construction semantics** copy referes to its own implementation.
Having **shallow copy-construction semantics** copy shares implementation with original, i.e. refers to the same data and meta-data.

By default `basic_tensor` with storage implementation exposes **deep copy-construction semantics**:

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a;
std::cout<<std::endl<<a;    //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<b;    //[(2,3){{1,2,3},{4,5,6}}]
a+=1;
std::cout<<std::endl<<a;    //[(2,3){{2,3,4},{5,6,7}}]
std::cout<<std::endl<<b;    //[(2,3){{1,2,3},{4,5,6}}]
```

Semantics can't be changed after `basic_tensor` object is constructed, but there is `clone()` interface to make copies which will expose specified semantics.

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a.clone(config::cloning_semantics::shallow,config::cloning_semantics::shallow);
auto c = b;
std::cout<<std::endl<<a;    //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<b;    //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<c;    //[(2,3){{1,2,3},{4,5,6}}]
b+=1;
std::cout<<std::endl<<a;    //[(2,3){{2,3,4},{5,6,7}}]
std::cout<<std::endl<<b;    //[(2,3){{2,3,4},{5,6,7}}]
std::cout<<std::endl<<c;    //[(2,3){{2,3,4},{5,6,7}}]
```

Here tensor `a` has **deep copy-construction semantics**.
Tensor `b` is shallow clone of `a` and has **shallow copy-construction semantics**.
Tensor `c` is copy constructed from `b`.
The effect is that `a`, `b` and `c` shares the same data elements.

To test if tensors share the same implementation:

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a;
auto c = b.clone_shallow();
std::cout<<std::endl<<a.is_same(b); //0
std::cout<<std::endl<<b.is_same(c); //1
std::cout<<std::endl<<c.is_same(a); //0
```

**Move construction** always has **shallow semantics**.
After move new tensor will refer to original's implementation, and original to no implementation. No data is copied.
Original is guaranteed to be **empty**.

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a.clone_shallow();
auto c = std::move(a);
std::cout<<std::endl<<a.empty();    //1
std::cout<<std::endl<<b.empty();    //0
std::cout<<std::endl<<c.empty();    //0
std::cout<<std::endl<<c.is_same(b); //1
```

Having such move semantics we can return local tensor objects by value without data copying:

```cpp
auto make_sequence = [](auto n){
    gtensor::tensor<double> t(n,0);
    std::iota(t.begin(),t.end(),0);
    return t;
};
auto seq = make_sequence(7);
std::cout<<std::endl<<seq;  //[(7){0,1,2,3,4,5,6}]
```

To make deep copy, regardless of tensor's copy construction semantics, `copy()` member function is provided.
It returnes new tensor with shape and elements copied from original.

```cpp
gtensor::tensor<double> a{{1,2,3},{4,5,6}};
auto b = a.clone_shallow();
auto c = b.copy();
c+=1;
std::cout<<std::endl<<a;    //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<b;    //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<c;    //[(2,3){{2,3,4},{5,6,7}}]
```

Default copy-construction semantics can be changed to be **shallow** by using custom argument for `Config` template parameter.
More about this in [GTensor config section](#section_10).

Views are always have shallow copy and move semantics. More about views in [section 5](#section_5) and [section 6](#section_6).

## 5 Expression view and lazy evaluation <a id=section_5></a>

In general, given multidimensional array object, view of this array is another object which provides us with some different way to look at original array's data.
And we can operate on this view object as it were true array. And no data copy required.

GTensor provides two kinds of views:
- View based on some index transformations, it can be slice of original tensor, reshape, transpose or selecting elements using tensor of indeces.
This kind of view always refers to original tensor's elements, using some indexing scheem. More on this kind of view in [next section](#section_6).
- **Expression view** is fundamentally different. It can produce new elements from original tensor or tensors by applying function object to its elements.
What is important that we can apply function object only when we refers to view element - **computation of values of expression view elements is lazy**.

Going back to last example from [section 3](#section_3) we extend it to show nature of lazy evaluation:

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
In fact such specialization type holds shallow copies of `t1` and `t2` (thanks to shallow copy semantic) and binary functor like std::plus\<void\> as data members.
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

GTensor library provides easy way to create expression views using custom function objects:

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<double> t2{7,8,9};
auto custom_f = [](const auto& a, const auto& b, const auto& c){return (a+b)*c;};
auto v1 = gtensor::n_operator(custom_f,t1,t2,t2);
auto v2 = gtensor::n_operator(custom_f,t1,5,3);
auto v3 = gtensor::n_operator(custom_f,t1,5,t2);
std::cout<<std::endl<<v1;   //[(2,3){{56,80,108},{77,104,135}}]
std::cout<<std::endl<<v2;   //[(2,3){{18,21,24},{27,30,33}}]
std::cout<<std::endl<<v3;   //[(2,3){{42,56,72},{63,80,99}}]
```

`n_operator()` free function takes function object and expression operands as arguments and returns expression view object.
Arity of function object must equal to number of operands.
Operands can be tensors or scalars, and must be broadcastable. At least one operand must be tensor.
In fact almost all library functions that perform lazy computations use `n_operator()` in their implementation.

For example possible implementation of `operator+()`:

```cpp
template<typename Impl1, typename Impl2>
auto operator+(const basic_tensor<Impl1>& op1, const basic_tensor<Impl2>& op2){
    return n_operator(std::plus<void>{},op1,op2);
}
```

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
- there is no any caching, each time you refer to element of view, computation is performed
- view holds shallow copies of its operands, so if you mutate operands it will affect view values
- view itself always exposes shallow copy and move semantics
- no temporary copies created, evaluation is elementwise
- evaluation can be easily parallelized
- `value_type` of expression veiw is determined by value_types of its operands and expression itself, look at next example

```cpp
gtensor::tensor<double> t1{{1,2,3},{4,5,6}};
gtensor::tensor<int> t2{3,8,2};
auto sum = t1+t2;   //sum is tensor of doubles
auto cmp = t1>t2;   //cmp is tensor of bools
std::cout<<std::endl<<std::is_same_v<typename decltype(sum)::value_type,double>;    //1
std::cout<<std::endl<<std::is_same_v<typename decltype(cmp)::value_type,bool>;      //1
```

Evaluation can be easily parallelized using overloaded versions of `copy()` and `eval()`:

```cpp
gtensor::tensor<double> t(1000000,0);
std::iota(t.begin(),t.end(),0);
auto res = sin((1/(t+1))).eval(multithreading::exec_pol<4>{});
std::cout<<std::endl<<res;  //[(1000000){0.841,0.479,0.327,...,1e-06,1e-06,1e-06}]
```

GTensor provides its own parallel execution subsystem defined in namespace `multithreading`.
To control level of parallelizm it uses type tags which is specializations of `multithreading::exec_pol` class template.
Integral constant in specialization means number of tasks original task is diveded into.
For example `multithreading::exec_pol<1>` means execute whole task in single thread.
`multithreading::exec_pol<4>` means divide task into four roughly equal parts and try to run they in parallel.
Whether these tasks will actually be executed in parallel mostly depends on the target system.

Almost all library routines that perform reductions have version that takes such tags as their first parameter to run reduction in multiple threads.

Worth note that parallel execution not always decrease computation time, it dependes on operands shapes, complexity of expression and many other factors.
For current example we have next mesurements of computation time:

| computation  method | min,ms  | max,ms  | mean,ms | stded,ms |
|---------------------|---------|---------|---------|----------|
| for loop            | 41.9813 | 76.4672 | 49.6582 | 7.11313  |
| eval(exec_pol<1>)   | 42.1112 | 75.4074 | 51.0164 | 6.87647  |
| eval(exec_pol<2>)   | 27.8526 | 66.9853 | 46.1142 | 7.29578  |
| eval(exec_pol<4>)   | 14.4759 | 33.8893 | 24.4906 | 7.21768  |
| eval(exec_pol<8>)   | 7.5464  | 17.6683 | 8.50008 | 1.8831   |
| eval(exec_pol<16>)  | 4.3118  | 5.7131  | 4.50992 | 0.281551 |

## 6 Slice, reshape, transpose and mapping view <a id=section_6></a>

Along with **expression view** GTensor library provides another kind of views which refers to original tensor's elements, but rearrage it in some way.
As with **expression view**, type of this kind of view is also specialization of `basic_tensor` class template.
View can be made from any other tensor object regardless of its implementation.

### Reshape view

```cpp
using gtensor::config::c_order;
using gtensor::config::f_order;
gtensor::tensor<double> t{{1,2,3,4,5,6},{7,8,9,10,11,12}};
auto v1 = t.reshape(std::vector<int>{-1,3});
auto v2 = t.reshape({3,4});
auto v3 = t.reshape({2,-1,3},c_order{});
auto v4 = t.reshape({6,-1},f_order{});
std::cout<<std::endl<<v1;   //[(4,3){{1,2,3},{4,5,6},{7,8,9},{10,11,12}}]
std::cout<<std::endl<<v2;   //[(3,4){{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
std::cout<<std::endl<<v3;   //[(2,2,3){{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}]
std::cout<<std::endl<<v4;   //[(6,2){{1,4},{7,10},{2,5},{8,11},{3,6},{9,12}}]
```

The first parameter of member function `reshape()` is shape of view, it should be container or std::initializer_list.
One of dimentions in view shape can be -1, in this case its size calculated automatically, based on other dimentions and tensor size.
The second argument is reshape order, if no oreder specified `c_order` is used. Effect of reshape order is the same as in **numpy**.

### Slice view

There are two interfaces to create slice view: using `slice_type` objects explicitly and using std::initializer_list.

```cpp
using tensor_type = gtensor::tensor<double>;
using slice_type = typename tensor_type::slice_type;
tensor_type t{1,2,3,4,5,6,7,8,9,10,11,12};
auto v1 = t(slice_type{1,-1,2});
auto v2 = t({{1,-1,2}});
std::cout<<std::endl<<v1;   //[(5){2,4,6,8,10}]
std::cout<<std::endl<<v2;   //[(5){2,4,6,8,10}]
```

In both cases `slice_type` object is constructed using three parameters: `start`,`stop`,`step`. Negative values are supported and interpreted as counting from the end.
Any of three parameters can be missed. To select all from axis, `slice_type` object should be constructed with no arguments.

```cpp
using tensor_type = gtensor::tensor<double>;
using slice_type = typename tensor_type::slice_type;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto v1 = t(slice_type{{},-1});
auto v2 = t(slice_type{},slice_type{{},{},2});
auto v3 = t(slice_type{{},{},-1},slice_type{1,3});
auto v4 = t(slice_type{5},slice_type{});
auto v5 = t(slice_type{{},5},slice_type{});
std::cout<<std::endl<<v1;   //[(2,4){{1,2,3,4},{5,6,7,8}}]
std::cout<<std::endl<<v2;   //[(3,2){{1,3},{5,7},{9,11}}]
std::cout<<std::endl<<v3;   //[(3,2){{10,11},{6,7},{2,3}}]
std::cout<<std::endl<<v4;   //[(0,4){}]
std::cout<<std::endl<<v5;   //[(3,4){{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
```

Any view from above example also can be created without using `slice_type` objects explicitly:

```cpp
using tensor_type = gtensor::tensor<double>;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto v1 = t({{{},{-1}}});
auto v2 = t({{},{{},{},2}});
auto v3 = t({{{},{},-1},{1,3}});
auto v4 = t({{5},{}});
auto v5 = t({{{},5},{}});
std::cout<<std::endl<<v1;   //[(2,4){{1,2,3,4},{5,6,7,8}}]
std::cout<<std::endl<<v2;   //[(3,2){{1,3},{5,7},{9,11}}]
std::cout<<std::endl<<v3;   //[(3,2){{10,11},{6,7},{2,3}}]
std::cout<<std::endl<<v4;   //[(0,4){}]
std::cout<<std::endl<<v5;   //[(3,4){{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
```

Both interfaces are equivalent and it is matter of taste which one to use.

Making slice view with dimension reduction only possible when using `slice_type` objects explicitly:

```cpp
using tensor_type = gtensor::tensor<double>;
using slice_type = typename tensor_type::slice_type;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto v1 = t(1);
auto v2 = t(2,1);
auto v3 = t(slice_type{},0);
auto v4 = t(slice_type{1},2);
auto v5 = t(1,slice_type{1,-1});
std::cout<<std::endl<<v1;   //[(4){5,6,7,8}]
std::cout<<std::endl<<v2;   //[(){10}]
std::cout<<std::endl<<v3;   //[(3){1,5,9}]
std::cout<<std::endl<<v4;   //[(2){7,11}]
std::cout<<std::endl<<v5;   //[(2){6,7}]
```

To create slice view dynamically at runtime, you can use container of `slice_type` objects:

```cpp
using tensor_type = gtensor::tensor<double>;
using slice_type = typename tensor_type::slice_type;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
std::vector<slice_type> subscripts{};
subscripts.push_back(slice_type{0,-1});
subscripts.push_back(slice_type{1,-1});
auto v = t(subscripts);
std::cout<<std::endl<<v;    //[(2,2){{2,3},{6,7}}]
```

To make `slice_type` object which causes dimension reduce, you should use special `reduce_tag_type`:

```cpp
using tensor_type = gtensor::tensor<double>;
using slice_type = typename tensor_type::slice_type;
using reduce_tag_type = typename slice_type::reduce_tag_type;
tensor_type t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
std::vector<slice_type> subscripts{};
subscripts.push_back(slice_type{});
subscripts.push_back(slice_type{2,reduce_tag_type{}});
auto v = t(subscripts);
std::cout<<std::endl<<v;    //[(3){3,7,11}]
```

### Transopose view

In general transpose view rearrange axes of original tensor. By default axes rearranged in reverse order.

```cpp
gtensor::tensor<double> t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto v1 = t.transpose();
auto v2 = t.transpose().transpose();
std::cout<<std::endl<<v1;   //[(4,3){{1,5,9},{2,6,10},{3,7,11},{4,8,12}}]
std::cout<<std::endl<<v2;   //[(3,4){{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
```

We can specify axes order explicitly:

```cpp
gtensor::tensor<double> t{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}};
auto v1 = t.transpose();
auto v2 = t.transpose(2,1,0);
auto v3 = t.transpose(std::vector<int>{1,0,2});
auto v4 = t.transpose(0,1,2);
std::cout<<std::endl<<v1;   //[(2,2,3){{{1,5,9},{3,7,11}},{{2,6,10},{4,8,12}}}]
std::cout<<std::endl<<v2;   //[(2,2,3){{{1,5,9},{3,7,11}},{{2,6,10},{4,8,12}}}]
std::cout<<std::endl<<v3;   //[(2,3,2){{{1,2},{5,6},{9,10}},{{3,4},{7,8},{11,12}}}]
std::cout<<std::endl<<v4;   //[(3,2,2){{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}]
```

### Mapping view

Mapping view uses another tensor with indexes or bools as subscript to select elements from original tensor.
Selecting is performed according to rules in **numpy**.

Next example uses tensor of integral indexes to select elements:

```cpp
using gtensor::tensor;
tensor<double> t{1,2,3,4,5,6,7,8,9,10,11,12};
auto v1 = t(tensor<int>{2,1,0,8,9,0});
auto v2 = t(tensor<int>{{2,1,0},{8,9,0}});
auto v3 = t.reshape(3,4)(tensor<int>{1,0,1,2});
auto v4 = t.reshape(3,4)(tensor<int>{0,0,2,2},tensor<int>{0,3,0,3});
std::cout<<std::endl<<v1;   //[(6){3,2,1,9,10,1}]
std::cout<<std::endl<<v2;   //[(2,3){{3,2,1},{9,10,1}}]
std::cout<<std::endl<<v3;   //[(4,4){{5,6,7,8},{1,2,3,4},{5,6,7,8},{9,10,11,12}}]
std::cout<<std::endl<<v4;   //[(4){1,4,9,12}]
```

Using container of tensors of indexes is also supported:

```cpp
using gtensor::tensor;
tensor<double> t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
std::vector<tensor<int>> subscript{};
subscript.push_back(tensor<int>{0,0,2,2});
subscript.push_back(tensor<int>{0,3,0,3});
auto v = t(subscript);
std::cout<<std::endl<<v;    //[(4){1,4,9,12}]
```

Next example uses tensor of bools to select elements, using `bool` type is mandatory:

```cpp
using gtensor::tensor;
tensor<double> t{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
auto v1 = t(tensor<bool>{false,true,false,true});
auto v2 = t(tensor<bool>{{false,true,false,true},{true,false,true,false},{false,true,true,false}});
std::cout<<std::endl<<v1;   //[(2,4){{1,2,3,4},{9,10,11,12}}]
std::cout<<std::endl<<v2;   //[(6){2,4,5,7,10,11}]
```

We can use **expression view** of `bool` value_type to select elements by condition:

```cpp
gtensor::tensor<double> t{{7,3,4,6},{1,5,6,2},{1,8,3,5},{0,2,6,2}};
auto v1 = t(t>3 && t.not_equal(6));
auto v2 = t((t*t)<(t+10));
std::cout<<std::endl<<v1;   //[(5){7,4,5,8,5}]
std::cout<<std::endl<<v2;   //[(8){3,1,2,1,3,0,2,2}]
```

## 7 `basic_tensor` assign semantic <a id=section_7></a>

`basic_tensor` objects can expose different assign semantic, depending on its type and assign expression:
- value assign semantic
- elementwise (or broadcast) assign semantic

Consider example:

```cpp
using tensor_type = gtensor::tensor<double>;
tensor_type a{1,2,3};
tensor_type b{{4,5,6},{7,8,9}};
a = b;
std::cout<<std::endl<<a;    //[(2,3){{4,5,6},{7,8,9}}]
a = a + b;
std::cout<<std::endl<<a;    //[(2,3){{8,10,12},{14,16,18}}]
a = 0;
std::cout<<std::endl<<a;    //[(){0}]
```

As expected after first assign `a` has the same value as `b`.
After second assign `a` has the same value as `a + b`.
All assignments expose **value assign semantic**.

In next example assignment has different semantic:

```cpp
using tensor_type = gtensor::tensor<double>;
tensor_type a{{1,2,3},{4,5,6}};
a.assign(tensor_type{7,8,9});
std::cout<<std::endl<<a;    //[(2,3){{7,8,9},{7,8,9}}]
a.assign(a+1);
std::cout<<std::endl<<a;    //[(2,3){{8,9,10},{8,9,10}}]
a.assign(0);
std::cout<<std::endl<<a;    //[(2,3){{0,0,0},{0,0,0}}]
```

Here we use `assign()` member function to assign to `a` - lhs.
This function has broadcast assign semantic.
It takes single argument, that can be tensor or scalar - rhs.
If rhs is tensor it must be broadcastable with lhs.

### Assign to view

In first example `operator=()` exposes value assign semantic, but this is not always the case.
The point is that definition of `operator=()` in `basic_tensor` class template differs for **lvalue** and **rvalue** objects i.e. operator is **ref-qualified**.
Being called on **lvalue** object assignment operator has value semantic, on **rvalue** object it has broadcast semantic.

It can be useful when **assigning to view**:

```cpp
using tensor_type = gtensor::tensor<double>;
tensor_type a{{7,3,4,6},{1,5,6,2},{1,8,3,5},{0,2,6,2}};
a(a>6) = 0;
std::cout<<std::endl<<a;    //[(4,4){{0,3,4,6},{1,5,6,2},{1,0,3,5},{0,2,6,2}}]
a({{1,-1}}) = tensor_type{-1,-1,-1,-1};
std::cout<<std::endl<<a;    //[(4,4){{0,3,4,6},{-1,-1,-1,-1},{-1,-1,-1,-1},{0,2,6,2}}]
auto v = a(0);
std::move(v) = 11;
std::cout<<std::endl<<a;    //[(4,4){{11,11,11,11},{-1,-1,-1,-1},{-1,-1,-1,-1},{0,2,6,2}}]
```

Using member function `assign()` would have the same effect.

```cpp
using tensor_type = gtensor::tensor<double>;
tensor_type a{{7,3,4,6},{1,5,6,2},{1,8,3,5},{0,2,6,2}};
a(a>6).assign(0);
std::cout<<std::endl<<a;    //[(4,4){{0,3,4,6},{1,5,6,2},{1,0,3,5},{0,2,6,2}}]
a({{1,-1}}).assign(tensor_type{-1,-1,-1,-1});
std::cout<<std::endl<<a;    //[(4,4){{0,3,4,6},{-1,-1,-1,-1},{-1,-1,-1,-1},{0,2,6,2}}]
auto v = a(0);
v.assign(11);
std::cout<<std::endl<<a;    //[(4,4){{11,11,11,11},{-1,-1,-1,-1},{-1,-1,-1,-1},{0,2,6,2}}]
```

As consequence `a=b` and `std::move(a)=b` usually have different effect.

Any assign to **expression view** will not compile. Value assign to ordinary view will not compile.
In fact we can value assign only to tensor with storage implementation i.e. created using `tensor` class template.


|                 | value assign |             broadcast assign            |
|:---------------:|:------------:|:---------------------------------------:|
|      tensor     |    lhs=rhs   | lhs.assign(rhs)<br/> std::move(lhs)=rhs |
|       view      |       X      | lhs.assign(rhs)<br/> std::move(lhs)=rhs |
| expression view |       X      |                    X                    |


## 8 `basic_tensor` equality <a id=section_8></a>

Two tensors are considered **equal** if their shapes and elements are equal.

```cpp
using tensor_type = tensor<double>;
tensor_type a{{1,2,3},{4,5,6}};
tensor_type b{1,2,3,4,5,6};
tensor_type c{{1,2,3},{3,2,1}};
std::cout<<std::endl<<(a==b);   //0
std::cout<<std::endl<<(a==b.reshape(2,3));  //1
std::cout<<std::endl<<(a.flatten()==b);     //1
std::cout<<std::endl<<(a==c);   //0
```

If value_type is **IEEE 754** floating-point type, `nan` elements optionaly can be compared as equal:

```cpp
using tensor_type = tensor<double>;
constexpr auto nan = std::numeric_limits<double>::quiet_NaN()
tensor_type a{{1,2,nan},{4,5,6}};
tensor_type b{{1,2,nan},{4,5,6}};
std::cout<<std::endl<<tensor_equal(a,b);        //0
std::cout<<std::endl<<tensor_equal(a,b,true);   //1
```

`tensor_equal()` free function takes two tensors to compare and optional bool argument, if `true` is passed then `nans` compared as equal. It is `false` by default.

For floating-point value_type strict elements equality is not very useful. More practical approach is to check whether tensors are close within tolerance.
Two tensors are considered **close** if their shapes are equal and elements are close within tolerance.

```cpp
using tensor_type = tensor<double>;
tensor_type a{{1.12345,2.12345,3.12345},{4.12345,5.12345,6.12345}};
tensor_type b{{1.12345,2.12345,3.12355},{4.12325,5.12345,6.12375}};
std::cout<<std::endl<<tensor_close(a,b);    //0
std::cout<<std::endl<<tensor_close(a,b,1E-6,1E-6);  //0
std::cout<<std::endl<<tensor_close(a,b,1E-3,1E-3);  //1
```

`tensor_close()` free function takes two tensors to compare and optional absolute and relative tolerance. By default both tolerance are equal to machine epsilon.

`allclose()` free function is similar to `tensor_close()` except tensors may be broadcastable.

```cpp
using tensor_type = tensor<double>;
tensor_type a{{1.12345,2.12345,3.12345},{1.12345,2.12345,3.12345}};
tensor_type b{1.12345,2.12345,3.12355};
std::cout<<std::endl<<allclose(a,b);    //0
std::cout<<std::endl<<allclose(a,b,1E-6,1E-6);  //0
std::cout<<std::endl<<allclose(a,b,1E-3,1E-3);  //1
```

`basic_tensor` class template also has member functions `equal()` and `not_equal()` which have quite different meaning than functions we just discussed above.
These functions provide **broadcast equality and inequality** the result is **expression view** of bool value_type.

```cpp
using tensor_type = tensor<double>;
tensor_type t{{1,2,3},{3,2,1}};
std::cout<<std::endl<<t.equal(3);   //[(2,3){{0,0,1},{1,0,0}}]
std::cout<<std::endl<<t.not_equal(3);   //[(2,3){{1,1,0},{0,1,1}}]
std::cout<<std::endl<<t.equal(tensor_type{{0,2,1},{3,2,0}});    //[(2,3){{0,1,0},{1,1,0}}]
std::cout<<std::endl<<t.not_equal(tensor_type{3,2,0});  //[(2,3){{1,0,1},{0,0,1}}]
```

## 9 `basic_tensor` data and meta-data interface <a id=section_9></a>

Next example shows member functions `basic_tensor` provides to access its **meta-data**.

```cpp
gtensor::tensor<double> t{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}};
auto dim = t.dim();
auto size = t.size();
auto empty = t.empty();
const auto& shape = t.shape();
const auto& strides = t.strides();
std::cout<<std::endl<<dim;  //3
std::cout<<std::endl<<size; //12
std::cout<<std::endl<<empty;    //0
std::cout<<std::endl;
std::copy(shape.begin(),shape.end(),std::ostream_iterator<int>(std::cout,",")); //2,2,3,
std::cout<<std::endl;
std::copy(strides.begin(),strides.end(),std::ostream_iterator<int>(std::cout,",")); //6,3,1,
```

To access its **data** `basic_tensor` provides iterator interface.

```cpp
using gtensor::config::c_order;
using gtensor::config::f_order;
gtensor::tensor<double,c_order> t_c{{1,2,3},{4,5,6}};
gtensor::tensor<double,f_order> t_f{{1,2,3},{4,5,6}};
std::cout<<std::endl;
std::copy(t_c.begin(),t_c.end(),std::ostream_iterator<double>(std::cout,","));  //1,2,3,4,5,6,
std::cout<<std::endl;
std::copy(t_f.begin(),t_f.end(),std::ostream_iterator<double>(std::cout,","));  //1,2,3,4,5,6,
std::cout<<std::endl;
std::copy(t_c.rbegin(),t_c.rend(),std::ostream_iterator<double>(std::cout,","));    //6,5,4,3,2,1,
std::cout<<std::endl;
std::copy(t_f.rbegin(),t_f.rend(),std::ostream_iterator<double>(std::cout,","));    //6,5,4,3,2,1,
```

By default iterator traverse order is **c_order** and it doesn't depend on tensor's **layout**.
Default traverse order can be changed using `Config` template parameter of `tensor` class template.
More about this in next section.

To specify traverse order explicitly in programm `basic_tensor` provide `traverse_order_adapder` helper.

```cpp
using gtensor::config::c_order;
using gtensor::config::f_order;
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto tr_adapt_c = t.traverse_order_adapter(c_order{});
std::cout<<std::endl;
std::copy(tr_adapt_c.begin(),tr_adapt_c.end(),std::ostream_iterator<double>(std::cout,","));    //1,2,3,4,5,6,
std::cout<<std::endl;
std::copy(tr_adapt_c.rbegin(),tr_adapt_c.rend(),std::ostream_iterator<double>(std::cout,","));  //6,5,4,3,2,1,

auto tr_adapt_f = t.traverse_order_adapter(f_order{});
std::cout<<std::endl;
std::copy(tr_adapt_f.begin(),tr_adapt_f.end(),std::ostream_iterator<double>(std::cout,","));    //1,4,2,5,3,6,
std::cout<<std::endl;
std::copy(tr_adapt_f.rbegin(),tr_adapt_f.rend(),std::ostream_iterator<double>(std::cout,","));  //6,3,5,2,4,1,
```

There are several important points regarding **traverse order** and **tensor layout**:
- the same `c_order` and `f_order` type tags are used to specify both traverse order and tensor layout
- although **traverse order** and **tensor layout** are related, they mean different things.
**Tensor layout** determines order of elements in underlaying storage, whereas **traverse order** determines order of elements when they are accessed using iterator.
- any combinations of layout and traverse order are possible
- iterator performs the best when traverse order and layout are the same. Traverse tensor elements in order different than tensor layout comes at some cost.

Most of the GTensor library functions that takes input tensor, and return result tensor doesn't change layout. But some does.

```cpp
using gtensor::config::c_order;
using gtensor::config::f_order;
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto t_original = t.copy();
auto t_c = t.copy(c_order{});
auto t_f = t.copy(f_order{});
std::cout<<std::endl<<t_original;  //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<t_c;         //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<t_f;         //[(2,3){{1,2,3},{4,5,6}}]
std::cout<<std::endl<<std::is_same_v<typename decltype(t_original)::order,typename decltype(t)::order>;    //1
std::cout<<std::endl<<std::is_same_v<typename decltype(t_c)::order,c_order>;    //1
std::cout<<std::endl<<std::is_same_v<typename decltype(t_f)::order,f_order>;    //1
```

In this example first copy has original layout, next two copies have `c_order` and `f_order` layouts accordingly.
Worth mention that `eval()` always returns result in original layout.

```cpp
using gtensor::config::c_order;
using gtensor::config::f_order;
gtensor::tensor<double> t{{1,2,3},{4,5,6}};
auto v = t.reshape({3,2});
auto v_c = t.reshape({3,2},c_order{});
auto v_f = t.reshape({3,2},f_order{});
std::cout<<std::endl<<v;    //[(3,2){{1,2},{3,4},{5,6}}]
std::cout<<std::endl<<v_c;  //[(3,2){{1,2},{3,4},{5,6}}]
std::cout<<std::endl<<v_f;  //[(3,2){{1,5},{4,3},{2,6}}]
```

`reshape()` has the same effect as in **numpy**, it uses `c_order` by default.

### Trivial iterator

It doesn't matter you traverse view tensor, expression view tensor, or tensor with storage implementation. Everything works the same.
But for **expression view** important optimization is possible.

Next example explains this:

```cpp
gtensor::tensor<double> t1{1,2,3};
gtensor::tensor<double> t2{4,5,6};
auto v_trivial = t1+t2;
auto v = t1+t2.reshape(-1,1)
std::cout<<std::endl<<v_trivial;    //[(3){5,7,9}]
std::cout<<std::endl<<v_trivial.is_trivial();   //1
std::cout<<std::endl<<v;    //[(3,3){{5,6,7},{6,7,8},{7,8,9}}]
std::cout<<std::endl<<v.is_trivial();   //0
```

The first expression view `v_trivial` is created using two operands of same shape,
to evaluate it we need go along its operands element by element and apply to each pair of elements `plus` functor. The logic is simple and introduce a little overhead.

On other hand, the second expression view `v` can't be evaluated in that way, due to different operands shapes.
In this case we need more complex logic to handle with broadcast, whiсh in turn introduces much more overhead.

We call expression views like first **trivial expression view**.
`basic_tensor` provides `is_trivial()` member function, that returns `true` if trensor is **trivial** and `false` otherwise.
Tensors with storage implementation is always considered **trivial**, whether **view** is trivial depends on its structure.

GTensor library provides special iterator interface to traverse trivial tensors.

```cpp
gtensor::tensor<double> t1{1,2,3};
gtensor::tensor<double> t2{4,5,6};
auto v_trivial = (t1+t2)*(t1-t2);
std::cout<<std::endl<<v_trivial;    //[(3){-15,-21,-27}]
std::cout<<std::endl<<v_trivial.is_trivial();   //1
std::cout<<std::endl;
std::copy(v_trivial.begin_trivial(),v_trivial.end_trivial(),std::ostream_iterator<double>(std::cout,","));  //-15,-21,-27,
std::cout<<std::endl;
std::copy(v_trivial.rbegin_trivial(),v_trivial.rend_trivial(),std::ostream_iterator<double>(std::cout,","));    //-27,-21,-15,
```
In fact library exploits this to optimize expression view evaluation.
Trivial iterator interface works the same as ordinary i.e. we can change traverse order, use reverse iteration and const iterators.

Using trivial iterator interface to traverse tensor for which `is_trivial()` returns `false` is **UB**.

## 10 GTensor config <a id=section_10></a>

To make tensor know its configuration, template type parmeter `Config` is used.
Default config is `gtensor::config::default_config`.

```cpp
enum class div_modes : std::size_t {native, libdivide};
enum class engines : std::size_t {expression_template};
enum class orders : std::size_t {c,f};
enum class cloning_semantics : std::size_t {deep,shallow};

using mode_div_native = std::integral_constant<div_modes, div_modes::native>;
using mode_div_libdivide = std::integral_constant<div_modes, div_modes::libdivide>;
using engine_expression_template = std::integral_constant<engines, engines::expression_template>;
using c_order = std::integral_constant<orders, orders::c>;
using f_order = std::integral_constant<orders, orders::f>;
using deep_semantics = std::integral_constant<cloning_semantics, cloning_semantics::deep>;
using shallow_semantics = std::integral_constant<cloning_semantics, cloning_semantics::shallow>;

struct default_config
{
    using engine = engine_expression_template;

    //specify whether to use optimized division
    using div_mode = mode_div_libdivide;
    //using div_mode = mode_div_native;

    //specify default traverse order of iterators
    using order = c_order;
    //using order = f_order;

    //cloning semantics - determines effect of tensor copy construction
    using semantics = deep_semantics;
    //using semantics = shallow_semantics;

    //data elements storage template
    template<typename T> using storage = gtensor::basic_storage<T>;

    //meta-data elements storage template i.e. shape, strides are specialization of shape
    //must provide std::vector like interface
    template<typename T> using shape = gtensor::stack_prealloc_vector<T,8>;

    //generally when public interface expected container parameter it may be any type providig usual container semantic and interface: iterators, aliases...
    //specialization of config_type::container uses as return type in public interface
    //it may be used by implementation as general purpose container
    //must provide std::vector like interface
    template<typename T> using container = std::vector<T>;

    //index_map specialization is used in mapping_descriptor that is descriptor type of mapping_view
    //it is natural to use storage as index_map in general, but if storage is specific e.g. map to file system or network, these should differ
    template<typename T> using index_map = storage<T>;
};
```

As we see `default_config` contain only type and template aliases and no data members.
`default_config` doesn't provide types for underlaying storage and shape but only aliase templates.
Motivation for such design is to make it possible to **rebind** to another value_type.

Look at `tensor` class template definition

```cpp
template<typename T, typename Layout = config::c_order, typename Config = config::extend_config_t<config::default_config,T>>
class tensor : public basic_tensor<typename tensor_factory_selector_t<Config,T,Layout>::result_type>
{
...
};
```

Default argument for `Config` parameter is `config::extend_config_t<config::default_config,T>`, result type is like `default_config` struct,
but with some new aliases defined: `storage_type`, `shape_type` which are **type aliases**.
We say that `gtensor::config::extend_config_t` trait **rebind** given config to given data element type.

Consider example:

```cpp
gtensor::tensor<int> t{1,2,3,4,5};
auto t_double = t.template copy<double>();
std::cout<<std::endl<<std::is_same_v<typename decltype(t)::value_type, int>;    //1
std::cout<<std::endl<<std::is_same_v<typename decltype(t_double)::value_type, double>;  //1
```

Here we explicitly specialize `copy()` to make copy of `t` but with `double` value_type.
Config of result type will be original config rebinded to `double`.

Worth mention that after **rebind** storage_type may be quiet different, consider `std::vector<bool>`.