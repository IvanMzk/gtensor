# Extending GTensor

GTensor library is designed to be easily extended in some different ways:

- library doesn't depend on particular data types for containers, data and meta-data elements.
So custom types can be used, provided they meet some common requirements.
- thanks to modular design new modules can be added, existing modules can be extended with new functionality, existing functionality can be customized.
- library provides generic way to implement routines to operate on tensors objects in broadcast elementwise and lazy manner.
As well as universal way to evaluate such lazy results. For efficient evaluation parallel execution subsystem is provided.

## Containers customization

Next example shows implementation of simple storage class template with maximum size predifined at compile time.

```cpp
template<typename T, std::size_t N>
class minimal_onstack_storage
{
    std::array<T,N> elements_;
    using base_type = std::array<T,N>;
public:
    using value_type = typename base_type::value_type;
    using size_type = typename base_type::size_type;
    using difference_type = typename base_type::difference_type;
    explicit minimal_onstack_storage(size_type n)
    {
        if (n > N){
            throw std::length_error("invalid storage size");
        }
    }
    const value_type& operator[](size_type i)const{
        return elements_[i];
    }
    value_type& operator[](size_type i){
        return elements_[i];
    }
};
```

To configure tensor to use custom storage, new **config type** should be defined and used as argument for `Config` template parameter of `tensor` class template.

```cpp
struct custom_config : gtensor::config::default_config
{
    template<typename T> using storage = minimal_onstack_storage<T,4>;
};
template<typename T, typename Layout=gtensor::config::c_order> using custom_tensor = gtensor::tensor<T,Layout,gtensor::config::extend_config_t<custom_config,T>>;
```

Thats all, we can use `custom_tensor` as usual. Of course we should take care not to exceed predefined storage size.

```cpp
custom_tensor<double> a{{1,2},{3,4}};
custom_tensor<double> b{5,6,7,8};
custom_tensor<double> c{2,1};
std::cout<<std::endl<<a;    //[(2,2){{1,2},{3,4}}]
std::cout<<std::endl<<b;    //[(4){5,6,7,8}]
std::cout<<std::endl<<c;    //[(2){2,1}]
std::cout<<std::endl<<(a+b.reshape(2,2))*(c+1); //[(2,2){{18,16},{30,24}}]
std::cout<<std::endl<<matmul(a,a.transpose());  //[(2,2){{5,11},{11,25}}]
```

This simple example shows the main idea. Similar approach can be used to customize other containers if needed.

Containers requirements are different.

`storage` must provide:
- type aliases: `value_type`, `difference_type`, `size_type`
- at least constructor that takes container `size` as its first parameter
- at least subscript operator interface or iterator `begin()`, `end()` interface

Constructors that takes **iterators range**, **size and value** will be considered if defined.

The same with `data()` member function, if it is defined then tensor will provide access to underlaying data buffer.

In general `storage` is abstraction which must provide interface to address data elements using flat index.
Flat index must have semantic of signed integral type, its type must be provided using `difference_type` type alias.
Implementation doesn't matter, data elements may be stored on file system, in memory, on network or database or something else, provided above requirements are met.

`shape` and `container` generally should provide interface like `std::vector`.

## Data type customization

GTensor library support all standart data types like `int`, `float`, `double` and `std::complex`.
Higher order tensors - types like `tensor<tensor<double>>` are also supported.

Next example shows possible implementation of `matrix22` class template that represents square 2x2 matrix and using it as tensor's element data type.
In `matrix22` implementation we will use `custom_tensor` defined in previous section.

```cpp
namespace mat22{

template<typename T>
class matrix22
{
    using base_type = custom_tensor<T>;
    base_type base_;
public:
    template<typename...Ts>
    explicit matrix22(const gtensor::basic_tensor<Ts...>& other):
        base_(other)
    {}
    matrix22():
        base_{{T{},T{}},{T{},T{}}}
    {}
    explicit matrix22(const T& e):
        base_{{e,T{}},{T{},e}}
    {}
    matrix22(const T& e00,const T& e01,const T& e10,const T& e11):
        base_{{e00,e01},{e10,e11}}
    {}
    auto& base()const{
        return base_;
    }
};
template<typename T>
auto& operator<<(std::ostream& os, const matrix22<T>& m){
    return os<<"["<<m.base().element(0,0)<<","<<m.base().element(0,1)<<","<<m.base().element(1,0)<<","<<m.base().element(1,1)<<"]";
}
template<typename T>
bool operator==(const matrix22<T>& a, const matrix22<T>& b){
    return a.base()==b.base();
}
template<typename T>
auto operator+(const matrix22<T>& a, const matrix22<T>& b){
    return matrix22<T>(a.base()+b.base());
}
template<typename T>
auto operator-(const matrix22<T>& a, const matrix22<T>& b){
    return matrix22<T>(a.base()-b.base());
}
template<typename T>
auto operator*(const matrix22<T>& a, const matrix22<T>& b){
    return matrix22<T>(matmul(a.base(),b.base()));
}
template<typename T>
auto norm(const matrix22<T>& m){
    const auto& b = m.base();
    return std::max({std::abs(b.element(0,0)),std::abs(b.element(0,1)),std::abs(b.element(1,0)),std::abs(b.element(1,1))});
}

}   //end of namespace mat22
```

Now we can use `matrix22` as tensor's element data type:

```cpp
using value_type = mat22::matrix22<double>;
using tensor_type = gtensor::tensor<value_type>;

tensor_type a{{value_type(1,0,2,1),value_type(2,1,1,0)},{value_type(1,4,2,3),value_type(3,3,2,2)}};
tensor_type b{{value_type(2,1,1,0),value_type(2,3,0,2)},{value_type(4,4,1,3),value_type(1,3,1,2)}};
tensor_type c(value_type(1));

std::cout<<std::endl<<a;    //[(2,2){{[1,0,2,1],[2,1,1,0]},{[1,4,2,3],[3,3,2,2]}}]
std::cout<<std::endl<<b;    //[(2,2){{[2,1,1,0],[2,3,0,2]},{[4,4,1,3],[1,3,1,2]}}]
std::cout<<std::endl<<c;    //[(){[1,0,0,1]}]
std::cout<<std::endl<<a*b;  //[(2,2){{[2,1,5,2],[4,8,2,3]},{[8,16,11,17],[6,15,4,10]}}]
std::cout<<std::endl<<a*c;  //[(2,2){{[1,0,2,1],[2,1,1,0]},{[1,4,2,3],[3,3,2,2]}}]
std::cout<<std::endl<<(a+b)*(a+c);  //[(2,2){{[8,2,8,2],[16,8,5,3]},{[26,52,18,36],[28,30,20,21]}}]
std::cout<<std::endl<<a.sum();  //[(){[7,8,7,6]}]
std::cout<<std::endl<<b.prod(); //[(){[64,152,28,67]}]
std::cout<<std::endl<<a.cumsum();   //[(4){[1,0,2,1],[3,1,3,1],[4,5,5,4],[7,8,7,6]}]
std::cout<<std::endl<<b.cumprod();  //[(4){[2,1,1,0],[4,8,2,3],[24,40,11,17],[64,152,28,67]}]
std::cout<<std::endl<<(a==b);   //0
std::cout<<std::endl<<tensor_close(a,a);    //1
```

## Modules customization

Most of GTensor library functionality is implemented in modules.
Each module has associated header file, to be included to use module.
Modules are defined in `gtensor` namespace.

Module contains module interface and module implementation.

Module interface is set of free functions template which use traits defined in `module_selector.hpp` header file to dispatch call to right module implementation.

Module implementation is defined in class or class template which provides public member functions interface to be called from module interface free function.

### Adding new functionality

Next example shows main idea of how to extend existing module with a new routine.

#### **`extended_tensor_math.hpp`**
```cpp
#include "tensor_math.hpp"
namespace gtensor{

//module implementation
struct custom_tensor_math : gtensor::tensor_math{
    template<typename...Args>
    static auto sum_reciprocal(const Args&...args){
        return n_operator([](const auto&...e){return ((1/e)+...);},args...);
    }
};

//module interface
template<typename...Args>
    static auto sum_reciprocal(const Args&...args){
    using config_type = detail::common_config_type_t<Args...>;
    return tensor_math_selector_t<config_type>::sum_reciprocal(args...);
}

//specialization of dispatching trait
template<typename Config>
struct tensor_math_selector<Config>
{
    using type = custom_tensor_math;
};

}   //end of namespace gtensor
```

Here we extend `tensor_math` module with routine to compute sum of reciprocals of given arguments.
Now we can use `sum_reciprocal` as well as all routines were defined in `tensor_math` module.

#### **`main.cpp`**
```cpp
#include "tensor.hpp"
#include "extended_tensor_math.hpp"

int main(int argc, const char*argv[])
{
    gtensor::tensor<double> a{{1,2,3},{4,5,6}};
    gtensor::tensor<double> b{3,2,1};

    std::cout<<std::endl<<sum_reciprocal(a);    //[(2,3){{1,0.5,0.333},{0.25,0.2,0.167}}]
    std::cout<<std::endl<<sum_reciprocal(a,b);  //[(2,3){{1.33,1,1.33},{0.583,0.7,1.17}}]
    std::cout<<std::endl<<(1/a + 1/b);  //[(2,3){{1.33,1,1.33},{0.583,0.7,1.17}}]
    std::cout<<std::endl<<sum_reciprocal(a,b,2.0);  //[(2,3){{1.83,1.5,1.83},{1.08,1.2,1.67}}]
    std::cout<<std::endl<<pow(sum_reciprocal(a,b),0.5); //[(2,3){{1.15,1,1.15},{0.764,0.837,1.08}}]
    std::cout<<std::endl<<sum(sum_reciprocal(a,b,2.0)); //[(){9.12}]

    return 0;
}
```

Although `sum_reciprocal(a,b)` is equivalent to `1/a + 1/b`, former may be more efficient when evaluated, due to more shallow expression tree.

### Customizing existing functionality

In next example we define custom `n_operator` which is **not lazy** and always returns evaluated result.

#### **`evaluating_expression_template_operator.hpp`**
```cpp
namespace gtensor{

struct evaluating_expression_template_operator : expression_template_operator{
    template<typename F, typename...Operands>
    static auto n_operator(F&& f, Operands&&...operands){
        using config_type = detail::common_config_type_t<std::decay_t<Operands>...>;
        return expression_template_operator::n_operator(std::forward<F>(f), std::forward<Operands>(operands)...).eval();
    }
};

template<typename Config>
struct generalized_operator_selector<Config>
{
    using type = evaluating_expression_template_operator;
};

}   //end of namespace gtensor
```

#### **`main.cpp`**
```cpp
#include "tensor.hpp"
#include "evaluating_expression_template_operator.hpp"

int main(int argc, const char*argv[])
{
    gtensor::tensor<double> a{{1,2,3},{4,5,6}};
    gtensor::tensor<double> b{1,2,0};

    auto c = a+b;   //c is not expression view, but has storage implementation, as if were constructed using tensor<double>{...} expression
    std::cout<<std::endl<<c.data()[2];  //3
    c+=1;
    std::cout<<std::endl<<c;    //[(2,3){{3,5,4},{6,8,7}}]
    c(c>4)+=1;
    std::cout<<std::endl<<c;    //[(2,3){{3,6,4},{7,9,8}}]

    return 0;
}
```

`n_operator` which parallelize evaluation can be implemented in similar manner:

```cpp
struct evaluating_expression_template_operator : expression_template_operator{
    template<typename F, typename...Operands>
    static auto n_operator(F&& f, Operands&&...operands){
        using config_type = typename detail::first_tensor_type_t<std::decay_t<Operands>...>::config_type;
        return expression_template_operator::n_operator(std::forward<F>(f), std::forward<Operands>(operands)...).eval(multithreading::exec_pol<4>{});
    }
};
```

Any other functionality, including operators, can be customized in ways described above to meet requirements.