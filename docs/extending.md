# Extending GTensor

GTensor library is designed to be easily extended in some different ways:

- library doesn't depend on particular data types for containers, data and meta-data elements.
So custom types can be used, provided they meet some common requirements.
- thanks to modular design new modules can be added, functionality and implementation of existing modules can be changed.
- library provides generic way to implement routines to operate on tensors objects in broadcast elementwise and lazy manner.
As well as universal way to evaluate such lazy results. For efficient evaluation of complex lazy expressions parallel execution subsystem is provided.

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
    minimal_onstack_storage(size_type n)
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

`shape` and `container` generally should provide interface like `std::vector`.

## Data type customization

GTensor library support all standart data types like `int`, `float`, `double` and `std::complex`.
Higher order tensors - types like `tensor<tensor<double>>` are also supported.

Next example shows possible implementation of `matrix22` class template that represents square 2x2 matrix and using it as tensor's element data type.
In `matrix22` implementation we will use `custom_tensor` defined in previous section.

```cpp
template<typename T>
class matrix22
{
    using base_type = custom_tensor<T>;
    base_type base_;
public:
    template<typename...Ts>
    matrix22(const gtensor::basic_tensor<Ts...>& other):
        base_(other)
    {}
    matrix22():
        base_{{T{},T{}},{T{},T{}}}
    {}
    matrix22(const T& e):
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
auto operator+(const matrix22<T>& a, const matrix22<T>& b){
    return matrix22<T>(a.base()+b.base());
}
template<typename T>
auto operator*(const matrix22<T>& a, const matrix22<T>& b){
    return matrix22<T>(matmul(a.base(),b.base()));
}
```

Now we can use `matrix22` as tensor's element data type:

```cpp
using value_type = matrix22<double>;
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
```
