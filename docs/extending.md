# Extending GTensor

GTensor library is designed to be easily extended in some different ways:

- library doesn't depend on particular data types for containers, data and meta-data elements.
So custom types can be used, provided they meet some common requirements.
- thanks to modular design new modules can be added, functionality and implementation of existing modules can be changed.
- library provides generic way to implement routines to operate on tensors objects in broadcast elementwise and lazy manner.
As well as universal way to evaluate such lazy results. For efficient evaluation of complex lazy expressions parallel execution subsystem is provided.


## Types customization

```cpp
template<typename T>
class matrix22
{
    std::array<T,4> elements_;
public:
    matrix22() = default;
    explicit matrix22(const T& v):elements_{v,v,v,v}{}
    matrix22(std::initializer_list<T> init_list):elements_{init_list}{}
    const T* data()const{return elements_.data();}
};
template<typename T>
std::ostream& operator<<(std::ostream& os, const matrix22<T>& m){
    os<<"[";
    std::copy(m.data(),m.data()+4,std::ostream_iterator<T>(os,","));
    return os<<"[";
}
template<typename U, typename V>
auto operator+(const matrix22<U>& u, const matrix22<V>& v){
    return matrix22<decltype(std::declval<U>()+std::declval<V>())>{
        u.data[0]+v.data[0],
        u.data[1]+v.data[1],
        u.data[2]+v.data[2],
        u.data[3]+v.data[3]
    };
}
template<typename U, typename V>
auto operator-(const matrix22<U>& u, const matrix22<V>& v){
    return matrix22<decltype(std::declval<U>()-std::declval<V>())>{
        u.data[0]-v.data[0],
        u.data[1]-v.data[1],
        u.data[2]-v.data[2],
        u.data[3]-v.data[3]
    };
}
template<typename U, typename V>
auto operator*(const matrix22<U>& u, const matrix22<V>& v){
    return matrix22<decltype(std::declval<U>()*std::declval<V>())>{
        u.data[0]*v.data[0]+u.data[1]*v.data[2],
        u.data[0]*v.data[1]+u.data[1]*v.data[3],
        u.data[2]*v.data[0]+u.data[3]*v.data[2],
        u.data[2]*v.data[1]+u.data[3]*v.data[3]
    };
}

using tensor_type = gtensor::tensor<matrix22<double>>;
tensor_type t{1,2,3,4};
std::cout<<std::endl<<t;
```
