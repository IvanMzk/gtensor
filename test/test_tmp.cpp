#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "test_config.hpp"

namespace test_tmp{

template<typename T>
class subscriptable_storage
{
    using inner_storage_type = std::vector<T>;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = typename inner_storage_type::size_type;
    using difference_type = typename inner_storage_type::difference_type;
    subscriptable_storage(size_type n):
        impl_(n)
    {}
    //decltype(auto) operator[](size_type i){return impl_[i];}
    decltype(auto) operator[](size_type i)const{return impl_[i];}
};

}

TEST_CASE("test_tmp","[test_tmp]")
{
    using value_type = int;
    using test_tmp::subscriptable_storage;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<subscriptable_storage>,value_type>;
    using gtensor::tensor;
    using tensor_type = tensor<value_type,config_type>;

    tensor_type t(std::vector<int>{1,2,3});
    t.begin();

    auto v = t.transpose();
    v.begin();
    auto e = v+v;
    //e.begin();


    //auto e = t+t;


    //std::cout<<std::endl<<t;

    // const tensor_type ct(std::vector<int>{2,2,2});
    // std::cout<<std::endl<<ct;
}