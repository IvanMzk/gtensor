#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "combine.hpp"
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
    decltype(auto) operator[](size_type i){return impl_[i];}
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


    //std::cout<<std::endl<<tensor_type{{1,2,3},{4,5,6},{7,8,9}}(tensor_type{1}, tensor_type{{0,2},{2,0}});
    //std::cout<<std::endl<<tensor_type{{1,2,3},{4,5,6},{7,8,9}}(tensor_type(1), tensor_type{0,2});

    std::cout<<std::endl<<tensor_type{{1,2,3},{4,5,6}};

    tensor_type a{1,2,3};
    tensor_type b{4,5,6};

    std::cout<<std::endl<<gtensor::stack(0, a,b);
    std::cout<<std::endl<<gtensor::stack(1, a,b);


}