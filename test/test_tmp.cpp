#include <iostream>
#include "catch.hpp"
#include <vector>
#include "tensor.hpp"

namespace test_tmp{

using gtensor::basic_tensor;

template<typename...Ts>
auto make_forward_traverser(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using traverser_type = gtensor::walker_forward_traverser<config_type,decltype(t.create_walker())>;
    return traverser_type{t.shape(),t.create_walker()};
}

}

TEST_CASE("test_tmp","[test_tmp]")
{

    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using test_tmp::make_forward_traverser;
    using tensor_type = tensor<double,f_order>;
    using config_type = tensor_type::config_type;
    using order = tensor_type::order;

    tensor_type t{{1,2,3,4,5}};
    auto v = t.transpose();

    auto w = v.create_walker();

    std::cout<<std::endl<<gtensor::detail::shape_to_str(v.shape());
    std::cout<<std::endl<<w.dim();
    std::cout<<std::endl<<*w;
    w.step(0);
    std::cout<<std::endl<<*w;
    w.step(0);
    std::cout<<std::endl<<*w;
    w.step(0);
    std::cout<<std::endl<<*w;
    w.step(0);
    std::cout<<std::endl<<*w;


    auto tr = make_forward_traverser(v);
    std::cout<<std::endl<<*tr;
    tr.template next<c_order>();
    std::cout<<std::endl<<*tr;
    tr.template next<c_order>();
    std::cout<<std::endl<<*tr;
    tr.template next<c_order>();
    std::cout<<std::endl<<*tr;
    tr.template next<c_order>();
    std::cout<<std::endl<<*tr;


    auto it = v.begin();
    std::cout<<std::endl<<*it;
    ++it;
    std::cout<<std::endl<<*it;
    ++it;
    std::cout<<std::endl<<*it;
    ++it;
    std::cout<<std::endl<<*it;
    ++it;
    std::cout<<std::endl<<*it;

}