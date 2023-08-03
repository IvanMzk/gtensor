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
    using tensor_type = tensor<double,c_order>;
    using config_type = tensor_type::config_type;
    using order = tensor_type::order;


    // tensor_type t{{{3,7}},{{4,8}}};
    // std::cout<<std::endl<<t;
    // //t.assign(tensor_type{{0,2},{1,3}});
    // t.assign(tensor_type{-1,1} + tensor_type{{1},{2}});
    // std::cout<<std::endl<<t;

    // std::cout<<std::endl<<tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}({{},{1}})+tensor_type{{0,2},{1,3}};
    // std::cout<<std::endl<<tensor_type{{{3,4}},{{7,8}}}+tensor_type{{0,2},{1,3}};


    tensor_type t{{{1,2},{3,4}},{{5,6},{7,8}}};
    std::cout<<std::endl<<t;
    //auto v = t.transpose()({{},{1}});
    auto v = t({{},{1}});
    std::cout<<std::endl<<v;
    auto rhs = tensor_type{{0,2},{1,3}};
    std::cout<<std::endl<<rhs;

    auto coupler = [](const auto& e1, const auto& e2){
        return std::make_pair(e1,e2);
    };

    auto pair_to_str = [](const auto& pair){
        std::stringstream ss{};
        ss<<pair.first<<" "<<pair.second;
        return ss.str();
    };

    auto ee = gtensor::n_operator(coupler,v,rhs);
    std::cout<<std::endl<<gtensor::detail::shape_to_str(ee.shape());
    auto w = ee.create_walker();
    std::cout<<std::endl<<pair_to_str(*w);
    w.step(2);
    std::cout<<std::endl<<pair_to_str(*w);
    w.reset_back(2);
    std::cout<<std::endl<<pair_to_str(*w);
    w.step(1);
    std::cout<<std::endl<<pair_to_str(*w);
    w.step(2);
    std::cout<<std::endl<<pair_to_str(*w);



    //v.assign(tensor_type{-1,1} + tensor_type{{1},{2}});
    //v.assign(tensor_type{{0,2},{1,3}});

    // std::cout<<std::endl<<t;
    //std::cout<<std::endl<<v;
    // for(auto e : t){
    //     std::cout<<std::endl<<e;
    // }

    // tensor_type t{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}};
    // auto v = t({{1},{1}});

    //  auto ee = tensor_type{-1,1} + tensor_type{{1},{2}};


    // for(auto e : ee){
    //     std::cout<<std::endl<<e;
    // }

    // std::cout<<std::endl<<t;
    // *v.begin() = 22;
    // std::cout<<std::endl<<t;
    // v.assign(tensor_type{2,0,1});
    // std::cout<<std::endl<<t;

    // t({{},{1},{1}}) = tensor_type{3,7};
    // std::cout<<std::endl<<t;

    // auto w = v.create_walker();
    // std::cout<<std::endl<<gtensor::detail::shape_to_str(v.shape());
    // std::cout<<std::endl<<w.dim();


    // std::cout<<std::endl<<*w;
    // w.step(2);
    // std::cout<<std::endl<<*w;
    // w.step(2);
    // std::cout<<std::endl<<*w;
    // w.reset_back(1);
    // w.step(0);
    // std::cout<<std::endl<<*w;
    // w.step(1);
    // std::cout<<std::endl<<*w;
    // w.step(1);
    // std::cout<<std::endl<<*w;




    // auto tr = make_forward_traverser(v);
    // std::cout<<std::endl<<*tr;
    // tr.template next<c_order>();
    // std::cout<<std::endl<<*tr;
    // tr.template next<c_order>();
    // std::cout<<std::endl<<*tr;
    // tr.template next<c_order>();
    // std::cout<<std::endl<<*tr;
    // tr.template next<c_order>();
    // std::cout<<std::endl<<*tr;


    // auto it = v.begin();
    // std::cout<<std::endl<<*it;
    // ++it;
    // std::cout<<std::endl<<*it;
    // ++it;
    // std::cout<<std::endl<<*it;
    // ++it;
    // std::cout<<std::endl<<*it;
    // ++it;
    // std::cout<<std::endl<<*it;

}