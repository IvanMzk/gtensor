#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

namespace test_tmp{

}

TEST_CASE("test_tmp","[test_tmp]")
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::walker_forward_traverser;
    using gtensor::config::c_order;
    using helpers_for_testing::range_to_str;
    using config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<c_order>,value_type>;
    using tensor_type = tensor<value_type,c_order,config_type>;
    using shape_type = typename tensor_type::shape_type;

    auto t = tensor<value_type,c_order,config_type>{1,2,3,4,5,6};
    std::cout<<std::endl<<t;

    shape_type br_shape{6,1};
    const auto max_dim = std::max(gtensor::detail::make_dim(br_shape),t.dim());
    auto w = t.create_walker(max_dim);


    std::cout<<std::endl<<*w;
    w.step(0);
    std::cout<<std::endl<<*w;
    w.step(0);
    std::cout<<std::endl<<*w;
    // w.step(1);
    // std::cout<<std::endl<<*w;
    // w.step(1);
    // std::cout<<std::endl<<*w;
    // w.reset_back();

    // walker_forward_traverser<config_type,decltype(t.create_walker())> tr{t.shape(),t.create_walker()};
    // std::cout<<std::endl<<*tr.walker();
    // tr.template next<c_order>();
    // std::cout<<std::endl<<*tr.walker();
    // tr.template next<c_order>();
    // std::cout<<std::endl<<*tr.walker();

    walker_forward_traverser<config_type,decltype(t.create_walker())> br_tr{br_shape,t.create_walker(max_dim)};
    std::cout<<std::endl<<*br_tr.walker()<<" "<<range_to_str(br_tr.index().begin(),br_tr.index().end());
    br_tr.template next<c_order>();
    std::cout<<std::endl<<*br_tr.walker()<<" "<<range_to_str(br_tr.index().begin(),br_tr.index().end());
    br_tr.template next<c_order>();
    std::cout<<std::endl<<*br_tr.walker()<<" "<<range_to_str(br_tr.index().begin(),br_tr.index().end());
    br_tr.template next<c_order>();
    std::cout<<std::endl<<*br_tr.walker()<<" "<<range_to_str(br_tr.index().begin(),br_tr.index().end());

}