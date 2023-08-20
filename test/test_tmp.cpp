#include <iostream>
#include <vector>
#include "statistic.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "benchmark_helpers.hpp"

namespace test_tmp{

using gtensor::basic_tensor;

}

TEST_CASE("test_tmp","[test_tmp]")
{


    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using tensor_type = tensor<int>;
    using slice_type = tensor_type::slice_type;
    using helpers_for_testing::range_to_str;
    using gtensor::detail::shape_to_str;

    auto t = tensor<int,c_order>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}};
    //std::cout<<std::endl<<t;

    //auto e = t(1,1);
    auto e = t(1)(1)(2);
    //auto e = t(1)(1);
    //auto e = t(1);

    auto w = e.create_walker();
    std::cout<<std::endl<<*w;
    w.reset_back();
    std::cout<<std::endl<<*w;






}