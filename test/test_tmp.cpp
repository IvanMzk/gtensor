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

    using benchmark_helpers::timing;
    using benchmark_helpers::shapes;

    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using tensor_type = tensor<int,c_order>;
    using slice_type = tensor_type::slice_type;
    using config_type = tensor_type::config_type;
    using shape_type = tensor_type::shape_type;
    using helpers_for_testing::range_to_str;
    using gtensor::detail::shape_to_str;

    auto t = tensor<double,c_order>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}};

    REQUIRE(std::is_same_v<decltype(t.begin()),decltype(t.begin_trivial())>);
    REQUIRE(std::is_same_v<decltype(t.traverse_order_adapter(f_order{}).begin()),decltype(t.traverse_order_adapter(f_order{}).begin_trivial())>);

}