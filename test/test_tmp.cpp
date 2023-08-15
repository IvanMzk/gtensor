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
    using helpers_for_testing::generate_lehmer;
    using benchmark_helpers::shapes;

    auto t = tensor<double>(shapes[0],2);

    auto t_copy = t.copy();


}