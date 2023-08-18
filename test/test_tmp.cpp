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

    auto t = tensor_type{{1,2,3},{4,5,6}};
    std::cout<<std::endl<<t;

    std::cout<<std::endl<<shape_to_str(tensor_type{1}.shape());
    std::cout<<std::endl<<shape_to_str(tensor_type{1}(slice_type{1}).shape());

    std::cout<<std::endl<<shape_to_str(tensor_type{1}.strides());
    std::cout<<std::endl<<shape_to_str(tensor_type{1}(slice_type{1}).strides());


}