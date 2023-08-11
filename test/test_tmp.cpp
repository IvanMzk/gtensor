#include <iostream>
#include "catch.hpp"
#include <vector>
#include "statistic.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

namespace test_tmp{

using gtensor::basic_tensor;

}

TEST_CASE("test_tmp","[test_tmp]")
{

    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using helpers_for_testing::generate_lehmer;


    auto t = tensor<double>({10},1);
    generate_lehmer(t.begin(),t.end(),123);

    std::cout<<std::endl<<str(t,10);

    //std::cout<<std::endl<<str(t.mean(),10);
}