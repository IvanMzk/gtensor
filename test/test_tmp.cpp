#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "statistic.hpp"

namespace test_tmp{
}   //end of namespace test_tmp

TEST_CASE("test_tmp_copy","[test_tmp]")
{

    using gtensor::tensor;
    const tensor<double> a{{1,2,3},{3,4}};

    std::cout<<std::endl<<a;
    std::cout<<std::endl<<a.mean();


}
