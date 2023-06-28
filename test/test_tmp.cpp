#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"

namespace test_tmp{


}

TEST_CASE("test_tmp","[test_tmp]")
{
    using value_type = double;
    using gtensor::tensor;

    tensor<value_type> t{1,2,3,4,5};

    //auto d = gtensor::diff2(t,0);
    auto g = gtensor::gradient(t,0);
    std::cout<<std::endl<<g;

}