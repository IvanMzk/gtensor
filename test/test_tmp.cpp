#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"

namespace test_tmp{
}   //end of namespace test_tmp

TEST_CASE("test_tmp_copy","[test_tmp]")
{

    using gtensor::tensor;

    tensor<double> t{1,2,3,4,5};
    auto t1 = t;
    std::cout<<std::endl<<t;
    std::cout<<std::endl<<t1;
    t.element(0)=1.1;
    std::cout<<std::endl<<t;
    std::cout<<std::endl<<t1;
}
