#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "statistic.hpp"

namespace test_tmp{
}   //end of namespace test_tmp

TEST_CASE("test_tmp_copy","[test_tmp]")
{

    using gtensor::tensor;
    using namespace std::complex_literals;
    const tensor<std::complex<double>> a{1.1+2.2i,2.2+1.1i,3.2+0.1i};

    std::cout<<std::endl<<a;
    std::cout<<std::endl<<a.mean();


}
