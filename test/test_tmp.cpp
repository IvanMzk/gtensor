#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"

namespace test_tmp{
}   //end of namespace test_tmp

TEST_CASE("test_tmp_copy","[test_tmp]")
{

    using gtensor::tensor;

    tensor<double> t0{1,0,3};
    tensor<double> t1{2,1,2};
    tensor<double> t2{3,0,1};
    tensor<double> t3{2,3,1};

    tensor<tensor<double>> a{{t0,t1},{t2,t3}};
    tensor<tensor<double>> b{{t1,t3},{t0,t2}};
    std::cout<<std::endl<<a;
    std::cout<<std::endl<<b;

    std::cout<<std::endl<<matmul(a,a+b);

}
