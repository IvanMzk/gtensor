#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"

TEST_CASE("test_tmp","[test_tmp]")
{
    using gtensor::tensor;

    // tensor<double> t{1,2,3,4,5};
    // //t(2) = 0;
    // t(2)+=1;
    // std::cout<<std::endl<<t;

    // tensor<double> t1{{1,2,3},{4,5,6},{7,8,9}};

    // t1({{},{1}})+=tensor<double>{1,2,3}.reshape(-1,1);
    // std::cout<<std::endl<<t1;

    // tensor<double> t2{{7,3,4,6},{1,5,6,2},{1,8,3,5},{0,2,6,2}};

    // t2(t2>5)+=1;
    // std::cout<<std::endl<<t2;

    auto r = tensor<double>{} + 1;
    std::cout<<std::endl<<r;

    auto t = tensor<double>{};
    auto lhs = t();
    auto rhs = 1;
    std::move(lhs)+=rhs;
    std::cout<<std::endl<<t;

}