#include <iostream>
#include "catch.hpp"
//#include "reduce.hpp"
#include "tensor.hpp"

namespace test_tmp{

}

TEST_CASE("test_tmp","[test_tmp]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;

    tensor_type t{1,2,3};
    std::cout<<std::endl<<t.reduce({0},[](auto f, auto l){return std::accumulate(f,l,0);});


}