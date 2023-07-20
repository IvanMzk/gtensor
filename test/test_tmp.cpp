#include <iostream>
#include "catch.hpp"
#include "random.hpp"
#include "tensor.hpp"

namespace test_tmp{



}

TEST_CASE("test_tmp","[test_tmp]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;

    for (auto i : gtensor::arange<int>(20)){
        //std::cout<<std::endl<<gtensor::default_rng().choice(tensor_type{1,2,3,4,5,6,7,8,9,10},9,false,tensor_type{10,10,10,10,1,10,10,10,10,10},0,false);
        //std::cout<<std::endl<<gtensor::default_rng().choice(tensor_type{1,2,3,4,5},4,false,tensor_type{1.0,1.0,0.1,1.0,1.0},0,false);
        //std::cout<<std::endl<<gtensor::default_rng().choice(tensor_type{1,2,3,4,5},1,false,tensor_type{1.0,1.0,0.1,1.0,1.0},0,false);
        std::cout<<std::endl<<gtensor::default_rng().choice(gtensor::arange<int>(1,21,1),1,false,tensor_type{1,1,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false);
    }



}