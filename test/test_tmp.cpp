#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"

namespace test_tmp{
}

TEST_CASE("test_tmp","[test_tmp]")
{
    //using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using gtensor::tensor;

    tensor<bool> bt{false,true,true,false};

}