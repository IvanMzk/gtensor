#include "catch.hpp"
#include "gtensor.hpp"


TEST_CASE("test_const_impl","[test_const_impl]"){

    using gtensor::tensor;
    using value_type = int;
    tensor<value_type> t{1,2,3};

    auto e = t+t;
}