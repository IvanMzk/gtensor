#include <iostream>
#include "catch.hpp"
#include <vector>
#include "tensor.hpp"
#include "integral_type.hpp"

namespace test_tmp{

auto f(std::initializer_list<integral_type::integral<std::ptrdiff_t>> init){

}

}

TEST_CASE("test_tmp","[test_tmp]")
{

    using gtensor::tensor;

    std::vector<int> v{1,2,3,4,5};
    //std::vector<int> v1{v.size()};

    test_tmp::f({v.size()});

    tensor<int> t({v.size()},v.begin(),v.end());
}