#include "catch.hpp"
#include "gtensor.hpp"


TEST_CASE("test_is_iterator","[test_tensor]"){
    using gtensor::detail::is_iterator;
    REQUIRE(!is_iterator<int>);
    REQUIRE(!is_iterator<std::vector<int>>);
    REQUIRE(is_iterator<std::vector<int>::iterator>);
}

TEST_CASE("test_is_tensor","[test_tensor]"){
    using gtensor::tensor;
    using gtensor::integral;
    using gtensor::detail::is_tensor_of_type_v;
    using gtensor::detail::is_bool_tensor_v;

    REQUIRE(is_tensor_of_type_v<tensor<int>,int>);
    REQUIRE(is_tensor_of_type_v<tensor<integral<std::int64_t>>,integral<std::int64_t>>);
    REQUIRE(is_tensor_of_type_v<tensor<int>,integral<std::int64_t>>);
    REQUIRE(is_tensor_of_type_v<tensor<std::size_t>,std::int64_t>);
    REQUIRE(is_tensor_of_type_v<tensor<bool>,std::int64_t>);
    REQUIRE(is_tensor_of_type_v<tensor<bool>,int>);

    REQUIRE(!is_tensor_of_type_v<tensor<float>,integral<std::int64_t>>);

    REQUIRE(!is_tensor_of_type_v<tensor<integral<std::int64_t>>,std::int64_t>);
    REQUIRE(!is_tensor_of_type_v<std::vector<int>,int>);
    REQUIRE(!is_tensor_of_type_v<std::string,int>);
    REQUIRE(!is_tensor_of_type_v<std::vector<bool>,int>);

    REQUIRE(is_bool_tensor_v<tensor<bool>>);
    REQUIRE(!is_bool_tensor_v<tensor<int>>);
    REQUIRE(!is_bool_tensor_v<tensor<float>>);
    REQUIRE(!is_bool_tensor_v<std::vector<int>>);
    REQUIRE(!is_bool_tensor_v<std::string>);
    REQUIRE(!is_bool_tensor_v<std::vector<bool>>);
}

