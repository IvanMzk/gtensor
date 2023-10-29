/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <tuple>
#include <vector>
#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "config_for_testing.hpp"

//test tensor operator==
TEST_CASE("test_tensor_operator==","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //0tensor0,1tensor1,2expected
    auto test_data = std::make_tuple(
        //equal
        std::make_tuple(tensor_type(2),tensor_type(2),true),
        std::make_tuple(tensor_type{},tensor_type{},true),
        std::make_tuple(tensor_type{1},tensor_type{1},true),
        std::make_tuple(tensor_type{1,2,3},tensor_type{1,2,3},true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}},true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},true),
        //not equal
        std::make_tuple(tensor_type(1),tensor_type(2),false),
        std::make_tuple(tensor_type{},tensor_type{1},false),
        std::make_tuple(tensor_type{1},tensor_type{2},false),
        std::make_tuple(tensor_type{1,2,3},tensor_type{{1,2,3}},false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,2},{4,5,6}},false)
    );
    SECTION("ten0_equals_ten0")
    {
        auto test = [](const auto& t){
            auto ten0 = std::get<0>(t);
            auto ten1 = std::get<1>(t);
            auto expected = true;
            auto result = ten0 == ten0;
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("ten0_equals_ten1")
    {
        auto test = [](const auto& t){
            auto ten0 = std::get<0>(t);
            auto ten1 = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = ten0 == ten1;
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("ten1_equals_ten0")
    {
        auto test = [](const auto& t){
            auto ten0 = std::get<0>(t);
            auto ten1 = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = ten1 == ten0;
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

//broadcast element-wise
TEST_CASE("test_tensor_equal","[test_tensor]"){
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using helpers_for_testing::apply_by_element;
    //0operand1,1operand2,2expected
    auto test_data = std::make_tuple(
        //other scalar
        std::make_tuple(tensor_type{},1,bool_tensor_type{}),
        std::make_tuple(tensor_type(1),1,bool_tensor_type(true)),
        std::make_tuple(tensor_type(1),2,bool_tensor_type(false)),
        std::make_tuple(tensor_type{{1,2},{3,1}},1,bool_tensor_type{{true,false},{false,true}}),
        //other tensor
        std::make_tuple(tensor_type{},tensor_type{},bool_tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type(1),bool_tensor_type{}),
        std::make_tuple(tensor_type(1),tensor_type{},bool_tensor_type{}),
        std::make_tuple(tensor_type(1),tensor_type(1),bool_tensor_type(true)),
        std::make_tuple(tensor_type(1),tensor_type(2),bool_tensor_type(false)),
        std::make_tuple(tensor_type(1),tensor_type{{1,2},{3,1}},bool_tensor_type{{true,false},{false,true}}),
        std::make_tuple(tensor_type{{1,2},{3,1}},tensor_type(1),bool_tensor_type{{true,false},{false,true}}),
        std::make_tuple(tensor_type{{1,2},{3,1}},tensor_type{2},bool_tensor_type{{false,true},{false,false}}),
        std::make_tuple(tensor_type{2},tensor_type{{1,2},{3,1}},bool_tensor_type{{false,true},{false,false}}),
        std::make_tuple(tensor_type{{1},{5}},tensor_type{{{1,2},{3,1}},{{4,5},{5,6}}},bool_tensor_type{{{true,false},{false,false}},{{false,false},{true,false}}})
    );
    auto test = [](const auto& t){
        auto operand1 = std::get<0>(t);
        auto operand2 = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = operand1.equal(operand2);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_not_equal","[test_tensor]"){
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using helpers_for_testing::apply_by_element;
    //0operand1,1operand2,2expected
    auto test_data = std::make_tuple(
        //other scalar
        std::make_tuple(tensor_type{},1,bool_tensor_type{}),
        std::make_tuple(tensor_type(1),1,bool_tensor_type(false)),
        std::make_tuple(tensor_type(1),2,bool_tensor_type(true)),
        std::make_tuple(tensor_type{{1,2},{3,1}},1,bool_tensor_type{{false,true},{true,false}}),
        //other tensor
        std::make_tuple(tensor_type{},tensor_type{},bool_tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type(1),bool_tensor_type{}),
        std::make_tuple(tensor_type(1),tensor_type{},bool_tensor_type{}),
        std::make_tuple(tensor_type(1),tensor_type(1),bool_tensor_type(false)),
        std::make_tuple(tensor_type(1),tensor_type(2),bool_tensor_type(true)),
        std::make_tuple(tensor_type(1),tensor_type{{1,2},{3,1}},bool_tensor_type{{false,true},{true,false}}),
        std::make_tuple(tensor_type{{1,2},{3,1}},tensor_type(1),bool_tensor_type{{false,true},{true,false}}),
        std::make_tuple(tensor_type{{1,2},{3,1}},tensor_type{2},bool_tensor_type{{true,false},{true,true}}),
        std::make_tuple(tensor_type{2},tensor_type{{1,2},{3,1}},bool_tensor_type{{true,false},{true,true}}),
        std::make_tuple(tensor_type{{1},{5}},tensor_type{{{1,2},{3,1}},{{4,5},{5,6}}},bool_tensor_type{{{false,true},{true,true}},{{true,true},{false,true}}})
    );
    auto test = [](const auto& t){
        auto operand1 = std::get<0>(t);
        auto operand2 = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = operand1.not_equal(operand2);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

//same tensor implementation
TEMPLATE_TEST_CASE("test_tensor_is_same_deep_cloning","[test_tensor]",
    gtensor::config::deep_semantics,
    gtensor::config::shallow_semantics
)
{
    using value_type = double;
    using gtensor::config::c_order;
    using config_type = gtensor::config::extend_config_t<test_config::config_semantics_selector_t<TestType>,value_type>;
    using tensor_type = gtensor::tensor<value_type,c_order,config_type>;
    using helpers_for_testing::apply_by_element;

    auto t = tensor_type{1,2,3};
    auto e = tensor_type{1,2,3}+tensor_type{0,0,0};
    auto v = tensor_type{1,2,3}.transpose().transpose();

    //true
    REQUIRE(t.is_same(t));
    REQUIRE(t.is_same(t.eval()));
    REQUIRE(e.is_same(e));
    REQUIRE(v.is_same(v));
    //false
    REQUIRE(!t.is_same(t.copy()));
    REQUIRE(!t.is_same(e));
    REQUIRE(!t.is_same(v));
    REQUIRE(!e.is_same(e.copy()));
    REQUIRE(!e.is_same(e.eval()));
    REQUIRE(!v.is_same(v.copy()));
    REQUIRE(!v.is_same(v.eval()));
}

