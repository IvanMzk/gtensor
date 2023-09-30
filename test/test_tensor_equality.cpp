/*
* GTensor - matrix computation library
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
#include "test_config.hpp"

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

//same tensor implementation
TEST_CASE("test_tensor_is_same","[test_tensor]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;

    const auto t = tensor_type{1,2,3};
    const auto e = tensor_type{1,2,3}+tensor_type{0,0,0};
    const auto v = tensor_type{1,2,3}.transpose().transpose();
    SECTION("test_not_same_tensor")
    {
        auto test_data = std::make_tuple(
            std::make_tuple(t,e,v),
            std::make_tuple(e,t,v),
            std::make_tuple(v,e,t)
        );
        auto test = [](const auto& t){
            auto first = std::get<0>(t);
            auto second = std::get<1>(t);
            auto third = std::get<2>(t);
            REQUIRE(!first.is_same(second));
            REQUIRE(!first.is_same(third));
            REQUIRE(!second.is_same(first));
            REQUIRE(!second.is_same(third));
            REQUIRE(!third.is_same(first));
            REQUIRE(!third.is_same(second));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_not_same_other"){
        REQUIRE(!t.is_same(1));
        REQUIRE(!e.is_same(1.0));
        REQUIRE(!v.is_same(std::string{}));
    }
    SECTION("test_same_tensor")
    {
        auto test_data = std::make_tuple(t,e,v);
        auto test = [](const auto& t){
            REQUIRE(t.is_same(t));
            auto t_ref_copy = t;
            REQUIRE(&t != &t_ref_copy);
            REQUIRE(t.is_same(t_ref_copy));
            auto t_ref_move = std::move(t_ref_copy);
            REQUIRE(t.is_same(t_ref_move));
            REQUIRE(!t.is_same(t_ref_copy));
        };
        apply_by_element(test,test_data);
    }
}

