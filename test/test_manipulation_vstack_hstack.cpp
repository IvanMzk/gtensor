/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include "manipulation.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "config_for_testing.hpp"

TEMPLATE_TEST_CASE("test_vstack","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using helpers_for_testing::apply_by_element;
    using gtensor::vstack;
    //0tensors,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}), tensor_type{}.reshape(1,0)),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}.reshape(3,0)),
        std::make_tuple(std::make_tuple(tensor_type{}.reshape(2,0), tensor_type{}.reshape(2,0), tensor_type{}.reshape(2,0)), tensor_type{}.reshape(6,0)),
        std::make_tuple(std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1},{2},{3}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3},tensor_type{4,5,6},tensor_type{7,8,9}), tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}), tensor_type{{1,2},{3,4}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{5,6}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}},{{5}}}, tensor_type{{{6}}}), tensor_type{{{1}},{{2}},{{3}},{{4}},{{5}},{{6}}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2}},{{3,4}}}, tensor_type{{{5,6}},{{7,8}}}, tensor_type{{{9,10}}}), tensor_type{{{1,2}},{{3,4}},{{5,6}},{{7,8}},{{9,10}}}),
        std::make_tuple(
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12},{13,14,15}},tensor_type{{16,17,18}},tensor_type{19,20,21}),
            tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15},{16,17,18},{19,20,21}}
        ),
        std::make_tuple(
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10},{11,12}}},tensor_type{{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}
        ),
        std::make_tuple(std::make_tuple(tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}, tensor_type{{{13,14},{15,16},{17,18}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    SECTION("test_vstack_variadic")
    {
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return vstack(tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_vstack_container")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(order{}))>;
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(order{})...};}, tensors);
            auto result = vstack(container);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_vstack_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::value_error;
    using gtensor::vstack;
    //0tensors
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{1}, tensor_type{1,1})),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{5,6,7})),
        std::make_tuple(std::make_tuple(tensor_type{{{1}}}, tensor_type{1})),
        std::make_tuple(std::make_tuple(tensor_type{{{1}}}, tensor_type{{1}})),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, tensor_type{{1,2},{3,4}}))
    );
    SECTION("test_vstack_variadic_exception")
    {
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return vstack(tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_vstack_container_exception")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy())>;
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy()...};}, tensors);
            REQUIRE_THROWS_AS(vstack(container), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_hstack","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using helpers_for_testing::apply_by_element;
    using gtensor::hstack;
    //0tensors,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{1,2,3}, tensor_type{}, tensor_type{4,5,6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7},{8}}, tensor_type{}.reshape(2,0), tensor_type{{9,10},{11,12}}), tensor_type{{1,2,3,7,9,10},{4,5,6,8,11,12}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, tensor_type{{{9,10}},{{11,12}}}, tensor_type{}.reshape(2,0,2)), tensor_type{{{1,2},{3,4},{9,10}},{{5,6},{7,8},{11,12}}})
    );
    SECTION("test_hstack_variadic")
    {
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return hstack(tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_hstack_container")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(order{}))>;
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(order{})...};}, tensors);
            auto result = hstack(container);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_hstack_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::config::c_order;
    using gtensor::value_error;
    using gtensor::hstack;
    //0tensors
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}.reshape(1,0))),
        std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7},{8},{9}})),
        std::make_tuple(std::make_tuple(tensor_type{}.reshape(2,2,0), tensor_type{}.reshape(3,2,0))),
        std::make_tuple(std::make_tuple(tensor_type{}.reshape(0,2,2), tensor_type{}.reshape(0,2,3)))
    );
    SECTION("test_hstack_variadic_exception")
    {
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return hstack(tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_hstack_container_exception")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(c_order{}))>;
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(c_order{})...};}, tensors);
            REQUIRE_THROWS_AS(hstack(container), value_error);
        };
        apply_by_element(test, test_data);
    }
}

