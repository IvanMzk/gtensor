/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <tuple>
#include <vector>
#include <iostream>
#include "catch.hpp"
#include "test_config.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

//test view_factory
//test create_reshape_view
TEST_CASE("test_create_reshape_view","[test_view_factory]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using shape_type = typename tensor_type::shape_type;
    using gtensor::basic_tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;
    //0parent,1order,2subs,3expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(tensor_type(2),c_order{},std::make_tuple(),tensor_type{shape_type{},2}),
        std::make_tuple(tensor_type(2),c_order{},std::make_tuple(1),tensor_type{shape_type{1},2}),
        std::make_tuple(tensor_type(2),c_order{},std::make_tuple(-1),tensor_type{shape_type{1},2}),
        std::make_tuple(tensor_type(3),c_order{},std::make_tuple(1,1),tensor_type{shape_type{1,1},3}),
        std::make_tuple(tensor_type(3),c_order{},std::make_tuple(1,-1),tensor_type{shape_type{1,1},3}),
        std::make_tuple(tensor_type(3),c_order{},std::make_tuple(-1,1),tensor_type{shape_type{1,1},3}),
        std::make_tuple(tensor_type(4),c_order{},std::make_tuple(1,1,1),tensor_type{shape_type{1,1,1},4}),
        std::make_tuple(tensor_type(4),c_order{},std::make_tuple(1,-1,1),tensor_type{shape_type{1,1,1},4}),
        std::make_tuple(tensor_type{},c_order{},std::make_tuple(),tensor_type{}),
        std::make_tuple(tensor_type{},c_order{},std::make_tuple(1,-1),tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{},c_order{},std::make_tuple(-1,1),tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{},c_order{},std::make_tuple(2,-1,1),tensor_type{}.reshape(2,0,1)),
        std::make_tuple(tensor_type{1},c_order{},std::make_tuple(),tensor_type{1}),
        std::make_tuple(tensor_type{1},c_order{},std::make_tuple(-1),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(-1),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(1,5),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(-1,5),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(5,1),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(5,-1),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(), tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(-1), tensor_type{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(6,2), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(6,-1), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(-1,2), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}),
        //f_order
        std::make_tuple(tensor_type(2),f_order{},std::make_tuple(),tensor_type{shape_type{},2}),
        std::make_tuple(tensor_type(2),f_order{},std::make_tuple(1),tensor_type{shape_type{1},2}),
        std::make_tuple(tensor_type(2),f_order{},std::make_tuple(-1),tensor_type{shape_type{1},2}),
        std::make_tuple(tensor_type(3),f_order{},std::make_tuple(1,1),tensor_type{shape_type{1,1},3}),
        std::make_tuple(tensor_type(3),f_order{},std::make_tuple(1,-1),tensor_type{shape_type{1,1},3}),
        std::make_tuple(tensor_type(3),f_order{},std::make_tuple(-1,1),tensor_type{shape_type{1,1},3}),
        std::make_tuple(tensor_type(4),f_order{},std::make_tuple(1,1,1),tensor_type{shape_type{1,1,1},4}),
        std::make_tuple(tensor_type(4),f_order{},std::make_tuple(1,-1,1),tensor_type{shape_type{1,1,1},4}),
        std::make_tuple(tensor_type{},f_order{},std::make_tuple(),tensor_type{}),
        std::make_tuple(tensor_type{},f_order{},std::make_tuple(1,-1),tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{},f_order{},std::make_tuple(-1,1),tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{},f_order{},std::make_tuple(2,-1,1),tensor_type{}.reshape(2,0,1)),
        std::make_tuple(tensor_type{1},f_order{},std::make_tuple(),tensor_type{1}),
        std::make_tuple(tensor_type{1},f_order{},std::make_tuple(-1),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(-1),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(1,5),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(-1,5),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(5,1),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(5,-1),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(), tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(-1), tensor_type{1,7,3,9,5,11,2,8,4,10,6,12}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(6,2), tensor_type{{1,2},{7,8},{3,4},{9,10},{5,6},{11,12}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(6,-1), tensor_type{{1,2},{7,8},{3,4},{9,10},{5,6},{11,12}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(-1,2), tensor_type{{1,2},{7,8},{3,4},{9,10},{5,6},{11,12}})
    );
    SECTION("test_create_reshape_view_variadic")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto order = std::get<1>(t);
            auto subs = std::get<2>(t);
            auto expected = std::get<3>(t);
            using order_type = decltype(order);
            auto apply_subs = [&parent](const auto&...subs_){
                return basic_tensor{view_factory_type::create_reshape_view<order_type>(parent, subs_...)};
            };
            auto result = std::apply(apply_subs, subs);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_create_reshape_view_container")
    {
        using container_type = std::vector<int>;
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto order = std::get<1>(t);
            auto subs = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto make_container = [](const auto&...subs_){
                return container_type{subs_...};
            };
            auto container = std::apply(make_container, subs);
            using order_type = decltype(order);
            auto result = basic_tensor{view_factory_type::template create_reshape_view<order_type>(parent, container)};
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_create_reshape_view_exception","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;

    //0parent,1subs
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(0),std::make_tuple(0)),
        std::make_tuple(tensor_type(0),std::make_tuple(1,0)),
        std::make_tuple(tensor_type(0),std::make_tuple(-1,0)),
        std::make_tuple(tensor_type(0),std::make_tuple(2)),
        std::make_tuple(tensor_type(0),std::make_tuple(1,2)),
        std::make_tuple(tensor_type{},std::make_tuple(-1,-1)),
        std::make_tuple(tensor_type{},std::make_tuple(-1,0)),
        std::make_tuple(tensor_type{},std::make_tuple(0,-1)),
        std::make_tuple(tensor_type{1},std::make_tuple(0)),
        std::make_tuple(tensor_type{1},std::make_tuple(2)),
        std::make_tuple(tensor_type{1},std::make_tuple(-1,0)),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(10)),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(3,3)),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(-1,-1)),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(-1,4)),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(4,-1)),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(0,2))
    );
    SECTION("test_create_reshape_view_exception_variadic")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto apply_subs = [&parent](const auto&...subs_){
                return view_factory_type::template create_reshape_view<order>(parent, subs_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_subs, subs), value_error);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_create_reshape_view_exception_container")
    {
        using container_type = std::vector<int>;
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto make_container = [](const auto&...subs_){
                return container_type{subs_...};
            };
            auto container = std::apply(make_container, subs);
            REQUIRE_THROWS_AS(view_factory_type::template create_reshape_view<order>(parent, container), value_error);
        };
        apply_by_element(test,test_data);
    }
}

//test create_transpose_view
TEMPLATE_TEST_CASE("test_create_transpose_view","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(2),std::make_tuple(),tensor_type(2)),
        std::make_tuple(tensor_type{},std::make_tuple(),tensor_type{}),
        std::make_tuple(tensor_type{},std::make_tuple(0),tensor_type{}),
        std::make_tuple(tensor_type{1},std::make_tuple(),tensor_type{1}),
        std::make_tuple(tensor_type{1},std::make_tuple(0),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(0),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3,4,5}},std::make_tuple(),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{1,2,3,4,5}},std::make_tuple(1,0),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{1,2,3,4,5}},std::make_tuple(0,1),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}},std::make_tuple(),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}},std::make_tuple(2,1,0),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}},std::make_tuple(2,0,1),tensor_type{{{1,3,5}},{{2,4,6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}},std::make_tuple(1,0,2),tensor_type{{{1,2}},{{3,4}},{{5,6}}})
    );
    SECTION("test_create_transpose_view_variadic")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto apply_subs = [&parent](const auto&...subs_){
                return basic_tensor{view_factory_type::create_transpose_view(parent, subs_...)};
            };
            auto result = std::apply(apply_subs, subs);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_create_transpose_view_container")
    {
        using container_type = std::vector<int>;
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto make_container = [](const auto&...subs_){
                return container_type{subs_...};
            };
            auto container = std::apply(make_container, subs);
            auto result = basic_tensor{view_factory_type::create_transpose_view(parent, container)};
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_create_transpose_view_exception","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,order>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using helpers_for_testing::apply_by_element;
    using gtensor::value_error;
    //0parent,1subs
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(2),std::make_tuple(0)),
        std::make_tuple(tensor_type(2),std::make_tuple(1)),
        std::make_tuple(tensor_type(2),std::make_tuple(0,1)),
        std::make_tuple(tensor_type{},std::make_tuple(0,0)),
        std::make_tuple(tensor_type{},std::make_tuple(1)),
        std::make_tuple(tensor_type{1},std::make_tuple(0,1)),
        std::make_tuple(tensor_type{1},std::make_tuple(1)),
        std::make_tuple(tensor_type{{1,2},{3,4}},std::make_tuple(0,2,1)),
        std::make_tuple(tensor_type{{1,2},{3,4}},std::make_tuple(0)),
        std::make_tuple(tensor_type{{1,2},{3,4}},std::make_tuple(1,1))
    );
    SECTION("test_create_transpose_view_exception_variadic")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto apply_subs = [&parent](const auto&...subs_){
                return basic_tensor{view_factory_type::create_transpose_view(parent, subs_...)};
            };
            REQUIRE_THROWS_AS(std::apply(apply_subs, subs), value_error);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_create_transpose_view_exception_container")
    {
        using container_type = std::vector<int>;
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto make_container = [](const auto&...subs_){
                return container_type{subs_...};
            };
            auto container = std::apply(make_container, subs);
            REQUIRE_THROWS_AS(view_factory_type::create_transpose_view(parent, container), value_error);
        };
        apply_by_element(test,test_data);
    }
}

