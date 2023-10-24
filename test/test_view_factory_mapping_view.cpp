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
#include "config_for_testing.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

//test create_index_mapping_view
TEMPLATE_TEST_CASE("test_create_index_mapping_view","[test_view_factory]",
    //parent's order, subs order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using parent_order = std::tuple_element_t<0,TestType>;
    using subs_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,parent_order>;
    using config_type = typename tensor_type::config_type;
    using index_tensor_type = gtensor::tensor<int, subs_order, config_type>;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{}),tensor_type{}),
        std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{}.reshape(2,3,0)),tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{}.reshape(1,0),std::make_tuple(index_tensor_type(0)),tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0),std::make_tuple(index_tensor_type{0}),tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(1,0),std::make_tuple(index_tensor_type{0,0,0}),tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(1,0),std::make_tuple(index_tensor_type{0},index_tensor_type{}),tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type(1)),tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{0,1,0,1,0}),tensor_type{}.reshape(5,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{4,1,2,1,3}),tensor_type{}.reshape(5,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{0,1},index_tensor_type{2}),tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{0,1},index_tensor_type{2},index_tensor_type{}.reshape(0,3,1)),tensor_type{}.reshape(0,3,2)),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{{0,1}},index_tensor_type{{0,2}},index_tensor_type{}.reshape(0,3,1)),tensor_type{}.reshape(0,3,2)),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{{0,1}},index_tensor_type{4},index_tensor_type{}.reshape(0,3,1)),tensor_type{}.reshape(0,3,2)),
        std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type(0)), tensor_type(1)),
        std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0}), tensor_type{1}),
        std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0,0,0}), tensor_type{1,1,1}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(index_tensor_type(3)), tensor_type(4)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(index_tensor_type{1,1,0,0}), tensor_type{2,2,1,1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(index_tensor_type{{1,2},{0,1}}), tensor_type{{{4,5,6},{7,8,9}},{{1,2,3},{4,5,6}}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},std::make_tuple(index_tensor_type{{0,0},{3,3}}, index_tensor_type{{0,2},{0,2}}), tensor_type{{1,3},{10,12}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1}), tensor_type{{{5,6},{7,8}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type(1)), tensor_type{{5,6},{7,8}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type(1),index_tensor_type(0)), tensor_type{5,6}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}), tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}, index_tensor_type{0,1}), tensor_type{{5,6},{15,16}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}, index_tensor_type(1)), tensor_type{{7,8},{15,16}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type(2), index_tensor_type{1,0}), tensor_type{{11,12},{9,10}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type(2), index_tensor_type{{1,0},{0,1}}), tensor_type{{{11,12},{9,10}},{{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}, index_tensor_type{1}), tensor_type{{7,8},{15,16}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}, index_tensor_type{{1,0},{0,1}}), tensor_type{{{7,8},{13,14}},{{5,6},{15,16}}})
    );
    SECTION("test_variadic")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto apply_subs = [&parent](const auto&...subs_){
                return basic_tensor{view_factory_type::create_index_mapping_view(parent, subs_...)};
            };
            auto result = std::apply(apply_subs, subs);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_container")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto expected = std::get<2>(t);
            using container_type = typename config_type::template container<index_tensor_type>;
            auto apply_subs = [&parent](const auto&...subs_){
                auto subs_container = container_type{subs_.copy(subs_order{})...};
                return basic_tensor{view_factory_type::create_index_mapping_view(parent, subs_container)};
            };
            auto result = std::apply(apply_subs, subs);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_create_index_mapping_view_exception","[test_view_factory]",
    //parent's order, subs order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using parent_order = std::tuple_element_t<0,TestType>;
    using subs_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,parent_order>;
    using index_tensor_type = gtensor::tensor<int,subs_order>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using gtensor::index_error;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs,2exception
    auto test_data = std::make_tuple(
        //0-dim tensor
        std::make_tuple(tensor_type(2),std::make_tuple(index_tensor_type{}),index_error{""}),
        //exception, parent zero size direction and non zero size subs
        std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type(0)),index_error{""}),
        std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{0}),index_error{""}),
        std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{1}),index_error{""}),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{1},index_tensor_type{2},index_tensor_type{0}),index_error{""}),
        //exception, subs number more than parent dim
        std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{},index_tensor_type{}),index_error{""}),
        std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0},index_tensor_type{0,0,0}),index_error{""}),
        std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0,1},index_tensor_type{0,1}),index_error{""}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(index_tensor_type{0,1},index_tensor_type{1,1},index_tensor_type{}),index_error{""}),
        //exception, subs shapes not broadcast
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(index_tensor_type{0,0},index_tensor_type{0,0,0}),value_error{""}),
        //exception, subs out of bounds
        std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0,4,0}),index_error{""}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(index_tensor_type{3}),index_error{""}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(index_tensor_type{0},index_tensor_type{1,2,3}),index_error{""})
    );
    SECTION("test_variadic")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto exception = std::get<2>(t);
            auto apply_subs = [&parent](const auto&...subs_){
                return basic_tensor{view_factory_type::create_index_mapping_view(parent, subs_...)};
            };
            REQUIRE_THROWS_AS(std::apply(apply_subs, subs), decltype(exception));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_container")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto exception = std::get<2>(t);
            using container_type = typename config_type::template container<index_tensor_type>;
            auto apply_subs = [&parent](const auto&...subs_){
                auto subs_container = container_type{subs_.copy(subs_order{})...};
                return basic_tensor{view_factory_type::create_index_mapping_view(parent, subs_container)};
            };
            REQUIRE_THROWS_AS(std::apply(apply_subs, subs), decltype(exception));
        };
        apply_by_element(test,test_data);
    }
}

//test create_bool_mapping_view
TEMPLATE_TEST_CASE("test_create_bool_mapping_view","[test_view_factory]",
    //parent's order, subs order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using parent_order = std::tuple_element_t<0,TestType>;
    using subs_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,parent_order>;
    using bool_tensor_type = gtensor::tensor<bool,subs_order>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, bool_tensor_type{}, tensor_type{}),
        std::make_tuple(tensor_type{}, bool_tensor_type(false), tensor_type{}.reshape(0,0)),
        std::make_tuple(tensor_type{}, bool_tensor_type(true), tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type(false), tensor_type{}.reshape(0,2,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type(true), tensor_type{}.reshape(1,2,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(2,0), tensor_type{}.reshape(0,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(2,3,0), tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{false,false}, tensor_type{}.reshape(0,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{true,false}, tensor_type{}.reshape(1,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{true,true}, tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{{true,true,false},{false,true,true}}, tensor_type{}.reshape(4,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(2,3,0), tensor_type{}),
        std::make_tuple(tensor_type{1}, bool_tensor_type(false), tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{1}, bool_tensor_type(true), tensor_type{{1}}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{}, tensor_type{}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{false}, tensor_type{}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{true}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type(false), tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type(true), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type{false,true,false,true,false}, tensor_type{2,4}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type{true,true,true,true,true}, tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type{false,false,false,false,false}, tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type{true,true}, tensor_type{1,2}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}, bool_tensor_type{{false,false,true,false,true}}, tensor_type{3,5}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}, bool_tensor_type{false}, tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{{1,2,3,4,5}}, bool_tensor_type{true}, tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}, bool_tensor_type{{true,false},{false,true}}, tensor_type{1,6}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}, bool_tensor_type{true,true}, tensor_type{{1,2,3,4},{5,6,7,8}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{true}, tensor_type{{{1,2},{3,4}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{true,true}, tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{false,true,false,true}, tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{{false,true},{true,false}}, tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{{{false,true}},{{true,false}}}, tensor_type{2,5})
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto subs = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = basic_tensor{view_factory_type::create_bool_mapping_view(parent, subs)};
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_create_bool_mapping_view_exception","[test_view_factory]",
    //parent's order, subs order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using parent_order = std::tuple_element_t<0,TestType>;
    using subs_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,parent_order>;
    using bool_tensor_type = gtensor::tensor<bool,subs_order>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using gtensor::index_error;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs
    auto test_data = std::make_tuple(
        //0-dim tensor
        std::make_tuple(tensor_type(2), bool_tensor_type{}),
        //exception, subs dim > parent dim
        std::make_tuple(tensor_type{}, bool_tensor_type{}.reshape(0,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(1,2,3,0)),
        std::make_tuple(tensor_type{1}, bool_tensor_type{{true}}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{{false}}),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}}, bool_tensor_type{{{true}}}),
        //exception, subs out of bounds
        std::make_tuple(tensor_type{}, bool_tensor_type{true}),
        std::make_tuple(tensor_type{}.reshape(1,0), bool_tensor_type{{true}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(3,3,0)),
        std::make_tuple(tensor_type{1}, bool_tensor_type{true,true}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{false,false}),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}}, bool_tensor_type{{true,false,true}})
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto subs = std::get<1>(t);
        REQUIRE_THROWS_AS(view_factory_type::create_bool_mapping_view(parent, subs), index_error);
    };
    apply_by_element(test,test_data);
}


