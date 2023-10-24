/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <list>
#include <iostream>
#include "catch.hpp"
#include "descriptor.hpp"
#include "config_for_testing.hpp"
#include "helpers_for_testing.hpp"

//test helpers
TEMPLATE_TEST_CASE("test_make_broadcast_shape","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using result_shape_type = shape_type;
    using helpers_for_testing::apply_by_element;
    using gtensor::detail::make_broadcast_shape;
    using gtensor::detail::make_broadcast_shape_container;
    //0shapes,1expected broadcast shape
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(shape_type{}), result_shape_type{}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{}), result_shape_type{}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{1}, shape_type{}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{2,3,4}, shape_type{}), result_shape_type{2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}), result_shape_type{1,2,3}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{1}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{1}, shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{5}, shape_type{5}), result_shape_type{5}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{0}), result_shape_type{1,0}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{1,1}), result_shape_type{1,0}),
        std::make_tuple(std::make_tuple(shape_type{2,1}, shape_type{0}), result_shape_type{2,0}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1}), result_shape_type{1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1}, shape_type{1,1,1}, shape_type{1,1}), result_shape_type{1,1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1,1}, shape_type{1,1}), result_shape_type{1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,5}, shape_type{5,1}), result_shape_type{5,5}),
        std::make_tuple(std::make_tuple(shape_type{1,5}, shape_type{5,1}, shape_type{1,5}, shape_type{1,1}), result_shape_type{5,5}),
        std::make_tuple(std::make_tuple(shape_type{1,2,0}, shape_type{3,1,1}), result_shape_type{3,2,0}),
        std::make_tuple(std::make_tuple(shape_type{2,3,4}, shape_type{3,4}), result_shape_type{2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,3,4}, shape_type{3,4}, shape_type{1,1,1,1}, shape_type{5,1,1,1}), result_shape_type{5,2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,1,4}, shape_type{3,1}, shape_type{3,4}), result_shape_type{2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{0,1,4}, shape_type{3,1}, shape_type{0,3,4}, shape_type{1,3,4}), result_shape_type{0,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4}),
        std::make_tuple(std::make_tuple(shape_type{2,1}, shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4})
    );
    SECTION("test_variadic")
    {
        auto test = [](const auto& t){
            auto shapes = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto apply_shapes = [](const auto&...shapes_){
                return make_broadcast_shape<result_shape_type>(shapes_...);
            };
            auto result = std::apply(apply_shapes, shapes);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_container")
    {
        auto test = [](const auto& t){
            auto shapes = std::get<0>(t);
            auto expected = std::get<1>(t);
            using container_type = std::vector<shape_type>;
            auto apply_shapes = [](const auto&...shapes_){
                auto shapes = container_type{shapes_...};
                return make_broadcast_shape_container<result_shape_type>(shapes);
            };
            auto result = std::apply(apply_shapes, shapes);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_make_broadcast_shape_exception","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::value_error;
    using gtensor::detail::make_broadcast_shape;
    using gtensor::detail::make_broadcast_shape_container;
    using helpers_for_testing::apply_by_element;
    //0shapes
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{0}, shape_type{2}),
        std::make_tuple(shape_type{2}, shape_type{0}),
        std::make_tuple(shape_type{3}, shape_type{2}),
        std::make_tuple(shape_type{2}, shape_type{3}),
        std::make_tuple(shape_type{1,2}, shape_type{0}),
        std::make_tuple(shape_type{1,2}, shape_type{3}),
        std::make_tuple(shape_type{1,2}, shape_type{4,3}),
        std::make_tuple(shape_type{3,2}, shape_type{4,2}),
        std::make_tuple(shape_type{5,1,2}, shape_type{4,4,2}),
        std::make_tuple(shape_type{3}, shape_type{0}, shape_type{3}),
        std::make_tuple(shape_type{3}, shape_type{3}, shape_type{2}),
        std::make_tuple(shape_type{1,2}, shape_type{3}, shape_type{1}),
        std::make_tuple(shape_type{1,2}, shape_type{1,1}, shape_type{4,4}),
        std::make_tuple(shape_type{5,1,0}, shape_type{2,1}, shape_type{5,2,2}),
        std::make_tuple(shape_type{5,1,2}, shape_type{2,2}, shape_type{4,4,2})
    );
    SECTION("test_variadic")
    {
        auto test = [](const auto& shapes){
            auto apply_shapes = [](const auto&...shapes_){
                return make_broadcast_shape<shape_type>(shapes_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_container")
    {
        auto test = [](const auto& shapes){
            using container_type = std::vector<shape_type>;
            auto apply_shapes = [](const auto&...shapes_){
                auto shapes_container = container_type{shapes_...};
                return make_broadcast_shape_container<shape_type>(shapes_container);
            };
            REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_make_strides","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_strides;
    using helpers_for_testing::apply_by_element;
    //0shape,1order,2expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(shape_type{}, c_order{}, shape_type{}),
        std::make_tuple(shape_type{0}, c_order{}, shape_type{1}),
        std::make_tuple(shape_type{1}, c_order{}, shape_type{1}),
        std::make_tuple(shape_type{5}, c_order{}, shape_type{1}),
        std::make_tuple(shape_type{0,0}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{1,0}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{0,1}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{5,0}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{0,5}, c_order{}, shape_type{5,1}),
        std::make_tuple(shape_type{1,1}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{5,1}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{1,5}, c_order{}, shape_type{5,1}),
        std::make_tuple(shape_type{2,3,4}, c_order{}, shape_type{12,4,1}),
        std::make_tuple(shape_type{0,0,0}, c_order{}, shape_type{1,1,1}),
        std::make_tuple(shape_type{2,2,0,2}, c_order{}, shape_type{4,2,2,1}),
        std::make_tuple(shape_type{4,3,2,0}, c_order{}, shape_type{6,2,1,1}),
        std::make_tuple(shape_type{0,3,2,1}, c_order{}, shape_type{6,2,1,1}),
        //f_order
        std::make_tuple(shape_type{}, f_order{}, shape_type{}),
        std::make_tuple(shape_type{0}, f_order{}, shape_type{1}),
        std::make_tuple(shape_type{1}, f_order{}, shape_type{1}),
        std::make_tuple(shape_type{5}, f_order{}, shape_type{1}),
        std::make_tuple(shape_type{0,0}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{1,0}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{0,1}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{5,0}, f_order{}, shape_type{1,5}),
        std::make_tuple(shape_type{0,5}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{1,1}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{5,1}, f_order{}, shape_type{1,5}),
        std::make_tuple(shape_type{1,5}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{2,3,4}, f_order{}, shape_type{1,2,6}),
        std::make_tuple(shape_type{0,0,0}, f_order{}, shape_type{1,1,1}),
        std::make_tuple(shape_type{2,2,0,2}, f_order{}, shape_type{1,2,4,4}),
        std::make_tuple(shape_type{4,3,2,0}, f_order{}, shape_type{1,4,12,24}),
        std::make_tuple(shape_type{0,3,2,1}, f_order{}, shape_type{1,1,3,6})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto order = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_strides(shape, order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_make_strides_div","[test_descriptor]",
    gtensor::config::mode_div_libdivide,
    gtensor::config::mode_div_native
)
{
    using config_type = gtensor::config::extend_config_t<test_config::config_div_mode_selector_t<TestType>, int>;
    using shape_type = typename config_type::shape_type;
    using strides_div_type = gtensor::detail::strides_div_t<config_type>;
    using divider_type = typename strides_div_type::value_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_strides_div;
    using helpers_for_testing::apply_by_element;
    //0shape,1order,2expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(shape_type{}, c_order{}, strides_div_type{}),
        std::make_tuple(shape_type{0}, c_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{1}, c_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{5}, c_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{0,0}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{1,0}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{5,0}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,1}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,5}, c_order{}, strides_div_type{divider_type(5),divider_type(1)}),
        std::make_tuple(shape_type{1,1}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{5,1}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{1,5}, c_order{}, strides_div_type{divider_type(5),divider_type(1)}),
        std::make_tuple(shape_type{0,0,0}, c_order{}, strides_div_type{divider_type(1),divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{2,3,4}, c_order{}, strides_div_type{divider_type(12),divider_type(4),divider_type(1)}),
        std::make_tuple(shape_type{2,2,0,2}, c_order{}, strides_div_type{divider_type(4),divider_type(2),divider_type(2),divider_type(1)}),
        std::make_tuple(shape_type{4,3,2,0}, c_order{}, strides_div_type{divider_type(6),divider_type(2),divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,3,2,1}, c_order{}, strides_div_type{divider_type(6),divider_type(2),divider_type(1),divider_type(1)}),
        //f_order
        std::make_tuple(shape_type{}, f_order{}, strides_div_type{}),
        std::make_tuple(shape_type{0}, f_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{1}, f_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{5}, f_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{0,0}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{1,0}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{5,0}, f_order{}, strides_div_type{divider_type(1),divider_type(5)}),
        std::make_tuple(shape_type{0,1}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,5}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{1,1}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{5,1}, f_order{}, strides_div_type{divider_type(1),divider_type(5)}),
        std::make_tuple(shape_type{1,5}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,0,0}, f_order{}, strides_div_type{divider_type(1),divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{2,3,4}, f_order{}, strides_div_type{divider_type(1),divider_type(2),divider_type(6)}),
        std::make_tuple(shape_type{2,2,0,2}, f_order{}, strides_div_type{divider_type(1),divider_type(2),divider_type(4),divider_type(4)}),
        std::make_tuple(shape_type{4,3,2,0}, f_order{}, strides_div_type{divider_type(1),divider_type(4),divider_type(12),divider_type(24)}),
        std::make_tuple(shape_type{0,3,2,1}, f_order{}, strides_div_type{divider_type(1),divider_type(1),divider_type(3),divider_type(6)})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto order = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_strides_div<config_type>(shape, order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_make_reset_strides","[test_descriptor]",std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::detail::make_reset_strides;
    //0shape,1strides,2expected reset strides
    using test_type = typename std::tuple<shape_type,shape_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},shape_type{},shape_type{}},
        test_type{shape_type{0},shape_type{1},shape_type{0}},
        test_type{shape_type{1},shape_type{1},shape_type{0}},
        test_type{shape_type{5},shape_type{1},shape_type{4}},
        test_type{shape_type{0,0},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{1,0},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{0,1},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{5,0},shape_type{1,1},shape_type{4,0}},
        test_type{shape_type{0,5},shape_type{5,1},shape_type{0,4}},
        test_type{shape_type{1,1},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{5,1},shape_type{1,1},shape_type{4,0}},
        test_type{shape_type{1,5},shape_type{5,1},shape_type{0,4}},
        test_type{shape_type{0,0,0},shape_type{1,1,1},shape_type{0,0,0}},
        test_type{shape_type{2,3,4},shape_type{12,4,1},{12,8,3}},
        test_type{shape_type{2,2,0,2},shape_type{4,2,2,1},shape_type{4,2,0,1}},
        test_type{shape_type{4,3,2,0},shape_type{6,2,1,1},shape_type{18,4,1,0}},
        test_type{shape_type{0,3,2,1},shape_type{6,2,1,1},shape_type{0,4,1,0}}
    );
    auto shape = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto reset_strides_expected = std::get<2>(test_data);
    auto reset_strides_result = make_reset_strides(shape,strides);
    REQUIRE(reset_strides_result == reset_strides_expected);
}

TEMPLATE_TEST_CASE("test_make_adapted_strides","[test_descriptor]",std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::detail::make_adapted_strides;
    //0shape,1strides,2expected
    using test_type = typename std::tuple<shape_type,shape_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},shape_type{},shape_type{}},
        test_type{shape_type{0},shape_type{1},shape_type{1}},
        test_type{shape_type{1},shape_type{1},shape_type{0}},
        test_type{shape_type{5},shape_type{1},shape_type{1}},
        test_type{shape_type{0,0},shape_type{1,1},shape_type{1,1}},
        test_type{shape_type{1,0},shape_type{1,1},shape_type{0,1}},
        test_type{shape_type{0,1},shape_type{1,1},shape_type{1,0}},
        test_type{shape_type{5,0},shape_type{1,1},shape_type{1,1}},
        test_type{shape_type{0,5},shape_type{5,1},shape_type{5,1}},
        test_type{shape_type{1,1},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{5,1},shape_type{1,1},shape_type{1,0}},
        test_type{shape_type{1,5},shape_type{5,1},shape_type{0,1}},
        test_type{shape_type{0,0,0},shape_type{1,1,1},shape_type{1,1,1}},
        test_type{shape_type{2,1,4},shape_type{4,4,1},{4,0,1}},
        test_type{shape_type{2,3,4},shape_type{12,4,1},{12,4,1}},
        test_type{shape_type{2,2,0,2},shape_type{4,2,2,1},shape_type{4,2,2,1}},
        test_type{shape_type{4,3,2,0},shape_type{6,2,1,1},shape_type{6,2,1,1}},
        test_type{shape_type{0,3,2,1},shape_type{6,2,1,1},shape_type{6,2,1,0}}
    );
    auto shape = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = make_adapted_strides(shape,strides);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_make_size","[test_descriptor]",std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using index_type = typename TestType::value_type;
    using gtensor::detail::make_size;
    //shape,expected
    using test_type = typename std::tuple<shape_type,index_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},index_type{1}},
        test_type{shape_type{0},index_type{0}},
        test_type{shape_type{1},index_type{1}},
        test_type{shape_type{5},index_type{5}},
        test_type{shape_type{0,0},index_type{0}},
        test_type{shape_type{1,0},index_type{0}},
        test_type{shape_type{0,1},index_type{0}},
        test_type{shape_type{0,5},index_type{0}},
        test_type{shape_type{5,0},index_type{0}},
        test_type{shape_type{1,1},index_type{1}},
        test_type{shape_type{5,1},index_type{5}},
        test_type{shape_type{1,5},index_type{5}},
        test_type{shape_type{0,0,0},index_type{0}},
        test_type{shape_type{2,3,4},index_type{24}},
        test_type{shape_type{2,2,0,2},index_type{0}},
        test_type{shape_type{4,3,2,0},index_type{0}},
        test_type{shape_type{0,3,2,1},index_type{0}}
    );
    auto shape = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    auto result = make_size(shape);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_flat_to_flat", "[test_descriptor]",
    gtensor::config::mode_div_native,
    gtensor::config::mode_div_libdivide
)
{
    using config_type = gtensor::config::extend_config_t<test_config::config_div_mode_selector_t<TestType>, int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_dividers;
    using gtensor::detail::flat_to_flat;
    using helpers_for_testing::apply_by_element;
    //0flat_idx,1strides,2cstrides,3offset,4order,5expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(index_type{0}, shape_type{1}, shape_type{1}, index_type{0}, c_order{}, index_type{0}),
        std::make_tuple(index_type{0}, shape_type{1}, shape_type{1}, index_type{1}, c_order{}, index_type{1}),
        std::make_tuple(index_type{5}, shape_type{1}, shape_type{1}, index_type{0}, c_order{}, index_type{5}),
        std::make_tuple(index_type{5}, shape_type{1}, shape_type{1}, index_type{1}, c_order{}, index_type{6}),
        std::make_tuple(index_type{5}, shape_type{1,1}, shape_type{1,1}, index_type{0}, c_order{}, index_type{5}),
        std::make_tuple(index_type{5}, shape_type{1,1}, shape_type{1,1}, index_type{10}, c_order{}, index_type{15}),
        std::make_tuple(index_type{0}, shape_type{3,1}, shape_type{2,1}, index_type{0}, c_order{}, index_type{0}),
        std::make_tuple(index_type{5}, shape_type{3,1}, shape_type{2,1}, index_type{0}, c_order{}, index_type{4}),
        std::make_tuple(index_type{5}, shape_type{3,1}, shape_type{2,1}, index_type{10}, c_order{}, index_type{14}),
        std::make_tuple(index_type{34}, shape_type{12,3,1}, shape_type{6,3,1}, index_type{0}, c_order{}, index_type{22}),
        std::make_tuple(index_type{34}, shape_type{12,3,1}, shape_type{6,3,1}, index_type{3}, c_order{}, index_type{25}),
        //f_order
        std::make_tuple(index_type{0}, shape_type{1}, shape_type{1}, index_type{0}, f_order{}, index_type{0}),
        std::make_tuple(index_type{0}, shape_type{1}, shape_type{1}, index_type{1}, f_order{}, index_type{1}),
        std::make_tuple(index_type{5}, shape_type{1}, shape_type{1}, index_type{0}, f_order{}, index_type{5}),
        std::make_tuple(index_type{5}, shape_type{1}, shape_type{1}, index_type{1}, f_order{}, index_type{6}),
        std::make_tuple(index_type{5}, shape_type{1,1}, shape_type{1,1}, index_type{0}, f_order{}, index_type{5}),
        std::make_tuple(index_type{5}, shape_type{1,1}, shape_type{1,1}, index_type{10}, f_order{}, index_type{15}),
        std::make_tuple(index_type{0}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{0}),
        std::make_tuple(index_type{1}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{3}),
        std::make_tuple(index_type{2}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{1}),
        std::make_tuple(index_type{3}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{4}),
        std::make_tuple(index_type{4}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{2}),
        std::make_tuple(index_type{5}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{5}),
        std::make_tuple(index_type{0}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{2}),
        std::make_tuple(index_type{1}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{3}),
        std::make_tuple(index_type{2}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{4}),
        std::make_tuple(index_type{3}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{5}),
        std::make_tuple(index_type{6}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{14}),
        std::make_tuple(index_type{7}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{15})
    );
    auto test = [](const auto& t){
        auto flat_idx = std::get<0>(t);
        auto strides = std::get<1>(t);
        auto strides_div = make_dividers<config_type>(strides);
        auto cstrides = std::get<2>(t);
        auto offset = std::get<3>(t);
        auto order = std::get<4>(t);
        auto expected = std::get<5>(t);
        using order_type = decltype(order);
        auto result = flat_to_flat<order_type>(strides_div, cstrides, offset, flat_idx);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_shape_of_type","[test_descriptor]"){
    using gtensor::detail::make_shape_of_type;
    using shape_type = std::vector<int>;
    auto s = shape_type{1,2,3};
    auto l = std::list<int>{1,2,3};
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(std::declval<std::size_t>())), shape_type>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(std::declval<int>())), shape_type>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(shape_type{1,2,3})), shape_type&&>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(s)), shape_type&>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(std::list{1,2,3})), shape_type>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(l)), shape_type>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>({1,2,3})), shape_type>);

    REQUIRE(make_shape_of_type<shape_type>(0) == shape_type{0});
    REQUIRE(make_shape_of_type<shape_type>(std::size_t{10}) == shape_type{10});
    REQUIRE(make_shape_of_type<shape_type>(shape_type{1,2,3}) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(s) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(std::list{1,2,3}) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(l) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>({1,2,3}) == shape_type{1,2,3});
}

//test basic_descriptor
TEST_CASE("test_basic_descriptor","[test_descriptor]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using gtensor::basic_descriptor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::change_order_t;
    using gtensor::detail::make_dividers;
    using gtensor::detail::make_strides_div;
    using helpers_for_testing::apply_by_element;
    //0order,1shape,2expected_strides,3expected_adapted_strides,4expected_reset_strides,5expected_size,6expected_dim
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(c_order{},shape_type{},shape_type{},shape_type{},shape_type{},index_type{1},dim_type{0}),
        std::make_tuple(c_order{},shape_type{0},shape_type{1},shape_type{1},shape_type{0},index_type{0},dim_type{1}),
        std::make_tuple(c_order{},shape_type{1},shape_type{1},shape_type{0},shape_type{0},index_type{1},dim_type{1}),
        std::make_tuple(c_order{},shape_type{5},shape_type{1},shape_type{1},shape_type{4},index_type{5},dim_type{1}),
        std::make_tuple(c_order{},shape_type{0,0},shape_type{1,1},shape_type{1,1},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{1,0},shape_type{1,1},shape_type{0,1},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{0,1},shape_type{1,1},shape_type{1,0},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{5,0},shape_type{1,1},shape_type{1,1},shape_type{4,0},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{0,5},shape_type{5,1},shape_type{5,1},shape_type{0,4},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{1,1},shape_type{1,1},shape_type{0,0},shape_type{0,0},index_type{1},dim_type{2}),
        std::make_tuple(c_order{},shape_type{1,5},shape_type{5,1},shape_type{0,1},shape_type{0,4},index_type{5},dim_type{2}),
        std::make_tuple(c_order{},shape_type{5,1},shape_type{1,1},shape_type{1,0},shape_type{4,0},index_type{5},dim_type{2}),
        std::make_tuple(c_order{},shape_type{5,4,3},shape_type{12,3,1},shape_type{12,3,1},shape_type{48,9,2},index_type{60},dim_type{3}),
        std::make_tuple(c_order{},shape_type{2,2,0,2},shape_type{4,2,2,1},shape_type{4,2,2,1},shape_type{4,2,0,1},index_type{0},dim_type{4}),
        std::make_tuple(c_order{},shape_type{4,3,2,0},shape_type{6,2,1,1},shape_type{6,2,1,1},shape_type{18,4,1,0},index_type{0},dim_type{4}),
        std::make_tuple(c_order{},shape_type{0,3,2,1},shape_type{6,2,1,1},shape_type{6,2,1,0},shape_type{0,4,1,0},index_type{0},dim_type{4}),
        //f_order
        std::make_tuple(f_order{},shape_type{},shape_type{},shape_type{},shape_type{},index_type{1},dim_type{0}),
        std::make_tuple(f_order{},shape_type{0},shape_type{1},shape_type{1},shape_type{0},index_type{0},dim_type{1}),
        std::make_tuple(f_order{},shape_type{1},shape_type{1},shape_type{0},shape_type{0},index_type{1},dim_type{1}),
        std::make_tuple(f_order{},shape_type{5},shape_type{1},shape_type{1},shape_type{4},index_type{5},dim_type{1}),
        std::make_tuple(f_order{},shape_type{0,0},shape_type{1,1},shape_type{1,1},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{1,0},shape_type{1,1},shape_type{0,1},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{0,1},shape_type{1,1},shape_type{1,0},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{5,0},shape_type{1,5},shape_type{1,5},shape_type{4,0},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{0,5},shape_type{1,1},shape_type{1,1},shape_type{0,4},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{1,1},shape_type{1,1},shape_type{0,0},shape_type{0,0},index_type{1},dim_type{2}),
        std::make_tuple(f_order{},shape_type{1,5},shape_type{1,1},shape_type{0,1},shape_type{0,4},index_type{5},dim_type{2}),
        std::make_tuple(f_order{},shape_type{5,1},shape_type{1,5},shape_type{1,0},shape_type{4,0},index_type{5},dim_type{2}),
        std::make_tuple(f_order{},shape_type{5,4,3},shape_type{1,5,20},shape_type{1,5,20},shape_type{4,15,40},index_type{60},dim_type{3}),
        std::make_tuple(f_order{},shape_type{2,2,0,2},shape_type{1,2,4,4},shape_type{1,2,4,4},shape_type{1,2,0,4},index_type{0},dim_type{4}),
        std::make_tuple(f_order{},shape_type{4,3,2,0},shape_type{1,4,12,24},shape_type{1,4,12,24},shape_type{3,8,12,0},index_type{0},dim_type{4}),
        std::make_tuple(f_order{},shape_type{0,3,2,1},shape_type{1,1,3,6},shape_type{1,1,3,0},shape_type{0,2,3,0},index_type{0},dim_type{4})
    );
    auto test = [](const auto& t){
        auto order = std::get<0>(t);
        using order_type = decltype(order);
        auto shape = std::get<1>(t);
        auto expected_shape = shape;
        auto expected_strides = std::get<2>(t);
        auto expected_strides_div = make_dividers<config_type>(expected_strides);
        auto expected_strides_div_c_order = make_strides_div<config_type>(shape, c_order{});
        auto expected_strides_div_f_order = make_strides_div<config_type>(shape, f_order{});
        auto expected_adapted_strides = std::get<3>(t);
        auto expected_reset_strides = std::get<4>(t);
        auto expected_size = std::get<5>(t);
        auto expected_dim = std::get<6>(t);
        using descriptor_type = basic_descriptor<config_type, order_type>;
        auto descriptor = descriptor_type{shape};

        auto result_shape = descriptor.shape();
        auto result_strides = descriptor.strides();
        auto result_strides_div = descriptor.strides_div();
        auto result_strides_div_c_order = descriptor.strides_div(c_order{});
        auto result_strides_div_f_order = descriptor.strides_div(f_order{});
        auto result_adapted_strides = descriptor.adapted_strides();
        auto result_reset_strides = descriptor.reset_strides();
        auto result_size = descriptor.size();
        auto result_dim = descriptor.dim();

        REQUIRE(result_shape == expected_shape);
        REQUIRE(result_strides == expected_strides);
        REQUIRE(result_strides_div == expected_strides_div);
        REQUIRE(result_strides_div_c_order == expected_strides_div_c_order);
        REQUIRE(result_strides_div_f_order == expected_strides_div_f_order);
        REQUIRE(result_adapted_strides == expected_adapted_strides);
        REQUIRE(result_reset_strides == expected_reset_strides);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_dim == expected_dim);
    };
    apply_by_element(test,test_data);
}
