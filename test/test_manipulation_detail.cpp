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
#include "test_config.hpp"

//test helpers
TEST_CASE("test_check_stack_args_nothrow","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::check_stack_variadic_args;
    using gtensor::detail::check_stack_container_args;
    using helpers_for_testing::apply_by_element;
    //0axis,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0}, shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}))
    );
    SECTION("test_check_stack_variadic_args_nothrow")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto apply_shapes = [&axis](const auto&...shapes_){
                check_stack_variadic_args(axis, shapes_...);
            };
            REQUIRE_NOTHROW(std::apply(apply_shapes, shapes));
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_stack_container_args_nothrow")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            REQUIRE_NOTHROW(check_stack_container_args(axis,container));
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_check_stack_args_exception","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::value_error;
    using gtensor::detail::check_stack_container_args;
    using gtensor::detail::check_stack_variadic_args;
    using helpers_for_testing::apply_by_element;
    //0axis,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{4}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1}, shape_type{1})),
        std::make_tuple(dim_type{4}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1}, shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{2,2,3}, shape_type{1,2,3}))
    );
    SECTION("test_check_stack_variadic_args_exception")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto apply_shapes = [&axis](const auto&...shapes_){
                check_stack_variadic_args(axis, shapes_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_stack_container_args_exception")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            REQUIRE_THROWS_AS(check_stack_container_args(axis,container), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_check_concatenate_args_nothrow","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::check_concatenate_variadic_args;
    using gtensor::detail::check_concatenate_container_args;
    using helpers_for_testing::apply_by_element;

    //0axis,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1}, shape_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{2,0}, shape_type{3,0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{2,0}, shape_type{2,0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{2,2}, shape_type{1,2})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{2,2}, shape_type{2,1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{5,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,20,3}, shape_type{1,10,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{1,2,3}, shape_type{5,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,22,3}, shape_type{1,12,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,13}, shape_type{1,2,33}))
    );
    SECTION("test_check_concatenate_variadic_args_nothrow")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            REQUIRE_NOTHROW(check_concatenate_variadic_args(axis, shapes));
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_concatenate_container_args_nothrow")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            REQUIRE_NOTHROW(check_concatenate_container_args(axis,container));
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_check_concatenate_args_exception","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::value_error;
    using gtensor::detail::check_concatenate_variadic_args;
    using gtensor::detail::check_concatenate_container_args;
    using helpers_for_testing::apply_by_element;

    //0axis,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0}, shape_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1}, shape_type{1})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0,1}, shape_type{1,0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,0}, shape_type{0,1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,20,3}, shape_type{1,20,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{10,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{10,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,12,3}, shape_type{1,2,3}))
    );
    SECTION("test_check_concatenate_variadic_args_exception")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            REQUIRE_THROWS_AS(check_concatenate_variadic_args(axis, shapes), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_concatenate_container_args_exception")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            REQUIRE_THROWS_AS(check_concatenate_container_args(axis,container), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_make_stack_shape","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_stack_shape;
    using test_type = std::tuple<shape_type,dim_type,index_type,shape_type>;
    //0shape,1axis,2tensors_number,3expected
    auto test_data = GENERATE(
        test_type{shape_type{0},dim_type{0},index_type{1},shape_type{1,0}},
        test_type{shape_type{0},dim_type{1},index_type{1},shape_type{0,1}},
        test_type{shape_type{5},dim_type{0},index_type{1},shape_type{1,5}},
        test_type{shape_type{5},dim_type{1},index_type{1},shape_type{5,1}},
        test_type{shape_type{3,4},dim_type{0},index_type{7},shape_type{7,3,4}},
        test_type{shape_type{3,4},dim_type{1},index_type{7},shape_type{3,7,4}},
        test_type{shape_type{3,4},dim_type{2},index_type{7},shape_type{3,4,7}},
        test_type{shape_type{3,4,5},dim_type{0},index_type{7},shape_type{7,3,4,5}},
        test_type{shape_type{3,4,5},dim_type{1},index_type{7},shape_type{3,7,4,5}},
        test_type{shape_type{3,4,5},dim_type{2},index_type{7},shape_type{3,4,7,5}},
        test_type{shape_type{3,4,5},dim_type{3},index_type{7},shape_type{3,4,5,7}}
    );
    auto shape = std::get<0>(test_data);
    auto axis = std::get<1>(test_data);
    auto tensors_number = std::get<2>(test_data);
    auto expected = std::get<3>(test_data);
    auto result = make_stack_shape(axis,shape,tensors_number);
    REQUIRE(result == expected);
}

TEST_CASE("test_make_concatenate_shape","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::make_concatenate_variadic_shape;
    using gtensor::detail::make_concatenate_container_shape;
    using helpers_for_testing::apply_by_element;
    //0axis,1shapes,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}), shape_type{0}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{0}), shape_type{0}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{3}, shape_type{0}), shape_type{3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{3}), shape_type{3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{1}, shape_type{2}, shape_type{0}), shape_type{3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0,3}, shape_type{2,3}), shape_type{2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{2,3}, shape_type{0,3}), shape_type{2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0,2}, shape_type{0,2}), shape_type{0,2}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0,2}, shape_type{0,2}), shape_type{0,4}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{2,0}, shape_type{2,3}), shape_type{2,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{2,3}, shape_type{2,0}), shape_type{2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{2,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{5,2,3}), shape_type{15,2,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,4,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,20,3}, shape_type{1,10,3}), shape_type{1,30,3}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,2,6}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,30}, shape_type{1,2,3}), shape_type{1,2,33}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{3,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{1,2,3}, shape_type{5,2,3}), shape_type{16,2,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,6,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,22,3}, shape_type{1,12,3}), shape_type{1,36,3}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,2,9}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,13}, shape_type{1,2,33}), shape_type{1,2,49})
    );

    SECTION("test_make_concatenate_variadic_shape")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = make_concatenate_variadic_shape(axis, shapes);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_make_concatenate_container_shape")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            auto result = make_concatenate_container_shape(axis,container);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_nested_tuple_depth", "[test_manipulation]"){
    using gtensor::detail::nested_tuple_depth_v;
    REQUIRE(nested_tuple_depth_v<std::tuple<int>> == 1);
    REQUIRE(nested_tuple_depth_v<std::tuple<int,int>> == 1);
    REQUIRE(nested_tuple_depth_v<std::tuple<int,int,int>> == 1);
    REQUIRE(nested_tuple_depth_v<std::tuple<std::tuple<int>,std::tuple<int>,std::tuple<int>>> == 2);
    REQUIRE(nested_tuple_depth_v<std::tuple<std::tuple<int,int>,std::tuple<int>,std::tuple<int,int,int>>> == 2);
    REQUIRE(nested_tuple_depth_v<std::tuple<std::tuple<std::tuple<int>, std::tuple<int>>,std::tuple<std::tuple<int>>,std::tuple<std::tuple<int>>>> == 3);
}

TEST_CASE("test_is_tensor_nested_tuple", "[test_manipulation]"){
    using tensor_int_type = gtensor::tensor<int>;
    using tensor_double_type = gtensor::tensor<double>;
    using gtensor::detail::is_tensor_nested_tuple_v;
    REQUIRE(!is_tensor_nested_tuple_v<int>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>, int>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>, tensor_int_type>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>, std::tuple<tensor_int_type,int>>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<std::tuple<tensor_int_type>>, std::tuple<tensor_int_type>>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type,std::tuple<tensor_double_type>>,std::tuple<tensor_int_type,tensor_double_type>>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type,tensor_int_type>>,std::tuple<tensor_int_type>>>);
    REQUIRE(!is_tensor_nested_tuple_v<
        std::tuple<
            std::tuple<std::tuple<tensor_int_type,tensor_int_type>,std::tuple<tensor_int_type>>,
            std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type>,std::tuple<tensor_int_type>>,
            tensor_int_type>
        >
    );
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<tensor_int_type>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<tensor_int_type,tensor_double_type>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<tensor_int_type,tensor_int_type,tensor_double_type>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type,tensor_double_type>,std::tuple<tensor_double_type,tensor_int_type>>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<
        std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type,tensor_int_type>>,
        std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_double_type>>,
        std::tuple<std::tuple<tensor_double_type,tensor_int_type,tensor_int_type>>>>
    );
}

TEST_CASE("test_widen_shape", "[test_manipulation]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using dim_type = config_type::dim_type;
    using gtensor::detail::widen_shape;
    //0shape,1new_dim,2expected
    using test_type = std::tuple<shape_type,dim_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{0},dim_type{1},shape_type{0}},
        test_type{shape_type{0},dim_type{2},shape_type{1,0}},
        test_type{shape_type{0},dim_type{3},shape_type{1,1,0}},
        test_type{shape_type{2,3},dim_type{1},shape_type{2,3}},
        test_type{shape_type{2,3},dim_type{2},shape_type{2,3}},
        test_type{shape_type{2,3},dim_type{3},shape_type{1,2,3}},
        test_type{shape_type{2,3},dim_type{4},shape_type{1,1,2,3}}
    );
    auto shape = std::get<0>(test_data);
    auto new_dim = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = widen_shape(shape,new_dim);
    REQUIRE(result == expected);
}

