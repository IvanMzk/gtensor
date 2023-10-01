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
#include "test_config.hpp"

//constructors
TEST_CASE("test_tensor_default_constructor","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = tensor_type::config_type;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;

    auto test_data = GENERATE(
        tensor_type(),
        tensor_type{}
    );
    REQUIRE(test_data.size() == index_type{0});
    REQUIRE(test_data.dim() == dim_type{1});
    REQUIRE(test_data.shape() == shape_type{0,});
}

TEST_CASE("test_0-dim_tensor_constructor","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = tensor_type::config_type;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0value
    auto test_data = std::make_tuple(
        0,
        1.0f
    );
    auto test = [](const auto& value){
        const shape_type expected_shape{};
        const index_type expected_size{1};
        const dim_type expected_dim{0};
        const value_type expected_value = static_cast<value_type>(value);
        auto result_tensor = tensor_type(value);
        auto result_shape = result_tensor.shape();
        auto result_size = result_tensor.size();
        auto result_dim = result_tensor.dim();
        auto result_value = *result_tensor.begin();
        REQUIRE(result_shape == expected_shape);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_value == expected_value);
        auto result_first = result_tensor.begin();
        auto result_last = result_tensor.end();
        REQUIRE(result_first+1 == result_last);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_constructor_from_list","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using dim_type = typename tensor_type::dim_type;
    using shape_type = typename tensor_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0result,1expected_shape,2expected_size,3expected_dim,4expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, shape_type{0}, index_type{0} , dim_type{1}, std::vector<value_type>{}),
        std::make_tuple(tensor_type{1}, shape_type{1}, index_type{1} , dim_type{1}, std::vector<value_type>{1}),
        std::make_tuple(tensor_type{1,2,3}, shape_type{3}, index_type{3} , dim_type{1}, std::vector<value_type>{1,2,3}),
        std::make_tuple(tensor_type{{1}}, shape_type{1,1}, index_type{1} , dim_type{2}, std::vector<value_type>{1}),
        std::make_tuple(tensor_type{{1,2,3}}, shape_type{1,3}, index_type{3} , dim_type{2}, std::vector<value_type>{1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, shape_type{2,3}, index_type{6} , dim_type{2}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{{1,2,3,4}}}, shape_type{1,1,4}, index_type{4} , dim_type{3}, std::vector<value_type>{1,2,3,4}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4}}}, shape_type{1,4,1}, index_type{4} , dim_type{3}, std::vector<value_type>{1,2,3,4}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}}, shape_type{1,4,3}, index_type{12} , dim_type{3}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected_shape = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_dim = std::get<3>(t);
        auto expected_elements = std::get<4>(t);
        REQUIRE(result.shape() == expected_shape);
        REQUIRE(result.size() == expected_size);
        REQUIRE(result.dim() == expected_dim);
        REQUIRE(static_cast<index_type>(std::distance(result.begin(),result.end())) == static_cast<index_type>(std::distance(expected_elements.begin(),expected_elements.end())));
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin()));
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_constructor_shape","[test_tensor]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = typename tensor_type::shape_type;
    using helpers_for_testing::apply_by_element;
    const value_type any{std::numeric_limits<value_type>::max()};
    //0shape,1expected
    auto test_data = std::make_tuple(
        //shape container
        std::make_tuple(shape_type{}, tensor_type(any)),
        std::make_tuple(shape_type{0}, tensor_type()),
        std::make_tuple(std::vector<int>{}, tensor_type(any)),
        std::make_tuple(std::vector<int>{1}, tensor_type{any}),
        std::make_tuple(shape_type{3}, tensor_type{any,any,any}),
        std::make_tuple(std::vector<int>{2,3}, tensor_type{{any,any,any},{any,any,any}})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto expected = std::get<1>(t);
        tensor_type result(shape);
        REQUIRE(result.shape() == expected.shape());
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_constructor_shape_value","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = typename tensor_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0shape,1value,2expected
    auto test_data = std::make_tuple(
        //shape scalar
        std::make_tuple(0,value_type{1},tensor_type{}),
        std::make_tuple(1,value_type{2},tensor_type{2}),
        std::make_tuple(std::size_t{5},value_type{3},tensor_type{3,3,3,3,3}),
        std::make_tuple(double{6},value_type{3},tensor_type{3,3,3,3,3,3}),
        //shape container
        std::make_tuple(shape_type{},value_type{1},tensor_type(1)),
        std::make_tuple(shape_type{},value_type{-1},tensor_type(-1)),
        std::make_tuple(shape_type{0},value_type{1},tensor_type{}),
        std::make_tuple(shape_type{1},value_type{1},tensor_type{1}),
        std::make_tuple(shape_type{5},value_type{2},tensor_type{2,2,2,2,2}),
        std::make_tuple(shape_type{1,1},value_type{2},tensor_type{{2}}),
        std::make_tuple(shape_type{1,3},value_type{0},tensor_type{{0,0,0}}),
        std::make_tuple(std::vector<std::size_t>{2,3},value_type{0},tensor_type{{0,0,0},{0,0,0}}),
        std::make_tuple(std::vector<int>{1,2,3},value_type{3},tensor_type{{{3,3,3},{3,3,3}}})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto value = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = tensor_type(shape, value);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_shape_init_list_value","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = tensor_type::index_type;
    using helpers_for_testing::apply_by_element;
    auto make_result = [](std::initializer_list<index_type> shape, const value_type& v){
        return tensor_type(shape,v);
    };
    //0result,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(make_result({},1), tensor_type(value_type{1})),
        std::make_tuple(make_result({},-1), tensor_type(value_type{-1})),
        std::make_tuple(make_result({0},1), tensor_type{}),
        std::make_tuple(make_result({1},1), tensor_type{1}),
        std::make_tuple(make_result({5},2), tensor_type{2,2,2,2,2}),
        std::make_tuple(make_result({1,1},2), tensor_type{{2}}),
        std::make_tuple(make_result({1,3},0), tensor_type{{0,0,0}}),
        std::make_tuple(make_result({2,3},1), tensor_type{{1,1,1},{1,1,1}}),
        std::make_tuple(make_result({1,2,3},1), tensor_type{{{1,1,1},{1,1,1}}})
    );
    auto test = [](auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_shape_range","[test_tensor]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = tensor_type::shape_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;
    SECTION("range_>=_size")
    {
        //0layout,1shape,2elements,3expected
        auto test_data = std::make_tuple(
            //shape scalar
            std::make_tuple(c_order{},0, std::vector<value_type>{1,2,3,4,5}, tensor_type{}),
            std::make_tuple(c_order{},4, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3,4}),
            std::make_tuple(c_order{},std::size_t{3}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3}),
            std::make_tuple(f_order{},0, std::vector<value_type>{1,2,3,4,5}, tensor_type{}),
            std::make_tuple(f_order{},4, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3,4}),
            std::make_tuple(f_order{},std::size_t{3}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3}),
            //shape container
            //c_order layout
            //0-dim
            std::make_tuple(c_order{},shape_type{}, std::vector<value_type>{1,2,3,4,5}, tensor_type(value_type{1})),
            //n-dim
            std::make_tuple(c_order{},shape_type{0}, std::vector<value_type>{}, tensor_type{}),
            std::make_tuple(c_order{},shape_type{0}, std::vector<value_type>{1,2,3,4,5}, tensor_type{}),
            std::make_tuple(c_order{},shape_type{3}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3}),
            std::make_tuple(c_order{},shape_type{5}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3,4,5}),
            std::make_tuple(c_order{},std::vector<std::size_t>{2,2}, std::vector<value_type>{1,2,3,4,5}, tensor_type{{1,2},{3,4}}),
            //f_order layout
            //0-dim
            std::make_tuple(f_order{},shape_type{}, std::vector<value_type>{1,2,3,4,5}, tensor_type(value_type{1})),
            //n-dim
            std::make_tuple(f_order{},shape_type{0}, std::vector<value_type>{}, tensor_type{}),
            std::make_tuple(f_order{},shape_type{0}, std::vector<value_type>{1,2,3,4,5}, tensor_type{}),
            std::make_tuple(f_order{},shape_type{3}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3}),
            std::make_tuple(f_order{},shape_type{5}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3,4,5}),
            std::make_tuple(f_order{},std::vector<std::size_t>{2,2}, std::vector<value_type>{1,2,3,4,5}, tensor_type{{1,3},{2,4}})
        );
        auto test = [](auto& t){
            auto layout = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto elements = std::get<2>(t);
            auto expected = std::get<3>(t);
            using layout_type = decltype(layout);
            using result_tensor_type = gtensor::tensor<value_type,layout_type>;
            auto result = result_tensor_type{shape, elements.begin(), elements.end()};
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("range_<_size")
    {
        using index_type = tensor_type::index_type;
        const value_type any{std::numeric_limits<value_type>::max()};

        //0layout,1shape,2elements,3range_size,4expected
        auto test_data = std::make_tuple(
            //shape scalar
            std::make_tuple(c_order{},8, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
            std::make_tuple(c_order{},std::size_t{6}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{1,2,3,4,5,any}),
            std::make_tuple(f_order{},8, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
            std::make_tuple(f_order{},std::size_t{6}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{1,2,3,4,5,any}),
            //shape container
            //c_order layout
            //0-dim
            std::make_tuple(c_order{},shape_type{}, std::vector<value_type>{}, index_type{0}, tensor_type(any)),
            //n-dim
            std::make_tuple(c_order{},shape_type{8}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
            std::make_tuple(c_order{},std::vector<int>{2,3}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{{1,2,3},{4,5,any}}),
            //f_order layout
            //0-dim
            std::make_tuple(f_order{},shape_type{}, std::vector<value_type>{}, index_type{0}, tensor_type(any)),
            //n-dim
            std::make_tuple(f_order{},shape_type{8}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
            std::make_tuple(f_order{},std::vector<int>{2,3}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{{1,3,5},{2,4,any}})
        );
        auto test = [](auto& t){
            auto layout = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto elements = std::get<2>(t);
            auto range_size = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto expected_shape = expected.shape();
            using layout_type = decltype(layout);
            using result_tensor_type = gtensor::tensor<value_type,layout_type>;
            auto result = result_tensor_type{shape, elements.begin(), elements.end()};
            auto result_size = result.size();
            auto result_shape = result.shape();
            REQUIRE(range_size < result_size);
            REQUIRE(result_shape == expected_shape);
            REQUIRE(std::equal(result.begin(),result.begin()+range_size,expected.begin()));
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_tensor_constructor_shape_init_list_range","[test_tensor]")
{
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = tensor_type::index_type;
    using helpers_for_testing::apply_by_element;
    auto make_result = [](auto layout, std::initializer_list<index_type> shape, const std::vector<value_type>& elements){
        using layout_type = decltype(layout);
        using result_tensor_type = gtensor::tensor<value_type,layout_type>;
        return result_tensor_type(shape,elements.begin(),elements.end());
    };
    SECTION("range_>=_size")
    {
        //0result,1expected
        auto test_data = std::make_tuple(
            //c_order layout
            //0-dim
            std::make_tuple(make_result(c_order{},{},std::vector<value_type>{1,2,3,4,5}), tensor_type(value_type{1})),
            //n-dim
            std::make_tuple(make_result(c_order{},{0},std::vector<value_type>{}), tensor_type{}),
            std::make_tuple(make_result(c_order{},{0},std::vector<value_type>{1,2,3,4,5}), tensor_type{}),
            std::make_tuple(make_result(c_order{},{3},std::vector<value_type>{1,2,3,4,5}), tensor_type{1,2,3}),
            std::make_tuple(make_result(c_order{},{5},std::vector<value_type>{1,2,3,4,5}), tensor_type{1,2,3,4,5}),
            std::make_tuple(make_result(c_order{},{2,2},std::vector<value_type>{1,2,3,4,5}), tensor_type{{1,2},{3,4}}),
            //f_order layout
            //0-dim
            std::make_tuple(make_result(f_order{},{},std::vector<value_type>{1,2,3,4,5}), tensor_type(value_type{1})),
            //n-dim
            std::make_tuple(make_result(f_order{},{0},std::vector<value_type>{}), tensor_type{}),
            std::make_tuple(make_result(f_order{},{0},std::vector<value_type>{1,2,3,4,5}), tensor_type{}),
            std::make_tuple(make_result(f_order{},{3},std::vector<value_type>{1,2,3,4,5}), tensor_type{1,2,3}),
            std::make_tuple(make_result(f_order{},{5},std::vector<value_type>{1,2,3,4,5}), tensor_type{1,2,3,4,5}),
            std::make_tuple(make_result(f_order{},{2,2},std::vector<value_type>{1,2,3,4,5}), tensor_type{{1,3},{2,4}})
        );
        auto test = [](auto& t){
            auto result = std::get<0>(t);
            auto expected = std::get<1>(t);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("range_<_size")
    {
        using index_type = typename tensor_type::index_type;
        const value_type any{-1};
        //0result,1range_size,2expected
        auto test_data = std::make_tuple(
            //c_order layout
            //0-dim
            std::make_tuple(make_result(c_order{},{},std::vector<value_type>{}), index_type{0}, tensor_type(any)),
            //n-dim
            std::make_tuple(make_result(c_order{},{8},std::vector<value_type>{1,2,3,4,5}), index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
            std::make_tuple(make_result(c_order{},{2,3},std::vector<value_type>{1,2,3,4,5}), index_type{5}, tensor_type{{1,2,3},{4,5,any}}),
            //f_order layout
            //0-dim
            std::make_tuple(make_result(f_order{},{},std::vector<value_type>{}), index_type{0}, tensor_type(any)),
            //n-dim
            std::make_tuple(make_result(f_order{},{8},std::vector<value_type>{1,2,3,4,5}), index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
            std::make_tuple(make_result(f_order{},{2,3},std::vector<value_type>{1,2,3,4,5}), index_type{5}, tensor_type{{1,3,5},{2,4,any}})
        );
        auto test = [](auto& t){
            auto result = std::get<0>(t);
            auto range_size = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto expected_shape = expected.shape();
            auto result_size = result.size();
            auto result_shape = result.shape();
            REQUIRE(range_size < result_size);
            REQUIRE(result_shape == expected_shape);
            REQUIRE(std::equal(result.begin(),result.begin()+range_size,expected.begin()));
        };
        apply_by_element(test,test_data);
    }
}

//related
TEST_CASE("test_tensor_swap","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //0tensor0,1tensor1,2expected_tensor0,3expected_tensor1
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3},tensor_type{{4},{5}},tensor_type{{4},{5}},tensor_type{1,2,3}),
        std::make_tuple(tensor_type{1,2,3}+1,tensor_type{{4},{5}}+1,tensor_type{{5},{6}},tensor_type{2,3,4}),
        std::make_tuple(tensor_type{{1,2,3}}.transpose(),tensor_type{{4},{5}}.transpose(),tensor_type{{4,5}},tensor_type{{1},{2},{3}})
    );
    auto test = [](const auto& t){
        auto ten1 = std::get<0>(t);
        auto ten2 = std::get<1>(t);
        auto expected_ten1 = std::get<2>(t);
        auto expected_ten2 = std::get<3>(t);
        auto ten1_copy{ten1};
        auto ten2_copy{ten2};
        REQUIRE(ten1_copy.is_same(ten1));
        REQUIRE(ten2_copy.is_same(ten2));
        swap(ten1,ten2);
        REQUIRE(ten1.is_same(ten2_copy));
        REQUIRE(ten2.is_same(ten1_copy));
        REQUIRE(ten1 == expected_ten1);
        REQUIRE(ten2 == expected_ten2);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_copy","[test_tensor]",
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using layout = std::tuple_element_t<0,TestType>;
    using copy_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,layout>;
    using helpers_for_testing::apply_by_element;
    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(2),tensor_type(2)),
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,3,2,1},tensor_type{1,2,3,4,3,2,1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type(-1) + tensor_type(1) + tensor_type(2) + tensor_type(3), tensor_type(5)),
        std::make_tuple(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1} + tensor_type{0,1,2}, tensor_type{{1,3,5},{4,6,8}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}), tensor_type{{5,6},{2,3}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose(), tensor_type{{5,2},{6,3}}),
        std::make_tuple((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1})({{{},{},-1},{1}}).transpose(), tensor_type{{5,2},{6,3}}),
        std::make_tuple(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1}, tensor_type{{5,2},{6,3}}),
        std::make_tuple(((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1})).reshape(4),tensor_type{5,2,6,3})
    );
    auto test_copy = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = ten.copy(policy...,copy_order{});
            using result_order = typename decltype(result)::order;
            REQUIRE(std::is_same_v<result_order,copy_order>);
            REQUIRE(result == expected);
            REQUIRE(!ten.is_same(result));
        };
        apply_by_element(test, test_data);
    };
    SECTION("default_policy")
    {
        test_copy();
    }
    SECTION("exec_pol<4>")
    {
        test_copy(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_copy(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_tensor_eval_of_view","[test_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,order>;
    using helpers_for_testing::apply_by_element;
    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(-1) + tensor_type(1) + tensor_type(2) + tensor_type(3), tensor_type(5)),
        std::make_tuple(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1} + tensor_type{0,1,2}, tensor_type{{1,3,5},{4,6,8}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}), tensor_type{{5,6},{2,3}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose(), tensor_type{{5,2},{6,3}}),
        std::make_tuple((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1})({{{},{},-1},{1}}).transpose(), tensor_type{{5,2},{6,3}}),
        std::make_tuple(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1}, tensor_type{{5,2},{6,3}}),
        std::make_tuple(((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1})).reshape(4),tensor_type{5,2,6,3})
    );
    auto test_eval = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = ten.eval(policy...);
            using ten_order = typename decltype(ten)::order;
            using result_order = typename decltype(result)::order;
            REQUIRE(std::is_same_v<result_order,ten_order>);
            REQUIRE(result == expected);
            REQUIRE(!ten.is_same(result));
        };
        apply_by_element(test, test_data);
    };
    SECTION("default_policy")
    {
        test_eval();
    }
    SECTION("exec_pol<4>")
    {
        test_eval(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_eval(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_tensor_eval_of_tensor","[test_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,order>;
    using helpers_for_testing::apply_by_element;
    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(2),tensor_type(2)),
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,3,2,1},tensor_type{1,2,3,4,3,2,1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}})
    );
    auto test_eval = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = ten.eval(policy...);
            using ten_order = typename decltype(ten)::order;
            using result_order = typename decltype(result)::order;
            REQUIRE(std::is_same_v<result_order,ten_order>);
            REQUIRE(result == expected);
            REQUIRE(ten.is_same(result));
        };
        apply_by_element(test, test_data);
    };
    SECTION("default_policy")
    {
        test_eval();
    }
    SECTION("exec_pol<4>")
    {
        test_eval(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_eval(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_tensor_resize","[test_tensor]")
{
    using value_type = int;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    SECTION("test_tensor_resize_to_not_bigger")
    {
        //0tensor,1new_shape,2expected
        auto test_data = std::make_tuple(
            //c_order
            std::make_tuple(tensor<value_type,c_order>{},std::vector<int>{0},tensor_type{}),
            std::make_tuple(tensor<value_type,c_order>{},std::array<int,3>{0,2,3},tensor_type{}.reshape(0,2,3)),
            std::make_tuple(tensor<value_type,c_order>(1),std::vector<int>{0},tensor_type{}),
            std::make_tuple(tensor<value_type,c_order>(2),std::vector<int>{},tensor_type(2)),
            std::make_tuple(tensor<value_type,c_order>(1),std::vector<int>{1,1},tensor_type{{1}}),
            std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{0},tensor_type{}),
            std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{},tensor_type(1)),
            std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{3},tensor_type{1,2,3}),
            std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,3},tensor_type{{1,2,3},{4,5,6}}),
            std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,2,2},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
            //f_order
            std::make_tuple(tensor<value_type,f_order>{},std::vector<int>{0},tensor_type{}),
            std::make_tuple(tensor<value_type,f_order>{},std::array<int,3>{0,2,3},tensor_type{}.reshape(0,2,3)),
            std::make_tuple(tensor<value_type,f_order>(1),std::vector<int>{0},tensor_type{}),
            std::make_tuple(tensor<value_type,f_order>(2),std::vector<int>{},tensor_type(2)),
            std::make_tuple(tensor<value_type,f_order>(1),std::vector<int>{1,1},tensor_type{{1}}),
            std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{0},tensor_type{}),
            std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{},tensor_type(1)),
            std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{3},tensor_type{1,5,3}),
            std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,3},tensor_type{{1,3,2},{5,7,6}}),
            std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,2,2},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}})
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto new_shape = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto ten_size = ten.size();
            auto expected_size = expected.size();
            REQUIRE(ten_size >= expected_size);
            ten.resize(new_shape);
            REQUIRE(ten == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_tensor_resize_to_bigger")
    {
        const value_type any{std::numeric_limits<value_type>::max()};
        //0tensor,1size,2new_shape,3expected
        auto test_data = std::make_tuple(
            //c_order
            std::make_tuple(tensor<value_type,c_order>{},std::vector<int>{1},tensor_type{any}),
            std::make_tuple(tensor<value_type,c_order>{},std::vector<int>{5},tensor_type{any,any,any,any,any}),
            std::make_tuple(tensor<value_type,c_order>{},std::vector<int>{2,3},tensor_type{{any,any,any},{any,any,any}}),
            std::make_tuple(tensor<value_type,c_order>(3),std::vector<int>{5},tensor_type{3,any,any,any,any}),
            std::make_tuple(tensor<value_type,c_order>(3),std::vector<int>{3,2},tensor_type{{3,any},{any,any},{any,any}}),
            std::make_tuple(tensor<value_type,c_order>{4},std::vector<int>{5},tensor_type{4,any,any,any,any}),
            std::make_tuple(tensor<value_type,c_order>{{1,2,3},{4,5,6}},std::vector<int>{2,2,2},tensor_type{{{1,2},{3,4}},{{5,6},{any,any}}}),
            //f_order
            std::make_tuple(tensor<value_type,f_order>{},std::vector<int>{1},tensor_type{any}),
            std::make_tuple(tensor<value_type,f_order>{},std::vector<int>{5},tensor_type{any,any,any,any,any}),
            std::make_tuple(tensor<value_type,f_order>{},std::vector<int>{2,3},tensor_type{{any,any,any},{any,any,any}}),
            std::make_tuple(tensor<value_type,f_order>(3),std::vector<int>{5},tensor_type{3,any,any,any,any}),
            std::make_tuple(tensor<value_type,f_order>(3),std::vector<int>{3,2},tensor_type{{3,any},{any,any},{any,any}}),
            std::make_tuple(tensor<value_type,f_order>{4},std::vector<int>{5},tensor_type{4,any,any,any,any}),
            std::make_tuple(tensor<value_type,f_order>{{1,2,3},{4,5,6}},std::vector<int>{2,2,2},tensor_type{{{1,3},{2,any}},{{4,6},{5,any}}})
        );
        auto test = [&any](const auto& t){
            auto ten = std::get<0>(t);
            auto new_shape = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto expected_shape = expected.shape();
            ten.resize(new_shape);
            auto result_shape = ten.shape();
            REQUIRE(result_shape == expected_shape);
            auto comparator = [&any](auto result_element, auto expected_element){
                return expected_element == any ? true : result_element == expected_element;
            };
            REQUIRE(std::equal(ten.begin(),ten.end(),expected.begin(),expected.end(),comparator));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_tensor_resize_init_list_interface")
    {
        tensor<value_type,c_order> t0{{1,2,3},{4,5,6}};
        t0.resize({2,2});
        REQUIRE(t0 == tensor_type{{1,2},{3,4}});
    }

}

