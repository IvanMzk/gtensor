/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "config_for_testing.hpp"
#include "sort_search.hpp"
#include "tensor.hpp"

//sort
TEST_CASE("test_sort_search_sort","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using gtensor::sort;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2comparator,3expected
    auto test_data = std::make_tuple(
        //no comparator
        std::make_tuple(tensor_type{},0,no_value{},tensor_type{}),
        std::make_tuple(tensor_type{1},0,no_value{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,no_value{},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,no_value{},tensor_type{0,1,1,2,2,3,5,6}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,no_value{},tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,no_value{},tensor_type{{-1,1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}}),
        //comparator
        std::make_tuple(tensor_type{},0,std::less<void>{},tensor_type{}),
        std::make_tuple(tensor_type{1},0,std::less<void>{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,std::less<void>{},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,std::greater<void>{},tensor_type{6,5,3,2,2,1,1,0}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,std::less<void>{},tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,std::greater<void>{},tensor_type{{6,3,2,1,-1},{8,5,2,1,0},{7,4,2,0,-1},{4,4,4,2,1},{6,4,3,3,1}})
    );
    auto test_sort = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto comparator = std::get<2>(t);
            auto expected = std::get<3>(t);

            auto result = sort(policy...,ten,axis,comparator);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_sort();
    }
    SECTION("exec_pol<4>")
    {
        test_sort(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_sort(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_sort_search_sort_overload_default_policy","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sort;

    //default comparator = std::less<void>
    REQUIRE(sort(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0) == tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}});
    //default comparator, default axis = -1
    REQUIRE(sort(tensor_type{2,1,6,3,2,1,0,5}) == tensor_type{0,1,1,2,2,3,5,6});
    REQUIRE(sort(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == tensor_type{{-1,1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}});
}

TEMPLATE_TEST_CASE("test_sort_search_sort_overload_policy","[test_sort_search]",
    multithreading::exec_pol<4>,
    multithreading::exec_pol<0>
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sort;

    //default comparator = std::less<void>
    REQUIRE(sort(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0) == tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}});
    //default comparator, default axis = -1
    REQUIRE(sort(policy{},tensor_type{2,1,6,3,2,1,0,5}) == tensor_type{0,1,1,2,2,3,5,6});
    REQUIRE(sort(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == tensor_type{{-1,1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}});
}

//argsort
TEST_CASE("test_sort_search_argsort","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::argsort;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argsort(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<no_value>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argsort(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<std::less<void>>())),result_tensor_type>);

    //0tensor,1axis,2comparator,3expected
    auto test_data = std::make_tuple(
        //no comparator
        std::make_tuple(tensor_type{},0,no_value{},result_tensor_type{}),
        std::make_tuple(tensor_type{1},0,no_value{},result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,no_value{},result_tensor_type{0,1,2,3,4,5}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,no_value{},result_tensor_type{6,1,5,0,4,3,7,2}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,no_value{},result_tensor_type{{2,0,0,1,2},{0,4,2,3,0},{4,1,1,2,4},{3,3,3,4,3},{1,2,4,0,1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,no_value{},result_tensor_type{{2,1,0,4,3},{3,2,1,4,0},{0,2,4,3,1},{3,2,0,1,4},{1,0,4,3,2}}),
        // //comparator
        std::make_tuple(tensor_type{},0,std::less<void>{},result_tensor_type{}),
        std::make_tuple(tensor_type{1},0,std::less<void>{},result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,std::less<void>{},result_tensor_type{0,1,2,3,4,5}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,std::greater<void>{},result_tensor_type{2,7,3,0,4,1,5,6}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,std::less<void>{},result_tensor_type{{2,0,0,1,2},{0,4,2,3,0},{4,1,1,2,4},{3,3,3,4,3},{1,2,4,0,1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,std::greater<void>{},result_tensor_type{{3,4,0,1,2},{0,4,1,2,3},{1,3,4,2,0},{0,1,4,2,3},{2,3,0,4,1}})
    );
    auto test_argsort = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto comparator = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = argsort(policy...,ten,axis,comparator);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_argsort();
    }
    SECTION("exec_pol<4>")
    {
        test_argsort(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_argsort(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_sort_search_argsort_overload_default_policy","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argsort;

    //default comparator
    REQUIRE(argsort(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0) == result_tensor_type{{2,0,0,1,2},{0,4,2,3,0},{4,1,1,2,4},{3,3,3,4,3},{1,2,4,0,1}});
    //default comparator, default axis
    REQUIRE(argsort(tensor_type{2,1,6,3,2,1,0,5}) == result_tensor_type{6,1,5,0,4,3,7,2});
    REQUIRE(argsort(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type{{2,1,0,4,3},{3,2,1,4,0},{0,2,4,3,1},{3,2,0,1,4},{1,0,4,3,2}});
}

TEMPLATE_TEST_CASE("test_sort_search_argsort_overload_policy","[test_sort_search]",
    multithreading::exec_pol<4>,
    multithreading::exec_pol<0>
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argsort;

    //default comparator
    REQUIRE(argsort(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0) == result_tensor_type{{2,0,0,1,2},{0,4,2,3,0},{4,1,1,2,4},{3,3,3,4,3},{1,2,4,0,1}});
    //default comparator, default axis
    REQUIRE(argsort(policy{},tensor_type{2,1,6,3,2,1,0,5}) == result_tensor_type{6,1,5,0,4,3,7,2});
    REQUIRE(argsort(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type{{2,1,0,4,3},{3,2,1,4,0},{0,2,4,3,1},{3,2,0,1,4},{1,0,4,3,2}});
}

