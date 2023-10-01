/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <limits>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"
#include "sort_search.hpp"
#include "tensor.hpp"

//argmin,nanargmin
TEST_CASE("test_sort_search_argmin_nanargmin","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::argmin;
    using gtensor::nanargmin;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argmin(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argmin(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(nanargmin(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(nanargmin(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<bool>())),result_tensor_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,false,result_tensor_type(6)),
        std::make_tuple(tensor_type{2,1,6,3,2,1,3,5},0,false,result_tensor_type(1)),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,false,result_tensor_type{2,0,0,1,2}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,false,result_tensor_type{2,3,0,3,1}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,true,result_tensor_type{{2,0,0,1,2}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,true,result_tensor_type{{2},{3},{0},{3},{1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{0,1},false,result_tensor_type(2))
    );
    auto test_argmin = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = argmin(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nanargmin = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = nanargmin(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };

    SECTION("test_argmin_default_policy")
    {
        test_argmin();
    }
    SECTION("test_nanargmin_default_policy")
    {
        test_nanargmin();
    }
    SECTION("test_argmin_exec_pol<4>")
    {
        test_argmin(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanargmin_exec_pol<4>")
    {
        test_nanargmin(multithreading::exec_pol<4>{});
    }
    SECTION("test_argmin_exec_pol<0>")
    {
        test_argmin(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanargmin_exec_pol<0>")
    {
        test_nanargmin(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_sort_search_argmin_nanargmin_nan_values_default_policy","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmin;
    using gtensor::nanargmin;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //argmin
        std::make_tuple(argmin(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(1)),
        std::make_tuple(argmin(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(argmin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(1)),
        std::make_tuple(argmin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), result_tensor_type(0)),
        std::make_tuple(argmin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), result_tensor_type{0,0,0,1,0,1}),
        std::make_tuple(argmin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), result_tensor_type{0,2}),
        std::make_tuple(argmin(tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), result_tensor_type{1,0,0,0}),
        std::make_tuple(argmin(tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), result_tensor_type{3,1}),
        std::make_tuple(argmin(tensor_type{{nan,nan,nan},{nan,nan,nan}}), result_tensor_type(0)),
        //nanargmin
        std::make_tuple(nanargmin(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(1)),
        std::make_tuple(nanargmin(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(nanargmin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(3)),
        std::make_tuple(nanargmin(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), result_tensor_type(6)),
        std::make_tuple(nanargmin(tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},0), result_tensor_type{2,1,0}),
        std::make_tuple(nanargmin(tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},1), result_tensor_type{2,1,0})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_argmin_nanargmin_nan_values_policy","[test_sort_search]",
    multithreading::exec_pol<4>,
    multithreading::exec_pol<0>
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmin;
    using gtensor::nanargmin;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //argmin
        std::make_tuple(argmin(policy{},tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(1)),
        std::make_tuple(argmin(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(argmin(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(1)),
        std::make_tuple(argmin(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), result_tensor_type(0)),
        std::make_tuple(argmin(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), result_tensor_type{0,0,0,1,0,1}),
        std::make_tuple(argmin(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), result_tensor_type{0,2}),
        std::make_tuple(argmin(policy{},tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), result_tensor_type{1,0,0,0}),
        std::make_tuple(argmin(policy{},tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), result_tensor_type{3,1}),
        std::make_tuple(argmin(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan}}), result_tensor_type(0)),
        //nanargmin
        std::make_tuple(nanargmin(policy{},tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(1)),
        std::make_tuple(nanargmin(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(nanargmin(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(3)),
        std::make_tuple(nanargmin(policy{},tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), result_tensor_type(6)),
        std::make_tuple(nanargmin(policy{},tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},0), result_tensor_type{2,1,0}),
        std::make_tuple(nanargmin(policy{},tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},1), result_tensor_type{2,1,0})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_sort_search_argmin_nanargmin_overload_default_policy","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmin;
    using gtensor::nanargmin;

    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();

    //default axes and keep_dims
    REQUIRE(argmin(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-4,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(10));
    REQUIRE(nanargmin(tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{nan,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}}) == result_tensor_type(2));
    //default axes
    REQUIRE(argmin(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-4,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{10}});
    REQUIRE(nanargmin(tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{nan,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}},true) == result_tensor_type{{2}});
}

TEMPLATE_TEST_CASE("test_sort_search_argmin_nanargmin_overload_policy","[test_sort_search]",
    multithreading::exec_pol<4>,
    multithreading::exec_pol<0>
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmin;
    using gtensor::nanargmin;

    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();

    //default axes and keep_dims
    REQUIRE(argmin(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-4,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(10));
    REQUIRE(nanargmin(policy{},tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{nan,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}}) == result_tensor_type(2));
    //default axes
    REQUIRE(argmin(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-4,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{10}});
    REQUIRE(nanargmin(policy{},tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{nan,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}},true) == result_tensor_type{{2}});
}

TEST_CASE("test_test_sort_search_argmin_nanargmin_exception","[test_test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::argmin;
    using gtensor::nanargmin;
    using gtensor::value_error;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    //empty
    REQUIRE_THROWS_AS(argmin(tensor_type{}), value_error);
    REQUIRE_THROWS_AS(argmin(tensor_type{}.reshape(0,2,3),0), value_error);
    REQUIRE_THROWS_AS(nanargmin(tensor_type{}), value_error);
    REQUIRE_THROWS_AS(nanargmin(tensor_type{}.reshape(0,2,3),0), value_error);
    //all nan
    REQUIRE_THROWS_AS(nanargmin(tensor_type{{nan,nan,nan},{nan,nan,nan}},0), value_error);
    REQUIRE_THROWS_AS(nanargmin(tensor_type{{nan,nan,nan},{nan,nan,nan}},1), value_error);
}

//argmax,nanargmax
TEST_CASE("test_sort_search_argmax_nanargmax","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::argmax;
    using gtensor::nanargmax;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argmax(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argmax(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(nanargmax(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(nanargmax(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<bool>())),result_tensor_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,false,result_tensor_type(5)),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,false,result_tensor_type(2)),
        std::make_tuple(tensor_type{6,1,6,3,2,1,6,5},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,false,result_tensor_type{1,2,4,0,1}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,false,result_tensor_type{3,0,1,0,2}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,true,result_tensor_type{{1,2,4,0,1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,true,result_tensor_type{{3},{0},{1},{0},{2}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{0,1},false,result_tensor_type(5))

    );
    auto test_argmax = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = argmax(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nanargmax = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = nanargmax(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };

    SECTION("test_argmax_default_policy")
    {
        test_argmax();
    }
    SECTION("test_nanargmax_default_policy")
    {
        test_nanargmax();
    }
    SECTION("test_argmax_exec_pol<4>")
    {
        test_argmax(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanargmax_exec_pol<4>")
    {
        test_nanargmax(multithreading::exec_pol<4>{});
    }
    SECTION("test_argmax_exec_pol<0>")
    {
        test_argmax(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanargmax_exec_pol<0>")
    {
        test_nanargmax(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_test_sort_search_argmax_nanargmax_nan_values_default_policy","[test_test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmax;
    using gtensor::nanargmax;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //argmax
        std::make_tuple(argmax(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(argmax(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(4)),
        std::make_tuple(argmax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(1)),
        std::make_tuple(argmax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), result_tensor_type(0)),
        std::make_tuple(argmax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), result_tensor_type{0,0,0,0,1,0}),
        std::make_tuple(argmax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), result_tensor_type{0,2}),
        std::make_tuple(argmax(tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), result_tensor_type{1,1,1,0}),
        std::make_tuple(argmax(tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), result_tensor_type{3,2}),
        std::make_tuple(argmax(tensor_type{{nan,nan,nan},{nan,nan,nan}}), result_tensor_type(0)),
        //nanargmax
        std::make_tuple(nanargmax(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(nanargmax(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(4)),
        std::make_tuple(nanargmax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(5)),
        std::make_tuple(nanargmax(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), result_tensor_type(7)),
        std::make_tuple(nanargmax(tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},0), result_tensor_type{2,2,0}),
        std::make_tuple(nanargmax(tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},1), result_tensor_type{2,1,1})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_test_sort_search_argmax_nanargmax_nan_values_policy","[test_test_sort_search]",
    multithreading::exec_pol<4>,
    multithreading::exec_pol<0>
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmax;
    using gtensor::nanargmax;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //argmax
        std::make_tuple(argmax(policy{},tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(argmax(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(4)),
        std::make_tuple(argmax(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(1)),
        std::make_tuple(argmax(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), result_tensor_type(0)),
        std::make_tuple(argmax(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), result_tensor_type{0,0,0,0,1,0}),
        std::make_tuple(argmax(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), result_tensor_type{0,2}),
        std::make_tuple(argmax(policy{},tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), result_tensor_type{1,1,1,0}),
        std::make_tuple(argmax(policy{},tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), result_tensor_type{3,2}),
        std::make_tuple(argmax(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan}}), result_tensor_type(0)),
        //nanargmax
        std::make_tuple(nanargmax(policy{},tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(nanargmax(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(4)),
        std::make_tuple(nanargmax(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(5)),
        std::make_tuple(nanargmax(policy{},tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), result_tensor_type(7)),
        std::make_tuple(nanargmax(policy{},tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},0), result_tensor_type{2,2,0}),
        std::make_tuple(nanargmax(policy{},tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},1), result_tensor_type{2,1,1})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_sort_search_argmax_nanargmax_overload_default_policy","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmax;
    using gtensor::nanargmax;

    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();

    //default axes and keep_dims
    REQUIRE(argmax(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(5));
    REQUIRE(nanargmax(tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{-1.0,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}}) == result_tensor_type(11));
    //default axes
    REQUIRE(argmax(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{5}});
    REQUIRE(nanargmax(tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{-1.0,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}},true) == result_tensor_type{{11}});
}

TEMPLATE_TEST_CASE("test_sort_search_argmax_nanargmax_overload_policy","[test_sort_search]",
    multithreading::exec_pol<4>,
    multithreading::exec_pol<0>
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmax;
    using gtensor::nanargmax;

    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();

    //default axes and keep_dims
    REQUIRE(argmax(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(5));
    REQUIRE(nanargmax(policy{},tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{-1.0,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}}) == result_tensor_type(11));
    //default axes
    REQUIRE(argmax(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{5}});
    REQUIRE(nanargmax(policy{},tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{-1.0,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}},true) == result_tensor_type{{11}});
}

TEST_CASE("test_test_sort_search_argmax_nanargmax_exception","[test_test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::argmax;
    using gtensor::nanargmax;
    using gtensor::value_error;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    //empty
    REQUIRE_THROWS_AS(argmax(tensor_type{}), value_error);
    REQUIRE_THROWS_AS(argmax(tensor_type{}.reshape(0,2,3),0), value_error);
    REQUIRE_THROWS_AS(nanargmax(tensor_type{}), value_error);
    REQUIRE_THROWS_AS(nanargmax(tensor_type{}.reshape(0,2,3),0), value_error);
    //all nan
    REQUIRE_THROWS_AS(nanargmax(tensor_type{{nan,nan,nan},{nan,nan,nan}},0), value_error);
    REQUIRE_THROWS_AS(nanargmax(tensor_type{{nan,nan,nan},{nan,nan,nan}},1), value_error);
}

