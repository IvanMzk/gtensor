/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "tensor_math.hpp"
#include "tensor.hpp"

//cumsum, nancumsum
TEMPLATE_TEST_CASE("test_math_cumsum_nancumsum","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumsum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(cumsum(std::declval<tensor_type>(),std::declval<int>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nancumsum(std::declval<tensor_type>(),std::declval<int>()))::value_type,value_type>);

    //0tensor,1axes,2expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{5},0,tensor_type{5}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,tensor_type{{{1,2,3},{4,5,6}},{{8,10,12},{14,16,18}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,tensor_type{{{1,2,3},{5,7,9}},{{7,8,9},{17,19,21}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,tensor_type{{{1,3,6},{4,9,15}},{{7,15,24},{10,21,33}}})
    );
    auto test_cumsum = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = cumsum(policy...,ten,axes);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nancumsum = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = nancumsum(policy...,ten,axes);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    //default policy
    SECTION("test_cumsum_default_policy")
    {
        test_cumsum();
    }
    SECTION("test_nancumsum_default_policy")
    {
        test_nancumsum();
    }
    //exec_pol<4>
    SECTION("test_cumsum_exec_pol<4>")
    {
        test_cumsum(multithreading::exec_pol<4>{});
    }
    SECTION("test_nancumsum_exec_pol<4>")
    {
        test_nancumsum(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_cumsum_exec_pol<0>")
    {
        test_cumsum(multithreading::exec_pol<0>{});
    }
    SECTION("test_nancumsum_exec_pol<0>")
    {
        test_nancumsum(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_math_cumsum_nancumsum_flatten","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumsum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(cumsum(std::declval<tensor_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nancumsum(std::declval<tensor_type>()))::value_type,value_type>);

    //0tensor,1expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{5},tensor_type{5}),
        std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor_type{1,3,6,10,15,21,28,36,45,55,66,78})
    );
    SECTION("test_cumsum")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = cumsum(ten);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_nancumsum")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = nancumsum(ten);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_math_cumsum_nancumsum_nan_values","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumsum;
    using gtensor::nancumsum;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //cumsum
        std::make_tuple(cumsum(tensor_type{1.0,0.5,2.0,pos_inf,3.0,4.0}), tensor_type{1.0,1.5,3.5,pos_inf,pos_inf,pos_inf}),
        std::make_tuple(cumsum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type{1.0,1.5,3.5,neg_inf,neg_inf,neg_inf}),
        std::make_tuple(cumsum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,1.5,3.5,neg_inf,neg_inf,nan}),
        std::make_tuple(cumsum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,nan,nan,nan,nan,nan}),
        std::make_tuple(cumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan}),
        std::make_tuple(cumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
        std::make_tuple(cumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
        std::make_tuple(cumsum(tensor_type{{0.5,1.0,0.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.5,1.5,1.5,0.0,3.0,nan,nan,nan,nan}),
        std::make_tuple(cumsum(tensor_type{{nan,1.0,0.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan}),
        std::make_tuple(
            cumsum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
            tensor_type{{nan,nan,nan,1.0},{nan,nan,nan,3.0},{nan,nan,nan,6.0},{nan,nan,nan,10.0}}
        ),
        std::make_tuple(
            cumsum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
            tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{0.5,2.5,nan,nan},{0.5,3.5,nan,nan}}
        ),
        //nancumsum
        std::make_tuple(nancumsum(tensor_type{1.0,0.5,2.0,pos_inf,3.0,4.0}), tensor_type{1.0,1.5,3.5,pos_inf,pos_inf,pos_inf}),
        std::make_tuple(nancumsum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type{1.0,1.5,3.5,neg_inf,neg_inf,neg_inf}),
        std::make_tuple(nancumsum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,1.5,3.5,neg_inf,neg_inf,nan}),
        std::make_tuple(nancumsum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,1.0,3.0,neg_inf,neg_inf,nan}),
        std::make_tuple(nancumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}),
        std::make_tuple(nancumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}),
        std::make_tuple(nancumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}),
        std::make_tuple(nancumsum(tensor_type{{0.5,1.0,0.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.5,1.5,1.5,0.0,3.0,3.0,3.5,5.5,5.5}),
        std::make_tuple(nancumsum(tensor_type{{nan,1.0,0.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.0,1.0,1.0,-0.5,2.5,2.5,3.0,5.0,5.0}),
        std::make_tuple(
            nancumsum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
            tensor_type{{0.0,0.0,0.0,1.0},{0.0,1.5,0.0,3.0},{0.5,3.5,0.0,6.0},{1.0,6.5,0.0,10.0}}
        ),
        std::make_tuple(
            nancumsum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
            tensor_type{{0.0,0.0,0.0,1.0},{0.0,1.5,1.5,3.50},{0.5,2.5,2.5,5.5},{0.5,3.5,3.5,7.5}}
        )
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_math_cumsum_nancumsum_nan_values_policy","test_math",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumsum;
    using gtensor::nancumsum;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    //0result,1expected
    auto test_data = std::make_tuple(
        //cumsum
        std::make_tuple(cumsum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
        std::make_tuple(cumsum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
        std::make_tuple(
            cumsum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
            tensor_type{{nan,nan,nan,1.0},{nan,nan,nan,3.0},{nan,nan,nan,6.0},{nan,nan,nan,10.0}}
        ),
        std::make_tuple(
            cumsum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
            tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{0.5,2.5,nan,nan},{0.5,3.5,nan,nan}}
        ),
        //nancumsum
        std::make_tuple(nancumsum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}),
        std::make_tuple(nancumsum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}),
        std::make_tuple(
            nancumsum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
            tensor_type{{0.0,0.0,0.0,1.0},{0.0,1.5,0.0,3.0},{0.5,3.5,0.0,6.0},{1.0,6.5,0.0,10.0}}
        ),
        std::make_tuple(
            nancumsum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
            tensor_type{{0.0,0.0,0.0,1.0},{0.0,1.5,1.5,3.50},{0.5,2.5,2.5,5.5},{0.5,3.5,3.5,7.5}}
        )
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

//cumprod,nancumprod
TEMPLATE_TEST_CASE("test_math_cumprod_nancumprod","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumprod;
    using gtensor::nancumprod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(cumprod(std::declval<tensor_type>(),std::declval<int>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nancumprod(std::declval<tensor_type>(),std::declval<int>()))::value_type,value_type>);

    //0tensor,1axes,2expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{5},0,tensor_type{5}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,2,6,24,120}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,tensor_type{{{1,2,3},{4,5,6}},{{7,16,27},{40,55,72}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,tensor_type{{{1,2,3},{4,10,18}},{{7,8,9},{70,88,108}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,tensor_type{{{1,2,6},{4,20,120}},{{7,56,504},{10,110,1320}}})
    );
    auto test_cumprod = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = cumprod(policy...,ten,axes);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nancumprod = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = nancumprod(policy...,ten,axes);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    //default policy
    SECTION("test_cumprod_default_policy")
    {
        test_cumprod();
    }
    SECTION("test_nancumprod_default_policy")
    {
        test_nancumprod();
    }
    //exec_pol<4>
    SECTION("test_cumprod_exec_pol<4>")
    {
        test_cumprod(multithreading::exec_pol<4>{});
    }
    SECTION("test_nancumprod_exec_pol<4>")
    {
        test_nancumprod(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_cumprod_exec_pol<0>")
    {
        test_cumprod(multithreading::exec_pol<0>{});
    }
    SECTION("test_nancumprod_exec_pol<0>")
    {
        test_nancumprod(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_math_cumprod_nancumprod_flatten","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumprod;
    using gtensor::nancumprod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(cumprod(std::declval<tensor_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nancumprod(std::declval<tensor_type>()))::value_type,value_type>);

    //0tensor,1expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{5},tensor_type{5}),
        std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{1,2,6,24,120}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{1,2,3},{0,4,5}}},tensor_type{1,2,6,24,120,720,720,1440,4320,0,0,0})
    );
    SECTION("test_cumprod")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = cumprod(ten);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_nancumprod")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = nancumprod(ten);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_math_cumprod_nancumprod_nan_values","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumprod;
    using gtensor::nancumprod;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //cumprod
        std::make_tuple(cumprod(tensor_type{1.0,0.5,2.0,pos_inf,4.0,3.0}), tensor_type{1.0,0.5,1.0,pos_inf,pos_inf,pos_inf}),
        std::make_tuple(cumprod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type{1.0,0.5,1.0,neg_inf,neg_inf,neg_inf}),
        std::make_tuple(cumprod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,0.5,1.0,neg_inf,neg_inf,neg_inf}),
        std::make_tuple(cumprod(tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,0.0,0.0,nan,nan,nan}),
        std::make_tuple(cumprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,nan,nan,nan,nan,nan}),
        std::make_tuple(cumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan}),
        std::make_tuple(cumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
        std::make_tuple(cumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
        std::make_tuple(cumprod(tensor_type{{0.5,1.0,2.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.5,0.5,1.0,-1.5,-4.5,nan,nan,nan,nan}),
        std::make_tuple(cumprod(tensor_type{{nan,1.0,2.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan}),
        std::make_tuple(
            cumprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
            tensor_type{{nan,nan,nan,1.0},{nan,nan,nan,2.0},{nan,nan,nan,6.0},{nan,nan,nan,24.0}}
        ),
        std::make_tuple(
            cumprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
            tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{0.5,1.0,nan,nan},{0.5,1.5,nan,nan}}
        ),
        //nancumprod
        std::make_tuple(nancumprod(tensor_type{1.0,nan,2.0,pos_inf,4.0,3.0}), tensor_type{1.0,1.0,2.0,pos_inf,pos_inf,pos_inf}),
        std::make_tuple(nancumprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type{1.0,1.0,2.0,neg_inf,neg_inf,neg_inf}),
        std::make_tuple(nancumprod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,0.5,1.0,neg_inf,neg_inf,neg_inf}),
        std::make_tuple(nancumprod(tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,0.0,0.0,nan,nan,nan}),
        std::make_tuple(nancumprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,1.0,2.0,neg_inf,neg_inf,neg_inf}),
        std::make_tuple(nancumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}),
        std::make_tuple(nancumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}}),
        std::make_tuple(nancumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}}),
        std::make_tuple(nancumprod(tensor_type{{0.5,1.0,2.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.5,0.5,1.0,-1.5,-4.5,-4.5,-2.25,-4.5,-4.5}),
        std::make_tuple(nancumprod(tensor_type{{nan,1.0,2.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{1.0,1.0,2.0,-3.0,-9.0,-9.0,-4.5,-9.0,-9.0}),
        std::make_tuple(
            nancumprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
            tensor_type{{1.0,1.0,1.0,1.0},{1.0,1.5,1.0,2.0},{0.5,3.0,1.0,6.0},{0.25,9.0,1.0,24.0}}
        ),
        std::make_tuple(
            nancumprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
            tensor_type{{1.0,1.0,1.0,1.0},{1.0,1.5,1.5,3.0},{0.5,1.0,1.0,3.0},{0.5,1.5,1.5,6.0}}
        )
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_math_cumprod_nancumprod_nan_values_policy","test_math",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumprod;
    using gtensor::nancumprod;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    //0result,1expected
    auto test_data = std::make_tuple(
        //cumprod
        std::make_tuple(cumprod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
        std::make_tuple(cumprod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
        std::make_tuple(
            cumprod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
            tensor_type{{nan,nan,nan,1.0},{nan,nan,nan,2.0},{nan,nan,nan,6.0},{nan,nan,nan,24.0}}
        ),
        std::make_tuple(
            cumprod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
            tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{0.5,1.0,nan,nan},{0.5,1.5,nan,nan}}
        ),
        //nancumprod
        std::make_tuple(nancumprod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}}),
        std::make_tuple(nancumprod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}}),
        std::make_tuple(
            nancumprod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
            tensor_type{{1.0,1.0,1.0,1.0},{1.0,1.5,1.0,2.0},{0.5,3.0,1.0,6.0},{0.25,9.0,1.0,24.0}}
        ),
        std::make_tuple(
            nancumprod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
            tensor_type{{1.0,1.0,1.0,1.0},{1.0,1.5,1.5,3.0},{0.5,1.0,1.0,3.0},{0.5,1.5,1.5,6.0}}
        )
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

