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
#include "statistic.hpp"
#include "tensor.hpp"

//var,nanvar
TEMPLATE_TEST_CASE("test_statistic_var_nanvar_normal_values","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::var;
    using gtensor::nanvar;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
    using result_tensor_type = gtensor::tensor<result_value_type>;
    REQUIRE(std::is_same_v<typename decltype(var(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(var(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanvar(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanvar(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,result_tensor_type{nan,nan}),
        std::make_tuple(tensor_type{5},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,result_tensor_type(2)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,result_tensor_type(2)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,result_tensor_type{{9,9,9},{9,9,9}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,result_tensor_type{{2.25,2.25,2.25},{2.25,2.25,2.25}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,result_tensor_type{{0.666,0.666},{0.666,0.666}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,result_tensor_type{11.25,11.25,11.25}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,result_tensor_type{2.916,2.916}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,result_tensor_type{0,0,0,0,0}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,result_tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}}),
        // //keep_dim true
        std::make_tuple(tensor_type{},0,true,result_tensor_type{nan}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,result_tensor_type{nan}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,result_tensor_type{{{nan},{nan}}}),
        std::make_tuple(tensor_type{5},0,true,result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,true,result_tensor_type{2}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,result_tensor_type{2}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,result_tensor_type{{{9,9,9},{9,9,9}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,result_tensor_type{{{2.25,2.25,2.25}},{{2.25,2.25,2.25}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,result_tensor_type{{{0.666},{0.666}},{{0.666},{0.666}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,result_tensor_type{{{11.25,11.25,11.25}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,result_tensor_type{{{2.916}},{{2.916}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,result_tensor_type{0,0,0,0,0}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,result_tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}})
    );
    auto test_var = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = var(policy...,ten,axes,keep_dims);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
        };
        apply_by_element(test,test_data);
    };
    auto test_nanvar = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = nanvar(policy...,ten,axes,keep_dims);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_var_default_policy")
    {
        test_var();
    }
    SECTION("test_nanvar_default_policy")
    {
        test_nanvar();
    }
    SECTION("test_var_exec_pol<4>")
    {
        test_var(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanvar_exec_pol<4>")
    {
        test_nanvar(multithreading::exec_pol<4>{});
    }
    SECTION("test_var_exec_pol<0>")
    {
        test_var(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanvar_exec_pol<0>")
    {
        test_nanvar(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_var_nanvar_overload_default_policy","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::var;
    using gtensor::nanvar;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(var(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanvar(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    //var
    REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{9,9,9},{9,9,9}}, 1E-2, 1E-2));
    REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{11.25,11.25,11.25}, 1E-2, 1E-2));
    REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{2.916}},{{2.916}}}, 1E-2, 1E-2));
    //all axes
    REQUIRE(tensor_close(var(tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
    REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(11.916), 1E-2, 1E-2));
    REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(11.916), 1E-2, 1E-2));
    REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{11.916}}}, 1E-2, 1E-2));

    //nanvar
    REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{9,9,9},{9,9,9}}, 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{11.25,11.25,11.25}, 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{2.916}},{{2.916}}}, 1E-2, 1E-2));
    //all axes
    REQUIRE(tensor_close(nanvar(tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(11.916), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(11.916), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{11.916}}}, 1E-2, 1E-2));
}

TEMPLATE_TEST_CASE("test_statistic_var_nanvar_overload_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::var;
    using gtensor::nanvar;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(var(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanvar(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    //var
    REQUIRE(tensor_close(var(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{9,9,9},{9,9,9}}, 1E-2, 1E-2));
    REQUIRE(tensor_close(var(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{11.25,11.25,11.25}, 1E-2, 1E-2));
    REQUIRE(tensor_close(var(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{2.916}},{{2.916}}}, 1E-2, 1E-2));
    //all axes
    REQUIRE(tensor_close(var(policy{},tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
    REQUIRE(tensor_close(var(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(11.916), 1E-2, 1E-2));
    REQUIRE(tensor_close(var(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(11.916), 1E-2, 1E-2));
    REQUIRE(tensor_close(var(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{11.916}}}, 1E-2, 1E-2));

    //nanvar
    REQUIRE(tensor_close(nanvar(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{9,9,9},{9,9,9}}, 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{11.25,11.25,11.25}, 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{2.916}},{{2.916}}}, 1E-2, 1E-2));
    //all axes
    REQUIRE(tensor_close(nanvar(policy{},tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(11.916), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(11.916), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanvar(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{11.916}}}, 1E-2, 1E-2));
}

TEST_CASE("test_statistic_var_nanvar_nan_values_default_policy","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::var;
    using gtensor::nanvar;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //var
        std::make_tuple(var(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
        std::make_tuple(var(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(var(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
        std::make_tuple(var(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(var(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(var(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(var(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(var(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(var(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(var(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,1.25}),
        std::make_tuple(var(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(var(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{0.0,nan,0.0,nan,0.0,nan}),
        //nanvar
        std::make_tuple(nanvar(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(1.64)),
        std::make_tuple(nanvar(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanvar(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
        std::make_tuple(nanvar(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanvar(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanvar(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanvar(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(nanvar(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(nanvar(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(1.111)),
        std::make_tuple(nanvar(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.0,0.166,nan,0.875}),
        std::make_tuple(nanvar(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{0.0,0.0625,3.5,0.5}),
        std::make_tuple(nanvar(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{0.0,nan,0.0,nan,0.0,nan})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_statistic_var_nanvar_nan_values_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::var;
    using gtensor::nanvar;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //var
        std::make_tuple(var(policy{},tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
        std::make_tuple(var(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(var(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
        std::make_tuple(var(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(var(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(var(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(var(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(var(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(var(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(var(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,1.25}),
        std::make_tuple(var(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(var(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{0.0,nan,0.0,nan,0.0,nan}),
        //nanvar
        std::make_tuple(nanvar(policy{},tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(1.64)),
        std::make_tuple(nanvar(policy{},tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanvar(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
        std::make_tuple(nanvar(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanvar(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanvar(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanvar(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(nanvar(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(nanvar(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(1.111)),
        std::make_tuple(nanvar(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.0,0.166,nan,0.875}),
        std::make_tuple(nanvar(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{0.0,0.0625,3.5,0.5}),
        std::make_tuple(nanvar(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{0.0,nan,0.0,nan,0.0,nan})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
    };
    apply_by_element(test,test_data);
}

//stdev,nanstdev
TEMPLATE_TEST_CASE("test_statistic_stdev_nanstdev_normal_values","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::nanstdev;
    using gtensor::stdev;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
    using result_tensor_type = gtensor::tensor<result_value_type>;
    REQUIRE(std::is_same_v<typename decltype(gtensor::stdev(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(gtensor::stdev(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanstdev(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanstdev(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,result_tensor_type{nan,nan}),
        std::make_tuple(tensor_type{5},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,result_tensor_type(1.414)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,result_tensor_type(1.414)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,result_tensor_type{{3,3,3},{3,3,3}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,result_tensor_type{{1.5,1.5,1.5},{1.5,1.5,1.5}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,result_tensor_type{{0.816,0.816},{0.816,0.816}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,result_tensor_type{3.354,3.354,3.354}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,result_tensor_type{1.707,1.707}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,result_tensor_type{0,0,0,0,0}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,result_tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}}),
        // //keep_dim true
        std::make_tuple(tensor_type{},0,true,result_tensor_type{nan}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,result_tensor_type{nan}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,result_tensor_type{{{nan},{nan}}}),
        std::make_tuple(tensor_type{5},0,true,result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,true,result_tensor_type{1.414}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,result_tensor_type{1.414}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,result_tensor_type{{{3,3,3},{3,3,3}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,result_tensor_type{{{1.5,1.5,1.5}},{{1.5,1.5,1.5}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,result_tensor_type{{{0.816},{0.816}},{{0.816},{0.816}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,result_tensor_type{{{3.354,3.354,3.354}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,result_tensor_type{{{1.707}},{{1.707}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,result_tensor_type{0,0,0,0,0}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,result_tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}})
    );
    auto test_stdev = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = stdev(policy...,ten,axes,keep_dims);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
        };
        apply_by_element(test,test_data);
    };
    auto test_nanstdev = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = nanstdev(policy...,ten,axes,keep_dims);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_stdev_default_policy")
    {
        test_stdev();
    }
    SECTION("test_nanstdev_default_policy")
    {
        test_nanstdev();
    }
    SECTION("test_stdev_exec_pol<4>")
    {
        test_stdev(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanstdev_exec_pol<4>")
    {
        test_nanstdev(multithreading::exec_pol<4>{});
    }
    SECTION("test_stdev_exec_pol<0>")
    {
        test_stdev(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanstdev_exec_pol<0>")
    {
        test_nanstdev(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_stdev_nanstdev_overload_default_policy","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::stdev;
    using gtensor::nanstdev;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(stdev(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanstdev(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    //std
    REQUIRE(tensor_close(stdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{3,3,3},{3,3,3}}, 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{3.354,3.354,3.354}, 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{1.707}},{{1.707}}}, 1E-2, 1E-2));
    //all axes
    REQUIRE(tensor_close(stdev(tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(3.452), 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(3.452), 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{3.452}}}, 1E-2, 1E-2));

    //nanstdev
    REQUIRE(tensor_close(nanstdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{3,3,3},{3,3,3}}, 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{3.354,3.354,3.354}, 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{1.707}},{{1.707}}}, 1E-2, 1E-2));
    //all axes
    REQUIRE(tensor_close(nanstdev(tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(3.452), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(3.452), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{3.452}}}, 1E-2, 1E-2));
}

TEMPLATE_TEST_CASE("test_statistic_stdev_nanstdev_overload_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::stdev;
    using gtensor::nanstdev;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(stdev(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanstdev(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    //std
    REQUIRE(tensor_close(stdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{3,3,3},{3,3,3}}, 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{3.354,3.354,3.354}, 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{1.707}},{{1.707}}}, 1E-2, 1E-2));
    //all axes
    REQUIRE(tensor_close(stdev(policy{},tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(3.452), 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(3.452), 1E-2, 1E-2));
    REQUIRE(tensor_close(stdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{3.452}}}, 1E-2, 1E-2));

    //nanstdev
    REQUIRE(tensor_close(nanstdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{3,3,3},{3,3,3}}, 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{3.354,3.354,3.354}, 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{1.707}},{{1.707}}}, 1E-2, 1E-2));
    //all axes
    REQUIRE(tensor_close(nanstdev(policy{},tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(3.452), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(3.452), 1E-2, 1E-2));
    REQUIRE(tensor_close(nanstdev(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{3.452}}}, 1E-2, 1E-2));
}

TEST_CASE("test_statistic_std_nanstd_nan_values_default_policy","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::stdev;
    using gtensor::nanstdev;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //std
        std::make_tuple(stdev(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
        std::make_tuple(stdev(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(stdev(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
        std::make_tuple(stdev(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(stdev(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(stdev(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(stdev(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(stdev(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(stdev(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(stdev(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,1.118}),
        std::make_tuple(stdev(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(stdev(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{0.0,nan,0.0,nan,0.0,nan}),
        //nanstdev
        std::make_tuple(nanstdev(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(1.28)),
        std::make_tuple(nanstdev(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanstdev(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
        std::make_tuple(nanstdev(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanstdev(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanstdev(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanstdev(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(nanstdev(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(nanstdev(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(1.054)),
        std::make_tuple(nanstdev(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.0,0.408,nan,0.935}),
        std::make_tuple(nanstdev(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{0.0,0.25,1.87,0.707}),
        std::make_tuple(nanstdev(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{0.0,nan,0.0,nan,0.0,nan})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_statistic_std_nanstd_nan_values_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::stdev;
    using gtensor::nanstdev;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //std
        std::make_tuple(stdev(policy{},tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
        std::make_tuple(stdev(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(stdev(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
        std::make_tuple(stdev(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(stdev(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(stdev(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(stdev(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(stdev(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(stdev(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(stdev(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,1.118}),
        std::make_tuple(stdev(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(stdev(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{0.0,nan,0.0,nan,0.0,nan}),
        //nanstdev
        std::make_tuple(nanstdev(policy{},tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(1.28)),
        std::make_tuple(nanstdev(policy{},tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanstdev(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
        std::make_tuple(nanstdev(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanstdev(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanstdev(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanstdev(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(nanstdev(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(nanstdev(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(1.054)),
        std::make_tuple(nanstdev(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.0,0.408,nan,0.935}),
        std::make_tuple(nanstdev(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{0.0,0.25,1.87,0.707}),
        std::make_tuple(nanstdev(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{0.0,nan,0.0,nan,0.0,nan})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
    };
    apply_by_element(test,test_data);
}

