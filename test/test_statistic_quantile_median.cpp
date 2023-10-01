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

//quantile,nanquantile
TEMPLATE_TEST_CASE("test_statistic_quantile_nanquantile_normal_values","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::quantile;
    using gtensor::nanquantile;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
    using result_tensor_type = gtensor::tensor<result_value_type>;
    REQUIRE(std::is_same_v<typename decltype(quantile(std::declval<tensor_type>(),std::declval<int>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(quantile(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanquantile(std::declval<tensor_type>(),std::declval<int>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanquantile(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,result_value_type>);

    //0tensor,1axes,2quantile,3keep_dims,4expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,result_value_type{0.5},false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{},std::vector<int>{0},result_value_type{0.5},false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},result_value_type{0.5},false,result_tensor_type{nan,nan}),
        std::make_tuple(tensor_type{5},0,result_value_type{0.5},false,result_tensor_type(5)),
        std::make_tuple(tensor_type{5},0,result_value_type{0.0},false,result_tensor_type(5)),
        std::make_tuple(tensor_type{5},0,result_value_type{0.2},false,result_tensor_type(5)),
        std::make_tuple(tensor_type{5},0,result_value_type{0.8},false,result_tensor_type(5)),
        std::make_tuple(tensor_type{5},0,result_value_type{1.0},false,result_tensor_type(5)),
        std::make_tuple(tensor_type{5,6},0,result_value_type{0.5},false,result_tensor_type(5.5)),
        std::make_tuple(tensor_type{6,5},0,result_value_type{0.0},false,result_tensor_type(5.0)),
        std::make_tuple(tensor_type{6,5},0,result_value_type{0.2},false,result_tensor_type(5.2)),
        std::make_tuple(tensor_type{5,6},0,result_value_type{0.8},false,result_tensor_type(5.8)),
        std::make_tuple(tensor_type{5,6},0,result_value_type{1.0},false,result_tensor_type(6.0)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{0.5},false,result_tensor_type(2.0)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{0.0},false,result_tensor_type(-1.0)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{0.2},false,result_tensor_type(1.0)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{0.8},false,result_tensor_type(4.0)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{1.0},false,result_tensor_type(6.0)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{0.5},false,result_tensor_type(2.5)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{0.0},false,result_tensor_type(-1.0)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{0.2},false,result_tensor_type(1.0)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{0.8},false,result_tensor_type(4.8)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{1.0},false,result_tensor_type(6.0)),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0,result_value_type{0.3},false,result_tensor_type{1.0,1.2,0.2,4.2,1.4,-0.6}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},1,result_value_type{0.3},false,result_tensor_type{0.5,0.5,1.5,0.5,2.0}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},std::vector<int>{1,0},result_value_type{0.3},false,result_tensor_type(1.0)),
        std::make_tuple(tensor_type{},std::vector<int>{},result_value_type{0.5},false,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::vector<int>{},result_value_type{0.3},false,result_tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},result_value_type{0.3},false,result_tensor_type{{1,2,3},{4,5,6}}),
        // //keep_dim true
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0,result_value_type{0.3},true,result_tensor_type{{1.0,1.2,0.2,4.2,1.4,-0.6}}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},1,result_value_type{0.3},true,result_tensor_type{{0.5},{0.5},{1.5},{0.5},{2.0}}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},std::vector<int>{1,0},result_value_type{0.3},true,result_tensor_type{{1.0}}),
        std::make_tuple(tensor_type{},std::vector<int>{},result_value_type{0.5},true,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::vector<int>{},result_value_type{0.3},true,result_tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},result_value_type{0.3},true,result_tensor_type{{1,2,3},{4,5,6}})

    );
    auto test_quantile = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto q = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = quantile(policy...,ten,axes,q,keep_dims);
            REQUIRE(tensor_close(result,expected,true));
        };
        apply_by_element(test,test_data);
    };
    auto test_nanquantile = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto q = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = nanquantile(policy...,ten,axes,q,keep_dims);
            REQUIRE(tensor_close(result,expected,true));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_quantile_default_policy")
    {
        test_quantile();
    }
    SECTION("test_nanquantile_default_policy")
    {
        test_nanquantile();
    }
    SECTION("test_quantile_exec_pol<4>")
    {
        test_quantile(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanquantile_exec_pol<4>")
    {
        test_nanquantile(multithreading::exec_pol<4>{});
    }
    SECTION("test_quantile_exec_pol<0>")
    {
        test_quantile(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanquantile_exec_pol<0>")
    {
        test_nanquantile(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_quantile_nanquantile_overload_default_policy","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::quantile;
    using gtensor::nanquantile;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(
        std::is_same_v<
            typename decltype(quantile(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,
            result_value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(nanquantile(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,
            result_value_type
        >
    );
    //quantile
    REQUIRE(tensor_close(quantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},0.3,false), result_tensor_type{1.0,1.2,0.2,4.2,1.4,-0.6}));
    REQUIRE(tensor_close(quantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1},0.3), result_tensor_type{0.5,0.5,1.5,0.5,2.0}));
    REQUIRE(tensor_close(quantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0},0.3), result_tensor_type(1.0)));
    REQUIRE(tensor_close(quantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0.3), result_tensor_type(1.0)));
    //nanquantile
    REQUIRE(tensor_close(nanquantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},0.3,false), result_tensor_type{1.0,1.2,0.2,4.2,1.4,-0.6}));
    REQUIRE(tensor_close(nanquantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1},0.3), result_tensor_type{0.5,0.5,1.5,0.5,2.0}));
    REQUIRE(tensor_close(nanquantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0},0.3), result_tensor_type(1.0)));
    REQUIRE(tensor_close(nanquantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0.3), result_tensor_type(1.0)));
}

TEMPLATE_TEST_CASE("test_statistic_quantile_nanquantile_overload_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::quantile;
    using gtensor::nanquantile;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(
        std::is_same_v<
            typename decltype(quantile(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,
            result_value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(nanquantile(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,
            result_value_type
        >
    );
    //quantile
    REQUIRE(tensor_close(quantile(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},0.3,false), result_tensor_type{1.0,1.2,0.2,4.2,1.4,-0.6}));
    REQUIRE(tensor_close(quantile(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1},0.3), result_tensor_type{0.5,0.5,1.5,0.5,2.0}));
    REQUIRE(tensor_close(quantile(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0},0.3), result_tensor_type(1.0)));
    REQUIRE(tensor_close(quantile(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0.3), result_tensor_type(1.0)));
    //nanquantile
    REQUIRE(tensor_close(nanquantile(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},0.3,false), result_tensor_type{1.0,1.2,0.2,4.2,1.4,-0.6}));
    REQUIRE(tensor_close(nanquantile(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1},0.3), result_tensor_type{0.5,0.5,1.5,0.5,2.0}));
    REQUIRE(tensor_close(nanquantile(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0},0.3), result_tensor_type(1.0)));
    REQUIRE(tensor_close(nanquantile(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0.3), result_tensor_type(1.0)));
}

TEMPLATE_TEST_CASE("test_statistic_quantile_nanquantile_exception","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::quantile;
    using gtensor::nanquantile;

    REQUIRE_THROWS_AS(quantile(tensor_type{1,2,3,4,5},1.1), value_error);
    REQUIRE_THROWS_AS(nanquantile(tensor_type{1,2,3,4,5},1.1), value_error);
}

TEST_CASE("test_statistic_quantile_nanquantile_nan_values_default_policy","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::quantile;
    using gtensor::nanquantile;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //quantile
        std::make_tuple(quantile(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(quantile(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}, value_type{0.5}), tensor_type(2.5)),
        std::make_tuple(quantile(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}, value_type{0.5}), tensor_type(1.5)),
        std::make_tuple(quantile(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(1.5)),
        std::make_tuple(quantile(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(quantile(tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(quantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(quantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0, value_type{0.5}), tensor_type{nan,nan,nan,3.0}),
        std::make_tuple(quantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1, value_type{0.5}), tensor_type{nan,nan,nan,nan,1.5}),
        std::make_tuple(quantile(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, std::vector<int>{}, value_type{0.3}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nanquantile
        std::make_tuple(nanquantile(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}, value_type{0.5}), tensor_type(2.0)),
        std::make_tuple(nanquantile(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}, value_type{0.5}), tensor_type(2.5)),
        std::make_tuple(nanquantile(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}, value_type{0.5}), tensor_type(1.5)),
        std::make_tuple(nanquantile(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(1.5)),
        std::make_tuple(nanquantile(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(2.0)),
        std::make_tuple(nanquantile(tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(nanquantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}, value_type{0.5}), tensor_type(2.0)),
        std::make_tuple(nanquantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0, value_type{0.5}), tensor_type{1.5,1.0,0.0,3.0}),
        std::make_tuple(nanquantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1, value_type{0.5}), tensor_type{1.5,-3.0,3.0,8.0,1.5}),
        std::make_tuple(nanquantile(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, std::vector<int>{}, value_type{0.3}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_statistic_quantile_nanquantile_nan_values_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::quantile;
    using gtensor::nanquantile;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //quantile
        std::make_tuple(quantile(policy{},tensor_type{1.0,0.5,nan,4.0,3.0,2.0}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(quantile(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}, value_type{0.5}), tensor_type(2.5)),
        std::make_tuple(quantile(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}, value_type{0.5}), tensor_type(1.5)),
        std::make_tuple(quantile(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(1.5)),
        std::make_tuple(quantile(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(quantile(policy{},tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(quantile(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(quantile(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0, value_type{0.5}), tensor_type{nan,nan,nan,3.0}),
        std::make_tuple(quantile(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1, value_type{0.5}), tensor_type{nan,nan,nan,nan,1.5}),
        std::make_tuple(quantile(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, std::vector<int>{}, value_type{0.3}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nanquantile
        std::make_tuple(nanquantile(policy{},tensor_type{1.0,0.5,nan,4.0,3.0,2.0}, value_type{0.5}), tensor_type(2.0)),
        std::make_tuple(nanquantile(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}, value_type{0.5}), tensor_type(2.5)),
        std::make_tuple(nanquantile(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}, value_type{0.5}), tensor_type(1.5)),
        std::make_tuple(nanquantile(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(1.5)),
        std::make_tuple(nanquantile(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(2.0)),
        std::make_tuple(nanquantile(policy{},tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}, value_type{0.5}), tensor_type(nan)),
        std::make_tuple(nanquantile(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}, value_type{0.5}), tensor_type(2.0)),
        std::make_tuple(nanquantile(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0, value_type{0.5}), tensor_type{1.5,1.0,0.0,3.0}),
        std::make_tuple(nanquantile(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1, value_type{0.5}), tensor_type{1.5,-3.0,3.0,8.0,1.5}),
        std::make_tuple(nanquantile(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, std::vector<int>{}, value_type{0.3}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

//median,nanmedian
TEMPLATE_TEST_CASE("test_statistic_median_nanmedian_normal_values","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::median;
    using gtensor::nanmedian;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
    using result_tensor_type = gtensor::tensor<result_value_type>;
    REQUIRE(std::is_same_v<typename decltype(median(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(median(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmedian(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmedian(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,result_tensor_type{nan,nan}),
        std::make_tuple(tensor_type{5},0,false,result_tensor_type(5)),
        std::make_tuple(tensor_type{5,6},0,false,result_tensor_type(5.5)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,false,result_tensor_type(2)),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,false,result_tensor_type(2.5)),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0,false,result_tensor_type{1,2,1,5,3,1}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},1,false,result_tensor_type{1.0,2.0,2.0,1.5,4.0}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},std::vector<int>{1,0},false,result_tensor_type(2.0)),
        std::make_tuple(tensor_type{},std::vector<int>{},false,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,result_tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},false,result_tensor_type{{1,2,3},{4,5,6}}),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,result_tensor_type{nan}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,result_tensor_type{nan}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,result_tensor_type{{{nan},{nan}}}),
        std::make_tuple(tensor_type{5},0,true,result_tensor_type{5}),
        std::make_tuple(tensor_type{5,6},0,true,result_tensor_type{5.5}),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,true,result_tensor_type{2}),
        std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,true,result_tensor_type{2.5}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0,true,result_tensor_type{{1,2,1,5,3,1}}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},1,true,result_tensor_type{{1.0},{2.0},{2.0},{1.5},{4.0}}),
        std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},std::vector<int>{1,0},true,result_tensor_type{{2.0}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,result_tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},true,result_tensor_type{{1,2,3},{4,5,6}})
    );
    auto test_median = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = median(policy...,ten,axes,keep_dims);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
        };
        apply_by_element(test,test_data);
    };
    auto test_nanmedian = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = nanmedian(policy...,ten,axes,keep_dims);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_median_default_policy")
    {
        test_median();
    }
    SECTION("test_nanmedian_default_policy")
    {
        test_nanmedian();
    }
    SECTION("test_median_exec_pol<4>")
    {
        test_median(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanmedian_exec_pol<4>")
    {
        test_nanmedian(multithreading::exec_pol<4>{});
    }
    SECTION("test_median_exec_pol<0>")
    {
        test_median(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanmedian_exec_pol<0>")
    {
        test_nanmedian(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_median_nanmedian_overload_default_policy","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::median;
    using gtensor::nanmedian;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(median(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmedian(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    //median
    REQUIRE(median(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},false) == result_tensor_type{1,2,1,5,3,1});
    REQUIRE(median(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1}) == result_tensor_type{1.0,2.0,2.0,1.5,4.0});
    REQUIRE(median(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0}) == result_tensor_type(2.0));
    REQUIRE(median(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}}) == result_tensor_type(2.0));
    //nanmedian
    REQUIRE(nanmedian(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},false) == result_tensor_type{1,2,1,5,3,1});
    REQUIRE(nanmedian(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1}) == result_tensor_type{1.0,2.0,2.0,1.5,4.0});
    REQUIRE(nanmedian(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0}) == result_tensor_type(2.0));
    REQUIRE(nanmedian(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}}) == result_tensor_type(2.0));
}

TEMPLATE_TEST_CASE("test_statistic_median_nanmedian_overload_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::median;
    using gtensor::nanmedian;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(median(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmedian(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    //median
    REQUIRE(median(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},false) == result_tensor_type{1,2,1,5,3,1});
    REQUIRE(median(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1}) == result_tensor_type{1.0,2.0,2.0,1.5,4.0});
    REQUIRE(median(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0}) == result_tensor_type(2.0));
    REQUIRE(median(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}}) == result_tensor_type(2.0));
    //nanmedian
    REQUIRE(nanmedian(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},false) == result_tensor_type{1,2,1,5,3,1});
    REQUIRE(nanmedian(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1}) == result_tensor_type{1.0,2.0,2.0,1.5,4.0});
    REQUIRE(nanmedian(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0}) == result_tensor_type(2.0));
    REQUIRE(nanmedian(policy{},tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}}) == result_tensor_type(2.0));
}

TEST_CASE("test_statistic_median_nanmedian_nan_values_default_policy","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::median;
    //using gtensor::nanmedian;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //median
        std::make_tuple(median(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
        std::make_tuple(median(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(2.5)),
        std::make_tuple(median(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(1.5)),
        std::make_tuple(median(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(1.5)),
        std::make_tuple(median(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(median(tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(median(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}), tensor_type(nan)),
        std::make_tuple(median(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0), tensor_type{nan,nan,nan,3.0}),
        std::make_tuple(median(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1), tensor_type{nan,nan,nan,nan,1.5}),
        std::make_tuple(median(tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf}),
        //nanmedian
        std::make_tuple(nanmedian(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(2.0)),
        std::make_tuple(nanmedian(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(2.5)),
        std::make_tuple(nanmedian(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(1.5)),
        std::make_tuple(nanmedian(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(1.5)),
        std::make_tuple(nanmedian(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(2.0)),
        std::make_tuple(nanmedian(tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanmedian(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}), tensor_type(2.0)),
        std::make_tuple(nanmedian(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0), tensor_type{1.5,1.0,0.0,3.0}),
        std::make_tuple(nanmedian(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1), tensor_type{1.5,-3.0,3.0,8.0,1.5}),
        std::make_tuple(nanmedian(tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_statistic_median_nanmedian_nan_values_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::median;
    using gtensor::nanmedian;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //median
        std::make_tuple(median(policy{},tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
        std::make_tuple(median(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(2.5)),
        std::make_tuple(median(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(1.5)),
        std::make_tuple(median(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(1.5)),
        std::make_tuple(median(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(median(policy{},tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(median(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}), tensor_type(nan)),
        std::make_tuple(median(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0), tensor_type{nan,nan,nan,3.0}),
        std::make_tuple(median(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1), tensor_type{nan,nan,nan,nan,1.5}),
        std::make_tuple(median(policy{},tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf}),
        //nanmedian
        std::make_tuple(nanmedian(policy{},tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(2.0)),
        std::make_tuple(nanmedian(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(2.5)),
        std::make_tuple(nanmedian(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(1.5)),
        std::make_tuple(nanmedian(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(1.5)),
        std::make_tuple(nanmedian(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(2.0)),
        std::make_tuple(nanmedian(policy{},tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanmedian(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}), tensor_type(2.0)),
        std::make_tuple(nanmedian(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0), tensor_type{1.5,1.0,0.0,3.0}),
        std::make_tuple(nanmedian(policy{},tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1), tensor_type{1.5,-3.0,3.0,8.0,1.5}),
        std::make_tuple(nanmedian(policy{},tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}