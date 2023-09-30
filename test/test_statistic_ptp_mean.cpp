/*
* GTensor - matrix computation library
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

//ptp
TEMPLATE_TEST_CASE("test_statistic_ptp","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::ptp;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{5},0,false,tensor_type(0)),
        std::make_tuple(tensor_type{5,6},0,false,tensor_type(1)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,tensor_type(4)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,tensor_type(4)),
        std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},0,false,tensor_type{{2,5,1},{2,2,9}}),
        std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},1,false,tensor_type{{1,1,8},{1,4,2}}),
        std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},2,false,tensor_type{{6,4},{4,7}}),
        std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},std::vector<int>{0,2},false,tensor_type{6,9}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,tensor_type{0,0,0,0,0}),
        std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},std::vector<int>{},false,tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}}),
        //keep_dim true
        std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},0,true,tensor_type{{{2,5,1},{2,2,9}}}),
        std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},std::vector<int>{2,1,0},true,tensor_type{{{9}}}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,tensor_type{0,0,0,0,0}),
        std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},std::vector<int>{},true,tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}})
    );
    auto test_ptp = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = ptp(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_ptp_default_policy")
    {
        test_ptp();
    }
    SECTION("test_ptp_exec_pol<4>")
    {
        test_ptp(multithreading::exec_pol<4>{});
    }
    SECTION("test_ptp_exec_pol<0>")
    {
        test_ptp(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_ptp_overloads_default_policy","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::ptp;

    REQUIRE(std::is_same_v<typename decltype(ptp(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);

    REQUIRE(ptp(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},{0},false) == tensor_type{{2,5,1},{2,2,9}});
    REQUIRE(ptp(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},{0,2},false) == tensor_type{6,9});
    REQUIRE(ptp(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},{1,0,2},true) == tensor_type{{{9}}});
    //all axes
    REQUIRE(ptp(tensor_type{{{5}}}) == tensor_type(0));
    REQUIRE(ptp(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},false) == tensor_type(9));
    REQUIRE(ptp(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}}) == tensor_type(9));
    REQUIRE(ptp(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},true) == tensor_type{{{9}}});
}

TEMPLATE_TEST_CASE("test_statistic_ptp_overloads_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::ptp;

    REQUIRE(std::is_same_v<typename decltype(ptp(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);

    REQUIRE(ptp(policy{},tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},{0},false) == tensor_type{{2,5,1},{2,2,9}});
    REQUIRE(ptp(policy{},tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},{0,2},false) == tensor_type{6,9});
    REQUIRE(ptp(policy{},tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},{1,0,2},true) == tensor_type{{{9}}});
    //all axes
    REQUIRE(ptp(policy{},tensor_type{{{5}}}) == tensor_type(0));
    REQUIRE(ptp(policy{},tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},false) == tensor_type(9));
    REQUIRE(ptp(policy{},tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}}) == tensor_type(9));
    REQUIRE(ptp(policy{},tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},true) == tensor_type{{{9}}});
}

TEST_CASE("test_statistic_ptp_exception","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::ptp;
    //zero size axis
    REQUIRE_THROWS_AS(ptp(tensor_type{},0), value_error);
}

//mean,nanmean
TEMPLATE_TEST_CASE("test_statistic_mean_nanmean_normal_values","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::mean;
    using gtensor::nanmean;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(mean(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(mean(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmean(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmean(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(mean(multithreading::exec_pol<4>{},std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(mean(multithreading::exec_pol<4>{},std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmean(multithreading::exec_pol<4>{},std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmean(multithreading::exec_pol<4>{},std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,result_tensor_type(nan)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,result_tensor_type{nan,nan}),
        std::make_tuple(tensor_type{5},0,false,result_tensor_type(5)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,result_tensor_type(3)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,result_tensor_type(3)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,result_tensor_type{{4,5,6},{7,8,9}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,result_tensor_type{{2.5,3.5,4.5},{8.5,9.5,10.5}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,result_tensor_type{{2,5},{8,11}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,result_tensor_type{5.5,6.5,7.5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,result_tensor_type{3.5,9.5}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,result_tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,result_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,result_tensor_type{nan}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,result_tensor_type{nan}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,result_tensor_type{{{nan},{nan}}}),
        std::make_tuple(tensor_type{5},0,true,result_tensor_type{5}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,true,result_tensor_type{3}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,result_tensor_type{3}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,result_tensor_type{{{4,5,6},{7,8,9}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,result_tensor_type{{{2.5,3.5,4.5}},{{8.5,9.5,10.5}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,result_tensor_type{{{2},{5}},{{8},{11}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,result_tensor_type{{{5.5,6.5,7.5}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,result_tensor_type{{{3.5}},{{9.5}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,result_tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,result_tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,result_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}})
    );
    auto test_mean = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = mean(policy...,ten,axes,keep_dims);
            REQUIRE(gtensor::tensor_equal(result,expected,true));
        };
        apply_by_element(test,test_data);
    };
    auto test_nanmean = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = nanmean(policy...,ten,axes,keep_dims);
            REQUIRE(gtensor::tensor_equal(result,expected,true));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_mean_default_policy")
    {
        test_mean();
    }
    SECTION("test_nanmean_default_policy")
    {
        test_nanmean();
    }
    SECTION("test_mean_exec_pol<4>")
    {
        test_mean(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanmean_exec_pol<4>")
    {
        test_nanmean(multithreading::exec_pol<4>{});
    }
    SECTION("test_mean_exec_pol<0>")
    {
        test_mean(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanmean_exec_pol<0>")
    {
        test_nanmean(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_mean_nanmean_overloads_default_policy","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::mean;
    using gtensor::nanmean;
    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(mean(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmean(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    //mean
    REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == result_tensor_type{{4,5,6},{7,8,9}});
    REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == result_tensor_type{5.5,6.5,7.5});
    REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == result_tensor_type{{{3.5}},{{9.5}}});
    //all axes
    REQUIRE(mean(tensor_type{{{5}}}) == result_tensor_type(5));
    REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == result_tensor_type(6.5));
    REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == result_tensor_type(6.5));
    REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == result_tensor_type{{{6.5}}});

    //nanmean
    REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == result_tensor_type{{4,5,6},{7,8,9}});
    REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == result_tensor_type{5.5,6.5,7.5});
    REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == result_tensor_type{{{3.5}},{{9.5}}});
    //all axes
    REQUIRE(nanmean(tensor_type{{{5}}}) == result_tensor_type(5));
    REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == result_tensor_type(6.5));
    REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == result_tensor_type(6.5));
    REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == result_tensor_type{{{6.5}}});
}

TEMPLATE_TEST_CASE("test_statistic_mean_nanmean_overloads_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::mean;
    using gtensor::nanmean;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(mean(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmean(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
    //mean
    REQUIRE(mean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == result_tensor_type{{4,5,6},{7,8,9}});
    REQUIRE(mean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == result_tensor_type{5.5,6.5,7.5});
    REQUIRE(mean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == result_tensor_type{{{3.5}},{{9.5}}});
    //all axes
    REQUIRE(mean(policy{},tensor_type{{{5}}}) == result_tensor_type(5));
    REQUIRE(mean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == result_tensor_type(6.5));
    REQUIRE(mean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == result_tensor_type(6.5));
    REQUIRE(mean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == result_tensor_type{{{6.5}}});

    //nanmean
    REQUIRE(nanmean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == result_tensor_type{{4,5,6},{7,8,9}});
    REQUIRE(nanmean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == result_tensor_type{5.5,6.5,7.5});
    REQUIRE(nanmean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == result_tensor_type{{{3.5}},{{9.5}}});
    //all axes
    REQUIRE(nanmean(policy{},tensor_type{{{5}}}) == result_tensor_type(5));
    REQUIRE(nanmean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == result_tensor_type(6.5));
    REQUIRE(nanmean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == result_tensor_type(6.5));
    REQUIRE(nanmean(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == result_tensor_type{{{6.5}}});
}

TEST_CASE("test_statistic_mean_nanmean_nan_values_default_policy","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::mean;
    using gtensor::nanmean;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //mean
        std::make_tuple(mean(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(mean(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(mean(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(mean(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(mean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(mean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(mean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(mean(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(mean(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,2.5}),
        std::make_tuple(mean(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(mean(tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf}),
        //nanmean
        std::make_tuple(nanmean(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(nanmean(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(nanmean(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanmean(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanmean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanmean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(nanmean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(nanmean(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(2.0)),
        std::make_tuple(nanmean(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.5,2.0,nan,3.0}),
        std::make_tuple(nanmean(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{2.5,1.75,2.5,1.5}),
        std::make_tuple(nanmean(tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_statistic_mean_nanmean_nan_values_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::mean;
    using gtensor::nanmean;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //mean
        std::make_tuple(mean(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(mean(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(mean(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(mean(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(mean(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(mean(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(mean(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(mean(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(mean(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,2.5}),
        std::make_tuple(mean(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(mean(policy{},tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf}),
        //nanmean
        std::make_tuple(nanmean(policy{},tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(nanmean(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(nanmean(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanmean(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanmean(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanmean(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(nanmean(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(nanmean(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(2.0)),
        std::make_tuple(nanmean(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.5,2.0,nan,3.0}),
        std::make_tuple(nanmean(policy{},tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{2.5,1.75,2.5,1.5}),
        std::make_tuple(nanmean(policy{},tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,-2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

