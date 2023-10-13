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

//moving mean
TEMPLATE_TEST_CASE("test_statistic_moving_mean","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using gtensor::tensor_close;
    using gtensor::moving_mean;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<
        typename decltype(moving_mean(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<index_type>(),std::declval<index_type>()))::value_type,
        result_value_type>
    );

    //0tensor,1axis,2window_size,3step,4expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}.reshape(0,4,5),1,2,1,result_tensor_type{}.reshape(0,4,5)),
        std::make_tuple(tensor_type{}.reshape(0,4,5),2,3,1,result_tensor_type{}.reshape(0,4,5)),
        std::make_tuple(tensor_type{5},0,1,1,result_tensor_type{5}),
        std::make_tuple(tensor_type{5},0,1,2,result_tensor_type{5}),
        std::make_tuple(tensor_type{5,6},0,1,1,result_tensor_type{5,6}),
        std::make_tuple(tensor_type{5,6},0,2,1,result_tensor_type{5.5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,4,1,result_tensor_type{2.5,3.5,4.5,5.5,6.5,7.5,8.5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,4,2,result_tensor_type{2.5,4.5,6.5,8.5}),
        std::make_tuple(tensor_type{1,2,3,4,5,5,4,3,2,1},0,4,3,result_tensor_type{2.5,4.5,2.5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,4,4,result_tensor_type{2.5,6.5}),
        std::make_tuple(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},0,3,1,result_tensor_type{{1.333,1.333,1.0,2.0,4.666},{2.0,1.666,1.0,3.666,3.666}}),
        std::make_tuple(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},1,3,1,result_tensor_type{{1.333,0.0,1.0},{2.666,3.0,3.333},{-0.333,1.333,3.333},{2.333,2.0,1.666}})
    );
    auto test_moving_mean = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto window_size = std::get<2>(t);
            auto step = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = moving_mean(policy...,ten,axis,window_size,step);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_moving_mean_default_policy")
    {
        test_moving_mean();
    }
    SECTION("test_moving_mean_exec_pol<4>")
    {
        test_moving_mean(multithreading::exec_pol<4>{});
    }
    SECTION("test_moving_mean_exec_pol<0>")
    {
        test_moving_mean(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_moving_mean_exception","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::moving_mean;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2window_size,3step
    auto test_data = std::make_tuple(
        //zero window size
        std::make_tuple(tensor_type{1,2,3},0,0,1),
        //window_size size greater than axis size
        std::make_tuple(tensor_type{1,2,3,4,5},0,6,1),
        //zero step
        std::make_tuple(tensor_type{1,2,3,4,5},0,3,0)
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto window_size = std::get<2>(t);
        auto step = std::get<3>(t);
        REQUIRE_THROWS_AS(moving_mean(ten,axis,window_size,step), value_error);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_statistic_moving_mean_overload_default_policy","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::moving_mean;
    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    //like over flatten, window_size=3,step=1
    REQUIRE(tensor_close(
        moving_mean(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},3,1),
        result_tensor_type{1.333,0.0,1.0,1.333,2.333,2.667,3.0,3.333,1.667,1.333,-0.333,1.333,3.333,5.667,4.667,2.333,2.0,1.667},1E-2,1E-2)
    );
    //like over flatten, window_size=3,step=3
    REQUIRE(tensor_close(
        moving_mean(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},3,3),
        result_tensor_type{1.333,1.333,3.0,1.333,3.333,2.333},1E-2,1E-2)
    );
}

TEMPLATE_TEST_CASE("test_statistic_moving_mean_overload_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::moving_mean;
    using result_value_type = typename gtensor::math::make_floating_point_like_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    //like over flatten, window_size=3,step=1
    REQUIRE(tensor_close(
        moving_mean(policy{},tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},3,1),
        result_tensor_type{1.333,0.0,1.0,1.333,2.333,2.667,3.0,3.333,1.667,1.333,-0.333,1.333,3.333,5.667,4.667,2.333,2.0,1.667},1E-2,1E-2)
    );
    //like over flatten, window_size=3,step=3
    REQUIRE(tensor_close(
        moving_mean(policy{},tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},3,3),
        result_tensor_type{1.333,1.333,3.0,1.333,3.333,2.333},1E-2,1E-2)
    );
}

//histogram
TEMPLATE_TEST_CASE("test_statistic_uniform_bins_histogram","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using fp_type = gtensor::math::make_floating_point_t<value_type>;
    using weights_tensor_type = gtensor::tensor<fp_type>;
    using result_tensor_type = gtensor::tensor<fp_type>;
    using gtensor::detail::no_value;
    using gtensor::tensor_close;
    using gtensor::histogram;
    using helpers_for_testing::apply_by_element;

    //0tensor,1bins,2range,3density,4weights,5expected_bins,6expected_intervals
    auto test_data = std::make_tuple(
        //bins integral, no range
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},1,no_value{},false,no_value{},result_tensor_type{13},result_tensor_type{-1,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},2,no_value{},false,no_value{},result_tensor_type{6,7},result_tensor_type{-1,3,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,no_value{},false,no_value{},result_tensor_type{2,4,2,2,3},result_tensor_type{-1.0,0.6,2.2,3.8,5.4,7.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,no_value{},false,no_value{},result_tensor_type{2,3,1,2,1,1,3},result_tensor_type{-1.0,0.143,1.286,2.429,3.571,4.714,5.857,7.0}),
        //bins integral, range
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,std::make_pair(2,6),false,no_value{},result_tensor_type{1,2,1,1,2},result_tensor_type{2.0,2.8,3.6,4.4,5.2,6.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(2,6),false,no_value{},result_tensor_type{1,2,0,1,0,1,2},result_tensor_type{2.0,2.571,3.142,3.714,4.285,4.857,5.428,6.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(-4,6),false,no_value{},result_tensor_type{0,0,2,3,3,1,3},result_tensor_type{-4.0,-2.571,-1.142,0.285,1.714,3.142,4.571,6.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(3,11),false,no_value{},result_tensor_type{3,1,2,1,0,0,0},result_tensor_type{3.0,4.142,5.285,6.428,7.571,8.714,9.857,11.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(-5.3,13.3),false,no_value{},result_tensor_type{0,2,4,4,3,0,0},result_tensor_type{-5.3,-2.643,0.014,2.671,5.329,7.986,10.643,13.3}),
        //density
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},1,no_value{},true,no_value{},result_tensor_type{0.125},result_tensor_type{-1,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},2,no_value{},true,no_value{},result_tensor_type{0.115,0.135},result_tensor_type{-1,3,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,no_value{},true,no_value{},result_tensor_type{0.096,0.192,0.096,0.096,0.144},result_tensor_type{-1.0,0.6,2.2,3.8,5.4,7.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,no_value{},true,no_value{},result_tensor_type{0.135,0.202,0.067,0.135,0.067,0.067,0.202},result_tensor_type{-1.0,0.143,1.286,2.429,3.571,4.714,5.857,7.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(2,6),true,no_value{},result_tensor_type{0.25,0.5,0.0,0.25,0.0,0.25,0.5},result_tensor_type{2.0,2.571,3.142,3.714,4.285,4.857,5.428,6.0}),
        //weights, no range
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},1,no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{12.5},result_tensor_type{-1,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},2,no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{5.5,7.0},result_tensor_type{-1,3,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{2.0,3.5,1.5,2.3,3.2},result_tensor_type{-1.0,0.6,2.2,3.8,5.4,7.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{2.0,2.5,1.0,1.5,1.0,1.3,3.2},result_tensor_type{-1.0,0.143,1.286,2.429,3.571,4.714,5.857,7.0}),
        //weights, range
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,std::make_pair(2,6),false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{1.0,1.5,1.0,1.3,2.5},result_tensor_type{2.0,2.8,3.6,4.4,5.2,6.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(2,6),false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{1.0,1.5,0.0,1.0,0.0,1.3,2.5},result_tensor_type{2.0,2.571,3.142,3.714,4.285,4.857,5.428,6.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(-4,6),false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{0.0,0.0,2.0,2.5,2.5,1.0,3.8},result_tensor_type{-4.0,-2.571,-1.142,0.285,1.714,3.142,4.571,6.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(3,11),false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{2.5,1.3,2.5,0.7,0.0,0.0,0.0},result_tensor_type{3.0,4.142,5.285,6.428,7.571,8.714,9.857,11.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(-5.3,13.3),false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{0.0,2.0,3.5,3.8,3.2,0.0,0.0},result_tensor_type{-5.3,-2.643,0.014,2.671,5.329,7.986,10.643,13.3}),
        //corner cases
        //empty source
        std::make_tuple(tensor_type{},1,no_value{},false,no_value{},result_tensor_type{0},result_tensor_type{0.0,1.0}),
        std::make_tuple(tensor_type{},5,no_value{},false,no_value{},result_tensor_type{0,0,0,0,0},result_tensor_type{0.0,0.2,0.4,0.6,0.8,1.0}),
        std::make_tuple(tensor_type{},5,std::make_pair(-3,3),false,no_value{},result_tensor_type{0,0,0,0,0},result_tensor_type{-3.0,-1.8,-0.6,0.6,1.8,3.0}),
        std::make_tuple(tensor_type{},5,std::make_pair(3,3),false,no_value{},result_tensor_type{0,0,0,0,0},result_tensor_type{2.5,2.7,2.9,3.1,3.3,3.5}),
        //empty range min==max
        std::make_tuple(tensor_type{3},1,no_value{},false,no_value{},result_tensor_type{1},result_tensor_type{2.5,3.5}),
        std::make_tuple(tensor_type{3},5,no_value{},false,no_value{},result_tensor_type{0,0,1,0,0},result_tensor_type{2.5,2.7,2.9,3.1,3.3,3.5}),
        std::make_tuple(tensor_type{3},7,no_value{},false,no_value{},result_tensor_type{0,0,0,1,0,0,0},result_tensor_type{2.5,2.643,2.786,2.929,3.071,3.214,3.357,3.5}),
        std::make_tuple(tensor_type{3,3,3},7,no_value{},false,no_value{},result_tensor_type{0,0,0,3,0,0,0},result_tensor_type{2.5,2.643,2.786,2.929,3.071,3.214,3.357,3.5}),
        //empty range rmin==rmax
        std::make_tuple(tensor_type{3,3,3,3,3},5,std::make_pair(1,1),false,no_value{},result_tensor_type{0,0,0,0,0},result_tensor_type{0.5,0.7,0.9,1.1,1.3,1.5}),
        std::make_tuple(tensor_type{3,3,3,3,3},5,std::make_pair(3,3),false,no_value{},result_tensor_type{0,0,5,0,0},result_tensor_type{2.5,2.7,2.9,3.1,3.3,3.5}),
        std::make_tuple(tensor_type{1,0,0,3,1,0,1,0,-1,1,0},5,std::make_pair(1,1),false,no_value{},result_tensor_type{0,0,4,0,0},result_tensor_type{0.5,0.7,0.9,1.1,1.3,1.5}),
        std::make_tuple(tensor_type{1,0,0,3,1,0,1,0,-1,1,0},7,std::make_pair(1,1),false,no_value{},result_tensor_type{0,0,0,4,0,0,0},result_tensor_type{0.5,0.643,0.786,0.929,1.071,1.214,1.357,1.5}),
        std::make_tuple(tensor_type{1,0,0,3,1,0,1,0,-1,1,0},7,std::make_pair(3,3),false,no_value{},result_tensor_type{0,0,0,1,0,0,0},result_tensor_type{2.5,2.643,2.786,2.929,3.071,3.214,3.357,3.5}),
        std::make_tuple(tensor_type{1,0,0,3,1,0,1,0,-1,1,0},7,std::make_pair(4,4),false,no_value{},result_tensor_type{0,0,0,0,0,0,0},result_tensor_type{3.5,3.643,3.786,3.929,4.071,4.214,4.357,4.5}),
        //source out of range
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,std::make_pair(-32,-10),false,no_value{},result_tensor_type{0,0,0,0,0},result_tensor_type{-32.0,-27.6,-23.2,-18.8,-14.4,-10.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(10,32),false,no_value{},result_tensor_type{0,0,0,0,0,0,0},result_tensor_type{10.0,13.143,16.286,19.429,22.571,25.714,28.857,32.0})
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto bins = std::get<1>(t);
        auto range = std::get<2>(t);
        auto density = std::get<3>(t);
        auto weights = std::get<4>(t);
        auto expected_bins = std::get<5>(t);
        auto expected_intervals = std::get<6>(t);

        auto result = histogram(ten,bins,range,density,weights);
        REQUIRE(tensor_close(result.first,expected_bins,1E-2,1E-2));
        REQUIRE(tensor_close(result.second,expected_intervals,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_statistic_not_uniform_bins_histogram","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using fp_type = gtensor::math::make_floating_point_t<value_type>;
    using weights_tensor_type = gtensor::tensor<fp_type>;
    using result_tensor_type = gtensor::tensor<fp_type>;
    using gtensor::detail::no_value;
    using gtensor::tensor_close;
    using gtensor::histogram;
    using helpers_for_testing::apply_by_element;

    //0tensor,1bins,2range,3density,4weights,5expected_bins,6expected_intervals
    auto test_data = std::make_tuple(
        //bins container, no weights
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,5},no_value{},false,no_value{},result_tensor_type{9},result_tensor_type{0,5}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,1,3},no_value{},false,no_value{},result_tensor_type{1,6},result_tensor_type{0,1,3}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,2,4,7},no_value{},false,no_value{},result_tensor_type{4,3,5},result_tensor_type{0,2,4,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},std::vector<value_type>{-2,1,2,3,6,8},no_value{},false,no_value{},result_tensor_type{2,3,1,4,3},result_tensor_type{-2,1,2,3,6,8}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{-2,1,1,3,6,8},no_value{},false,no_value{},result_tensor_type{2,0,4,4,3},result_tensor_type{-2,1,1,3,6,8}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{3,4,6,15},no_value{},false,no_value{},result_tensor_type{2,2,3},result_tensor_type{3,4,6,15}),
        //bins container, density
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,5},no_value{},true,no_value{},result_tensor_type{0.2},result_tensor_type{0,5}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,1,3},no_value{},true,no_value{},result_tensor_type{0.143,0.429},result_tensor_type{0,1,3}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,2,4,6,8},no_value{},true,no_value{},result_tensor_type{0.167,0.125,0.083,0.125},result_tensor_type{0,2,4,6,8}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},std::vector<value_type>{0,2,4,7},no_value{},true,no_value{},result_tensor_type{0.167,0.125,0.139},result_tensor_type{0,2,4,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{-2,1,2,3,6,8},no_value{},true,no_value{},result_tensor_type{0.051,0.231,0.077,0.103,0.115},result_tensor_type{-2,1,2,3,6,8}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{3,4,6,15},no_value{},true,no_value{},result_tensor_type{0.286,0.143,0.048},result_tensor_type{3,4,6,15}),
        //bins container, weights
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,5},no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{8.3},result_tensor_type{0,5}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,2,4,7},no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{3.5,2.5,5.5},result_tensor_type{0,2,4,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},std::vector<value_type>{-2,1,2,3,6,8},no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{2.0,2.5,1.0,3.8,3.2},result_tensor_type{-2,1,2,3,6,8}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{-2,1,1,3,6,8},no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{2.0,0.0,3.5,3.8,3.2},result_tensor_type{-2,1,1,3,6,8}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{3,4,6,15},no_value{},false,weights_tensor_type{0.5,0.7,1.0,1.0,1.0,1.3,1.5,1.3,1.0,1.0,1.0,0.7,0.5},result_tensor_type{1.5,2.3,3.2},result_tensor_type{3,4,6,15}),
        //corner cases
        //too few edges
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{3},no_value{},false,no_value{},result_tensor_type{},result_tensor_type{3}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{},no_value{},false,no_value{},result_tensor_type{},result_tensor_type{}),
        //equal edges
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{3,3},no_value{},false,no_value{},result_tensor_type{2},result_tensor_type{3,3}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{3,3,3,3},no_value{},false,no_value{},result_tensor_type{0,0,2},result_tensor_type{3,3,3,3})
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto bins = std::get<1>(t);
        auto range = std::get<2>(t);
        auto density = std::get<3>(t);
        auto weights = std::get<4>(t);
        auto expected_bins = std::get<5>(t);
        auto expected_intervals = std::get<6>(t);

        auto result = histogram(ten,bins,range,density,weights);
        REQUIRE(tensor_close(result.first,expected_bins,1E-2,1E-2));
        REQUIRE(tensor_close(result.second,expected_intervals,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_statistic_histogram_bins_density_parameters_overload","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using fp_type = gtensor::math::make_floating_point_t<value_type>;
    using result_tensor_type = gtensor::tensor<fp_type>;
    using gtensor::detail::no_value;
    using gtensor::tensor_close;
    using gtensor::histogram;
    using helpers_for_testing::apply_by_element;

    //0tensor,1bins,2density,3expected_bins,4expected_intervals
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,false,result_tensor_type{2,4,2,2,3},result_tensor_type{-1.0,0.6,2.2,3.8,5.4,7.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,true,result_tensor_type{0.135,0.202,0.067,0.135,0.067,0.067,0.202},result_tensor_type{-1.0,0.143,1.286,2.429,3.571,4.714,5.857,7.0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{0,2,4,7},false,result_tensor_type{4,3,5},result_tensor_type{0,2,4,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{-2,1,2,3,6,8},true,result_tensor_type{0.051,0.231,0.077,0.103,0.115},result_tensor_type{-2,1,2,3,6,8})

    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto bins = std::get<1>(t);
        auto density = std::get<2>(t);
        auto expected_bins = std::get<3>(t);
        auto expected_intervals = std::get<4>(t);

        auto result = histogram(ten,bins,density);
        REQUIRE(tensor_close(result.first,expected_bins,1E-2,1E-2));
        REQUIRE(tensor_close(result.second,expected_intervals,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}
TEST_CASE("test_statistic_histogram_overload","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::histogram;

    const tensor_type gauss_100{-0.154,1.988,-0.454,-0.878,1.902,1.247,-0.843,-0.988,0.562,1.295,0.226,-1.387,-0.099,-0.118,0.226,0.15,-0.158,0.966,1.814,0.122,0.645,-1.309,0.798,
    -0.13,-0.697,-0.716,-0.245,2.16,0.678,-0.839,-0.335,-1.944,0.864,0.887,-0.318,-1.775,0.542,-0.394,-0.09,-0.454,1.496,0.167,0.906,0.55,0.562,0.101,-0.174,-0.382,-0.269,1.606,
    -1.981,0.431,0.3,0.311,-2.907,0.697,-1.206,1.924,1.224,1.062,-1.086,-0.08,-0.102,1.021,1.223,1.191,-0.928,-0.252,-0.365,0.01,-0.805,-0.665,0.003,-0.661,0.543,-1.08,0.613,
    -0.837,0.097,0.216,1.177,0.683,-0.64,-2.341,0.325,1.146,-1.286,0.817,0.768,-0.595,-0.377,0.187,0.811,1.074,0.735,0.009,0.609,-0.445,-1.491,0.985};

    auto hist = histogram(gauss_100);
    REQUIRE(hist.first == tensor_type{1,3,3,7,17,21,17,18,8,5});
    REQUIRE(tensor_close(hist.second,tensor_type{-2.907,-2.4003,-1.8936,-1.3869,-0.8802,-0.3735,0.1332,0.6399,1.1466,1.6533,2.16},1E-2,1E-2));
}

TEMPLATE_TEST_CASE("test_statistic_histogram_exception","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using gtensor::value_error;
    using gtensor::histogram;
    using helpers_for_testing::apply_by_element;

    //0tensor,1bins,2range,3density,4weights
    auto test_data = std::make_tuple(
        //bins less equal zero
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},-1,no_value{},false,no_value{}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},0,no_value{},false,no_value{}),
        //bins tensor, dim!=1
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{{1,2,3}},no_value{},false,no_value{}),
        //bins container, edges doesnt increase monotonically
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},std::vector<value_type>{3,1},no_value{},false,no_value{}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},tensor_type{3,1},no_value{},false,no_value{}),
        //range first > second
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},4,std::make_pair(5,1),false,no_value{}),
        //weights shape not much
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},3,no_value{},false,tensor_type{0,1,2,2,1,0})
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto bins = std::get<1>(t);
        auto range = std::get<2>(t);
        auto density = std::get<3>(t);
        auto weights = std::get<4>(t);
        REQUIRE_THROWS_AS(histogram(ten,bins,range,density,weights), value_error);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_statistic_histogram_algorithm","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using fp_type = gtensor::math::make_floating_point_t<value_type>;
    using result_tensor_type = gtensor::tensor<fp_type>;
    using gtensor::detail::no_value;
    using gtensor::tensor_close;
    using gtensor::histogram_algorithm;
    using gtensor::histogram;
    using helpers_for_testing::apply_by_element;

    const tensor_type gauss_100{-0.154,1.988,-0.454,-0.878,1.902,1.247,-0.843,-0.988,0.562,1.295,0.226,-1.387,-0.099,-0.118,0.226,0.15,-0.158,0.966,1.814,0.122,0.645,-1.309,0.798,
    -0.13,-0.697,-0.716,-0.245,2.16,0.678,-0.839,-0.335,-1.944,0.864,0.887,-0.318,-1.775,0.542,-0.394,-0.09,-0.454,1.496,0.167,0.906,0.55,0.562,0.101,-0.174,-0.382,-0.269,1.606,
    -1.981,0.431,0.3,0.311,-2.907,0.697,-1.206,1.924,1.224,1.062,-1.086,-0.08,-0.102,1.021,1.223,1.191,-0.928,-0.252,-0.365,0.01,-0.805,-0.665,0.003,-0.661,0.543,-1.08,0.613,
    -0.837,0.097,0.216,1.177,0.683,-0.64,-2.341,0.325,1.146,-1.286,0.817,0.768,-0.595,-0.377,0.187,0.811,1.074,0.735,0.009,0.609,-0.445,-1.491,0.985};

    const tensor_type uniform_100{0.977,0.045,0.512,0.879,0.89,0.245,0.226,0.542,0.779,0.901,0.196,0.116,0.384,0.864,0.106,0.035,0.576,0.832,0.679,0.74,0.569,0.067,0.252,0.467,0.176,0.336,
    0.016,0.333,0.681,0.382,0.436,0.123,0.399,0.096,0.086,0.047,0.759,0.836,0.394,0.171,0.108,0.787,0.667,0.906,0.826,0.866,0.406,0.515,0.752,0.371,0.043,0.062,0.918,0.079,0.321,
    0.342,0.038,0.781,0.342,0.375,0.164,0.919,0.265,0.528,0.21,0.824,0.971,0.535,0.13,0.962,0.993,0.793,0.088,0.677,0.575,0.908,0.95,0.563,0.761,0.488,0.683,0.146,0.682,0.979,0.342,
    0.468,0.96,0.387,0.469,0.343,0.529,0.913,0.881,0.014,0.905,0.226,0.626,0.192,0.277,0.608};

    //0tensor,1bins,2range,3density,4weights,5expected_bins,6expected_intervals
    auto test_data = std::make_tuple(
        //integral
        std::make_tuple(gauss_100,10,no_value{},false,no_value{},result_tensor_type{1,3,3,7,17,21,17,18,8,5},result_tensor_type{-2.907,-2.4,-1.894,-1.387,-0.88,-0.373,0.133,0.64,1.147,1.653,2.16}),
        std::make_tuple(gauss_100,10,std::make_pair(-1,1),false,no_value{},result_tensor_type{7,5,4,9,9,9,6,6,9,7},result_tensor_type{-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0}),
        std::make_tuple(uniform_100,10,no_value{},false,no_value{},result_tensor_type{15,9,7,14,6,10,8,8,9,14},result_tensor_type{0.014,0.112,0.21,0.308,0.406,0.503,0.601,0.699,0.797,0.895,0.993}),
        std::make_tuple(uniform_100,10,std::make_pair(0.15,0.85),false,no_value{},result_tensor_type{6,6,7,8,5,8,4,6,5,7},result_tensor_type{0.15,0.22,0.29,0.36,0.43,0.5,0.57,0.64,0.71,0.78,0.85}),
        //integral,density
        std::make_tuple(gauss_100,10,no_value{},true,no_value{},result_tensor_type{0.02,0.059,0.059,0.138,0.336,0.414,0.336,0.355,0.158,0.099},result_tensor_type{-2.907,-2.4,-1.894,-1.387,-0.88,-0.373,0.133,0.64,1.147,1.653,2.16}),
        std::make_tuple(gauss_100,10,std::make_pair(-1,1),true,no_value{},result_tensor_type{0.493,0.352,0.282,0.634,0.634,0.634,0.423,0.423,0.634,0.493},result_tensor_type{-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0}),
        std::make_tuple(uniform_100,10,no_value{},true,no_value{},result_tensor_type{1.532,0.919,0.715,1.43,0.613,1.021,0.817,0.817,0.919,1.43},result_tensor_type{0.014,0.112,0.21,0.308,0.406,0.503,0.601,0.699,0.797,0.895,0.993}),
        std::make_tuple(uniform_100,10,std::make_pair(0.15,0.85),true,no_value{},result_tensor_type{1.382,1.382,1.613,1.843,1.152,1.843,0.922,1.382,1.152,1.613},result_tensor_type{0.15,0.22,0.29,0.36,0.43,0.5,0.57,0.64,0.71,0.78,0.85}),
        //integral,weights
        std::make_tuple(gauss_100,10,no_value{},false,uniform_100,result_tensor_type{0.321,1.145,0.44,3.65,8.805,10.85,9.347,8.619,3.684,2.728},result_tensor_type{-2.907,-2.4,-1.894,-1.387,-0.88,-0.373,0.133,0.64,1.147,1.653,2.16}),
        std::make_tuple(gauss_100,10,std::make_pair(-1,1),false,uniform_100,result_tensor_type{4.556,2.664,1.218,3.835,5.252,5.378,2.129,3.907,4.94,3.87},result_tensor_type{-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0}),
        //edges
        std::make_tuple(uniform_100,tensor_type{0.1,0.22,0.37,0.4,0.5,0.77,0.9},no_value{},false,no_value{},result_tensor_type{12.0,13.0,7.0,6.0,22.0,13.0},result_tensor_type{0.1,0.22,0.37,0.4,0.5,0.77,0.9}),
        std::make_tuple(uniform_100,std::vector<value_type>{0.1,0.22,0.37,0.4,0.5,0.77,0.9},std::make_pair(0.15,0.85),false,no_value{},result_tensor_type{12.0,13.0,7.0,6.0,22.0,13.0},result_tensor_type{0.1,0.22,0.37,0.4,0.5,0.77,0.9}),
        //edges, density
        std::make_tuple(uniform_100,tensor_type{0.1,0.22,0.37,0.4,0.5,0.77,0.9},no_value{},true,no_value{},result_tensor_type{1.37,1.187,3.196,0.822,1.116,1.37},result_tensor_type{0.1,0.22,0.37,0.4,0.5,0.77,0.9}),
        std::make_tuple(uniform_100,std::vector<value_type>{0.1,0.22,0.37,0.4,0.5,0.77,0.9},no_value{},true,no_value{},result_tensor_type{1.37,1.187,3.196,0.822,1.116,1.37},result_tensor_type{0.1,0.22,0.37,0.4,0.5,0.77,0.9}),
        //automatic
        std::make_tuple(gauss_100,histogram_algorithm::automatic,no_value{},false,no_value{},result_tensor_type{1,3,3,7,17,21,17,18,8,5},result_tensor_type{-2.907,-2.4,-1.894,-1.387,-0.88,-0.373,0.133,0.64,1.147,1.653,2.16}),
        std::make_tuple(uniform_100,histogram_algorithm::automatic,no_value{},false,no_value{},result_tensor_type{18,11,11,11,11,8,13,17},result_tensor_type{0.014,0.136,0.259,0.381,0.503,0.626,0.748,0.871,0.993}),
        //fd
        std::make_tuple(gauss_100,histogram_algorithm::fd,no_value{},false,no_value{},result_tensor_type{1,3,3,7,17,21,17,18,8,5},result_tensor_type{-2.907,-2.4,-1.894,-1.387,-0.88,-0.373,0.133,0.64,1.147,1.653,2.16}),
        std::make_tuple(uniform_100,histogram_algorithm::fd,no_value{},false,no_value{},result_tensor_type{24,21,16,16,23},result_tensor_type{0.014,0.21,0.406,0.601,0.797,0.993}),
        //sturges
        std::make_tuple(gauss_100,histogram_algorithm::sturges,no_value{},false,no_value{},result_tensor_type{2,3,7,19,27,22,14,6},result_tensor_type{-2.907,-2.274,-1.64,-1.007,-0.373,0.26,0.893,1.527,2.16}),
        std::make_tuple(uniform_100,histogram_algorithm::sturges,no_value{},false,no_value{},result_tensor_type{18,11,11,11,11,8,13,17},result_tensor_type{0.014,0.136,0.259,0.381,0.503,0.626,0.748,0.871,0.993}),
        //scott
        std::make_tuple(gauss_100,histogram_algorithm::scott,no_value{},false,no_value{},result_tensor_type{2,4,13,27,27,20,7},result_tensor_type{-2.907,-2.183,-1.459,-0.735,-0.012,0.712,1.436,2.16}),
        std::make_tuple(uniform_100,histogram_algorithm::scott,no_value{},false,no_value{},result_tensor_type{24,21,16,16,23},result_tensor_type{0.014,0.21,0.406,0.601,0.797,0.993}),
        //rice
        std::make_tuple(gauss_100,histogram_algorithm::rice,no_value{},false,no_value{},result_tensor_type{1,3,3,7,17,21,17,18,8,5},result_tensor_type{-2.907,-2.4,-1.894,-1.387,-0.88,-0.373,0.133,0.64,1.147,1.653,2.16}),
        std::make_tuple(uniform_100,histogram_algorithm::rice,no_value{},false,no_value{},result_tensor_type{15,9,7,14,6,10,8,8,9,14},result_tensor_type{0.014,0.112,0.21,0.308,0.406,0.503,0.601,0.699,0.797,0.895,0.993}),
        //sqrt
        std::make_tuple(gauss_100,histogram_algorithm::sqrt,no_value{},false,no_value{},result_tensor_type{1,3,3,7,17,21,17,18,8,5},result_tensor_type{-2.907,-2.4,-1.894,-1.387,-0.88,-0.373,0.133,0.64,1.147,1.653,2.16}),
        std::make_tuple(uniform_100,histogram_algorithm::sqrt,no_value{},false,no_value{},result_tensor_type{15,9,7,14,6,10,8,8,9,14},result_tensor_type{0.014,0.112,0.21,0.308,0.406,0.503,0.601,0.699,0.797,0.895,0.993})
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto bins = std::get<1>(t);
        auto range = std::get<2>(t);
        auto density = std::get<3>(t);
        auto weights = std::get<4>(t);
        auto expected_bins = std::get<5>(t);
        auto expected_intervals = std::get<6>(t);

        auto result = histogram(ten,bins,range,density,weights);
        REQUIRE(tensor_close(result.first,expected_bins,1E-2,1E-2));
        REQUIRE(tensor_close(result.second,expected_intervals,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

