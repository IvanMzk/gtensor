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

//diff
TEST_CASE("test_math_diff","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::diff;
    using helpers_for_testing::apply_by_element;

    //0tensor,1n,2axis,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},0,0,tensor_type{1,3,2,5,7,4,6,7,8}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},1,0,tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},2,0,tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},3,0,tensor_type{7,-5,-4,10,-6,1}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},0,0,tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},1,0,tensor_type{{2,4,2,-2,-2},{-6,0,0,1,0},{6,-3,-1,0,2},{-2,1,2,-1,-4}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},2,0,tensor_type{{-8,-4,-2,3,2},{12,-3,-1,-1,2},{-8,4,3,-1,-6}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},3,0,tensor_type{{20,1,1,-4,0},{-20,7,4,0,-8}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},1,1,tensor_type{{-3,1,2,1},{-1,-1,-2,1},{5,-1,-1,0},{-4,1,0,2},{-1,2,-3,-1}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},2,1,tensor_type{{4,1,-1},{0,-1,3},{-6,0,1},{5,-1,2},{3,-5,2}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},3,1,tensor_type{{-3,-2},{-1,4},{6,1},{-6,3},{-8,7}})
    );
    auto test_diff = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto n = std::get<1>(t);
            auto axis = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = diff(policy...,ten,n,axis);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    //default policy
    SECTION("test_diff_default_policy")
    {
        test_diff();
    }
    //exec_pol<4>
    SECTION("test_diff_exec_pol<4>")
    {
        test_diff(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_diff_exec_pol<0>")
    {
        test_diff(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_math_diff2","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::diff2;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},0,tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},0,tensor_type{{-8,-4,-2,3,2},{12,-3,-1,-1,2},{-8,4,3,-1,-6}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},1,tensor_type{{4,1,-1},{0,-1,3},{-6,0,1},{5,-1,2},{3,-5,2}})
    );
    auto test_diff2 = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = diff2(policy...,ten,axis);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    //default policy
    SECTION("test_diff2_default_policy")
    {
        test_diff2();
    }
    //exec_pol<4>
    SECTION("test_diff2_exec_pol<4>")
    {
        test_diff2(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_diff2_exec_pol<0>")
    {
        test_diff2(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_math_diff_diff2_default_arguments","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::diff;
    using gtensor::diff2;
    using helpers_for_testing::apply_by_element;
    //default last axis
    REQUIRE(diff(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},3) == tensor_type{{-3,-2},{-1,4},{6,1},{-6,3},{-8,7}});
    //default n=1 and last axis
    REQUIRE(diff(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}}) == tensor_type{{-3,1,2,1},{-1,-1,-2,1},{5,-1,-1,0},{-4,1,0,2},{-1,2,-3,-1}});
    //default last axis
    REQUIRE(diff2(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}}) == tensor_type{{4,1,-1},{0,-1,3},{-6,0,1},{5,-1,2},{3,-5,2}});
}

//gradient
TEMPLATE_TEST_CASE("test_math_gradient","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using gtensor::gradient;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = gtensor::math::make_floating_point_like_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<typename decltype(gradient(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<value_type>()))::value_type,result_value_type>);
    REQUIRE(std::is_same_v<typename decltype(gradient(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<std::vector<value_type>>()))::value_type,result_value_type>);

    //0tensor,1axis,2spacing,3expected
    auto test_data = std::make_tuple(
        //spacing is scalar
        std::make_tuple(tensor_type{1,1},0,1,result_tensor_type{0.0,0.0}),
        std::make_tuple(tensor_type{1,2},0,1,result_tensor_type{1.0,1.0}),
        std::make_tuple(tensor_type{3,1},0,0.8,result_tensor_type{-2.5,-2.5}),
        std::make_tuple(tensor_type{1,2,4,7,11,16},0,1,result_tensor_type{1.0,1.5,2.5,3.5,4.5,5.0}),
        std::make_tuple(tensor_type{1,2,4,7,11,16,12,12,4},0,0.2,result_tensor_type{5.0,7.5,12.5,17.5,22.5,2.5,-10.0,-20.0,-40.0}),
        std::make_tuple(
            tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},
            0,
            0.8,
            result_tensor_type{{2.5,5.0,2.5,-2.5,-2.5},{-2.5,2.5,1.25,-0.625,-1.25},{0.0,-1.875,-0.625,0.625,1.25},{2.5,-1.25,0.625,-0.625,-1.25},{-2.5,1.25,2.5,-1.25,-5.0}}
        ),
        std::make_tuple(
            tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},
            1,
            0.8,
            result_tensor_type{{-3.75,-1.25,1.875,1.875,1.25},{-1.25,-1.25,-1.875,-0.625,1.25},{6.25,2.5,-1.25,-0.625,0.0},{-5.0,-1.875,0.625,1.25,2.5},{-1.25,0.625,-0.625,-2.5,-1.25}}
        ),
        //spacing is container
        std::make_tuple(tensor_type{1,1},0,tensor_type{1,2},result_tensor_type{0.0,0.0}),
        std::make_tuple(tensor_type{1,2},0,tensor_type{1,3},result_tensor_type{0.5,0.5}),
        std::make_tuple(tensor_type{3,1},0,std::vector<double>{0.8,1.0},result_tensor_type{-10.0,-10.0}),
        std::make_tuple(tensor_type{1,2,4,7,11,16},0,std::vector<double>{0.0,1.0,1.5,3.5,4.0,6.0},result_tensor_type{1.0,3.0,3.5,6.7,6.9,2.5}),
        std::make_tuple(tensor_type{1,2,4,7,11,16,12,12,4},0,std::vector<double>{0.0,1.0,1.5,3.5,4.0,5.0,5.5,6.0,6.5},result_tensor_type{1.0,3.0,3.5,6.7,7.0,-3.666,-4.0,-8.0,-16.0}),
        std::make_tuple(
            tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},
            0,
            std::vector<double>{-1,0.2,0.8,1.5,2.0},
            result_tensor_type{{1.666,3.333,1.666,-1.666,-1.666},{-6.111,1.111,0.555,0.555,-0.555},{-1.428,-1.978,-0.659,0.897,1.318},{1.238,-0.619,1.738,-1.166,-3.476},{-4.0,2.0,4.0,-2.0,-8.0}}
        ),
        std::make_tuple(
            tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},
            1,
            std::vector<double>{0.0,0.7,1.2,2.3,2.5},
            result_tensor_type{{-4.285,-0.619,1.943,4.510,5.0},{-1.428,-1.761,-1.943,3.951,5.0},{7.142,1.809,-1.659,-0.139,0.0},{-5.714,-1.214,1.375,8.461,10.0},{-1.428,1.738,1.897,-4.650,-5.0}}
        )
    );
    auto test_gradient = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto spacing = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = gradient(policy...,ten,axis,spacing);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2));
        };
        apply_by_element(test,test_data);
    };
    //default policy
    SECTION("test_gradient_default_policy")
    {
        test_gradient();
    }
    //exec_pol<4>
    SECTION("test_gradient_exec_pol<4>")
    {
        test_gradient(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_gradient_exec_pol<0>")
    {
        test_gradient(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_math_gradient_exception","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::gradient;
    //too few points
    REQUIRE_THROWS_AS(gradient(tensor_type{1},0), value_error);
    //coordinates not match size along axis
    REQUIRE_THROWS_AS(gradient(tensor_type{1,2,3,4,5},0,std::vector<double>{1,2}), value_error);
    REQUIRE_THROWS_AS(gradient(tensor_type{1,2,3,4,5},0,std::vector<double>{1,2,3,4,5,6}), value_error);
}

