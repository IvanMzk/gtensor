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

TEST_CASE("test_statistic_cov", "[test_statistic]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;

    //0ten,1rowvar,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{2,3},true,tensor_type(0.5)),
        std::make_tuple(tensor_type{2,3},true,tensor_type(0.5)),
        std::make_tuple(tensor_type{1,2,3,4,5},true,tensor_type(2.5)),
        std::make_tuple(tensor_type{1,2,3,4,5},false,tensor_type(2.5)),
        std::make_tuple(tensor_type{{3,3,3,0,0,1,4,3,0,3},{0,2,3,1,0,3,4,2,1,1},{2,0,4,1,1,4,0,1,4,2},{1,3,0,2,3,1,0,2,3,0},{3,2,1,4,3,2,3,2,3,4}},true,
            tensor_type{{2.44444444,1.0,-0.77777778,-1.22222222,-0.44444444},{1.0,1.78888889,0.07777778,-0.83333333,-0.65555556},{-0.77777778,0.07777778,2.54444444,-0.38888889,-0.47777778},{-1.22222222,-0.83333333,-0.38888889,1.61111111,0.05555556},{-0.44444444,-0.65555556,-0.47777778,0.05555556,0.9}}
        ),
        std::make_tuple(tensor_type{{3,0,2,1,3},{3,2,0,3,2},{3,3,4,0,1},{0,1,1,2,4},{0,0,1,3,3},{1,3,4,1,2},{4,4,0,0,3},{3,2,1,2,2},{0,1,4,3,3},{3,1,2,0,4}},false,
            tensor_type{{2.44444444,1.0,-0.77777778,-1.22222222,-0.44444444},{1.0,1.78888889,0.07777778,-0.83333333,-0.65555556},{-0.77777778,0.07777778,2.54444444,-0.38888889,-0.47777778},{-1.22222222,-0.83333333,-0.38888889,1.61111111,0.05555556},{-0.44444444,-0.65555556,-0.47777778,0.05555556,0.9}}
        )
    );

    auto test_cov = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){

            auto ten = std::get<0>(t);
            auto rowvar = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = cov(policy...,ten,rowvar);
            REQUIRE(tensor_close(result,expected,1E-6,1E-6));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_cov_default_policy")
    {
        test_cov();
    }
    SECTION("test_cov_exec_pol<4>")
    {
        test_cov(multithreading::exec_pol<4>{});
    }
    SECTION("test_cov_exec_pol<10>")
    {
        test_cov(multithreading::exec_pol<10>{});
    }
}