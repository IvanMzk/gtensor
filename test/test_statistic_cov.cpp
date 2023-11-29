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

TEST_CASE("test_statistic_cov_overload", "[test_statistic]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    tensor_type a{{3,3,3,0,0,1,4,3,0,3,0,2,3,1,0},{3,4,2,1,1,2,0,4,1,1,4,0,1,4,2},{1,3,0,2,3,1,0,2,3,0,3,2,1,4,3},{2,3,2,3,4,3,0,0,3,1,1,0,1,0,3},{2,3,2,2,4,1,4,1,2,4,3,2,0,3,3},{1,4,2,0,4,0,4,3,3,0,2,1,0,4,3},{4,4,1,3,4,1,0,2,1,1,1,2,1,4,2},{0,3,3,1,0,3,0,3,4,3,2,4,1,2,2},{1,3,0,4,2,0,1,4,3,0,4,2,3,3,2},{1,2,0,4,0,1,2,4,4,2,3,3,3,4,1},{1,2,4,4,0,3,1,1,4,2,1,2,1,2,4},{2,2,1,0,0,0,4,2,4,0,0,4,0,2,3},{3,3,3,0,4,0,2,1,4,1,3,2,4,4,4},{3,3,1,3,0,2,1,3,4,4,3,1,0,2,1},{1,2,3,2,1,2,0,1,2,0,0,0,1,4,4}};
    tensor_type expected{{2.20952381,-0.07142857,-1.32380952,-1.07619048,-0.17142857,-0.12380952,-0.40952381,-0.12380952,-0.81904762,-0.28095238,-0.67619048,0.38571429,-0.34761905,-0.12380952,-0.7047619},{-0.07142857,2.14285714,0.85714286,-0.14285714,-0.14285714,0.71428571,0.85714286,0.21428571,0.78571429,0.28571429,-0.28571429,-0.42857143,0.28571429,0.71428571,0.64285714},{-1.32380952,0.85714286,1.6952381,0.39047619,0.2,1.08095238,1.00952381,0.22380952,1.3047619,0.68095238,0.01904762,0.3,0.86190476,0.15238095,0.71904762},{-1.07619048,-0.14285714,0.39047619,1.92380952,0.11428571,0.01904762,0.59047619,-0.1952381,-0.17619048,-0.92380952,0.68095238,-0.61428571,0.22380952,0.01904762,0.65238095},{-0.17142857,-0.14285714,0.2,0.11428571,1.4,0.97142857,0.25714286,-0.38571429,-0.34285714,-0.47142857,-0.27142857,0.24285714,0.27142857,0.11428571,-0.15714286},{-0.12380952,0.71428571,1.08095238,0.01904762,0.97142857,2.63809524,0.63809524,-0.21904762,0.56190476,-0.09047619,-0.43809524,1.24285714,1.24761905,-0.29047619,0.6047619},{-0.40952381,0.85714286,1.00952381,0.59047619,0.25714286,0.63809524,1.92380952,-0.5047619,0.56190476,-0.09047619,-0.36666667,-0.18571429,0.46190476,0.06666667,0.6047619},{-0.12380952,0.21428571,0.22380952,-0.1952381,-0.38571429,-0.21904762,-0.5047619,1.92380952,-0.00952381,0.55238095,0.91904762,0.52857143,-0.32380952,0.70952381,0.24761905},{-0.81904762,0.78571429,1.3047619,-0.17619048,-0.34285714,0.56190476,0.56190476,-0.00952381,2.12380952,1.53333333,-0.23333333,0.05714286,0.35238095,0.34761905,0.06666667},{-0.28095238,0.28571429,0.68095238,-0.92380952,-0.47142857,-0.09047619,-0.09047619,0.55238095,1.53333333,2.06666667,0.1047619,0.47142857,-0.2952381,0.83809524,-0.15238095},{-0.67619048,-0.28571429,0.01904762,0.68095238,-0.27142857,-0.43809524,-0.36666667,0.91904762,-0.23333333,0.1047619,1.83809524,0.34285714,-0.29047619,0.41904762,1.13809524},{0.38571429,-0.42857143,0.3,-0.61428571,0.24285714,1.24285714,-0.18571429,0.52857143,0.05714286,0.47142857,0.34285714,2.54285714,0.58571429,0.02857143,0.15714286},{-0.34761905,0.28571429,0.86190476,0.22380952,0.27142857,1.24761905,0.46190476,-0.32380952,0.35238095,-0.2952381,-0.29047619,0.58571429,2.12380952,-0.68095238,0.62380952},{-0.12380952,0.71428571,0.15238095,0.01904762,0.11428571,-0.29047619,0.06666667,0.70952381,0.34761905,0.83809524,0.41904762,0.02857143,-0.68095238,1.78095238,-0.18095238},{-0.7047619,0.64285714,0.71904762,0.65238095,-0.15714286,0.6047619,0.6047619,0.24761905,0.06666667,-0.15238095,1.13809524,0.15714286,0.62380952,-0.18095238,1.83809524}};
    REQUIRE(tensor_close(cov(a),expected,1E-6,1E-6));
    REQUIRE(tensor_close(cov(multithreading::exec_pol<4>{},a),expected,1E-6,1E-6));
}

TEST_CASE("test_statistic_corrcoef", "[test_statistic]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;

    //0ten,1rowvar,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3,4,5},true,tensor_type(1.0)),
        std::make_tuple(tensor_type{1,2,3,4,5},false,tensor_type(1.0)),
        std::make_tuple(tensor_type{{3,3,3,0,0,1,4,3,0,3},{0,2,3,1,0,3,4,2,1,1},{2,0,4,1,1,4,0,1,4,2},{1,3,0,2,3,1,0,2,3,0},{3,2,1,4,3,2,3,2,3,4}},true,
            tensor_type{{1.0,0.47820953,-0.31186667,-0.61588176,-0.29964438},{0.47820953,1.0,0.03645586,-0.49086755,-0.51665016},{-0.31186667,0.03645586,1.0,-0.19207299,-0.31572444},{-0.61588176,-0.49086755,-0.19207299,1.0,0.04613638},{-0.29964438,-0.51665016,-0.31572444,0.04613638,1.0}}
        ),
        std::make_tuple(tensor_type{{3,0,2,1,3},{3,2,0,3,2},{3,3,4,0,1},{0,1,1,2,4},{0,0,1,3,3},{1,3,4,1,2},{4,4,0,0,3},{3,2,1,2,2},{0,1,4,3,3},{3,1,2,0,4}},false,
            tensor_type{{1.0,0.47820953,-0.31186667,-0.61588176,-0.29964438},{0.47820953,1.0,0.03645586,-0.49086755,-0.51665016},{-0.31186667,0.03645586,1.0,-0.19207299,-0.31572444},{-0.61588176,-0.49086755,-0.19207299,1.0,0.04613638},{-0.29964438,-0.51665016,-0.31572444,0.04613638,1.0}}
        )
    );

    auto test_corrcoef = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){

            auto ten = std::get<0>(t);
            auto rowvar = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = corrcoef(policy...,ten,rowvar);
            REQUIRE(tensor_close(result,expected,1E-6,1E-6));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_corrcoef_default_policy")
    {
        test_corrcoef();
    }
    SECTION("test_corrcoef_exec_pol<4>")
    {
        test_corrcoef(multithreading::exec_pol<4>{});
    }
    SECTION("test_corrcoef_exec_pol<10>")
    {
        test_corrcoef(multithreading::exec_pol<10>{});
    }
}

TEST_CASE("test_statistic_corrcoef_overload", "[test_statistic]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    tensor_type a{{3,3,3,0,0,1,4,3,0,3,0,2,3,1,0},{3,4,2,1,1,2,0,4,1,1,4,0,1,4,2},{1,3,0,2,3,1,0,2,3,0,3,2,1,4,3},{2,3,2,3,4,3,0,0,3,1,1,0,1,0,3},{2,3,2,2,4,1,4,1,2,4,3,2,0,3,3},{1,4,2,0,4,0,4,3,3,0,2,1,0,4,3},{4,4,1,3,4,1,0,2,1,1,1,2,1,4,2},{0,3,3,1,0,3,0,3,4,3,2,4,1,2,2},{1,3,0,4,2,0,1,4,3,0,4,2,3,3,2},{1,2,0,4,0,1,2,4,4,2,3,3,3,4,1},{1,2,4,4,0,3,1,1,4,2,1,2,1,2,4},{2,2,1,0,0,0,4,2,4,0,0,4,0,2,3},{3,3,3,0,4,0,2,1,4,1,3,2,4,4,4},{3,3,1,3,0,2,1,3,4,4,3,1,0,2,1},{1,2,3,2,1,2,0,1,2,0,0,0,1,4,4}};
    tensor_type expected{{1.0,-0.03282661,-0.68400741,-0.52198589,-0.09746975,-0.05128135,-0.1986318,-0.06005147,-0.37809595,-0.13147651,-0.33553336,0.16272542,-0.16047096,-0.0624135,-0.34971082},{-0.03282661,1.0,0.44971901,-0.07035975,-0.08247861,0.30042088,0.42215853,0.10553963,0.36830724,0.13576885,-0.14396315,-0.18359702,0.13392991,0.36563621,0.32391709},{-0.68400741,0.44971901,1.0,0.21622115,0.1298227,0.51114719,0.55901078,0.12393163,0.68763525,0.36380291,0.01079049,0.14449237,0.45424081,0.0876979,0.40734116},{-0.52198589,-0.07035975,0.21622115,1.0,0.0696381,0.00845502,0.30693069,-0.10148515,-0.0871653,-0.46330414,0.36211933,-0.27773358,0.11072348,0.01029043,0.34692551},{-0.09746975,-0.08247861,0.1298227,0.0696381,1.0,0.50547726,0.15668572,-0.23502858,-0.19883434,-0.27715114,-0.1692028,0.12871403,0.15741052,0.0723772,-0.09795951},{-0.05128135,0.30042088,0.51114719,0.00845502,0.50547726,1.0,0.28324303,-0.09723268,0.23738851,-0.03874841,-0.19894787,0.47986049,0.52708296,-0.1340106,0.27463456},{-0.1986318,0.42215853,0.55901078,0.30693069,0.15668572,0.28324303,1.0,-0.26237624,0.27798662,-0.04537515,-0.19498733,-0.08396597,0.22851443,0.0360165,0.32160248},{-0.06005147,0.10553963,0.12393163,-0.10148515,-0.23502858,-0.09723268,-0.26237624,1.0,-0.00471164,0.27702722,0.48873447,0.23898006,-0.16019568,0.3833185,0.13167975},{-0.37809595,0.36830724,0.68763525,-0.0871653,-0.19883434,0.23738851,0.27798662,-0.00471164,1.0,0.73188623,-0.11809595,0.02458913,0.16591928,0.17873913,0.0337417},{-0.13147651,0.13576885,0.36380291,-0.46330414,-0.27715114,-0.03874841,-0.04537515,0.27702722,0.73188623,1.0,0.0537507,0.20564573,-0.14092219,0.43684966,-0.07818284},{-0.33553336,-0.14396315,0.01079049,0.36211933,-0.1692028,-0.19894787,-0.19498733,0.48873447,-0.11809595,0.0537507,1.0,0.15858723,-0.14701741,0.23160782,0.61917098},{0.16272542,-0.18359702,0.14449237,-0.27773358,0.12871403,0.47986049,-0.08396597,0.23898006,0.02458913,0.20564573,0.15858723,1.0,0.2520386,0.01342594,0.07268581},{-0.16047096,0.13392991,0.45424081,0.11072348,0.15741052,0.52708296,0.22851443,-0.16019568,0.16591928,-0.14092219,-0.14701741,0.2520386,1.0,-0.35013281,0.3157259},{-0.0624135,0.36563621,0.0876979,0.01029043,0.0723772,-0.1340106,0.0360165,0.3833185,0.17873913,0.43684966,0.23160782,0.01342594,-0.35013281,1.0,-0.10001247},{-0.34971082,0.32391709,0.40734116,0.34692551,-0.09795951,0.27463456,0.32160248,0.13167975,0.0337417,-0.07818284,0.61917098,0.07268581,0.3157259,-0.10001247,1.0}};
    REQUIRE(tensor_close(corrcoef(a),expected,1E-6,1E-6));
    REQUIRE(tensor_close(corrcoef(multithreading::exec_pol<4>{},a),expected,1E-6,1E-6));
}

TEST_CASE("test_statistic_cov_corrcoef_exception", "[test_statistic]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    REQUIRE_THROWS_AS(cov(tensor_type(1)),value_error);
    REQUIRE_THROWS_AS(cov(tensor_type{{{2}}}),value_error);
    REQUIRE_THROWS_AS(corrcoef(tensor_type(1)),value_error);
    REQUIRE_THROWS_AS(corrcoef(tensor_type{{{2}}}),value_error);
}
