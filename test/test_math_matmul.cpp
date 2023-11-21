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

// TEMPLATE_TEST_CASE("test_math_matmul","test_math",
//     (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::f_order>),
//     (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::c_order>)
// )
// {
//     using layout1 = std::tuple_element_t<0,TestType>;
//     using layout2 = std::tuple_element_t<1,TestType>;
//     using value_type = double;
//     using tensor_type1 = gtensor::tensor<value_type,layout1>;
//     using tensor_type2 = gtensor::tensor<value_type,layout2>;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::matmul;
//     using helpers_for_testing::apply_by_element;

//     //0ten_a,1ten_b,2expected
//     auto test_data = std::make_tuple(
//         // //1d x 1d
//         std::make_tuple(tensor_type1{},tensor_type2{},tensor_type(0)),
//         std::make_tuple(tensor_type1{1,2,3,4},tensor_type2{5,6,7,8},tensor_type(70)),
//         //1d x nd
//         std::make_tuple(tensor_type1{1,2,3},tensor_type2{{1,2,3,4},{4,5,6,7},{7,8,9,10}},tensor_type{30,36,42,48}),
//         std::make_tuple(tensor_type1{1,2,3},tensor_type2{{{1,2,3,4},{4,5,6,7},{7,8,9,10}}},tensor_type{{30,36,42,48}}),
//         std::make_tuple(tensor_type1{1,2,3},tensor_type2{{{1,2,3,4},{4,5,6,7},{7,8,9,10}},{{2,3,4,5},{5,6,7,8},{8,9,10,11}}},tensor_type{{30,36,42,48},{36,42,48,54}}),
//         std::make_tuple(tensor_type1{1,2,3},tensor_type2{{{{1,2,3,4},{4,5,6,7},{7,8,9,10}}},{{{2,3,4,5},{5,6,7,8},{8,9,10,11}}}},tensor_type{{{30,36,42,48}},{{36,42,48,54}}}),
//         std::make_tuple(tensor_type1{2,4},
//             tensor_type2{{{{0,0,1},{4,1,4}},{{3,4,0},{0,4,0}}},{{{3,1,4},{2,0,0}},{{1,1,4},{2,3,2}}},{{{2,2,3},{1,3,1}},{{3,2,3},{1,1,0}}}},
//             tensor_type{{{16,4,18},{6,24,0}},{{14,2,8},{10,14,16}},{{8,16,10},{10,8,6}}}
//         ),
//         //nd x 1d
//         std::make_tuple(tensor_type1{{1,2,3,4},{4,5,6,7},{7,8,9,10}},tensor_type2{4,3,2,1},tensor_type{20,50,80}),
//         std::make_tuple(tensor_type1{{{1,2,3,4},{4,5,6,7},{7,8,9,10}}},tensor_type2{4,3,2,1},tensor_type{{20,50,80}}),
//         std::make_tuple(tensor_type1{{{1,2,3,4},{4,5,6,7},{7,8,9,10}},{{2,3,4,5},{5,6,7,8},{8,9,10,11}}},tensor_type2{4,3,2,1},tensor_type{{20,50,80},{30,60,90}}),
//         std::make_tuple(tensor_type1{{{{1,2,3,4},{4,5,6,7},{7,8,9,10}}},{{{2,3,4,5},{5,6,7,8},{8,9,10,11}}}},tensor_type2{4,3,2,1},tensor_type{{{20,50,80}},{{30,60,90}}}),
//         std::make_tuple(tensor_type1{{{{0,0,1},{4,1,4}},{{3,4,0},{0,4,0}}},{{{3,1,4},{2,0,0}},{{1,1,4},{2,3,2}}},{{{2,2,3},{1,3,1}},{{3,2,3},{1,1,0}}}},
//             tensor_type2{2,0,3},
//             tensor_type{{{3,20},{6,0}},{{18,4},{14,10}},{{13,5},{15,2}}}
//         ),
//         //nd x nd
//         std::make_tuple(tensor_type1{{1,2,3}},tensor_type2{{7,8,1,2},{2,3,4,5},{5,6,8,9}},tensor_type{{26,32,33,39}}),
//         std::make_tuple(tensor_type1{{7,8,1,2},{2,3,4,5},{5,6,8,9}},tensor_type2{{1},{0},{2},{2}},tensor_type{{13},{20},{39}}),
//         std::make_tuple(tensor_type1{{{1,2}}},
//             tensor_type2{{{{0,0,1},{4,1,4}},{{3,4,0},{0,4,0}}},{{{3,1,4},{2,0,0}},{{1,1,4},{2,3,2}}},{{{2,2,3},{1,3,1}},{{3,2,3},{1,1,0}}}},
//             tensor_type{{{{8,2,9}},{{3,12,0}}},{{{7,1,4}},{{5,7,8}}},{{{4,8,5}},{{5,4,3}}}}
//         ),
//         std::make_tuple(tensor_type1{{{{0,0,1},{4,1,4}},{{3,4,0},{0,4,0}}},{{{3,1,4},{2,0,0}},{{1,1,4},{2,3,2}}},{{{2,2,3},{1,3,1}},{{3,2,3},{1,1,0}}}},
//             tensor_type2{{1},{0},{2}},
//             tensor_type{{{{2},{12}},{{3},{0}}},{{{11},{2}},{{9},{6}}},{{{8},{3}},{{9},{1}}}}
//         ),
//         std::make_tuple(tensor_type1{{1,2,3},{4,5,6}},tensor_type2{{7,8,1,2},{2,3,4,5},{5,6,8,9}},tensor_type{{26,32,33,39},{68,83,72,87}}),
//         std::make_tuple(tensor_type1{{1,2,3},{4,5,6}},tensor_type2{{{1,2,3},{0,1,2},{2,3,4}},{{0,1,2},{3,4,5},{4,5,6}}},tensor_type{{{7,13,19},{16,31,46}},{{18,24,30},{39,54,69}}}),
//         std::make_tuple(tensor_type1{{0,2},{3,2},{1,3},{2,2}},
//             tensor_type2{{{{0,0,1},{4,1,4}},{{3,4,0},{0,4,0}}},{{{3,1,4},{2,0,0}},{{1,1,4},{2,3,2}}},{{{2,2,3},{1,3,1}},{{3,2,3},{1,1,0}}}},
//             tensor_type{{{{8,2,8},{8,2,11},{12,3,13},{8,2,10}},{{0,8,0},{9,20,0},{3,16,0},{6,16,0}}},{{{4,0,0},{13,3,12},{9,1,4},{10,2,8}},{{4,6,4},{7,9,16},{7,10,10},{6,8,12}}},{{{2,6,2},{8,12,11},{5,11,6},{6,10,8}},{{2,2,0},{11,8,9},{6,5,3},{8,6,6}}}}
//         ),
//         std::make_tuple(tensor_type1{{{{0,0,1},{4,1,4}},{{3,4,0},{0,4,0}}},{{{3,1,4},{2,0,0}},{{1,1,4},{2,3,2}}},{{{2,2,3},{1,3,1}},{{3,2,3},{1,1,0}}}},
//             tensor_type2{{1,1,2,4},{4,2,3,4},{0,2,0,2}},
//             tensor_type{{{{0,2,0,2},{8,14,11,28}},{{19,11,18,28},{16,8,12,16}}},{{{7,13,9,24},{2,2,4,8}},{{5,11,5,16},{14,12,13,24}}},{{{10,12,10,22},{13,9,11,18}},{{11,13,12,26},{5,3,5,8}}}}
//         ),
//         std::make_tuple(tensor_type1{{{{0,0,1},{4,1,4}},{{3,4,0},{0,4,0}}},{{{3,1,4},{2,0,0}},{{1,1,4},{2,3,2}}},{{{2,2,3},{1,3,1}},{{3,2,3},{1,1,0}}}},
//             tensor_type2{{{{2,2},{3,4},{1,1}}},{{{2,4},{2,2},{0,1}}},{{{4,2},{4,4},{4,1}}}},
//             tensor_type{{{{1,1},{15,16}},{{18,22},{12,16}}},{{{8,18},{4,8}},{{4,10},{10,16}}},{{{28,15},{20,15}},{{32,17},{8,6}}}}
//         )
//     );

//     auto test_matmul = [&test_data](auto...policy){
//         auto test = [policy...](const auto& t){

//             auto ten1 = std::get<0>(t);
//             auto ten2 = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto result = matmul(policy...,ten1,ten2);
//             REQUIRE(result==expected);
//         };
//         apply_by_element(test,test_data);
//     };

//     SECTION("test_matmul_default_policy")
//     {
//         test_matmul();
//     }
//     SECTION("test_matmul_exec_pol<2>")
//     {
//         test_matmul(multithreading::exec_pol<2>{});
//     }
//     SECTION("test_matmul_exec_pol<3>")
//     {
//         test_matmul(multithreading::exec_pol<3>{});
//     }
//     SECTION("test_matmul_exec_pol<4>")
//     {
//         test_matmul(multithreading::exec_pol<4>{});
//     }
//     SECTION("test_matmul_exec_pol<5>")
//     {
//         test_matmul(multithreading::exec_pol<5>{});
//     }
//     SECTION("test_matmul_exec_pol<10>")
//     {
//         test_matmul(multithreading::exec_pol<10>{});
//     }
//     SECTION("test_matmul_exec_pol<16>")
//     {
//         test_matmul(multithreading::exec_pol<16>{});
//     }
// }

// TEST_CASE("test_math_matmul_exception","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::value_error;
//     using gtensor::matmul;
//     using helpers_for_testing::apply_by_element;

//     //scalar arg
//     REQUIRE_THROWS_AS(matmul(tensor_type(1),tensor_type(2)),value_error);
//     REQUIRE_THROWS_AS(matmul(tensor_type{1,2,3},tensor_type(2)),value_error);
//     REQUIRE_THROWS_AS(matmul(tensor_type(2),tensor_type{1,2,3}),value_error);
//     //incompatible matices
//     REQUIRE_THROWS_AS(matmul(tensor_type{1,2,3,4},tensor_type{1,2,3}),value_error);
//     REQUIRE_THROWS_AS(matmul(tensor_type{1,2,3},tensor_type{1,2,3,4}),value_error);
//     REQUIRE_THROWS_AS(matmul(tensor_type{1,2,3},tensor_type{{1,2,3},{3,2,1}}),value_error);
//     REQUIRE_THROWS_AS(matmul(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{3,2,1}}),value_error);
//     //not broadcast
//     REQUIRE_THROWS_AS(matmul(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}),value_error);
//     REQUIRE_THROWS_AS(matmul(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),value_error);
// }

TEMPLATE_TEST_CASE("test_math_matmul_big","test_math",
    // (std::tuple<gtensor::config::c_order,gtensor::config::c_order,double>),
    // (std::tuple<gtensor::config::f_order,gtensor::config::f_order,double>),
    // (std::tuple<gtensor::config::c_order,gtensor::config::f_order,double>),
    // (std::tuple<gtensor::config::f_order,gtensor::config::c_order,double>),
    // (std::tuple<gtensor::config::c_order,gtensor::config::c_order,float>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order,float>)
    // (std::tuple<gtensor::config::c_order,gtensor::config::f_order,float>),
    // (std::tuple<gtensor::config::f_order,gtensor::config::c_order,float>)
)
{
    using layout1 = std::tuple_element_t<0,TestType>;
    using layout2 = std::tuple_element_t<1,TestType>;
    using value_type = std::tuple_element_t<2,TestType>;
    using tensor_type1 = gtensor::tensor<value_type,layout1>;
    using tensor_type2 = gtensor::tensor<value_type,layout2>;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::matmul;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;

    const auto m = 1025;
    const auto n = 1029;
    const auto k = 255;

    tensor_type1 a({m,k},0);
    tensor_type2 b({k,n},0);
    tensor_type res({m,n},0);

    helpers_for_testing::generate_lehmer(a.begin(),a.end(),[](auto e){return e%3;},123);
    helpers_for_testing::generate_lehmer(b.begin(),b.end(),[](auto e){return e%3;},456);

    for (auto i=0; i!=m; ++i){
        for (auto j=0; j!=n; ++j){
            for (auto r=0; r!=k; ++r){
                res.element(i,j)+=a.element(i,r)*b.element(r,j);
            }
        }
    }

    REQUIRE(res==matmul(a,b));
    // REQUIRE(res==matmul(multithreading::exec_pol<4>{},a,b));
    // REQUIRE(res==matmul(multithreading::exec_pol<10>{},a,b));
    // REQUIRE(res==matmul(multithreading::exec_pol<16>{},a,b));
}

