/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <tuple>
#include <vector>
#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"


TEST_CASE("test_tensor_view_interface","[test_tensor]")
{

    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using shape_type = typename tensor_type::shape_type;
    using slice_type = typename tensor_type::slice_type;
    using nop_type = typename slice_type::nop_type;
    using helpers_for_testing::apply_by_element;
    const nop_type nop;
    //0result,1expected
    auto test_data = std::make_tuple(
        //slice view
        //init-list subs
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{nop,nop,-1},{0,-1}}), tensor_type{{4,5},{1,2}}),
        //variadic slices subs
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(slice_type{},slice_type{1}), tensor_type{{2,3},{5,6}}),
        //variadic mixed subs
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(slice_type{},1), tensor_type{2,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(1,slice_type{}), tensor_type{4,5,6}),
        //variadic index subs
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(0), tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(0,1), tensor_type(2)),
        //slice container subs
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(std::vector<slice_type>{slice_type{},slice_type{1,-1}}), tensor_type{{2},{5}}),
        //transpose view
        //variadic, no subs
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(), tensor_type{{{1,5},{3,7}},{{2,6},{4,8}}}),
        //variadic, index subs
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(1,0,2), tensor_type{{{1,2},{5,6}},{{3,4},{7,8}}}),
        //container subs
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(std::vector<int>{2,0,1}), tensor_type{{{1,3},{5,7}},{{2,4},{6,8}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose({2,0,1}), tensor_type{{{1,3},{5,7}},{{2,4},{6,8}}}),
        //reshape view
        //variadic, no subs
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(), tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        //variadic, index subs
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(-1,1), tensor_type{{1},{2},{3},{4},{5},{6},{7},{8}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(-1,4), tensor_type{{1,2,3,4},{5,6,7,8}}),
        //container subs
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(std::vector<int>{2,-1}), tensor_type{{1,2,3,4},{5,6,7,8}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(shape_type{2,2,2}, f_order{}), tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(shape_type{-1,4}, f_order{}), tensor_type{{1,3,2,4},{5,7,6,8}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape({-1,2}, f_order{}), tensor_type{{1,2},{5,6},{3,4},{7,8}}),
        //index mapping view
        //variadic subs
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(tensor<int>{1,0},tensor<int>{0,1}), tensor_type{{5,6},{3,4}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(tensor<int>(1),tensor<int>{{0,1},{1,0}}), tensor_type{{{5,6},{7,8}},{{7,8},{5,6}}}),
        //container subs
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(std::vector<tensor<int>>{tensor<int>{1,0},tensor<int>{0,1}}), tensor_type{{5,6},{3,4}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(std::vector<tensor<int>>{tensor<int>(1),tensor<int>{{0,1},{1,0}}}), tensor_type{{{5,6},{7,8}},{{7,8},{5,6}}}),
        //bool mapping view
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(tensor<bool>{{{true,false},{false,true}}}), tensor_type{1,4})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);

        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_view_chain","[test_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type=double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using index_type = typename tensor_type::index_type;
    using slice_type = typename tensor_type::slice_type;
    using helpers_for_testing::apply_by_element;

    const auto input = tensor_type{
        {{{{7,5,8,5},{0,5,5,1},{3,8,0,8}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}}}},
        {{{{4,8,0,7},{0,0,2,4},{1,5,8,5}}},{{{6,8,4,8},{4,1,3,2},{7,0,6,2}}}},
        {{{{7,3,6,4},{2,6,4,7},{0,3,3,1}}},{{{2,1,3,0},{4,7,4,4},{7,6,3,3}}}}
    };  //(3,2,1,3,4)

    //0tensor,1view_maker,2expected
    auto test_data = std::make_tuple(
        //transpose
        std::make_tuple(input,[](const auto& t_){return t_.transpose().transpose(1,0,2,4,3);},
            tensor_type{{{{{7,0},{4,6},{7,2}}},{{{5,0},{8,8},{3,1}}},{{{8,2},{0,4},{6,3}}},{{{5,5},{7,8},{4,0}}}},{{{{0,1},{0,4},{2,4}}},{{{5,2},{0,1},{6,7}}},{{{5,3},{2,3},{4,4}}},{{{1,0},{4,2},{7,4}}}},{{{{3,6},{1,7},{0,7}}},{{{8,7},{5,0},{3,6}}},{{{0,3},{8,6},{3,3}}},{{{8,7},{5,2},{1,3}}}}}
        ),
        //slice
        std::make_tuple(input,[](const auto& t_){return t_(1,slice_type{},0,slice_type{{},{},-1},slice_type{{},-1})(slice_type{},1,slice_type{{},{},2});},
            tensor_type{{0,2},{4,3}}
        ),
        //reshape
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,6,4},c_order{}).reshape({3,24},c_order{});},
            tensor_type{{7,5,8,5,0,5,5,1,3,8,0,8,0,0,2,5,1,2,3,0,6,7,3,7},{4,8,0,7,0,0,2,4,1,5,8,5,6,8,4,8,4,1,3,2,7,0,6,2},{7,3,6,4,2,6,4,7,0,3,3,1,2,1,3,0,4,7,4,4,7,6,3,3}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,6,4},f_order{}).reshape({3,24},f_order{});},
            tensor_type{{7,0,0,1,3,6,5,0,5,2,8,7,8,2,5,3,0,3,5,5,1,0,8,7},{4,6,0,4,1,7,8,8,0,1,5,0,0,4,2,3,8,6,7,8,4,2,5,2},{7,2,2,4,0,7,3,1,6,7,3,6,6,3,4,4,3,3,4,0,7,4,1,3}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,6,4},c_order{}).reshape({3,24},f_order{});},
            tensor_type{{7,0,3,0,1,6,5,5,8,0,2,7,8,5,0,2,3,3,5,1,8,5,0,7},{4,0,1,6,4,7,8,0,5,8,1,0,0,2,8,4,3,6,7,4,5,8,2,2},{7,2,0,2,4,7,3,6,3,1,7,6,6,4,3,3,4,3,4,7,1,0,4,3}}
        ),
        //mapping
        std::make_tuple(
            input,
            [](const auto& t_){
                return t_(tensor<index_type,f_order>{1,0,2,0},tensor<index_type,c_order>{0,1,0,1},tensor<index_type,c_order>{0})(tensor<index_type,c_order>{{2,1},{1,0}},tensor<index_type,c_order>{0,2});
            },
            tensor_type{{{7,3,6,4},{6,7,3,7}},{{0,0,2,5},{1,5,8,5}}}
        ),
        std::make_tuple(
            input,
            [](const auto& t_){
                return t_(tensor<index_type,f_order>{1,0,2,0},tensor<index_type,c_order>{0,1,0,1},tensor<index_type,c_order>{0})(tensor<bool,f_order>{{true,false,true},{false,false,false},{false,false,true},{true,false,true}});
            },
            tensor_type{{4,8,0,7},{1,5,8,5},{0,3,3,1},{0,0,2,5},{6,7,3,7}}
        ),
        std::make_tuple(
            input,
            [](const auto& t_){
                return t_(tensor<bool,c_order>{{{false},{false}},{{true},{false}},{{true},{false}}})(tensor<bool,f_order>{{{true,false,false,false},{true,true,true,true},{true,true,false,false}},{{false,false,true,false},{false,false,true,false},{true,false,false,false}}});
            },
            tensor_type{4,0,0,2,4,1,5,6,4,0}
        ),
        std::make_tuple(
            input,
            [](const auto& t_){
                return t_(tensor<bool,c_order>{{{false},{false}},{{true},{false}},{{true},{false}}})(tensor<index_type,f_order>{1,0,0},tensor<index_type,f_order>{{2,0,1},{1,0,2}});
            },
            tensor_type{{{0,3,3,1},{4,8,0,7},{0,0,2,4}},{{2,6,4,7},{4,8,0,7},{1,5,8,5}}}
        ),
        //transpose reshape
        std::make_tuple(input,[](const auto& t_){return t_.transpose(2,0,1,3,4).reshape({6,12},c_order{});},
            tensor_type{{7,5,8,5,0,5,5,1,3,8,0,8},{0,0,2,5,1,2,3,0,6,7,3,7},{4,8,0,7,0,0,2,4,1,5,8,5},{6,8,4,8,4,1,3,2,7,0,6,2},{7,3,6,4,2,6,4,7,0,3,3,1},{2,1,3,0,4,7,4,4,7,6,3,3}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.transpose(2,0,1,3,4).reshape({6,12},f_order{});},
            tensor_type{{7,0,3,5,5,8,8,5,0,5,1,8},{4,0,1,8,0,5,0,2,8,7,4,5},{7,2,0,3,6,3,6,4,3,4,7,1},{0,1,6,0,2,7,2,3,3,5,0,7},{6,4,7,8,1,0,4,3,6,8,2,2},{2,4,7,1,7,6,3,4,3,0,4,3}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,1,4,6},c_order{}).transpose(2,0,1,3);},
            tensor_type{{{{7,5,8,5,0,5}},{{4,8,0,7,0,0}},{{7,3,6,4,2,6}}},{{{5,1,3,8,0,8}},{{2,4,1,5,8,5}},{{4,7,0,3,3,1}}},{{{0,0,2,5,1,2}},{{6,8,4,8,4,1}},{{2,1,3,0,4,7}}},{{{3,0,6,7,3,7}},{{3,2,7,0,6,2}},{{4,4,7,6,3,3}}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,1,4,6},f_order{}).transpose(2,0,1,3);},
            tensor_type{{{{7,3,5,8,0,1}},{{4,1,0,0,8,4}},{{7,0,6,6,3,7}}},{{{0,6,2,2,3,0}},{{6,7,1,4,6,2}},{{2,7,7,3,3,4}}},{{{0,5,8,5,5,8}},{{0,8,5,2,7,5}},{{2,3,3,4,4,1}}},{{{1,0,7,3,5,7}},{{4,8,0,3,8,2}},{{4,1,6,4,0,3}}}}
        ),
        //transpose slice
        std::make_tuple(input,[](const auto& t_){return t_({{},{1},{},{{},{},-1},{1,-1}}).transpose(3,2,1,0,4)({{{},{},2},{},{},{{},{},2},{}});},
            tensor_type{{{{{7,3},{6,3}}}},{{{{0,2},{1,3}}}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.transpose(3,2,1,4,0)(slice_type{},0,slice_type{},2,slice_type{{},{},2});},
            tensor_type{{{8,6},{2,3}},{{5,4},{3,4}},{{0,3},{3,3}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(2,1,0).transpose();},
            tensor_type{{2,4,7},{1,7,6},{3,4,3},{0,4,3}}
        ),
        //slice reshape
        std::make_tuple(input,[](const auto& t_){return t_.reshape({6,3,4},c_order{})(slice_type{},1,slice_type{1,{},2});},
            tensor_type{{5,1},{2,0},{0,4},{1,2},{6,7},{7,4}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({6,3,4},f_order{})(slice_type{},1,slice_type{1,{},2});},
            tensor_type{{5,1},{0,4},{6,7},{2,0},{1,2},{7,4}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(1,slice_type{},0,slice_type{{},{},-1}).reshape({-1},c_order{});},
            tensor_type{1,5,8,5,0,0,2,4,4,8,0,7,7,0,6,2,4,1,3,2,6,8,4,8}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(1,slice_type{},0,slice_type{{},{},-1}).reshape({-1},f_order{});},
            tensor_type{1,7,0,4,4,6,5,0,0,1,8,8,8,6,2,3,0,4,5,2,4,2,7,8}
        ),
        //mapping transpose
        std::make_tuple(input,[](const auto& t_){return t_.transpose(2,0,1,3,4)(tensor<index_type,c_order>{0},tensor<index_type,c_order>{0,2,1},tensor<index_type,c_order>{{0,1,1},{1,1,0}});},
            tensor_type{{{{7,5,8,5},{0,5,5,1},{3,8,0,8}},{{2,1,3,0},{4,7,4,4},{7,6,3,3}},{{6,8,4,8},{4,1,3,2},{7,0,6,2}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}},{{2,1,3,0},{4,7,4,4},{7,6,3,3}},{{4,8,0,7},{0,0,2,4},{1,5,8,5}}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.transpose(2,0,1,3,4)(tensor<bool>{{{{true,true,false},{true,true,true}},{{false,true,true},{true,false,false}},{{false,true,false},{false,true,false}}}});},
            tensor_type{{7,5,8,5},{0,5,5,1},{0,0,2,5},{1,2,3,0},{6,7,3,7},{0,0,2,4},{1,5,8,5},{6,8,4,8},{2,6,4,7},{4,7,4,4}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(tensor<index_type,c_order>{1,2,0},tensor<index_type,c_order>{{0,1,0},{1,0,0}}).transpose();},
            tensor_type{{{{{4,6},{2,7},{7,7}}},{{{0,4},{4,2},{0,0}}},{{{1,7},{7,0},{3,3}}}},{{{{8,8},{1,3},{5,5}}},{{{0,1},{7,6},{5,5}}},{{{5,0},{6,3},{8,8}}}},{{{{0,4},{3,6},{8,8}}},{{{2,3},{4,4},{5,5}}},{{{8,6},{3,3},{0,0}}}},{{{{7,8},{0,4},{5,5}}},{{{4,2},{4,7},{1,1}}},{{{5,2},{3,1},{8,8}}}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(tensor<bool>{{false,false},{true,true},{true,false}}).transpose();},
            tensor_type{{{{4,6,7}},{{0,4,2}},{{1,7,0}}},{{{8,8,3}},{{0,1,6}},{{5,0,3}}},{{{0,4,6}},{{2,3,4}},{{8,6,3}}},{{{7,8,4}},{{4,2,7}},{{5,2,1}}}}
        ),
        //mapping slice
        std::make_tuple(input,[](const auto& t_){return t_({{},{1},{},{{},{},-1},{1,-1}})(tensor<index_type,c_order>{1,0,2,1},tensor<index_type>{0},tensor<index_type>{0},tensor<index_type,f_order>{1,0,1,2});},
            tensor_type{{1,3},{7,3},{7,4},{8,4}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_({{},{1},{},{{},{},-1},{1,-1}})(tensor<bool>{{{{true,false,true}}},{{{true,false,true}}},{{{true,true,false}}}});},
            tensor_type{{7,3},{0,2},{0,6},{8,4},{6,3},{7,4}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(tensor<index_type,c_order>{1,2,0},tensor<index_type,c_order>{{0,1,0},{1,0,0}})({{1},{{},{},2},{},{3,{},-1}});},
            tensor_type{{{{{7,0,6,2},{4,1,3,2},{6,8,4,8}}},{{{3,8,0,8},{0,5,5,1},{7,5,8,5}}}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(tensor<bool>{{false,false},{true,true},{true,false}})({{{},-1},{},{},{1,-1}});},
            tensor_type{{{{8,0},{0,2},{5,8}}},{{{8,4},{1,3},{0,6}}}}
        ),
        //mapping reshape
        std::make_tuple(input,[](const auto& t_){return t_.reshape({6,3,4},c_order{})(tensor<index_type,f_order>{{3,5,1},{0,2,3}},tensor<index_type,c_order>{0,2,1});},
            tensor_type{{{6,8,4,8},{7,6,3,3},{1,2,3,0}},{{7,5,8,5},{1,5,8,5},{4,1,3,2}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({6,3,4},f_order{})(tensor<bool,f_order>{{true,false,true},{true,true,true},{false,false,true},{false,false,true},{false,true,false},{false,true,false}});},
            tensor_type{{7,5,8,5},{3,8,0,8},{4,8,0,7},{0,0,2,4},{1,5,8,5},{0,3,3,1},{6,7,3,7},{4,1,3,2},{4,7,4,4}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(tensor<index_type,c_order>{1,2,0},tensor<index_type,c_order>{{0,1,0},{1,0,0}}).reshape({9,8},f_order{});},
            tensor_type{{4,2,8,6,0,4,7,7},{6,0,8,5,4,5,8,1},{2,0,1,5,3,5,0,1},{7,1,3,5,6,8,4,5},{7,7,5,0,8,6,5,2},{7,7,5,6,8,3,5,3},{0,0,0,3,2,3,4,1},{4,3,1,8,3,0,2,8},{4,3,7,8,4,0,4,8}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(tensor<bool>{{false,false},{true,true},{true,false}}).reshape({4,9},c_order{});},
            tensor_type{{4,8,0,7,0,0,2,4,1},{5,8,5,6,8,4,8,4,1},{3,2,7,0,6,2,7,3,6},{4,2,6,4,7,0,3,3,1}}
        ),
        //expression input
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_.reshape({-1},c_order{});},
            tensor_type{21,19,22,19,17,20,19,17,20,22,20,23,20,18,21,20,18,21,17,15,18,15,13,16,15,13,16}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_.reshape({-1},f_order{});},
            tensor_type{21,22,17,19,20,15,19,20,15,19,20,15,17,18,13,17,18,13,22,23,18,20,21,16,20,21,16}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_.transpose();},
            tensor_type{{{21,22,17},{19,20,15},{19,20,15}},{{19,20,15},{17,18,13},{17,18,13}},{{22,23,18},{20,21,16},{20,21,16}}}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_(slice_type{1},0,slice_type{1,{},-1});},
            tensor_type{{20,22},{15,17}}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_(tensor<index_type,c_order>{{1,2,0},{2,0,1}});},
            tensor_type{{{{22,20,23},{20,18,21},{20,18,21}},{{17,15,18},{15,13,16},{15,13,16}},{{21,19,22},{19,17,20},{19,17,20}}},{{{17,15,18},{15,13,16},{15,13,16}},{{21,19,22},{19,17,20},{19,17,20}},{{22,20,23},{20,18,21},{20,18,21}}}}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_(tensor<index_type,f_order>{{1,2,0},{2,0,1}});},
            tensor_type{{{{22,20,23},{20,18,21},{20,18,21}},{{17,15,18},{15,13,16},{15,13,16}},{{21,19,22},{19,17,20},{19,17,20}}},{{{17,15,18},{15,13,16},{15,13,16}},{{21,19,22},{19,17,20},{19,17,20}},{{22,20,23},{20,18,21},{20,18,21}}}}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_(tensor<index_type,f_order>{{1,2,0},{2,0,1}},tensor<index_type,c_order>{{2,1,0}},tensor<index_type,f_order>{{1,0,2},{0,2,1}});},
            tensor_type{{18,15,22},{15,20,20}}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_(tensor<bool,c_order>{{{false,true,false},{true,false,false},{false,false,false}},{{false,false,true},{false,false,false},{true,false,false}},{{true,false,false},{false,true,true},{false,true,false}}});},
            tensor_type{19,19,23,20,17,13,16,13}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_(tensor<bool,f_order>{{{false,true,false},{true,false,false},{false,false,false}},{{false,false,true},{false,false,false},{true,false,false}},{{true,false,false},{false,true,true},{false,true,false}}});},
            tensor_type{19,19,23,20,17,13,16,13}
        )
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto view_maker = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = view_maker(ten);
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_ravel","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;

    //0tensor,1order,2expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(tensor_type(2),c_order{},tensor_type{2}),
        std::make_tuple(tensor_type{},c_order{},tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),c_order{},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},c_order{},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},c_order{},tensor_type{1,2,3,4,5,6,7,8,9,10,11,12}),
        //f_order
        std::make_tuple(tensor_type(2),f_order{},tensor_type{2}),
        std::make_tuple(tensor_type{},f_order{},tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),f_order{},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},f_order{},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},f_order{},tensor_type{1,7,4,10,2,8,5,11,3,9,6,12})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto order = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = ten.ravel(order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}