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

//value assignment operator=
TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_result","[test_tensor]",
    //0lhs_value_type,1rhs_value_type
    (std::tuple<int,int>),
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        //rhs scalar
        std::make_tuple(lhs_tensor_type{},1,lhs_tensor_type(1),1),
        std::make_tuple(lhs_tensor_type(1),2,lhs_tensor_type(2),2),
        std::make_tuple(lhs_tensor_type{1,2,3},4,lhs_tensor_type(4),4),
        //rhs tensor
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(3)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{4,5},{6,7}}),
        //rhs view
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}.reshape(2,3,0),lhs_tensor_type{}.reshape(2,3,0),rhs_tensor_type{}.reshape(2,3,0)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(4).transpose(),lhs_tensor_type(4),rhs_tensor_type(4)),
        std::make_tuple(lhs_tensor_type{{1,2},{3,4}},rhs_tensor_type{{5,6},{7,8}}(1,1),lhs_tensor_type(8),rhs_tensor_type(8)),
        //rhs expression
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}+rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{1,2,3}+rhs_tensor_type{4,5,6}+rhs_tensor_type(1),lhs_tensor_type{6,8,10},rhs_tensor_type{6,8,10})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto& result = lhs = rhs;
        REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_value_semantic","[test_tensor]",
//0lhs_value_type,1rhs_value_type
    (std::tuple<int,int>),
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    //0lhs,1rhs,2expected_lhs
    auto test_data = std::make_tuple(
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        lhs = rhs;
        *rhs.begin() = -1;
        REQUIRE(lhs == expected_lhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_move_assignment_result","[test_tensor]",
//0lhs_value_type,1rhs_value_type
    (std::tuple<int,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    REQUIRE(std::is_same_v<lhs_value_type,rhs_value_type>);
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{1}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{1}),
        std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{{1},{2},{3}}})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto& result = lhs = std::move(rhs);
        REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_converting_move_assignment_result","[test_tensor]",
//0lhs_value_type,1rhs_value_type
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    REQUIRE(!std::is_same_v<lhs_value_type,rhs_value_type>);
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(3)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{4,5},{6,7}}),
        //rhs view
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}.reshape(2,3,0),lhs_tensor_type{}.reshape(2,3,0),rhs_tensor_type{}.reshape(2,3,0)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(4).transpose(),lhs_tensor_type(4),rhs_tensor_type(4)),
        std::make_tuple(lhs_tensor_type{{1,2},{3,4}},rhs_tensor_type{{5,6},{7,8}}(1,1),lhs_tensor_type(8),rhs_tensor_type(8)),
        //rhs expression
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}+rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{1,2,3}+rhs_tensor_type{4,5,6}+rhs_tensor_type(1),lhs_tensor_type{6,8,10},rhs_tensor_type{6,8,10})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto& result = lhs = std::move(rhs);
        REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

//broadcast elementwise assignment
//operator=
TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_lhs_is_rvalue_view","[test_tensor]",
    (std::tuple<int,int>),
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using gtensor::tensor;
    using lhs_tensor_type = tensor<lhs_value_type>;
    using rhs_tensor_type = tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    //0parent,1lhs_view_maker,2rhs,3expected_parent,4expected_lhs,5expected_rhs
    auto test_data = std::make_tuple(
        //rhs scalar
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},1,lhs_tensor_type{},lhs_tensor_type{},1),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},2,lhs_tensor_type(2),lhs_tensor_type(2),2),
        std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},7,lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7},7),
        std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t(tensor<int>{3,4,1});},7,lhs_tensor_type{1,7,3,7,7,6},lhs_tensor_type{7,7,7},7),
        std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t(tensor<bool>{true,true,false,true});},7,lhs_tensor_type{7,7,3,7,5,6},lhs_tensor_type{7,7,7},7),
        //rhs 0-dim
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type(1),lhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type(2),lhs_tensor_type(2),lhs_tensor_type(2),rhs_tensor_type(2)),
        std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},rhs_tensor_type(7),lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7},rhs_tensor_type(7)),
        std::make_tuple(lhs_tensor_type{{1,2,3},{4,5,6}},[](const auto& t){return t(tensor<int>(1),tensor<int>{0,0,1});},rhs_tensor_type(7),lhs_tensor_type{{1,2,3},{7,7,6}},lhs_tensor_type{7,7,7},rhs_tensor_type(7)),
        std::make_tuple(lhs_tensor_type{{1,2,3},{4,5,6}},[](const auto& t){return t(tensor<bool>{{false,true},{true,true}});},rhs_tensor_type(7),lhs_tensor_type{{1,7,3},{7,7,6}},lhs_tensor_type{7,7,7},rhs_tensor_type(7)),
        //rhs n-dim
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type{1},lhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{1}),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type{2},lhs_tensor_type(2),lhs_tensor_type(2),rhs_tensor_type{2}),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1},
            lhs_tensor_type{{{1,2},{-1,1}},{{5,6},{-1,1}}},
            lhs_tensor_type{{{-1,1}},{{-1,1}}},
            rhs_tensor_type{-1,1}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1},
            lhs_tensor_type{{{1,5},{-1,1}},{{2,6},{-1,1}}},
            lhs_tensor_type{{{-1,1}},{{-1,1}}},
            rhs_tensor_type{-1,1}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1} + rhs_tensor_type{{1},{2}},
            lhs_tensor_type{{{1,5},{1,3}},{{2,6},{1,3}}},
            lhs_tensor_type{{{1,3}},{{1,3}}},
            rhs_tensor_type{{0,2},{1,3}}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
            [](const auto& t){return t(tensor<int>{0,1},tensor<int>(1),tensor<int>(0));},
            rhs_tensor_type{9,10},
            lhs_tensor_type{{{1,2},{9,4}},{{5,6},{10,8}}},
            lhs_tensor_type{9,10},
            rhs_tensor_type{9,10}
        )
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto lhs_view_maker = std::get<1>(t);
        auto rhs = std::get<2>(t);
        auto expected_parent = std::get<3>(t);
        auto expected_lhs = std::get<4>(t);
        auto expected_rhs = std::get<5>(t);
        auto lhs = lhs_view_maker(parent);
        auto& result = std::move(lhs) = rhs;
        REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(parent == expected_parent);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_move_assignment_converting_move_assignment_lhs_is_rvalue_view","[test_tensor]",
    (std::tuple<int,int>),
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using gtensor::assign;
    using helpers_for_testing::apply_by_element;
    //0parent,1lhs_view_maker,2rhs,3expected_parent,4expected_lhs
    auto test_data = std::make_tuple(
        //rhs 0-dim
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type(1),lhs_tensor_type{},lhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type(2),lhs_tensor_type(2),lhs_tensor_type(2)),
        std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},rhs_tensor_type(7),lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7}),
        //rhs n-dim
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type{1},lhs_tensor_type{},lhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type{2},lhs_tensor_type(2),lhs_tensor_type(2)),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1},
            lhs_tensor_type{{{1,2},{-1,1}},{{5,6},{-1,1}}},
            lhs_tensor_type{{{-1,1}},{{-1,1}}}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1},
            lhs_tensor_type{{{1,5},{-1,1}},{{2,6},{-1,1}}},
            lhs_tensor_type{{{-1,1}},{{-1,1}}}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1} + rhs_tensor_type{{1},{2}},
            lhs_tensor_type{{{1,5},{1,3}},{{2,6},{1,3}}},
            lhs_tensor_type{{{1,3}},{{1,3}}}
        )
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto lhs_view_maker = std::get<1>(t);
        auto rhs = std::get<2>(t);
        auto expected_parent = std::get<3>(t);
        auto expected_lhs = std::get<4>(t);
        auto lhs = lhs_view_maker(parent);
        auto& result = std::move(lhs) = std::move(rhs);
        REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(parent == expected_parent);
        REQUIRE(rhs.empty());
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_assign_to_rvalue_view_or_tensor","[test_tensor]")
{
    using gtensor::tensor;
    tensor<double> t{1,2,3,4,5,6,7,8,9,10,11,12};

    SECTION("assign1")
    {
        t.reshape(2,2,3) = tensor<double>{{3,2,1},{1,2,3}};
        REQUIRE(t == tensor<double>{3,2,1,1,2,3,3,2,1,1,2,3});
    }
    SECTION("assign2")
    {
        t.reshape(2,2,3).transpose() = tensor<int>{{2,1},{3,2}};
        REQUIRE(t == tensor<double>{2,2,2,3,3,3,1,1,1,2,2,2});
    }
    SECTION("assign3")
    {
        using slice_type = tensor<double>::slice_type;
        t.reshape(2,2,3)(slice_type{},slice_type{},1) = 0;
        REQUIRE(t == tensor<double>{1,0,3,4,0,6,7,0,9,10,0,12});
    }
    SECTION("assign4")
    {
        t(t>3&&t<10) = 0;
        REQUIRE(t == tensor<double>{1,2,3,0,0,0,0,0,0,10,11,12});
    }
    SECTION("assign5")
    {
        t.reshape(2,2,3)(tensor<int>{{1,0},{0,1}},tensor<int>{{0,1},{1,0}}) = tensor<int>{{8},{9}};
        REQUIRE(t == tensor<double>{1,2,3,8,8,8,9,9,9,10,11,12});
    }
    SECTION("assign6")
    {
        std::move(t) = 5;
        REQUIRE(t == tensor<double>{5,5,5,5,5,5,5,5,5,5,5,5});
    }
}

TEST_CASE("test_tensor_compaund_assign","[test_tensor]")
{
    using gtensor::tensor;
    tensor<double> t{{7,3,4,6},{1,5,6,2},{1,8,3,5},{0,2,6,2}};
    SECTION("compaund_assign1")
    {
        t(t>3&&t.not_equal(6)) += 1;
        REQUIRE(t==tensor<double>{{8,3,5,6},{1,6,6,2},{1,9,3,6},{0,2,6,2}});
    }
    SECTION("compaund_assign2")
    {
        t({{1},{{},{-1}}}) *= 2;
        REQUIRE(t==tensor<double>{{7,3,4,6},{2,10,12,2},{2,16,6,5},{0,4,12,2}});
    }
    SECTION("compaund_assign3")
    {
        t({{1},{{},{-1}}}) += tensor<double>{1,0,-1};
        REQUIRE(t==tensor<double>{{7,3,4,6},{2,5,5,2},{2,8,2,5},{1,2,5,2}});
    }
}

namespace test_tensor_assignment_corner_cases{

struct assign_exception{};
struct throw_on_assign{
    inline static bool is_throw = false;
    throw_on_assign() = default;
    throw_on_assign(const throw_on_assign&) = default;
    throw_on_assign& operator=(const throw_on_assign&){
        if (is_throw){
            throw assign_exception{};
        }
        return *this;
    }
};

}

TEST_CASE("test_tensor_assignment_corner_cases","[test_tensor]")
{
    using helpers_for_testing::apply_by_element;
    using test_tensor_assignment_corner_cases::throw_on_assign;
    throw_on_assign::is_throw = false;
    REQUIRE_NOTHROW(throw_on_assign{} = throw_on_assign{});
    throw_on_assign::is_throw = true;
    REQUIRE_THROWS(throw_on_assign{} = throw_on_assign{});
    throw_on_assign::is_throw = false;
    SECTION("copy_assign_to_same_lvalue")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        //assign may be required during construction
        tensor_type t({10},value_type{});
        throw_on_assign::is_throw = true;   //enable throwing on assign
        const auto ptr_to_first_expected = &(*t.begin());
        REQUIRE_NOTHROW(t.operator=(t));    //self assignment
        const auto ptr_to_first_result = &(*t.begin());
        REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
    }
    SECTION("copy_assign_to_same_rvalue")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        //assign may be required during construction
        tensor_type t({10},value_type{});
        throw_on_assign::is_throw = true;   //enable throwing on assign
        REQUIRE_NOTHROW(std::move(t).operator=(t));    //self assignment
    }
    SECTION("move_assign_to_same_lvalue")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        //assign may be required during construction
        tensor_type t({10},value_type{});
        throw_on_assign::is_throw = true;   //enable throwing on assign
        const auto ptr_to_first_expected = &(*t.begin());
        REQUIRE_NOTHROW(t.operator=(std::move(t)));    //self assignment
        const auto ptr_to_first_result = &(*t.begin());
        REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
    }
    SECTION("move_assign_to_same_rvalue")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        //assign may be required during construction
        tensor_type t({10},value_type{});
        throw_on_assign::is_throw = true;   //enable throwing on assign
        REQUIRE_NOTHROW(std::move(t).operator=(std::move(t)));    //self assignment
    }
    SECTION("move_assign_to_lvalue")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        tensor_type lhs({10},value_type{});
        tensor_type rhs({10},value_type{});
        throw_on_assign::is_throw = true;   //enable throwing on assign
        const auto ptr_to_first_expected = &(*rhs.begin());
        REQUIRE_NOTHROW(lhs.operator=(std::move(rhs)));    //no assignment
        const auto ptr_to_first_result = &(*lhs.begin());
        REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //swap impl
    }
    SECTION("move_assign_to_rvalue")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        tensor_type lhs({10},value_type{});
        tensor_type rhs({10},value_type{});
        throw_on_assign::is_throw = true;   //enable throwing on assign
        REQUIRE_THROWS_AS(std::move(lhs).operator=(std::move(rhs)),test_tensor_assignment_corner_cases::assign_exception);    //elementwise assignment
    }
    SECTION("copy_assign_convert_copy_assign_to_lvalue_same_shape")
    {
        using gtensor::tensor;
        //0lhs,1rhs,2expected_lhs,3expected_rhs
        auto test_data = std::make_tuple(
            std::make_tuple(tensor<int>(1),tensor<int>(2),tensor<int>(2),tensor<int>(2)),
            std::make_tuple(tensor<double>(1),tensor<int>(2),tensor<double>(2),tensor<int>(2)),
            std::make_tuple(tensor<int>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}}),
            std::make_tuple(tensor<double>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}}+0,tensor<double>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}})
        );
        auto test = [](const auto& t){
            auto lhs = std::get<0>(t);
            auto rhs = std::get<1>(t);
            auto expected_lhs = std::get<2>(t);
            auto expected_rhs = std::get<3>(t);
            const auto ptr_to_first_expected = &(*lhs.begin());
            lhs.operator=(rhs);
            const auto ptr_to_first_result = &(*lhs.begin());
            REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
            REQUIRE(lhs == expected_lhs);
            REQUIRE(rhs == expected_rhs);
        };
        apply_by_element(test,test_data);
    }
    SECTION("convert_move_assign_to_lvalue_same_shape")
    {
        using gtensor::tensor;
        //0lhs,1rhs,2expected_lhs,3expected_rhs
        auto test_data = std::make_tuple(
            std::make_tuple(tensor<double>(1),tensor<int>(2),tensor<double>(2),tensor<int>(2)),
            std::make_tuple(tensor<double>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}}+0,tensor<double>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}})
        );
        auto test = [](const auto& t){
            auto lhs = std::get<0>(t);
            auto rhs = std::get<1>(t);
            auto expected_lhs = std::get<2>(t);
            auto expected_rhs = std::get<3>(t);
            const auto ptr_to_first_expected = &(*lhs.begin());
            lhs.operator=(std::move(rhs));
            const auto ptr_to_first_result = &(*lhs.begin());
            REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
            REQUIRE(lhs == expected_lhs);
            REQUIRE(rhs == expected_rhs);
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_tensor_elementwise_assignment_exception","[test_tensor]")
{
    using tensor_type = gtensor::tensor<int>;
    using gtensor::value_error;
    auto lhs = tensor_type{{1,2,3},{4,5,6}}(1);
    tensor_type rhs{1,2};
    //shapes not broadcast
    REQUIRE_THROWS_AS((std::move(lhs) = rhs), value_error);
    REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(1) = rhs), value_error);
    REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}.reshape(-1) = tensor_type{1,2}), value_error);
}

//assign
TEST_CASE("test_tensor_assign","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    SECTION("lvalue_rhs")
    {
        //0lhs,1rhs,2expected_lhs,3expected_rhs
        auto test_data = std::make_tuple(
            //rhs scalar
            std::make_tuple(tensor_type{},1,tensor_type{},1),
            std::make_tuple(tensor_type(1),2,tensor_type(2),2),
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},7,tensor_type{{7,7,7},{7,7,7}},7),
            //rhs 0-dim
            std::make_tuple(tensor_type{},tensor_type(2),tensor_type{},tensor_type(2)),
            std::make_tuple(tensor_type(1),tensor_type(2),tensor_type(2),tensor_type(2)),
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type(7),tensor_type{{7,7,7},{7,7,7}},tensor_type(7)),
            //rhs n-dim
            std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
            std::make_tuple(tensor_type(1),tensor_type{},tensor_type(1),tensor_type{}),
            std::make_tuple(tensor_type(1),tensor_type{2},tensor_type(2),tensor_type{2}),
            std::make_tuple(tensor_type(1),tensor_type{{1,2},{3,4}},tensor_type(4),tensor_type{{1,2},{3,4}}),
            std::make_tuple(tensor_type{2},tensor_type{},tensor_type{2},tensor_type{}),
            std::make_tuple(tensor_type{{3}},tensor_type{},tensor_type{{3}},tensor_type{}),
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{9},{10}},tensor_type{{{9,9},{10,10}},{{9,9},{10,10}}},tensor_type{{9},{10}})
        );
        auto test = [](const auto& t){
            auto lhs = std::get<0>(t);
            auto rhs = std::get<1>(t);
            auto expected_lhs = std::get<2>(t);
            auto expected_rhs = std::get<3>(t);
            auto& result = lhs.assign(rhs);
            REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
            REQUIRE(&result == &lhs);
            REQUIRE(result == expected_lhs);
            REQUIRE(rhs == expected_rhs);
        };
        apply_by_element(test,test_data);
    }
    SECTION("rvalue_rhs")
    {
        //0lhs,1rhs,2expected_lhs,3expected_rhs
        auto test_data = std::make_tuple(
            //rhs 0-dim
            std::make_tuple(tensor_type{},tensor_type(2),tensor_type{},tensor_type(2)),
            std::make_tuple(tensor_type(1),tensor_type(2),tensor_type(2),tensor_type(2)),
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type(7),tensor_type{{7,7,7},{7,7,7}},tensor_type(7)),
            //rhs n-dim
            std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
            std::make_tuple(tensor_type(1),tensor_type{},tensor_type(1),tensor_type{}),
            std::make_tuple(tensor_type(1),tensor_type{2},tensor_type(2),tensor_type{2}),
            std::make_tuple(tensor_type(1),tensor_type{{1,2},{3,4}},tensor_type(4),tensor_type{{1,2},{3,4}}),
            std::make_tuple(tensor_type{2},tensor_type{},tensor_type{2},tensor_type{}),
            std::make_tuple(tensor_type{{3}},tensor_type{},tensor_type{{3}},tensor_type{}),
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{9},{10}},tensor_type{{{9,9},{10,10}},{{9,9},{10,10}}},tensor_type{{9},{10}})
        );
        auto test = [](const auto& t){
            auto lhs = std::get<0>(t);
            auto rhs = std::get<1>(t);
            auto expected_lhs = std::get<2>(t);
            auto expected_rhs = std::get<3>(t);
            auto& result = lhs.assign(std::move(rhs));
            REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
            REQUIRE(&result == &lhs);
            REQUIRE(result == expected_lhs);
            REQUIRE(rhs.empty());
        };
        apply_by_element(test,test_data);
    }
}

//related
