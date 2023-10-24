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
#include "reduce.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "config_for_testing.hpp"

//reduce_range
TEST_CASE("test_tensor_reduce_range","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using helpers_for_testing::apply_by_element;
    auto sum = [](auto first, auto last){
        const auto& init = *first;
        return std::accumulate(++first,last,init,std::plus{});
    };

    //0tensor,1axes,2operation,3keep_dims,4expected
    auto test_data = std::make_tuple(
        //single axis
        std::make_tuple(tensor_type{1,2,3,4,5},dim_type{0},sum,false,tensor_type(15)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{0},sum,false,tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{1},sum,false,tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),dim_type{0},sum,false,tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),dim_type{1},sum,false,tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),dim_type{0},sum,false,tensor_type{11,15,19}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),dim_type{1},sum,false,tensor_type{18,27}),
        //axes container
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<dim_type>{0},sum,false,tensor_type(15)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{0},sum,false,tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{1},sum,false,tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{0,1},sum,false,tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{0},sum,false,tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{1},sum,false,tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{1,0},sum,false,tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{0},sum,false,tensor_type{11,15,19}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{1},sum,false,tensor_type{18,27}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{0,1},sum,false,tensor_type(45)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<dim_type>{},sum,false,tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{},sum,false,tensor_type{{1,2,3},{4,5,6}}),
        //keep_dims true
        std::make_tuple(tensor_type{1,2,3,4,5},dim_type{0},sum,true,tensor_type{15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{1},sum,true,tensor_type{{6},{15}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{0},sum,true,tensor_type{{5,7,9}})
    );
    auto test_reduce_range = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto operation = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = ten.reduce_range(policy...,axes,operation,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_reduce_range();
    }
    SECTION("exec_pol<4>")
    {
        test_reduce_range(multithreading::exec_pol<4>{});
    }
}

TEST_CASE("test_tensor_reduce_range_overload_default_policy","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    auto sum = [](auto first, auto last){
        const auto& init = *first;
        return std::accumulate(++first,last,init,std::plus{});
    };
    //defaulf keep_dims
    //axes scalar
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(0,sum) == tensor_type{{6,8},{10,12}});
    //axes container
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(std::vector<int>{0,1},sum) == tensor_type{16,20});
    //axes initializer_list
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range({0},sum) == tensor_type{{6,8},{10,12}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range({0,2},sum) == tensor_type{14,22});
    //like over flatten
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(sum) == tensor_type(36));

    //axes initializer_list and keep_dims
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range({0},sum,true) == tensor_type{{{6,8},{10,12}}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range({0},sum,false) == tensor_type{{6,8},{10,12}});

    //like over flatten and keep_dims
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(sum,true) == tensor_type{{{36}}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(sum,false) == tensor_type(36));
}

TEST_CASE("test_tensor_reduce_range_overload_policy","[test_tensor]")
{
    using policy = multithreading::exec_pol<4>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    auto sum = [](auto first, auto last){
        const auto& init = *first;
        return std::accumulate(++first,last,init,std::plus{});
    };
    //defaulf keep_dims
    //axes scalar
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},0,sum) == tensor_type{{6,8},{10,12}});
    //axes container
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},std::vector<int>{0,1},sum) == tensor_type{16,20});
    //axes initializer_list
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},{0},sum) == tensor_type{{6,8},{10,12}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},{0,2},sum) == tensor_type{14,22});
    //like over flatten
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},sum) == tensor_type(36));

    //axes initializer_list and keep_dims
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},{0},sum,true) == tensor_type{{{6,8},{10,12}}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},{0},sum,false) == tensor_type{{6,8},{10,12}});

    //like over flatten and keep_dims
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},sum,true) == tensor_type{{{36}}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_range(policy{},sum,false) == tensor_type(36));
}

//reduce_binary
TEST_CASE("test_tensor_reduce_binary","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axes,2operation,3keep_dims,4initial,5expected
    auto test_data = std::make_tuple(
        //single axis
        std::make_tuple(tensor_type{1,2,3,4,5},dim_type{0},std::plus<void>{},false,value_type{0},tensor_type(15)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{0},std::plus<void>{},false,value_type{0},tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{1},std::plus<void>{},false,value_type{0},tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),dim_type{0},std::plus<void>{},false,value_type{0},tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),dim_type{1},std::plus<void>{},false,value_type{0},tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),dim_type{0},std::plus<void>{},false,value_type{0},tensor_type{11,15,19}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),dim_type{1},std::plus<void>{},false,value_type{0},tensor_type{18,27}),
        //axes container
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<dim_type>{0},std::plus<void>{},false,value_type{0},tensor_type(15)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{0},std::plus<void>{},false,value_type{0},tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{1},std::plus<void>{},false,value_type{0},tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{0,1},std::plus<void>{},false,value_type{0},tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{0},std::plus<void>{},false,value_type{0},tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{1},std::plus<void>{},false,value_type{0},tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{1,0},std::plus<void>{},false,value_type{0},tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{0},std::plus<void>{},false,value_type{0},tensor_type{11,15,19}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{1},std::plus<void>{},false,value_type{0},tensor_type{18,27}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{0,1},std::plus<void>{},false,value_type{0},tensor_type(45)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<dim_type>{},std::plus<void>{},false,value_type{0},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{},std::plus<void>{},false,value_type{0},tensor_type{{1,2,3},{4,5,6}}),
        //keep_dims true
        std::make_tuple(tensor_type{1,2,3,4,5},dim_type{0},std::plus<void>{},true,value_type{0},tensor_type{15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{1},std::plus<void>{},true,value_type{0},tensor_type{{6},{15}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{0},std::plus<void>{},true,value_type{0},tensor_type{{5,7,9}}),
        //initial
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{1},std::plus<void>{},true,value_type{-1},tensor_type{{5},{14}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{0,1},std::plus<void>{},false,value_type{10},tensor_type(31))
    );
    auto test_reduce_binary = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto operation = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto initial = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto result = ten.reduce_binary(policy...,axes,operation,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_reduce_binary();
    }
    SECTION("exec_pol<4>")
    {
        test_reduce_binary(multithreading::exec_pol<4>{});
    }
}

TEST_CASE("test_tensor_reduce_binary_overload_default_policy","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //defaulf keep_dims and initial
    //axes scalar
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(0,std::plus<void>{}) == tensor_type{{6,8},{10,12}});
    //axes container
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(std::vector<int>{0,1},std::plus<void>{}) == tensor_type{16,20});
    //axes initializer_list
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary({0},std::plus<void>{}) == tensor_type{{6,8},{10,12}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary({0,2},std::plus<void>{}) == tensor_type{14,22});
    //like over flatten
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(std::plus<void>{}) == tensor_type(36));

    //defaulf initial
    //axes scalar
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(0,std::plus<void>{},false) == tensor_type{{6,8},{10,12}});
    //axes container
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(std::vector<int>{0,1},std::plus<void>{},true) == tensor_type{{{16,20}}});
    //axes initializer_list
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary({0},std::plus<void>{},false) == tensor_type{{6,8},{10,12}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary({0,2},std::plus<void>{},true) == tensor_type{{{14},{22}}});
    //like over flatten
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(std::plus<void>{},false) == tensor_type(36));
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(std::plus<void>{},true) == tensor_type{{{36}}});

    //axes initializer_list, keep_dims and initial
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary({0},std::plus<void>{},true,value_type{1}) == tensor_type{{{7,9},{11,13}}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary({0},std::plus<void>{},false,value_type{-1}) == tensor_type{{5,7},{9,11}});

    //like over flatten, keep_dims and initial
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(std::plus<void>{},true,value_type{1}) == tensor_type{{{37}}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(std::plus<void>{},false,value_type{-1}) == tensor_type(35));
}

TEST_CASE("test_tensor_reduce_binary_overload_policy","[test_tensor]")
{
    using policy = multithreading::exec_pol<4>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //defaulf keep_dims and initial
    //axes scalar
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},0,std::plus<void>{}) == tensor_type{{6,8},{10,12}});
    //axes container
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},std::vector<int>{0,1},std::plus<void>{}) == tensor_type{16,20});
    //axes initializer_list
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},{0},std::plus<void>{}) == tensor_type{{6,8},{10,12}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},{0,2},std::plus<void>{}) == tensor_type{14,22});
    //like over flatten
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},std::plus<void>{}) == tensor_type(36));

    //defaulf initial
    //axes scalar
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},0,std::plus<void>{},false) == tensor_type{{6,8},{10,12}});
    //axes container
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},std::vector<int>{0,1},std::plus<void>{},true) == tensor_type{{{16,20}}});
    //axes initializer_list
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},{0},std::plus<void>{},false) == tensor_type{{6,8},{10,12}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},{0,2},std::plus<void>{},true) == tensor_type{{{14},{22}}});
    //like over flatten
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},std::plus<void>{},false) == tensor_type(36));
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},std::plus<void>{},true) == tensor_type{{{36}}});

    //axes initializer_list, keep_dims and initial
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},{0},std::plus<void>{},true,value_type{1}) == tensor_type{{{7,9},{11,13}}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},{0},std::plus<void>{},false,value_type{-1}) == tensor_type{{5,7},{9,11}});

    //like over flatten, keep_dims and initial
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},std::plus<void>{},true,value_type{1}) == tensor_type{{{37}}});
    REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce_binary(policy{},std::plus<void>{},false,value_type{-1}) == tensor_type(35));
}

//slide
TEST_CASE("test_tensor_slide","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using helpers_for_testing::apply_by_element;

    auto cumsum = [](auto first, auto, auto dfirst, auto dlast){
        auto cumsum_ = *first;
        *dfirst = cumsum_;
        for(++dfirst,++first;dfirst!=dlast;++dfirst,++first){
            cumsum_+=*first;
            *dfirst = cumsum_;
        }
    };

    auto diff_1 = [](auto first, auto, auto dfirst, auto dlast){
        for(;dfirst!=dlast;++dfirst){
            auto prev = *first;
            *dfirst = *++first - prev;
        }
    };
    //0tensor,1axis,2operation,3window_size,4window_step,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3,4,5},dim_type{0},cumsum,index_type{1},index_type{1},tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{1,2,0,4,3,2,5},dim_type{0},diff_1,index_type{2},index_type{1},tensor_type{1,-2,4,-1,-1,3}),
        std::make_tuple(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}},dim_type{0},diff_1,index_type{2},index_type{1},tensor_type{{1,-1,0,-4},{1,0,2,5},{-1,0,0,-4}}),
        std::make_tuple(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}},dim_type{1},diff_1,index_type{2},index_type{1},tensor_type{{1,-2,4},{-1,-1,0},{-2,1,3},{-1,1,-1}})
    );
    auto test_slide = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto operation = std::get<2>(t);
            auto window_size = std::get<3>(t);
            auto window_step = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto result = ten.slide(policy...,axis,operation,window_size,window_step);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_slide();
    }
    SECTION("exec_pol<4>")
    {
        test_slide(multithreading::exec_pol<4>{});
    }
}

TEST_CASE("test_tensor_slide_overload","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;

    auto diff_1 = [](auto first, auto, auto dfirst, auto dlast){
        for(;dfirst!=dlast;++dfirst){
            auto prev = *first;
            *dfirst = *++first - prev;
        }
    };
    //like over flatten default policy
    REQUIRE(tensor_type{2,3,1,0,2,4,2,1,6,3}.slide(diff_1,2,1) == tensor_type{1,-2,-1,2,2,-2,-1,5,-3});
    REQUIRE(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}}.slide(diff_1,2,1) == tensor_type{1,-2,4,-2,-1,-1,0,3,-2,1,3,-3,-1,1,-1});

    //like over flatten, exec_pol<4>
    REQUIRE(tensor_type{2,3,1,0,2,4,2,1,6,3}.slide(multithreading::exec_pol<4>{},diff_1,2,1) == tensor_type{1,-2,-1,2,2,-2,-1,5,-3});
    REQUIRE(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}}.slide(multithreading::exec_pol<4>{},diff_1,2,1) == tensor_type{1,-2,4,-2,-1,-1,0,3,-2,1,3,-3,-1,1,-1});
}

//transform
TEST_CASE("test_tensor_transform","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;

    auto sort = [](auto first, auto last){
        std::sort(first,last);
    };

    //0tensor,1axis,2operation,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,0,4,3,2,5},0,sort,tensor_type{0,1,2,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}},0,sort,tensor_type{{1,1,0,0},{2,1,0,1},{2,1,2,4},{3,2,2,5}}),
        std::make_tuple(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}},1,sort,tensor_type{{0,1,2,4},{0,0,1,2},{1,2,3,5},{1,1,2,2}})
    );
    auto test_transform = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto operation = std::get<2>(t);
            auto expected = std::get<3>(t);
            ten.transform(policy...,axis,operation);
            REQUIRE(ten == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_transform();
    }
    SECTION("exec_pol<4>")
    {
        test_transform(multithreading::exec_pol<4>{});
    }
}
