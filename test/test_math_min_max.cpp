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

//amin,nanmin
TEMPLATE_TEST_CASE("test_math_amin_nanmin","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amin;
    using gtensor::min;
    using gtensor::nanmin;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type, value_type>);
    REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type, value_type>);
    REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type, value_type>);
    REQUIRE(std::is_same_v<typename decltype(min(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type, value_type>);
    REQUIRE(std::is_same_v<typename decltype(min(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type, value_type>);
    REQUIRE(std::is_same_v<typename decltype(min(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type, value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type, value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type, value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type, value_type>);

    //0tensor,1axes,2keep_dims,3initial,4expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,value_type{100},tensor_type(value_type{100})),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,value_type{100},tensor_type(value_type{100})),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,value_type{100},tensor_type{value_type{100},value_type{100}}),
        std::make_tuple(tensor_type{5},0,false,value_type{100},tensor_type(5)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,value_type{100},tensor_type(-1)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},false,value_type{100},tensor_type(-1)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,false,value_type{100},tensor_type{{1,4,3},{1,0,-1}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,false,value_type{100},tensor_type{{1,0,-1},{1,4,2}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,value_type{100},tensor_type{{1,-1},{4,1}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},false,value_type{100},tensor_type{1,0,-1}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},false,value_type{100},tensor_type{-1,1}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,value_type{100},tensor_type{}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},false,value_type{2},tensor_type{2,2,1,-1,2,2}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},false,value_type{2},tensor_type{{{1,2,2},{2,0,-1}},{{2,2,2},{1,2,2}}}),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,value_type{100},tensor_type{value_type{100}}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,true,value_type{100},tensor_type{-1}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,true,value_type{100},tensor_type{{{1,0,-1}},{{1,4,2}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},true,value_type{100},tensor_type{{{-1}},{{1}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,value_type{100},tensor_type{}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},true,value_type{2},tensor_type{{{1,2,2},{2,0,-1}},{{2,2,2},{1,2,2}}}),
        //initial is min
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,value_type{-2},tensor_type(-2)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,value_type{1},tensor_type{{1,-1},{1,1}})
    );
    auto test_amin = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = amin(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_min = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = min(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nanmin = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = nanmin(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };

    //default policy
    SECTION("test_amin_default_policy")
    {
        test_amin();
    }
    SECTION("test_min_default_policy")
    {
        test_min();
    }
    SECTION("test_nanmin_default_policy")
    {
        test_nanmin();
    }
    //exec_pol<4>
    SECTION("test_amin_exec_pol<4>")
    {
        test_amin(multithreading::exec_pol<4>{});
    }
    SECTION("test_min_exec_pol<4>")
    {
        test_min(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanmin_exec_pol<4>")
    {
        test_nanmin(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_amin_exec_pol<0>")
    {
        test_amin(multithreading::exec_pol<0>{});
    }
    SECTION("test_min_exec_pol<0>")
    {
        test_min(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanmin_exec_pol<0>")
    {
        test_nanmin(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_math_amin_nanmin_flatten","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amin;
    using gtensor::min;
    using gtensor::nanmin;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<bool>()))::value_type,value_type>);

    //0tensor,1keep_dims,2initial,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{3,2,1,5,2,9,5,3,2},false,value_type{100},tensor_type(1)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,value_type{100},tensor_type(-1)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,value_type{-2},tensor_type(-2)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true,value_type{100},tensor_type{{{-1}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true,value_type{-2},tensor_type{{{-2}}})
    );
    auto test_amin = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto keep_dims = std::get<1>(t);
            auto initial = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = amin(policy...,ten,keep_dims,initial);
            REQUIRE(result==expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_min = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto keep_dims = std::get<1>(t);
            auto initial = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = min(policy...,ten,keep_dims,initial);
            REQUIRE(result==expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nanmin = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto keep_dims = std::get<1>(t);
            auto initial = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = nanmin(policy...,ten,keep_dims,initial);
            REQUIRE(result==expected);
        };
        apply_by_element(test,test_data);
    };

    //default policy
    SECTION("test_amin_default_policy")
    {
        test_amin();
    }
    SECTION("test_min_default_policy")
    {
        test_min();
    }
    SECTION("test_nanmin_default_policy")
    {
        test_nanmin();
    }
    //exec_pol<4>
    SECTION("test_amin_exec_pol<4>")
    {
        test_amin(multithreading::exec_pol<4>{});
    }
    SECTION("test_min_exec_pol<4>")
    {
        test_min(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanmin_exec_pol<4>")
    {
        test_nanmin(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_amin_exec_pol<0>")
    {
        test_amin(multithreading::exec_pol<0>{});
    }
    SECTION("test_min_exec_pol<0>")
    {
        test_min(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanmin_exec_pol<0>")
    {
        test_nanmin(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_math_amin_nanmin_overloads","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amin;
    using gtensor::nanmin;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);

    //amin
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}}) == tensor_type(-1));
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(-1));
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{-1}}});
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1}) == tensor_type{-1,1});
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{1,0,-1},{1,4,2}});
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{-1,1});
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{1,0,-1}}});
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true,value_type{0}) == tensor_type{{{0,0,-1}}});
    //nanmin
    REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}}) == tensor_type(-1));
    REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(-1));
    REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{-1}}});
    REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{1,0,-1},{1,4,2}});
    REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{-1,1});
    REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1}) == tensor_type{-1,1});
    REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{1,0,-1}}});
    REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true,value_type{0}) == tensor_type{{{0,0,-1}}});
}

TEMPLATE_TEST_CASE("test_math_amin_nanmin_exception","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::amin;
    using gtensor::nanmin;

    //amin
    REQUIRE_THROWS_AS(amin(tensor_type{}),value_error);
    REQUIRE_THROWS_AS(amin(tensor_type{}.reshape(0,2,3),{0,1}),value_error);
    REQUIRE_NOTHROW(amin(tensor_type{}.reshape(0,2,3),{1,2}));
    //nanmin
    REQUIRE_THROWS_AS(nanmin(tensor_type{}),value_error);
    REQUIRE_THROWS_AS(nanmin(tensor_type{}.reshape(0,2,3),{0,1}),value_error);
    REQUIRE_NOTHROW(nanmin(tensor_type{}.reshape(0,2,3),{1,2}));
}

TEST_CASE("test_math_amin_nanmin_nan_values_default_policy","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amin;
    using gtensor::nanmin;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //amin
        std::make_tuple(amin(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(0.5)),
        std::make_tuple(amin(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(neg_inf)),
        std::make_tuple(amin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(amin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), tensor_type(nan)),
        std::make_tuple(amin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), tensor_type{nan,nan,nan,-1.0,0.0,1.0}),
        std::make_tuple(amin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), tensor_type{nan,nan}),
        std::make_tuple(amin(tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), tensor_type{nan,-1.0,3.0,nan}),
        std::make_tuple(amin(tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), tensor_type{nan,0.1}),
        std::make_tuple(amin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nanmin
        std::make_tuple(nanmin(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(0.5)),
        std::make_tuple(nanmin(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(neg_inf)),
        std::make_tuple(nanmin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
        std::make_tuple(nanmin(tensor_type{{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanmin(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), tensor_type(0.1)),
        std::make_tuple(nanmin(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},0), tensor_type{0.1,1.1,nan}),
        std::make_tuple(nanmin(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},1), tensor_type{nan,1.1,0.1}),
        std::make_tuple(nanmin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_math_amin_nanmin_nan_values_policy","test_math",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amin;
    using gtensor::nanmin;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //amin
        std::make_tuple(amin(policy{},tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(0.5)),
        std::make_tuple(amin(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(neg_inf)),
        std::make_tuple(amin(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(amin(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), tensor_type(nan)),
        std::make_tuple(amin(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), tensor_type{nan,nan,nan,-1.0,0.0,1.0}),
        std::make_tuple(amin(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), tensor_type{nan,nan}),
        std::make_tuple(amin(policy{},tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), tensor_type{nan,-1.0,3.0,nan}),
        std::make_tuple(amin(policy{},tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), tensor_type{nan,0.1}),
        std::make_tuple(amin(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nanmin
        std::make_tuple(nanmin(policy{},tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(0.5)),
        std::make_tuple(nanmin(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(neg_inf)),
        std::make_tuple(nanmin(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
        std::make_tuple(nanmin(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanmin(policy{},tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), tensor_type(0.1)),
        std::make_tuple(nanmin(policy{},tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},0), tensor_type{0.1,1.1,nan}),
        std::make_tuple(nanmin(policy{},tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},1), tensor_type{nan,1.1,0.1}),
        std::make_tuple(nanmin(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}


//amax,nanmax
TEMPLATE_TEST_CASE("test_math_amax_nanmax","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amax;
    using gtensor::max;
    using gtensor::nanmax;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(max(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(max(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(max(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);

    //0tensor,1axes,2keep_dims,3initial,4expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,value_type{-100},tensor_type(value_type{-100})),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,value_type{-100},tensor_type(value_type{-100})),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,value_type{-100},tensor_type{value_type{-100},value_type{-100}}),
        std::make_tuple(tensor_type{5},0,false,value_type{-100},tensor_type(5)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,value_type{-100},tensor_type(5)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},false,value_type{-100},tensor_type(5)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,false,value_type{-100},tensor_type{{7,5,9},{2,11,2}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,false,value_type{-100},tensor_type{{2,5,3},{7,11,9}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,value_type{-100},tensor_type{{5,2},{9,11}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},false,value_type{-100},tensor_type{7,11,9}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},false,value_type{-100},tensor_type{5,11}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,value_type{-100},tensor_type{}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},false,value_type{2},tensor_type{5,2,2,2,4,4}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},false,value_type{2},tensor_type{{{2,5,3},{2,2,2}},{{7,4,9},{2,11,2}}}),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,value_type{-100},tensor_type{value_type{-100}}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},true,value_type{-100},tensor_type{5}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,true,value_type{-100},tensor_type{{{2,5,3}},{{7,11,9}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},true,value_type{-100},tensor_type{{{5}},{{11}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,value_type{-100},tensor_type{}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},true,value_type{2},tensor_type{5,2,2,2,4,4}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},true,value_type{2},tensor_type{{{2,5,3},{2,2,2}},{{7,4,9},{2,11,2}}}),
        //initial is max
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,value_type{6},tensor_type(6)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,value_type{3},tensor_type{{5,3},{9,11}})
    );
    auto test_amax = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = amax(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_max = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = max(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nanmax = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = nanmax(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };

    //default policy
    SECTION("test_amax_default_policy")
    {
        test_amax();
    }
    SECTION("test_max_default_policy")
    {
        test_max();
    }
    SECTION("test_nanmax_default_policy")
    {
        test_nanmax();
    }
    //exec_pol<4>
    SECTION("test_amax_exec_pol<4>")
    {
        test_amax(multithreading::exec_pol<4>{});
    }
    SECTION("test_max_exec_pol<4>")
    {
        test_max(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanmax_exec_pol<4>")
    {
        test_nanmax(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_amax_exec_pol<0>")
    {
        test_amax(multithreading::exec_pol<0>{});
    }
    SECTION("test_max_exec_pol<0>")
    {
        test_max(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanmax_exec_pol<0>")
    {
        test_nanmax(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_math_amax_nanmax_flatten","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amax;
    using helpers_for_testing::apply_by_element;

    //0tensor,1keep_dims,2initial,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,value_type{0},tensor_type(11)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,value_type{12},tensor_type(12)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true,value_type{0},tensor_type{{{11}}})
    );

    auto test_amax = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto keep_dims = std::get<1>(t);
            auto initial = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = amax(policy...,ten,keep_dims,initial);
            REQUIRE(result==expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_max = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto keep_dims = std::get<1>(t);
            auto initial = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = max(policy...,ten,keep_dims,initial);
            REQUIRE(result==expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nanmax = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto keep_dims = std::get<1>(t);
            auto initial = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = nanmax(policy...,ten,keep_dims,initial);
            REQUIRE(result==expected);
        };
        apply_by_element(test,test_data);
    };

    //default policy
    SECTION("test_amax_default_policy")
    {
        test_amax();
    }
    SECTION("test_max_default_policy")
    {
        test_max();
    }
    SECTION("test_nanmax_default_policy")
    {
        test_nanmax();
    }
    //exec_pol<4>
    SECTION("test_amax_exec_pol<4>")
    {
        test_amax(multithreading::exec_pol<4>{});
    }
    SECTION("test_max_exec_pol<4>")
    {
        test_max(multithreading::exec_pol<4>{});
    }
    SECTION("test_nanmax_exec_pol<4>")
    {
        test_nanmax(multithreading::exec_pol<4>{});
    }
    //exec_pol<0>
    SECTION("test_amax_exec_pol<0>")
    {
        test_amax(multithreading::exec_pol<0>{});
    }
    SECTION("test_max_exec_pol<0>")
    {
        test_max(multithreading::exec_pol<0>{});
    }
    SECTION("test_nanmax_exec_pol<0>")
    {
        test_nanmax(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_math_amax_nanmax_overloads","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amax;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);

    //amax
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}}) == tensor_type(11));
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(11));
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{11}}});
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1}) == tensor_type{5,11});
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{2,5,3},{7,11,9}});
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{5,11});
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{7,11,9}}});
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true,value_type{8}) == tensor_type{{{8,11,9}}});
    //nanmax
    REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}}) == tensor_type(11));
    REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(11));
    REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{11}}});
    REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1}) == tensor_type{5,11});
    REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{2,5,3},{7,11,9}});
    REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{5,11});
    REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{7,11,9}}});
    REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true,value_type{8}) == tensor_type{{{8,11,9}}});
}

TEMPLATE_TEST_CASE("test_math_amax_nanmax_exception","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::amax;
    using gtensor::nanmax;

    //amax
    REQUIRE_THROWS_AS(amax(tensor_type{}),value_error);
    REQUIRE_THROWS_AS(amax(tensor_type{}.reshape(0,2,3),{0,1}),value_error);
    REQUIRE_NOTHROW(amax(tensor_type{}.reshape(0,2,3),{1,2}));
    //nanmax
    REQUIRE_THROWS_AS(nanmax(tensor_type{}),value_error);
    REQUIRE_THROWS_AS(nanmax(tensor_type{}.reshape(0,2,3),{0,1}),value_error);
    REQUIRE_NOTHROW(nanmax(tensor_type{}.reshape(0,2,3),{1,2}));
}

TEST_CASE("test_math_amax_nanmax_nan_values_default_policy","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amax;
    using gtensor::nanmax;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //amax
        std::make_tuple(amax(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(pos_inf)),
        std::make_tuple(amax(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(3.0)),
        std::make_tuple(amax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(amax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), tensor_type(nan)),
        std::make_tuple(amax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), tensor_type{nan,nan,nan,4.0,1.0,3.0}),
        std::make_tuple(amax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), tensor_type{nan,nan}),
        std::make_tuple(amax(tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), tensor_type{nan,0.1,5.0,nan}),
        std::make_tuple(amax(tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), tensor_type{nan,5.0}),
        std::make_tuple(amax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nanmax
        std::make_tuple(nanmax(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(pos_inf)),
        std::make_tuple(nanmax(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(3.0)),
        std::make_tuple(nanmax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(nanmax(tensor_type{{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanmax(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), tensor_type(2.0)),
        std::make_tuple(nanmax(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},0), tensor_type{0.1,2.0,nan}),
        std::make_tuple(nanmax(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},1), tensor_type{nan,1.1,2.0}),
        std::make_tuple(nanmax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_math_amax_nanmax_nan_values_policy","test_math",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amax;
    using gtensor::nanmax;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //amax
        std::make_tuple(amax(policy{},tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(pos_inf)),
        std::make_tuple(amax(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(3.0)),
        std::make_tuple(amax(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(amax(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), tensor_type(nan)),
        std::make_tuple(amax(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), tensor_type{nan,nan,nan,4.0,1.0,3.0}),
        std::make_tuple(amax(policy{},tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), tensor_type{nan,nan}),
        std::make_tuple(amax(policy{},tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), tensor_type{nan,0.1,5.0,nan}),
        std::make_tuple(amax(policy{},tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), tensor_type{nan,5.0}),
        std::make_tuple(amax(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nanmax
        std::make_tuple(nanmax(policy{},tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(pos_inf)),
        std::make_tuple(nanmax(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(3.0)),
        std::make_tuple(nanmax(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(nanmax(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(nanmax(policy{},tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), tensor_type(2.0)),
        std::make_tuple(nanmax(policy{},tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},0), tensor_type{0.1,2.0,nan}),
        std::make_tuple(nanmax(policy{},tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},1), tensor_type{nan,1.1,2.0}),
        std::make_tuple(nanmax(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

