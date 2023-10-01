/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"
#include "sort_search.hpp"
#include "tensor.hpp"

//count_nonzero
TEST_CASE("test_sort_search_count_nonzero","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::count_nonzero;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(count_nonzero(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>()))::value_type,index_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1},0,false,result_tensor_type(1)),
        std::make_tuple(tensor_type{0},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},0,false,result_tensor_type(7)),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,false,result_tensor_type{5,5,4,4,5}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,false,result_tensor_type{5,4,4,5,5}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{0,1},false,result_tensor_type(23)),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},0,true,result_tensor_type{7}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,true,result_tensor_type{{5,5,4,4,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,true,result_tensor_type{{5},{4},{4},{5},{5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{0,1},true,result_tensor_type{{23}})

    );
    auto test_count_nonzero = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = count_nonzero(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_count_nonzero();
    }
    SECTION("exec_pol<4>")
    {
        test_count_nonzero(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_count_nonzero(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_sort_search_count_nonzero_overload_default_policy","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::count_nonzero;

    REQUIRE(count_nonzero(tensor_type{1,3,0,0,1,4,6,-2,1,0}) == result_tensor_type(7));
    REQUIRE(count_nonzero(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(23));
    REQUIRE(count_nonzero(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},{1,0}) == result_tensor_type(23));
    REQUIRE(count_nonzero(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{23}});
    REQUIRE(count_nonzero(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},{0},true) == result_tensor_type{{5,5,4,4,5}});
}

TEMPLATE_TEST_CASE("test_sort_search_count_nonzero_overload_policy","[test_sort_search]",
    multithreading::exec_pol<4>,
    multithreading::exec_pol<0>
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::count_nonzero;

    REQUIRE(count_nonzero(policy{},tensor_type{1,3,0,0,1,4,6,-2,1,0}) == result_tensor_type(7));
    REQUIRE(count_nonzero(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(23));
    REQUIRE(count_nonzero(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},{1,0}) == result_tensor_type(23));
    REQUIRE(count_nonzero(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{23}});
    REQUIRE(count_nonzero(policy{},tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},{0},true) == result_tensor_type{{5,5,4,4,5}});
}

//nonzero
TEMPLATE_TEST_CASE("test_sort_search_nonzero","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = int;
    using layout = TestType;
    using tensor_type = gtensor::tensor<value_type,layout>;
    using config_type = typename tensor_type::config_type;
    using index_type = typename tensor_type::index_type;
    using result_config_type = gtensor::config::extend_config_t<config_type,index_type>;
    using result_tensor_type = gtensor::tensor<index_type,layout,result_config_type>;
    using result_container_type = typename config_type::template container<result_tensor_type>;
    using result_tensor_c_order_type = gtensor::tensor<index_type,gtensor::config::c_order,result_config_type>;
    using result_container_c_order_type = typename config_type::template container<result_tensor_c_order_type>;

    using gtensor::nonzero;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(nonzero(std::declval<tensor_type>())),result_container_type>);

    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},result_container_type{result_tensor_type{}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),result_container_c_order_type{result_tensor_c_order_type{},result_tensor_c_order_type{},result_tensor_c_order_type{}}),
        std::make_tuple(tensor_type{1},result_container_type{result_tensor_type{0}}),
        std::make_tuple(tensor_type{0},result_container_type{result_tensor_type{}}),
        std::make_tuple(tensor_type{0,0,0,0},result_container_type{result_tensor_type{}}),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},result_container_type{result_tensor_type{0,1,4,5,6,7,8}}),
        std::make_tuple(
            tensor_type{{2,1,-1,6,0},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{0,0,6,0,3}},
            result_container_type{result_tensor_type{0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4},result_tensor_type{0,1,2,3,0,1,2,4,0,1,3,4,0,1,2,3,4,2,4}}
        ),
        std::make_tuple(
            tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}},
            result_container_type{result_tensor_type{0,0,0,0,0,1,1,1,1,2,2,2},result_tensor_type{0,0,1,1,1,0,0,0,1,0,1,1},result_tensor_type{0,1,0,1,2,0,1,2,2,1,0,2}}
        ),
        std::make_tuple(
            tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}},
            result_container_type{result_tensor_type{},result_tensor_type{},result_tensor_type{}}
        ),
        //trivial view expression
        std::make_tuple(
            tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}}+tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}}+tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}},
            result_container_type{result_tensor_type{0,0,0,0,0,1,1,1,1,2,2,2},result_tensor_type{0,0,1,1,1,0,0,0,1,0,1,1},result_tensor_type{0,1,0,1,2,0,1,2,2,1,0,2}}
        ),
        //non trivial view expression
        std::make_tuple(
            tensor_type{{1,-1,0,0}}+tensor_type{{-1},{0},{1},{1}},
            result_container_type{result_tensor_type{0,0,0,1,1,2,2,2,3,3,3},result_tensor_type{1,2,3,0,1,0,2,3,0,2,3}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = nonzero(ten);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_nonzero_index_map_view","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::nonzero;
    using helpers_for_testing::apply_by_element;

    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),tensor_type{}),
        std::make_tuple(tensor_type{1},tensor_type{1}),
        std::make_tuple(tensor_type{0},tensor_type{}),
        std::make_tuple(tensor_type{0,0,0,0},tensor_type{}),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},tensor_type{1,3,1,4,6,-2,1}),
        std::make_tuple(tensor_type{{2,1,-1,6,0},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{0,0,6,0,3}},tensor_type{2,1,-1,6,8,2,1,5,-1,7,4,2,4,4,2,1,4,6,3}),
        std::make_tuple(tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}},tensor_type{1,3,2,2,2,1,5,2,1,1,1,1})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = ten(nonzero(ten));
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//argwhere
TEMPLATE_TEST_CASE("test_sort_search_argwhere","[test_sort_search]",
    //0tensor layout, 1traverse order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using value_type = int;
    using layout = typename std::tuple_element_t<0,TestType>;
    using traverse_order = typename std::tuple_element_t<1,TestType>;
    using config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<traverse_order>,value_type>;
    using tensor_type = gtensor::tensor<value_type,layout,config_type>;
    using index_type = typename tensor_type::index_type;
    using result_config_type = gtensor::config::extend_config_t<config_type,index_type>;
    using result_tensor_type = gtensor::tensor<index_type,layout,result_config_type>;

    using gtensor::argwhere;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argwhere(std::declval<tensor_type>())),result_tensor_type>);

    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},result_tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),result_tensor_type{}.reshape(0,3)),
        std::make_tuple(tensor_type{1},result_tensor_type{{0}}),
        std::make_tuple(tensor_type{0},result_tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{0,0,0,0},result_tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},result_tensor_type{{0},{1},{4},{5},{6},{7},{8}}),
        std::make_tuple(
            tensor_type{{2,1,-1,6,0},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{0,0,6,0,3}},
            result_tensor_type{{0,0},{0,1},{0,2},{0,3},{1,0},{1,1},{1,2},{1,4},{2,0},{2,1},{2,3},{2,4},{3,0},{3,1},{3,2},{3,3},{3,4},{4,2},{4,4}}
        ),
        std::make_tuple(
            tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}},
            result_tensor_type{{0,0,0},{0,0,1},{0,1,0},{0,1,1},{0,1,2},{1,0,0},{1,0,1},{1,0,2},{1,1,2},{2,0,1},{2,1,0},{2,1,2}}
        ),
        std::make_tuple(
            tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}},
            result_tensor_type{}.reshape(0,3)
        ),
        //trivial expression view
        std::make_tuple(
            tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}}+tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}}+tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}},
            result_tensor_type{{0,0,0},{0,0,1},{0,1,0},{0,1,1},{0,1,2},{1,0,0},{1,0,1},{1,0,2},{1,1,2},{2,0,1},{2,1,0},{2,1,2}}
        ),
        //non trivial view expression
        std::make_tuple(
            tensor_type{{1,-1,0,0}}+tensor_type{{-1},{0},{1},{1}},
            result_tensor_type{{0,1},{0,2},{0,3},{1,0},{1,1},{2,0},{2,2},{2,3},{3,0},{3,2},{3,3}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = argwhere(ten);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//searchsorted
TEMPLATE_TEST_CASE("test_sort_search_searchsorted","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using value_tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::detail::no_value;
    using gtensor::argsort;
    using gtensor::searchsorted;
    using helpers_for_testing::apply_by_element;

    //0tensor,1v,2side,3sorter,4expected
    auto test_data = std::make_tuple(
        //scalar
        //no sorter
        std::make_tuple(tensor_type{1,2,3,4,5},3,std::false_type{},no_value{},2),
        std::make_tuple(tensor_type{1,2,3,4,5},3,std::true_type{},no_value{},3),
        std::make_tuple(tensor_type{2,3,1,5,4},4,std::false_type{},argsort(tensor_type{2,3,1,5,4}),3),
        std::make_tuple(tensor_type{2,3,1,5,4},4,std::true_type{},argsort(tensor_type{2,3,1,5,4}),4),
        //tensor
        //no sorter
        std::make_tuple(tensor_type{1,2,3,4,5},value_tensor_type{2,3,1,5,4,2,1,1,3,2,5,4},std::false_type{},no_value{},tensor<int>{1,2,0,4,3,1,0,0,2,1,4,3}),
        std::make_tuple(tensor_type{1,2,3,4,5},value_tensor_type{2,3,1,5,4,2,1,1,3,2,5,4},std::true_type{},no_value{},tensor<int>{2,3,1,5,4,2,1,1,3,2,5,4}),
        std::make_tuple(tensor_type{1,2,3,4,5},value_tensor_type{{5,4,3,5},{2,2,5,2},{3,1,5,4},{5,5,2,2}},std::false_type{},no_value{},tensor<int>{{4,3,2,4},{1,1,4,1},{2,0,4,3},{4,4,1,1}}),
        std::make_tuple(tensor_type{1,2,3,4,5},value_tensor_type{{5,4,3,5},{2,2,5,2},{3,1,5,4},{5,5,2,2}},std::true_type{},no_value{},tensor<int>{{5,4,3,5},{2,2,5,2},{3,1,5,4},{5,5,2,2}}),
        //sorter
        std::make_tuple(tensor_type{4,3,3,5,2,3,1,2,3,2},value_tensor_type{2,3,1,5,4,2,1,1,3,2,5,4},std::false_type{},argsort(tensor_type{4,3,3,5,2,3,1,2,3,2}),tensor<int>{1,4,0,9,8,1,0,0,4,1,9,8}),
        std::make_tuple(tensor_type{4,3,3,5,2,3,1,2,3,2},value_tensor_type{2,3,1,5,4,2,1,1,3,2,5,4},std::true_type{},argsort(tensor_type{4,3,3,5,2,3,1,2,3,2}),tensor<int>{4,8,1,10,9,4,1,1,8,4,10,9})
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto v = std::get<1>(t);
        auto side = std::get<2>(t);
        auto sorter = std::get<3>(t);
        auto expected = std::get<4>(t);

        auto result = searchsorted(ten,v,side,sorter);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_sort_search_searchsorted_exception","[test_sort_search]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using gtensor::value_error;
    using gtensor::searchsorted;

    REQUIRE_THROWS_AS(searchsorted(tensor_type(2),2),value_error);
    REQUIRE_THROWS_AS(searchsorted(tensor_type{{1,2,3},{4,5,6}},2),value_error);
    REQUIRE_THROWS_AS(searchsorted(tensor_type{{1,2,3},{4,5,6}},tensor_type{2,3,1}),value_error);
    REQUIRE_THROWS_AS(searchsorted(tensor_type{1,2,3,4,5,6},tensor_type{2,3,1},std::false_type{},tensor<int>{{0,1,2},{3,4,5}}),value_error);
}

