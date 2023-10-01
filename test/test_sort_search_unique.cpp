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

//unique
TEMPLATE_TEST_CASE("test_sort_search_unique_return_index_inverse_count","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using order = typename tensor_type::order;
    using config_type = typename tensor_type::config_type;
    using index_type = typename tensor_type::index_type;
    using index_tensor_type = gtensor::tensor<index_type,order,config_type>;

    using gtensor::unique;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2expected_unique,3expected_counts
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{4},0,tensor_type{4},index_tensor_type{1}),
        std::make_tuple(tensor_type{4,4,4,4},0,tensor_type{4},index_tensor_type{4}),
        std::make_tuple(tensor_type{4,3,1,2,1,3,3,4,4,5,2,5},0,tensor_type{1,2,3,4,5},index_tensor_type{2,2,3,3,2}),
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},0,tensor_type{1,2,3,4,5},index_tensor_type{9,3,8,6,4}),
        std::make_tuple(tensor_type{{1,2,3}},0,tensor_type{{1,2,3}},index_tensor_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{1,2,3},{1,2,3},{1,2,3}},0,tensor_type{{1,2,3}},index_tensor_type{4}),
        std::make_tuple(tensor_type{{7,8,9},{1,2,3},{1,2,3},{4,5,6},{7,8,9},{1,2,3},{10,11,12},{4,5,6}},0,tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},index_tensor_type{3,2,2,1}),
        std::make_tuple(tensor_type{{7,1,1,4,7,10,4},{8,2,2,5,8,11,5},{9,3,3,6,9,12,6}},1,tensor_type{{1,4,7,10},{2,5,8,11},{3,6,9,12}},index_tensor_type{2,2,2,1}),
        std::make_tuple(
            tensor_type{{0,1,1,1,0,1,0,1,1,0,2,0,2,2,1,0,2,0,2,0,2,1,1,0,2,2,2,1,2,0},{2,0,2,2,0,0,0,1,2,0,2,0,1,2,1,2,0,0,0,1,0,0,0,1,1,2,2,0,0,0},{0,0,1,1,0,2,1,1,0,1,2,2,2,1,1,0,2,1,2,1,1,1,1,1,0,2,0,2,1,1}},
            1,
            tensor_type{{0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2},{0,0,0,1,2,0,0,0,1,2,2,0,0,1,1,2,2,2},{0,1,2,1,0,0,1,2,1,0,1,1,2,0,2,0,1,2}},
            index_tensor_type{1,4,1,2,2,1,2,2,2,1,2,2,2,1,1,1,1,2}
        ),
        std::make_tuple(
            tensor_type{
                {{1,0},{1,0},{0,1},{1,1},{0,1},{0,0},{0,1},{1,0},{1,1},{1,1},{0,0},{1,1},{0,1},{1,1},{0,1},{1,1},{0,0},{0,0},{0,0},{1,1},{0,0},{0,1},{0,0},{0,0},{0,0},{1,1},{1,1},{1,0},{1,0},{0,0}},
                {{0,0},{0,0},{1,1},{0,1},{0,1},{0,1},{1,1},{0,0},{1,1},{1,1},{1,0},{0,0},{1,0},{1,1},{1,0},{1,1},{1,1},{1,1},{1,0},{1,1},{1,0},{1,1},{0,0},{1,0},{0,1},{0,0},{0,0},{1,1},{1,1},{1,0}}
            },
            1,
            tensor_type{{{0,0},{0,0},{0,0},{0,0},{0,1},{0,1},{0,1},{1,0},{1,0},{1,1},{1,1},{1,1}},{{0,0},{0,1},{1,0},{1,1},{0,1},{1,0},{1,1},{0,0},{1,1},{0,0},{0,1},{1,1}}},
            index_tensor_type{1,2,5,2,1,2,3,3,2,3,1,5}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto expected_unique = std::get<2>(t);
        auto expected_counts = std::get<3>(t);

        auto result = unique(ten,std::true_type{},std::true_type{},std::true_type{},axis);
        auto result_unique = std::get<0>(result);
        auto result_index = std::get<1>(result);
        auto result_inverse = std::get<2>(result);
        auto result_counts = std::get<3>(result);
        auto result_elements_at_index = gtensor::take(ten,result_index,axis);
        auto result_reconstruction = gtensor::take(result_unique,result_inverse,axis);

        REQUIRE(result_unique == expected_unique);
        REQUIRE(result_counts == expected_counts);
        REQUIRE(result_elements_at_index == expected_unique);
        REQUIRE(result_reconstruction == ten);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_unique_return_selected","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using order = typename tensor_type::order;
    using config_type = typename tensor_type::config_type;
    using index_type = typename tensor_type::index_type;
    using index_tensor_type = gtensor::tensor<index_type,order,config_type>;

    using gtensor::unique;
    using helpers_for_testing::apply_by_element;

    //0ten,1axis,2expected_unique,3expected_counts
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},0,tensor_type{1,2,3,4,5},index_tensor_type{9,3,8,6,4}),
        std::make_tuple(tensor_type{{0,1},{1,2},{0,2},{0,1},{1,0},{1,2},{0,2},{2,0},{1,0},{2,1},{1,2},{0,2},{1,1}},0,tensor_type{{0,1},{0,2},{1,0},{1,1},{1,2},{2,0},{2,1}},index_tensor_type{2,3,2,1,3,1,1})
    );
    SECTION("unique")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto expected_unique = std::get<2>(t);
            auto result_unique = unique(ten,std::false_type{},std::false_type{},std::false_type{},axis);
            REQUIRE(result_unique == expected_unique);
        };
        apply_by_element(test,test_data);
    }
    SECTION("unique_index")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto expected_unique = std::get<2>(t);
            auto result = unique(ten,std::true_type{},std::false_type{},std::false_type{},axis);
            auto result_unique = std::get<0>(result);
            auto result_index = std::get<1>(result);
            auto result_elements_at_index = gtensor::take(ten,result_index,axis);
            REQUIRE(result_unique == expected_unique);
            REQUIRE(result_elements_at_index == expected_unique);
        };
        apply_by_element(test,test_data);
    }
    SECTION("unique_inverse")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto expected_unique = std::get<2>(t);
            auto result = unique(ten,std::false_type{},std::true_type{},std::false_type{},axis);
            auto result_unique = std::get<0>(result);
            auto result_inverse = std::get<1>(result);
            auto result_reconstruction = gtensor::take(result_unique,result_inverse,axis);
            REQUIRE(result_unique == expected_unique);
            REQUIRE(result_reconstruction == ten);
        };
        apply_by_element(test,test_data);
    }
    SECTION("unique_counts")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto expected_unique = std::get<2>(t);
            auto expected_counts = std::get<3>(t);
            auto result = unique(ten,std::false_type{},std::false_type{},std::true_type{},axis);
            auto result_unique = std::get<0>(result);
            auto result_counts = std::get<1>(result);
            REQUIRE(result_unique == expected_unique);
            REQUIRE(result_counts == expected_counts);
        };
        apply_by_element(test,test_data);
    }
    SECTION("unique_index_counts")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto expected_unique = std::get<2>(t);
            auto expected_counts = std::get<3>(t);
            auto result = unique(ten,std::true_type{},std::false_type{},std::true_type{},axis);
            auto result_unique = std::get<0>(result);
            auto result_index = std::get<1>(result);
            auto result_counts = std::get<2>(result);
            auto result_elements_at_index = gtensor::take(ten,result_index,axis);
            REQUIRE(result_unique == expected_unique);
            REQUIRE(result_elements_at_index == expected_unique);
            REQUIRE(result_counts == expected_counts);
        };
        apply_by_element(test,test_data);
    }
    SECTION("unique_inverse_counts")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto expected_counts = std::get<3>(t);
            auto expected_unique = std::get<2>(t);
            auto result = unique(ten,std::false_type{},std::true_type{},std::true_type{},axis);
            auto result_unique = std::get<0>(result);
            auto result_inverse = std::get<1>(result);
            auto result_counts = std::get<2>(result);
            auto result_reconstruction = gtensor::take(result_unique,result_inverse,axis);
            REQUIRE(result_unique == expected_unique);
            REQUIRE(result_reconstruction == ten);
            REQUIRE(result_counts == expected_counts);
        };
        apply_by_element(test,test_data);
    }
    SECTION("unique_index_inverse")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto expected_unique = std::get<2>(t);
            auto result = unique(ten,std::true_type{},std::true_type{},std::false_type{},axis);
            auto result_unique = std::get<0>(result);
            auto result_index = std::get<1>(result);
            auto result_inverse = std::get<2>(result);
            auto result_elements_at_index = gtensor::take(ten,result_index,axis);
            auto result_reconstruction = gtensor::take(result_unique,result_inverse,axis);
            REQUIRE(result_unique == expected_unique);
            REQUIRE(result_elements_at_index == expected_unique);
            REQUIRE(result_reconstruction == ten);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_sort_search_unique_default_args","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;

    using gtensor::unique;
    using helpers_for_testing::apply_by_element;

    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{4,3,1,2,1,3,3,4,4,5,2,5},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5}},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{2},{1},{3},{5},{4}},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{2},{1},{1},{5},{4},{3},{5},{2}},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},tensor_type{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(tensor_type{{7,8,9},{1,2,3},{1,2,3},{4,5,6},{7,8,9},{10,11,12},{4,5,6}},tensor_type{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{1,5}},{{1,2},{1,2},{1,3}},{{1,2},{3,4},{1,5}}},tensor_type{1,2,3,4,5})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = unique(ten);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_unique_no_axis","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::unique;
    using helpers_for_testing::apply_by_element;

    //0ten,1expected_unique,2expected_counts
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{4,3,1,2,1,3,3,4,4,5,2,5},tensor_type{1,2,3,4,5},tensor<int>{2,2,3,3,2}),
        std::make_tuple(tensor_type{{{1,3,4,3,1},{5,1,5,2,2}},{{3,4,3,3,1},{5,1,3,2,1}},{{1,4,5,4,4},{1,4,1,3,4}}},tensor_type{1,2,3,4,5},tensor<int>{9,3,7,7,4})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected_unique = std::get<1>(t);
        auto expected_counts = std::get<2>(t);

        auto result = unique(ten,std::true_type{},std::true_type{},std::true_type{});
        auto result_unique = std::get<0>(result);
        auto result_index = std::get<1>(result);
        auto result_inverse = std::get<2>(result);
        auto result_counts = std::get<3>(result);
        auto result_elements_at_index = gtensor::take(ten,result_index);
        auto result_reconstruction = gtensor::take(result_unique,result_inverse);

        REQUIRE(result_unique == expected_unique);
        REQUIRE(result_counts == expected_counts);
        REQUIRE(result_elements_at_index == expected_unique);
        REQUIRE(result_reconstruction == ten.flatten());
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_unique_empty","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using index_type = typename tensor_type::index_type;
    using gtensor::unique;
    using helpers_for_testing::apply_by_element;
    //0ten,1axis,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},0,std::make_tuple(tensor_type{},tensor<index_type>{},tensor<index_type>{},tensor<index_type>{})),
        std::make_tuple(tensor_type{}.reshape(2,3,0),0,std::make_tuple(tensor_type{}.reshape(1,3,0),tensor<index_type>{0},tensor<index_type>{0,0},tensor<index_type>{2})),
        std::make_tuple(tensor_type{}.reshape(2,3,0),1,std::make_tuple(tensor_type{}.reshape(2,1,0),tensor<index_type>{0},tensor<index_type>{0,0,0},tensor<index_type>{3})),
        std::make_tuple(tensor_type{}.reshape(2,3,0),2,std::make_tuple(tensor_type{}.reshape(2,3,0),tensor<index_type>{},tensor<index_type>{},tensor<index_type>{}))
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = unique(ten,std::true_type{},std::true_type{},std::true_type{},axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_unique_empty_no_axis","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using index_type = typename tensor_type::index_type;
    using gtensor::unique;
    using helpers_for_testing::apply_by_element;
    //0ten,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},std::make_tuple(tensor_type{},tensor<index_type>{},tensor<index_type>{},tensor<index_type>{})),
        std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(tensor_type{},tensor<index_type>{},tensor<index_type>{},tensor<index_type>{}))
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = unique(ten,std::true_type{},std::true_type{},std::true_type{});
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_unique_0d","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using index_type = typename tensor_type::index_type;
    using gtensor::unique;
    using helpers_for_testing::apply_by_element;

    const auto test_tensor = tensor_type(2);
    auto result = unique(test_tensor,std::true_type{},std::true_type{},std::true_type{});
    auto expected = std::make_tuple(tensor_type{2},tensor<index_type>{0},tensor<index_type>{0},tensor<index_type>{1});
    REQUIRE(result == expected);
}

TEST_CASE("test_sort_search_unique_exception","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::unique;

    REQUIRE_THROWS_AS(unique(tensor_type(1),std::false_type{},std::false_type{},std::false_type{},0),value_error);
    REQUIRE_THROWS_AS(unique(tensor_type(1),std::false_type{},std::false_type{},std::false_type{},1),value_error);
    REQUIRE_THROWS_AS(unique(tensor_type{1,2,3,4,5},std::false_type{},std::false_type{},std::false_type{},1),value_error);
    REQUIRE_THROWS_AS(unique(tensor_type{{1,2,3},{4,5,6}},std::false_type{},std::false_type{},std::false_type{},2),value_error);
}

