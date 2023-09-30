/*
* GTensor - matrix computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include "manipulation.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_split_split_points","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::split;
    using helpers_for_testing::apply_by_element;

    //0ten,1split_points,2axis,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, std::vector<int>{}, dim_type{0}, result_type{tensor_type{}}),
        std::make_tuple(tensor_type{}, std::vector<int>{0}, dim_type{0}, result_type{tensor_type{},tensor_type{}}),
        std::make_tuple(tensor_type{}, std::vector<int>{0,1}, dim_type{0}, result_type{tensor_type{},tensor_type{},tensor_type{}}),
        std::make_tuple(tensor_type{}.reshape(5,0), std::vector<int>{}, dim_type{0}, result_type{tensor_type{}.reshape(5,0).copy(order{})}),
        std::make_tuple(tensor_type{}.reshape(5,0), std::vector<int>{0}, dim_type{0}, result_type{tensor_type{}.reshape(0,0).copy(order{}), tensor_type{}.reshape(5,0).copy(order{})}),
        std::make_tuple(
            tensor_type{}.reshape(5,0),
            std::vector<int>{0,1},
            dim_type{0},
            result_type{tensor_type{}.reshape(0,0).copy(order{}), tensor_type{}.reshape(1,0).copy(order{}), tensor_type{}.reshape(4,0).copy(order{})}
        ),
        std::make_tuple(tensor_type{1}, std::vector<int>{}, dim_type{0}, result_type{tensor_type{1}}),
        std::make_tuple(tensor_type{1}, std::vector<int>{0}, dim_type{0}, result_type{tensor_type{},tensor_type{1}}),
        std::make_tuple(tensor_type{1}, std::vector<int>{1}, dim_type{0}, result_type{tensor_type{1},tensor_type{}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{}, dim_type{0}, result_type{tensor_type{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{2}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{2,4}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{0,3,5}, dim_type{0}, result_type{tensor_type{}, tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{1,2,3,4}, dim_type{0}, result_type{tensor_type{1}, tensor_type{2}, tensor_type{3}, tensor_type{4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::initializer_list<int>{2,4}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, gtensor::tensor<int>{2,4}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            std::initializer_list<index_type>{0,2},
            dim_type{0},
            result_type{tensor_type{}.reshape(0,4).copy(order{}), tensor_type{{1,2,3,4},{5,6,7,8}}, tensor_type{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            std::initializer_list<index_type>{1,2},
            dim_type{0},
            result_type{tensor_type{{1,2,3,4}}, tensor_type{{5,6,7,8}}, tensor_type{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            std::initializer_list<int>{1,2},
            dim_type{1},
            result_type{tensor_type{{1},{5},{9},{13}}, tensor_type{{2},{6},{10},{14}}, tensor_type{{3,4},{7,8},{11,12},{15,16}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            std::vector<int>{1,2},
            dim_type{0},
            result_type{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            std::vector<int>{1,2},
            dim_type{-3},
            result_type{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            gtensor::tensor<index_type>{1},
            dim_type{1},
            result_type{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            gtensor::tensor<index_type>{1},
            dim_type{-2},
            result_type{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            std::initializer_list<std::size_t>{1,2},
            dim_type{2},
            result_type{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            std::initializer_list<std::size_t>{1,2},
            dim_type{-1},
            result_type{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto split_points = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = split(ten, split_points, axis);
        REQUIRE(expected.size() == result.size());
        auto result_it = result.begin();
        for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
            REQUIRE(*result_it == *expected_it);
        }
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_split_equal_parts","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::split;
    using helpers_for_testing::apply_by_element;

    //0ten,1parts_number,2axis,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, index_type{1}, dim_type{0}, result_type{tensor_type{}}),
        std::make_tuple(tensor_type{}, index_type{2}, dim_type{0}, result_type{tensor_type{},tensor_type{}}),
        std::make_tuple(tensor_type{}, index_type{3}, dim_type{0}, result_type{tensor_type{},tensor_type{},tensor_type{}}),
        std::make_tuple(
            tensor_type{}.reshape(3,0),
            index_type{3},
            dim_type{0},
            result_type{tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{})}
        ),
        std::make_tuple(tensor_type{1}, index_type{1}, dim_type{0}, result_type{tensor_type{1}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, index_type{1}, dim_type{0}, result_type{tensor_type{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, int{5}, dim_type{0}, result_type{tensor_type{1}, tensor_type{2}, tensor_type{3}, tensor_type{4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, index_type{2}, dim_type{0}, result_type{tensor_type{1,2,3}, tensor_type{4,5,6}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::size_t{3}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5,6}}),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            index_type{2},
            dim_type{0},
            result_type{tensor_type{{1,2,3,4},{5,6,7,8}}, tensor_type{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            index_type{2},
            dim_type{1},
            result_type{tensor_type{{1,2},{5,6},{9,10},{13,14}}, tensor_type{{3,4},{7,8},{11,12},{15,16}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            index_type{3},
            dim_type{0},
            result_type{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            index_type{3},
            dim_type{-3},
            result_type{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{2},
            dim_type{1},
            result_type{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{2},
            dim_type{-2},
            result_type{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{3},
            dim_type{2},
            result_type{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{3},
            dim_type{-1},
            result_type{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto parts_number = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = split(ten, parts_number, axis);
        REQUIRE(expected.size() == result.size());
        auto result_it = result.begin();
        for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
            REQUIRE(*result_it == *expected_it);
        }
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_split_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::split;
    using helpers_for_testing::apply_by_element;

    //0tensor,1split_arg,2axis
    auto test_data = std::make_tuple(
        //split by points
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{1},1),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::vector<int>{1},2),
        //split by equal parts
        std::make_tuple(tensor_type{1,2,3,4,5},1,1),
        std::make_tuple(tensor_type{1,2,3,4,5},0,0),
        std::make_tuple(tensor_type{1,2,3,4,5},2,0),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},1,2),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},0,0),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},4,0)
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto split_arg = std::get<1>(t);
        auto axis = std::get<2>(t);
        REQUIRE_THROWS_AS(split(ten,split_arg,axis), value_error);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_vsplit","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::vsplit;
    using helpers_for_testing::apply_by_element;

    SECTION("test_vsplit_nothrow")
    {
        //0tensor,1split_arg,2expected
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::vector<int>{2},result_type{tensor_type{{1,2},{3,4}},tensor_type{{5,6},{7,8},{9,10}}}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::vector<int>{1,3},result_type{tensor_type{{1,2}},tensor_type{{3,4},{5,6}},tensor_type{{7,8},{9,10}}}),
            std::make_tuple(
                tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},
                std::initializer_list<int>{1,2,3,4},
                result_type{tensor_type{{1,2}},tensor_type{{3,4}},tensor_type{{5,6}},tensor_type{{7,8}},tensor_type{{9,10}}}
            ),
            std::make_tuple(
                tensor_type{{{1},{2}},{{3},{4}},{{5},{6}},{{7},{8}},{{9},{10}}},
                gtensor::tensor<std::size_t>{1,3},
                result_type{tensor_type{{{1},{2}}},tensor_type{{{3},{4}},{{5},{6}}},tensor_type{{{7},{8}},{{9},{10}}}}
            ),
            //split by equal parts
            std::make_tuple(tensor_type{}.reshape(1,0),1,result_type{tensor_type{}.reshape(1,0).copy(order{})}),
            std::make_tuple(tensor_type{}.reshape(3,0),3,result_type{tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{})}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},1,result_type{tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}}}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},5,result_type{tensor_type{{1,2}},tensor_type{{3,4}},tensor_type{{5,6}},tensor_type{{7,8}},tensor_type{{9,10}}}),
            std::make_tuple(
                tensor_type{{{1},{2}},{{3},{4}},{{5},{6}},{{7},{8}}},
                2,
                result_type{tensor_type{{{1},{2}},{{3},{4}}},tensor_type{{{5},{6}},{{7},{8}}}}
            )
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = vsplit(ten, split_arg);
            REQUIRE(expected.size() == result.size());
            auto result_it = result.begin();
            for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
                REQUIRE(*result_it == *expected_it);
            }
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_vsplit_exception")
    {
        using gtensor::value_error;
        //0tensor,1split_arg
        auto test_data = std::make_tuple(
            std::make_tuple(tensor_type{1,2,3},std::vector<int>{1}),
            std::make_tuple(tensor_type{1,2,3},3),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},2)
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            REQUIRE_THROWS_AS(vsplit(ten, split_arg), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_hsplit","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::hsplit;
    using helpers_for_testing::apply_by_element;

    SECTION("test_hsplit_nothrow")
    {
        //0tensor,1split_arg,2expected
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{2},result_type{tensor_type{1,2},tensor_type{3,4,5}}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},std::vector<int>{1},result_type{tensor_type{{1},{5},{9}},tensor_type{{2,3,4},{6,7,8},{10,11,12}}}),
            std::make_tuple(
                tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},
                std::vector<int>{1,3},
                result_type{tensor_type{{1},{5},{9}},tensor_type{{2,3},{6,7},{10,11}},tensor_type{{4},{8},{12}}}
            ),
            std::make_tuple(
                tensor_type{{{1},{2},{3},{4}},{{5},{6},{7},{8}},{{9},{10},{11},{12}}},
                gtensor::tensor<std::size_t>{1,3},
                result_type{tensor_type{{{1}},{{5}},{{9}}},tensor_type{{{2},{3}},{{6},{7}},{{10},{11}}},tensor_type{{{4}},{{8}},{{12}}}}
            ),
            //split by equal parts
            std::make_tuple(tensor_type{}.reshape(0,3),3,result_type{tensor_type{}.reshape(0,1).copy(order{}),tensor_type{}.reshape(0,1).copy(order{}),tensor_type{}.reshape(0,1).copy(order{})}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1,result_type{tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}}}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},2,result_type{tensor_type{{1,2},{5,6},{9,10}},tensor_type{{3,4},{7,8},{11,12}}}),
            std::make_tuple(
                tensor_type{{{1},{2},{3},{4}},{{5},{6},{7},{8}},{{9},{10},{11},{12}}},
                2,
                result_type{tensor_type{{{1},{2}},{{5},{6}},{{9},{10}}},tensor_type{{{3},{4}},{{7},{8}},{{11},{12}}}}
            )
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = hsplit(ten, split_arg);
            REQUIRE(expected.size() == result.size());
            auto result_it = result.begin();
            for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
                REQUIRE(*result_it == *expected_it);
            }
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_hsplit_exception")
    {
        using gtensor::value_error;
        //0tensor,1split_arg
        auto test_data = std::make_tuple(
            std::make_tuple(tensor_type{1},2),
            std::make_tuple(tensor_type{1,2,3},2),
            std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},2)
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            REQUIRE_THROWS_AS(hsplit(ten, split_arg), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_hsplit_hstack","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using order = typename tensor_type::order;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::hsplit;
    using gtensor::hstack;
    using helpers_for_testing::apply_by_element;

    SECTION("test_hsplit_hstack")
    {
        //0tensor,1split_arg
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{2}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},std::vector<int>{1}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},std::vector<int>{1,3}),
            std::make_tuple(tensor_type{{{1},{2},{3},{4}},{{5},{6},{7},{8}},{{9},{10},{11},{12}}},gtensor::tensor<std::size_t>{1,3}),
            //split by equal parts
            std::make_tuple(tensor_type{}.reshape(0,3),3),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},2),
            std::make_tuple(tensor_type{{{1},{2},{3},{4}},{{5},{6},{7},{8}},{{9},{10},{11},{12}}},2)
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            auto expected = ten;
            auto result = hstack(hsplit(ten, split_arg));
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_hstack_hsplit")
    {
        //0split_arg,1parts
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(std::vector<int>{2},result_type{tensor_type{1,2},tensor_type{3,4,5}}),
            std::make_tuple(std::vector<int>{1},result_type{tensor_type{{1},{5},{9}},tensor_type{{2,3,4},{6,7,8},{10,11,12}}}),
            std::make_tuple(std::vector<int>{1,3},result_type{tensor_type{{1},{5},{9}},tensor_type{{2,3},{6,7},{10,11}},tensor_type{{4},{8},{12}}}),
            std::make_tuple(gtensor::tensor<std::size_t>{1,3},result_type{tensor_type{{{1}},{{5}},{{9}}},tensor_type{{{2},{3}},{{6},{7}},{{10},{11}}},tensor_type{{{4}},{{8}},{{12}}}}),
            //split by equal parts
            std::make_tuple(3,result_type{tensor_type{}.reshape(0,1).copy(order{}),tensor_type{}.reshape(0,1).copy(order{}),tensor_type{}.reshape(0,1).copy(order{})}),
            std::make_tuple(1,result_type{tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}}}),
            std::make_tuple(2,result_type{tensor_type{{1,2},{5,6},{9,10}},tensor_type{{3,4},{7,8},{11,12}}}),
            std::make_tuple(2,result_type{tensor_type{{{1},{2}},{{5},{6}},{{9},{10}}},tensor_type{{{3},{4}},{{7},{8}},{{11},{12}}}})
        );
        auto test = [](const auto& t){
            auto split_arg = std::get<0>(t);
            auto parts = std::get<1>(t);
            auto expected = parts;
            auto result = hsplit(hstack(parts), split_arg);
            REQUIRE(expected.size() == result.size());
            auto result_it = result.begin();
            for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
                REQUIRE(*result_it == *expected_it);
            }
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_vsplit_vstack","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using order = typename tensor_type::order;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::vstack;
    using gtensor::vsplit;
    using helpers_for_testing::apply_by_element;

    SECTION("test_vsplit_vstack")
    {
        //0tensor,1split_arg
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::vector<int>{2}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::vector<int>{1,3}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::initializer_list<int>{1,2,3,4}),
            std::make_tuple(tensor_type{{{1},{2}},{{3},{4}},{{5},{6}},{{7},{8}},{{9},{10}}},gtensor::tensor<std::size_t>{1,3}),
            //split by equal parts
            std::make_tuple(tensor_type{}.reshape(1,0),1),
            std::make_tuple(tensor_type{}.reshape(3,0),3),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},1),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},5),
            std::make_tuple(tensor_type{{{1},{2}},{{3},{4}},{{5},{6}},{{7},{8}}},2)
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            auto expected = ten;
            auto result = vstack(vsplit(ten, split_arg));
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_vstack_vsplit")
    {
        //0split_arg,1parts
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(std::vector<int>{2},result_type{tensor_type{{1,2},{3,4}},tensor_type{{5,6},{7,8},{9,10}}}),
            std::make_tuple(std::vector<int>{1,3},result_type{tensor_type{{1,2}},tensor_type{{3,4},{5,6}},tensor_type{{7,8},{9,10}}}),
            std::make_tuple(std::initializer_list<int>{1,2,3,4},result_type{tensor_type{{1,2}},tensor_type{{3,4}},tensor_type{{5,6}},tensor_type{{7,8}},tensor_type{{9,10}}}),
            std::make_tuple(gtensor::tensor<std::size_t>{1,3},result_type{tensor_type{{{1},{2}}},tensor_type{{{3},{4}},{{5},{6}}},tensor_type{{{7},{8}},{{9},{10}}}}),
            //split by equal parts
            std::make_tuple(1,result_type{tensor_type{}.reshape(1,0).copy(order{})}),
            std::make_tuple(3,result_type{tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{})}),
            std::make_tuple(1,result_type{tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}}}),
            std::make_tuple(5,result_type{tensor_type{{1,2}},tensor_type{{3,4}},tensor_type{{5,6}},tensor_type{{7,8}},tensor_type{{9,10}}}),
            std::make_tuple(2,result_type{tensor_type{{{1},{2}},{{3},{4}}},tensor_type{{{5},{6}},{{7},{8}}}})
        );
        auto test = [](const auto& t){
            auto split_arg = std::get<0>(t);
            auto parts = std::get<1>(t);
            auto expected = parts;
            auto result = vsplit(vstack(parts), split_arg);
            REQUIRE(expected.size() == result.size());
            auto result_it = result.begin();
            for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
                REQUIRE(*result_it == *expected_it);
            }
        };
        apply_by_element(test, test_data);
    }
}

//tensor of tensors variadic arg should be treated by combine routines as basic_tensor not as container of tensors
TEST_CASE("test_tensor_of_tensors_variadic_arg","[test_manipulation]")
{
    using gtensor::tensor;
    using value_type = tensor<double>;
    using tensor_type = tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    using gtensor::vstack;
    using gtensor::hstack;
    SECTION("stack")
    {
        using gtensor::stack;
        REQUIRE(std::is_same_v<decltype(stack(0,tensor_type{value_type{1}})),tensor_type>);
        REQUIRE(stack(0,tensor_type{value_type{1}}) == tensor_type{{value_type{1}}});
        REQUIRE(stack(0,tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}}) == tensor_type{{value_type{1},value_type{2,3},value_type{4,5,6}}});
    }
    SECTION("concatenate")
    {
        using gtensor::concatenate;
        REQUIRE(std::is_same_v<decltype(concatenate(0,tensor_type{value_type{1}})),tensor_type>);
        REQUIRE(concatenate(0,tensor_type{value_type{1}}) == tensor_type{value_type{1}});
        REQUIRE(concatenate(0,tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}}) == tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}});
    }
    SECTION("vstack")
    {
        using gtensor::vstack;
        REQUIRE(std::is_same_v<decltype(vstack(tensor_type{value_type{1}})),tensor_type>);
        REQUIRE(vstack(tensor_type{value_type{1}}) == tensor_type{{value_type{1}}});
        REQUIRE(vstack(tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}}) == tensor_type{{value_type{1},value_type{2,3},value_type{4,5,6}}});
    }
    SECTION("hstack")
    {
        using gtensor::hstack;
        REQUIRE(std::is_same_v<decltype(hstack(tensor_type{value_type{1}})),tensor_type>);
        REQUIRE(hstack(tensor_type{value_type{1}}) == tensor_type{value_type{1}});
        REQUIRE(hstack(tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}}) == tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}});
    }
}

