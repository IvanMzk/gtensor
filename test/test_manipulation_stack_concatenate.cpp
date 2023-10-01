/*
* GTensor - computation library
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

TEMPLATE_TEST_CASE("test_stack_nothrow","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using dim_type = typename tensor_type::dim_type;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}), tensor_type{}.reshape(1,0)),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{}), tensor_type{}.reshape(1,0)),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}), tensor_type{}.reshape(0,1)),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}.reshape(3,0)),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}.reshape(0,3)),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(3,1,0)),
        std::make_tuple(dim_type{-2}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(3,1,0)),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(1,3,0)),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(1,3,0)),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(1,0,3)),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1},{2},{3}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1,2,3}}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1,2,3,4},tensor_type{5,6,7,8},tensor_type{9,10,11,12}), tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1,2,3,4},tensor_type{5,6,7,8},tensor_type{9,10,11,12}), tensor_type{{1,5,9},{2,6,10},{3,7,11},{4,8,12}}),
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{-2},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{7,8,9},{13,14,15},{19,20,21}},{{4,5,6},{10,11,12},{16,17,18},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{-1},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{7,8,9},{13,14,15},{19,20,21}},{{4,5,6},{10,11,12},{16,17,18},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,7,13,19},{2,8,14,20},{3,9,15,21}},{{4,10,16,22},{5,11,17,23},{6,12,18,24}}}
        )
    );
    SECTION("test_stack_variadic_nothrow")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);

            auto apply_tensors = [&axis](const auto&...tensors_){
                return stack(axis, tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_stack_container_nothrow")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(order{}))>;
        auto test_concatenate_container = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(order{})...};}, tensors);
            auto result = stack(axis, container);
            REQUIRE(result == expected);
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEST_CASE("test_stack_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0axis,1tensors
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{})),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{1})),
        std::make_tuple(dim_type{4}, std::make_tuple(tensor_type({1,2,3},value_type{}))),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{1}, tensor_type{1})),
        std::make_tuple(dim_type{4}, std::make_tuple(tensor_type({1,2,3},value_type{}), tensor_type({1,2,3},value_type{}))),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}, tensor_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1}, tensor_type{})),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type({1,2,3},value_type{}), tensor_type{})),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}, tensor_type({1,2,3},value_type{}))),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type({1,2,3},value_type{}), tensor_type({1,2,3},value_type{}), tensor_type({1,1,2,3},value_type{}))),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type({1,2,3},value_type{}), tensor_type({2,2,3},value_type{}), tensor_type({1,2,3},value_type{})))
    );
    SECTION("test_stack_variadic_exception")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto apply_tensors = [&axis](const auto&...tensors_){
                return stack(axis, tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_stack_container_exception")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy())>;
        auto test_concatenate_container = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy()...};}, tensors);
            REQUIRE_THROWS_AS(stack(axis, container), value_error);
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEST_CASE("test_stack_common_type","[test_manipulation]")
{
    using tensor_int32_type = gtensor::tensor<int>;
    using tensor_int64_type = gtensor::tensor<std::int64_t>;
    using tensor_double_type = gtensor::tensor<double>;
    using dim_type = std::common_type_t<typename tensor_int32_type::dim_type, typename tensor_int64_type::dim_type, typename tensor_double_type::dim_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_int32_type{1},tensor_int32_type{2},tensor_int64_type{3}), tensor_int64_type{{1},{2},{3}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_int32_type{1},tensor_double_type{2},tensor_int64_type{3}), tensor_double_type{{1,2,3}})
    );
    auto test = [](const auto& t){
        auto axis = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&axis](const auto&...tensors_){
            return stack(axis, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_stack_common_order","[test_manipulation]")
{

    using value_type = int;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using dim_type = typename tensor<value_type>::dim_type;
    using traverse_order = typename tensor<value_type>::config_type::order;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(
            dim_type{0},
            std::make_tuple(
                tensor<value_type, c_order>{{1,2,3},{4,5,6}},
                tensor<value_type, c_order>{{7,8,9},{10,11,12}},
                tensor<value_type, c_order>{{13,14,15},{16,17,18}},
                tensor<value_type, c_order>{{19,20,21},{22,23,24}}
            ),
            tensor<value_type, c_order>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(
                tensor<value_type, f_order>{{1,2,3},{4,5,6}},
                tensor<value_type, f_order>{{7,8,9},{10,11,12}},
                tensor<value_type, f_order>{{13,14,15},{16,17,18}},
                tensor<value_type, f_order>{{19,20,21},{22,23,24}}
            ),
            tensor<value_type, f_order>{{{1,2,3},{7,8,9},{13,14,15},{19,20,21}},{{4,5,6},{10,11,12},{16,17,18},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(
                tensor<value_type, f_order>{{1,2,3},{4,5,6}},
                tensor<value_type, c_order>{{7,8,9},{10,11,12}},
                tensor<value_type, c_order>{{13,14,15},{16,17,18}},
                tensor<value_type, f_order>{{19,20,21},{22,23,24}}),
            tensor<value_type, traverse_order>{{{1,7,13,19},{2,8,14,20},{3,9,15,21}},{{4,10,16,22},{5,11,17,23},{6,12,18,24}}}
        )
    );
    auto test = [](const auto& t){
        auto axis = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&axis](const auto&...tensors_){
            return stack(axis, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_concatenate","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using dim_type = typename tensor_type::dim_type;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1}), tensor_type{1}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{1,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{},tensor_type{1},tensor_type{},tensor_type{2},tensor_type{3},tensor_type{}), tensor_type{1,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1},tensor_type{2,3},tensor_type{4,5,6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{1},tensor_type{2,3},tensor_type{4,5,6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}.reshape(1,0),tensor_type{}.reshape(2,0)), tensor_type{}.reshape(3,0)),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}.reshape(0,1),tensor_type{}.reshape(0,2)), tensor_type{}.reshape(0,3)),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(dim_type{-2}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}.reshape(0,2),tensor_type{{1,2},{3,4}},tensor_type{}.reshape(0,2),tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}.reshape(2,0),tensor_type{{1,2},{3,4}},tensor_type{}.reshape(2,0),tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{}.reshape(0,2,2),tensor_type{{{9,10},{11,12}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{}.reshape(2,0,2),tensor_type{{{9,10}},{{11,12}}}),
            tensor_type{{{1,2},{3,4},{9,10}},{{5,6},{7,8},{11,12}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{}.reshape(2,2,0),tensor_type{{{9},{10}},{{11},{12}}}),
            tensor_type{{{1,2,9},{3,4,10}},{{5,6,11},{7,8,12}}}
        ),
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12},{13,14,15}},tensor_type{{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15},{16,17,18},{19,20,21},{22,23,24}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13},{14}}),
            tensor_type{{1,2,5,7,8,9,13},{3,4,6,10,11,12,14}}
        ),
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10},{11,12}}},tensor_type{{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}
        ),
        std::make_tuple(
            dim_type{-3},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10},{11,12}}},tensor_type{{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10}},{{11,12}}},tensor_type{{{13,14},{15,16},{17,18}},{{19,20},{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4},{9,10},{13,14},{15,16},{17,18}},{{5,6},{7,8},{11,12},{19,20},{21,22},{23,24}}}
        ),
        std::make_tuple(
            dim_type{-2},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10}},{{11,12}}},tensor_type{{{13,14},{15,16},{17,18}},{{19,20},{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4},{9,10},{13,14},{15,16},{17,18}},{{5,6},{7,8},{11,12},{19,20},{21,22},{23,24}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9},{10}},{{11},{12}}},tensor_type{{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}),
            tensor_type{{{1,2,9,13,14,15},{3,4,10,16,17,18}},{{5,6,11,19,20,21},{7,8,12,22,23,24}}}
        ),
        std::make_tuple(
            dim_type{-1},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9},{10}},{{11},{12}}},tensor_type{{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}),
            tensor_type{{{1,2,9,13,14,15},{3,4,10,16,17,18}},{{5,6,11,19,20,21},{7,8,12,22,23,24}}}
        ),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}), tensor_type{{{1,3}},{{2,4}}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{{{1,2}},{{3,4}}}, tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}), tensor_type{{{1,2},{5,6},{7,8}},{{3,4},{9,10},{11,12}}}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}, tensor_type{{{13,14},{15,16},{17,18}}}),
            tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    SECTION("test_concatenate_variadic")
    {
        auto test_concatenate_variadic = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);

            auto apply_tensors = [&axis](const auto&...tensors_){
                return concatenate(axis, tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result == expected);
        };
        apply_by_element(test_concatenate_variadic, test_data);
    }
    SECTION("test_concatenate_container")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(order{}))>;
        auto test_concatenate_container = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(order{})...};}, tensors);
            auto result = concatenate(axis, container);
            REQUIRE(result == expected);
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEST_CASE("test_concatenate_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using shape_type = typename tensor_type::shape_type;
    using gtensor::config::c_order;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0axis,1tensors
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{})),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1})),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{{1,2,3},{4,5,6}})),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{0}, tensor_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1}, tensor_type{1})),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{1,2,3},{4,5,6}})),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}.reshape(0,1), tensor_type{}.reshape(1,0))),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(0,1))),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{2,4,4}}, tensor_type{shape_type{2,3,4}})),
        std::make_tuple(dim_type{-3}, std::make_tuple(tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{2,4,4}}, tensor_type{shape_type{2,3,4}})),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{shape_type{2,3,5}}, tensor_type{shape_type{2,4,4}}, tensor_type{shape_type{2,3,4}})),
        std::make_tuple(dim_type{-2}, std::make_tuple(tensor_type{shape_type{2,3,5}}, tensor_type{shape_type{2,4,4}}, tensor_type{shape_type{2,3,4}})),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{3,3,4}})),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{3,3,4}}))
    );
    SECTION("test_concatenate_variadic_exception")
    {
        auto test_concatenate_variadic = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto apply_tensors = [&axis](const auto&...tensors_){
                return concatenate(axis, tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), value_error);
        };
        apply_by_element(test_concatenate_variadic, test_data);
    }
    SECTION("test_concatenate_container")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(c_order{}))>;
        auto test_concatenate_container = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(c_order{})...};}, tensors);
            REQUIRE_THROWS_AS(concatenate(axis, container), value_error);
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEST_CASE("test_concatenate_common_type","[test_manipulation]")
{
    using tensor_int32_type = gtensor::tensor<int>;
    using tensor_int64_type = gtensor::tensor<std::int64_t>;
    using tensor_double_type = gtensor::tensor<double>;
    using dim_type = std::common_type_t<typename tensor_int32_type::dim_type, typename tensor_int64_type::dim_type, typename tensor_double_type::dim_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_int32_type{{1,2},{3,4}},tensor_int64_type{{5,6}}), tensor_int64_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_int64_type{{1,2},{3,4}},tensor_double_type{{5},{6}}), tensor_double_type{{1,2,5},{3,4,6}})
    );
    auto test = [](const auto& t){
        auto axis = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&axis](const auto&...tensors_){
            return concatenate(axis, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_concatenate_common_order","[test_manipulation]")
{
    using value_type = int;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using dim_type = typename tensor<value_type>::dim_type;
    using traverse_order = typename tensor<value_type>::config_type::order;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor<value_type, c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},tensor<value_type, c_order>{}.reshape(0,2,2),tensor<value_type, c_order>{{{9,10},{11,12}}}),
            tensor<value_type, c_order>{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor<value_type, f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},tensor<value_type, f_order>{}.reshape(2,0,2),tensor<value_type, f_order>{{{9,10}},{{11,12}}}),
            tensor<value_type, f_order>{{{1,2},{3,4},{9,10}},{{5,6},{7,8},{11,12}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(tensor<value_type, f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},tensor<value_type, c_order>{}.reshape(2,2,0),tensor<value_type, c_order>{{{9},{10}},{{11},{12}}}),
            tensor<value_type, traverse_order>{{{1,2,9},{3,4,10}},{{5,6,11},{7,8,12}}}
        )
    );
    auto test = [](const auto& t){
        auto axis = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&axis](const auto&...tensors_){
            return concatenate(axis, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

