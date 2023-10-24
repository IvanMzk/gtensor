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
#include "config_for_testing.hpp"

TEMPLATE_TEST_CASE("test_block_tuple","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;

    //0blocks,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{1,2,3}, tensor_type{}, tensor_type{4,5}, tensor_type{}), tensor_type{1,2,3,4,5}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{{1,2,3}}, tensor_type{}, tensor_type{{{4,5}}}, tensor_type{}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{})), tensor_type{}.reshape(1,0)),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{}), std::make_tuple(tensor_type{})), tensor_type{}.reshape(2,0)),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{},tensor_type{}),std::make_tuple(tensor_type{},tensor_type{})), tensor_type{}.reshape(2,0)),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{},tensor_type{1,2,3}),std::make_tuple(tensor_type{4,5,6},tensor_type{})), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5}), tensor_type{1,2,3,4,5}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7,8},{9,10}}), tensor_type{{1,2,3,7,8},{4,5,6,9,10}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5})), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5}))), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}})), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}))), tensor_type{{{1,2,3},{4,5,6}}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6})), tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2}), std::make_tuple(tensor_type{3,4}), std::make_tuple(tensor_type{5,6})), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1}, tensor_type{2,3}), std::make_tuple(tensor_type{4,5,6})), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}}})),
            std::make_tuple(std::make_tuple(tensor_type{{{3}},{{4}},{{5}}}))),
            tensor_type{{{1}},{{2}},{{3}},{{4}},{{5}}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor_type{1,2})),std::make_tuple(std::make_tuple(tensor_type{3,4})),std::make_tuple(std::make_tuple(tensor_type{5,6}))),
            tensor_type{{{1,2}},{{3,4}},{{5,6}}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor_type{1,2}),std::make_tuple(tensor_type{3,4})),std::make_tuple(std::make_tuple(tensor_type{5,6}),std::make_tuple(tensor_type{7,8}))),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}
        ),
        std::make_tuple(std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2}, tensor_type{3,4,5}, tensor_type{6,7,8,9}))), tensor_type{{{1,2,3,4,5,6,7,8,9}}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}},{{3}}}, tensor_type{{{4}},{{5}},{{6}}}), tensor_type{{{1,4}},{{2,5}},{{3,6}}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2}, tensor_type{{3,4,5}}), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(std::make_tuple(tensor_type{{3,4,5}}, tensor_type{1,2}), tensor_type{{3,4,5,1,2}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2}, tensor_type{{{3,4,5}}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}, tensor_type{{11},{12}}), tensor_type{{1,2,5,6,7,11},{3,4,8,9,10,12}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2}), std::make_tuple(tensor_type{{3,4},{5,6}})), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{3,4,5}}, tensor_type{{6,7,8,9},{10,11,12,13}})), tensor_type{{1,2,3,6,7,8,9},{3,4,5,10,11,12,13}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}), std::make_tuple(tensor_type{{7,8},{9,10},{11,12}})), tensor_type{{1,2},{3,4},{7,8},{9,10},{11,12}}),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}), std::make_tuple(tensor_type{{11,12,13,14,15}})),
            tensor_type{{1,2,5,6,7},{3,4,8,9,10},{11,12,13,14,15}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{{5,6},{7,8}}), std::make_tuple(tensor_type{{9},{10}}, tensor_type{{11,12,13},{14,15,16}})),
            tensor_type{{1,2,5,6},{3,4,7,8},{9,11,12,13},{10,14,15,16}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor_type{{1,2},{3,4}}), std::make_tuple(tensor_type{{5,6},{7,8}}), std::make_tuple(tensor_type{{9},{10}}, tensor_type{{11},{12}})),
            tensor_type{{1,2},{3,4},{5,6},{7,8},{9,11},{10,12}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}), std::make_tuple(tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}})),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}), std::make_tuple(tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}})),
            std::make_tuple(std::make_tuple(tensor_type{13,14}), std::make_tuple(tensor_type{15,16}), std::make_tuple(tensor_type{17,18}))),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    auto test = [](const auto& t){
        auto blocks = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = block(blocks);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_block_tuple_common_order","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;

    //0blocks,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor<value_type,c_order>{{1,2},{3,4}}, tensor<value_type,f_order>{{5,6,7},{8,9,10}}), std::make_tuple(tensor<value_type,c_order>{{11,12,13,14,15}})),
            tensor_type{{1,2,5,6,7},{3,4,8,9,10},{11,12,13,14,15}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor<value_type,f_order>{{1,2},{3,4}}, tensor<value_type,f_order>{{5,6},{7,8}}), std::make_tuple(tensor<value_type,c_order>{{9},{10}}, tensor<value_type,c_order>{{11,12,13},{14,15,16}})),
            tensor_type{{1,2,5,6},{3,4,7,8},{9,11,12,13},{10,14,15,16}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor<value_type,c_order>{{1,2},{3,4}}), std::make_tuple(tensor<value_type,f_order>{{5,6},{7,8}}), std::make_tuple(tensor<value_type,c_order>{{9},{10}}, tensor<value_type,f_order>{{11},{12}})),
            tensor_type{{1,2},{3,4},{5,6},{7,8},{9,11},{10,12}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor<value_type,f_order>{{{1}},{{2}}}, tensor<value_type,c_order>{{{3}},{{4}}}), std::make_tuple(tensor<value_type,c_order>{{{5,6},{7,8}},{{9,10},{11,12}}})),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor<value_type,c_order>{{{1}},{{2}}}, tensor<value_type,f_order>{{{3}},{{4}}}), std::make_tuple(tensor<value_type,c_order>{{{5,6},{7,8}},{{9,10},{11,12}}})),
            std::make_tuple(std::make_tuple(tensor<value_type,f_order>{13,14}), std::make_tuple(tensor<value_type,c_order>{15,16}), std::make_tuple(tensor<value_type,f_order>{17,18}))),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    auto test = [](const auto& t){
        auto blocks = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = block(blocks);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_block_tuple_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;
    //blocks
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2},tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{1},{2},{3}},tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{3,4},{5,6}}, tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}},{{5}}}),
        std::make_tuple(std::make_tuple(tensor_type{},tensor_type{1,2,3}),std::make_tuple(tensor_type{})),
        std::make_tuple(std::make_tuple(tensor_type{1,2}),std::make_tuple(tensor_type{})),
        std::make_tuple(std::make_tuple(tensor_type{1,2}),std::make_tuple(tensor_type{3,4,5})),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{},tensor_type{}),std::make_tuple(tensor_type{})), std::make_tuple(std::make_tuple(tensor_type{})))
    );
    auto test = [](const auto& blocks){
        REQUIRE_THROWS_AS(block(blocks),value_error);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_block_init_list","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;
    //0result,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{}}), tensor_type{}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{}, tensor_type{}, tensor_type{}}), tensor_type{}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{}, tensor_type{1,2,3}, tensor_type{}, tensor_type{4,5}, tensor_type{}}), tensor_type{1,2,3,4,5}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{}, tensor_type{{1,2,3}}, tensor_type{}, tensor_type{{{4,5}}}, tensor_type{}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{}}}), tensor_type{}.reshape(1,0)),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{},tensor_type{}},{tensor_type{},tensor_type{}}}), tensor_type{}.reshape(2,0)),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{},tensor_type{1,2,3}},{tensor_type{4,5,6},tensor_type{}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2,3,4,5}}), tensor_type{1,2,3,4,5}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{1,2,3},{4,5,6}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7,8},{9,10}}}), tensor_type{{1,2,3,7,8},{4,5,6,9,10}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1,2,3,4,5}}}), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{1,2,3,4,5}}}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2,3},{4,5,6}}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{{1,2,3},{4,5,6}}}}}), tensor_type{{{1,2,3},{4,5,6}}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}}), tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1,2}}, {tensor_type{3,4}}, {tensor_type{5,6}}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1}, tensor_type{2,3}}, {tensor_type{4,5,6}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{1,2}}},{{tensor_type{3,4}}},{{tensor_type{5,6}}}}), tensor_type{{{1,2}},{{3,4}},{{5,6}}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{1,2}},{tensor_type{3,4}}},{{tensor_type{5,6}},{tensor_type{7,8}}}}), tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{1,2}, tensor_type{3,4,5}, tensor_type{6,7,8,9}}}}), tensor_type{{{1,2,3,4,5,6,7,8,9}}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{{1}},{{2}},{{3}}}, tensor_type{{{4}},{{5}},{{6}}}}), tensor_type{{{1,4}},{{2,5}},{{3,6}}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2}, tensor_type{{3,4,5}}}), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{3,4,5}}, tensor_type{1,2}}), tensor_type{{3,4,5,1,2}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2}, tensor_type{{{3,4,5}}}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}, tensor_type{{11},{12}}}), tensor_type{{1,2,5,6,7,11},{3,4,8,9,10,12}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1,2}}, {tensor_type{{3,4},{5,6}}}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2,3},{3,4,5}}, tensor_type{{6,7,8,9},{10,11,12,13}}}}), tensor_type{{1,2,3,6,7,8,9},{3,4,5,10,11,12,13}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2},{3,4}}}, {tensor_type{{7,8},{9,10},{11,12}}}}), tensor_type{{1,2},{3,4},{7,8},{9,10},{11,12}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}}, {tensor_type{{11,12,13,14,15}}}}), tensor_type{{1,2,5,6,7},{3,4,8,9,10},{11,12,13,14,15}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2},{3,4}}, tensor_type{{5,6},{7,8}}}, {tensor_type{{9},{10}}, tensor_type{{11,12,13},{14,15,16}}}}), tensor_type{{1,2,5,6},{3,4,7,8},{9,11,12,13},{10,14,15,16}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2},{3,4}}}, {tensor_type{{5,6},{7,8}}}, {tensor_type{{9},{10}}, tensor_type{{11},{12}}}}), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,11},{10,12}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{
            {tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}},
            {tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{
            {{tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}}, {tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}}},
            {{tensor_type{13,14}}, {tensor_type{15,16}}, {tensor_type{17,18}}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result == expected);
    };

    apply_by_element(test, test_data);
}

TEST_CASE("test_block_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{1,2},tensor_type{{3,4},{5,6}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{{1},{2},{3}},tensor_type{{3,4},{5,6}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{{3,4},{5,6}}, tensor_type{}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{},tensor_type{{3,4},{5,6}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}},{{5}}} }), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list2<tensor_type>{{tensor_type{},tensor_type{1,2,3}},{tensor_type{}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list2<tensor_type>{{tensor_type{1,2}},{tensor_type{}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list2<tensor_type>{{tensor_type{1,2}},{tensor_type{3,4,5}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list3<tensor_type>{{{tensor_type{},tensor_type{}},{tensor_type{}}}, {{tensor_type{}}}}), value_error);
}

