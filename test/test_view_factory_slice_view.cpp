/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <tuple>
#include <vector>
#include <iostream>
#include "catch.hpp"
#include "test_config.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

//test create_slice_view
TEMPLATE_TEST_CASE("test_create_slice_view","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,order>;
    using config_type = typename tensor_type::config_type;
    using slice_type = typename tensor_type::slice_type;
    using nop_type = typename slice_type::nop_type;
    using rtag_type = typename slice_type::reduce_tag_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(2),std::make_tuple(),tensor_type(2)),
        std::make_tuple(tensor_type{},std::make_tuple(),tensor_type{}),
        std::make_tuple(tensor_type{},std::make_tuple(slice_type{}),tensor_type{}),
        std::make_tuple(tensor_type{},std::make_tuple(slice_type{1,-1}),tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,-1}),tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{nop_type{},nop_type{},2}),tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,-1},slice_type{nop_type{},nop_type{},2}),tensor_type{}.reshape(2,2,0)),
        std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,-1},slice_type{nop_type{},nop_type{},-3},slice_type{1,-1}),tensor_type{}.reshape(2,1,0)),
        std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,rtag_type{}}),tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,-1},slice_type{1,rtag_type{}}),tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,rtag_type{}},slice_type{1,rtag_type{}}),tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(),tensor_type{1,2,3,4,5,6,7,8,9,10}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{0,rtag_type{}}),tensor_type(1)),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{1,rtag_type{}}),tensor_type(2)),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-1,rtag_type{}}),tensor_type(10)),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{}),tensor_type{1,2,3,4,5,6,7,8,9,10}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-20,20}),tensor_type{1,2,3,4,5,6,7,8,9,10}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-20,5}),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-20,-5}),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{5,20}),tensor_type{6,7,8,9,10}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{20,-20,-1}),tensor_type{10,9,8,7,6,5,4,3,2,1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{20,5,-1}),tensor_type{10,9,8,7}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{20,-5,-1}),tensor_type{10,9,8,7}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{5,-20,-1}),tensor_type{6,5,4,3,2,1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{nop_type{},nop_type{},-1}),tensor_type{10,9,8,7,6,5,4,3,2,1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{nop_type{},nop_type{},-3}),tensor_type{10,7,4,1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{2,-2}),tensor_type{3,4,5,6,7,8}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-2,2,-1}),tensor_type{9,8,7,6,5,4}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{2,-2,2}),tensor_type{3,5,7}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-2,2,-2}),tensor_type{9,7,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{}),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{1}),tensor_type{{4,5,6},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-10,10}),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-10,2}),tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{10,-3,-1}),tensor_type{{7,8,9},{4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{nop_type{},nop_type{},-1}),tensor_type{{7,8,9},{4,5,6},{1,2,3}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{}),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-10,10},slice_type{-10,10}),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{1}),tensor_type{{2,3},{5,6},{8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{1,2}),tensor_type{{2},{5},{8}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{0,1},slice_type{1,2}),tensor_type{{2}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{0,rtag_type{}}),tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{1,rtag_type{}}),tensor_type{4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{2,rtag_type{}}),tensor_type{7,8,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-1,rtag_type{}}),tensor_type{7,8,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-2,rtag_type{}}),tensor_type{4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-3,rtag_type{}}),tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{0,rtag_type{}},slice_type{0,rtag_type{}}),tensor_type(1)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{1,rtag_type{}},slice_type{1,rtag_type{}}),tensor_type(5)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-1,rtag_type{}},slice_type{-1,rtag_type{}}),tensor_type(9)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{2,rtag_type{}}),tensor_type{3,6,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{1,rtag_type{}}),tensor_type{2,5,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{0,rtag_type{}}),tensor_type{1,4,7}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{-3,rtag_type{}}),tensor_type{1,4,7}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{-2,rtag_type{}}),tensor_type{2,5,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{-1,rtag_type{}}),tensor_type{3,6,9}),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{}),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{nop_type{},nop_type{},-1}),
            tensor_type{{{10,11,12},{13,14,15},{16,17,18}},{{1,2,3},{4,5,6},{7,8,9}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{1}),
            tensor_type{{{10,11,12},{13,14,15},{16,17,18}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{1,rtag_type{}}),
            tensor_type{{10,11,12},{13,14,15},{16,17,18}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{-2,rtag_type{}}),
            tensor_type{{1,2,3},{4,5,6},{7,8,9}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{},slice_type{}),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{},slice_type{nop_type{},nop_type{},-1}),
            tensor_type{{{7,8,9},{4,5,6},{1,2,3}},{{16,17,18},{13,14,15},{10,11,12}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{1},slice_type{1}),
            tensor_type{{{13,14,15},{16,17,18}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{-2},slice_type{-1,1,-1}),
            tensor_type{{{7,8,9}},{{16,17,18}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{1,rtag_type{}},slice_type{}),
            tensor_type{{10,11,12},{13,14,15},{16,17,18}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{1,rtag_type{}},slice_type{0,rtag_type{}}),
            tensor_type{10,11,12}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{},slice_type{2,rtag_type{}}),
            tensor_type{{7,8,9},{16,17,18}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{},slice_type{-1,rtag_type{}}),
            tensor_type{{7,8,9},{16,17,18}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{},slice_type{},slice_type{}),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{},slice_type{},slice_type{nop_type{},nop_type{},-2}),
            tensor_type{{{3,1},{6,4},{9,7}},{{12,10},{15,13},{18,16}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{},slice_type{1,-1},slice_type{1,-1}),
            tensor_type{{{5}},{{14}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{1},slice_type{1,-1},slice_type{1,-1}),
            tensor_type{{{14}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{1,rtag_type{}},slice_type{1,-1},slice_type{1,-1}),
            tensor_type{{14}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{1,rtag_type{}},slice_type{1,rtag_type{}}),
            tensor_type{13,14,15}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{-2,rtag_type{}},slice_type{-1,rtag_type{}},slice_type{nop_type{},nop_type{},-1}),
            tensor_type{9,8,7}
        )
    );
    SECTION("test_create_slice_view_variadic")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto apply_subs = [&parent](const auto&...subs_){
                return basic_tensor{view_factory_type::create_slice_view(parent, subs_...)};
            };
            auto result = std::apply(apply_subs, subs);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_create_slice_view_container")
    {
        using container_type = std::vector<slice_type>;
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto make_container = [](const auto&...subs_){
                return container_type{subs_...};
            };
            auto container = std::apply(make_container, subs);
            auto result = basic_tensor{view_factory_type::create_slice_view(parent, container)};
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_create_slice_view_mixed_subs","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,order>;
    using config_type = typename tensor_type::config_type;
    using slice_type = typename tensor_type::slice_type;
    using nop_type = typename slice_type::nop_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(0),tensor_type(1)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(1),tensor_type(2)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(-1),tensor_type(6)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(-2),tensor_type(5)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(0),tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(1),tensor_type{4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(2),tensor_type{7,8,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-3),tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-2),tensor_type{4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-1),tensor_type{7,8,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(0,0),tensor_type(1)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(1,1),tensor_type(5)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-1,-1),tensor_type(9)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},0),tensor_type{1,4,7}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},1),tensor_type{2,5,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},-1),tensor_type{3,6,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},-2),tensor_type{2,5,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(0,slice_type{}),tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(1,slice_type{nop_type{},nop_type{},-1}),tensor_type{6,5,4}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-1,slice_type{nop_type{},nop_type{},-2}),tensor_type{9,7}),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(0,1,2),
            tensor_type(6)
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(0,2),
            tensor_type{7,8,9}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(1,1),
            tensor_type{13,14,15}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(0),
            tensor_type{{1,2,3},{4,5,6},{7,8,9}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(1),
            tensor_type{{10,11,12},{13,14,15},{16,17,18}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(slice_type{},1,slice_type{}),
            tensor_type{{4,5,6},{13,14,15}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(-2,-1,slice_type{nop_type{},nop_type{},-1}),
            tensor_type{9,8,7}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(1,slice_type{1},0),
            tensor_type{13,16}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
            std::make_tuple(1,slice_type{1},1),
            tensor_type{14,17}
        )
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto subs = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto apply_subs = [&parent](const auto&...subs_){
            return basic_tensor{view_factory_type::create_slice_view(parent, subs_...)};
        };
        auto result = std::apply(apply_subs, subs);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_create_slice_view_init_list_interface","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,order>;
    using config_type = typename tensor_type::config_type;
    using slice_type = typename tensor_type::slice_type;
    using rtag_type = typename slice_type::reduce_tag_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using slice_item_type = typename slice_type::slice_item_type;
    using list_type = std::initializer_list<std::initializer_list<slice_item_type>>;
    using gtensor::basic_tensor;
    using helpers_for_testing::apply_by_element;
    //0result,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{},list_type{})},tensor_type{}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{},list_type{{-3,3}})},tensor_type{}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{})},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{0,10}})},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{-10,10}})},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{3,10}})},tensor_type{4,5,6}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{10,-10,-1}})},tensor_type{6,5,4,3,2,1}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{3,-10,-1}})},tensor_type{4,3,2,1}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{1,-1}})},tensor_type{2,3,4,5}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{-1,{},-1}})},tensor_type{6,5,4,3,2,1}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{-1,2,-1}})},tensor_type{6,5,4}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{}})},tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{{},{},-1}})},tensor_type{{3,2,1},{6,5,4},{9,8,7}}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{{},2}})},tensor_type{{1,2},{4,5},{7,8}}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{{},{},-1},{}})},tensor_type{{7,8,9},{4,5,6},{1,2,3}}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{0,rtag_type{}}})},tensor_type{1,4,7}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{-1,rtag_type{}}})},tensor_type{3,6,9}),
        std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{1,rtag_type{}}})},tensor_type{2,5,8})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_create_slice_exception","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,order>;
    using config_type = typename tensor_type::config_type;
    using slice_type = typename tensor_type::slice_type;
    using rtag_type = typename slice_type::reduce_tag_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using gtensor::index_error;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(2),std::make_tuple(slice_type{})),
        std::make_tuple(tensor_type(2),std::make_tuple(slice_type{0,rtag_type{}})),
        std::make_tuple(tensor_type(2),std::make_tuple(slice_type{},slice_type{})),
        std::make_tuple(tensor_type{},std::make_tuple(slice_type{0,rtag_type{}})),
        std::make_tuple(tensor_type{},std::make_tuple(slice_type{},slice_type{})),
        std::make_tuple(tensor_type{}.reshape(4,0),std::make_tuple(slice_type{},slice_type{0,rtag_type{}})),
        std::make_tuple(tensor_type{}.reshape(4,0),std::make_tuple(slice_type{},slice_type{1,rtag_type{}})),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(slice_type{},slice_type{})),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},slice_type{},slice_type{})),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},slice_type{3,rtag_type{}})),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},slice_type{-4,rtag_type{}}))
    );
    SECTION("test_create_slice_view_variadic_exception")
    {
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto apply_subs = [&parent](const auto&...subs_){
                return basic_tensor{view_factory_type::create_slice_view(parent, subs_...)};
            };
            REQUIRE_THROWS_AS(std::apply(apply_subs, subs), index_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_create_slice_view_container_exception")
    {
        using container_type = std::vector<slice_type>;
        auto test = [](const auto& t){
            auto parent = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto make_container = [](const auto&...subs_){
                return container_type{subs_...};
            };
            auto container = std::apply(make_container, subs);
            REQUIRE_THROWS_AS(view_factory_type::create_slice_view(parent, container), index_error);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_create_slice_exception_mixed_subs_exception","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,order>;
    using config_type = typename tensor_type::config_type;
    using slice_type = typename tensor_type::slice_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using gtensor::index_error;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(2),std::make_tuple(0)),
        std::make_tuple(tensor_type(2),std::make_tuple(slice_type{},0)),
        std::make_tuple(tensor_type(2),std::make_tuple(0,slice_type{})),
        std::make_tuple(tensor_type{},std::make_tuple(0)),
        std::make_tuple(tensor_type{},std::make_tuple(0,0)),
        std::make_tuple(tensor_type{}.reshape(4,0),std::make_tuple(0,0)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(2)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(-3)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(2,slice_type{})),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(-3,slice_type{})),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},3)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},-4))
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto subs = std::get<1>(t);
        auto apply_subs = [&parent](const auto&...subs_){
            return basic_tensor{view_factory_type::create_slice_view(parent, subs_...)};
        };
        REQUIRE_THROWS_AS(std::apply(apply_subs, subs), index_error);
    };
    apply_by_element(test, test_data);
}

