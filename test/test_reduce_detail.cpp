/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <algorithm>
#include "catch.hpp"
#include "builder.hpp"
#include "reduce.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "config_for_testing.hpp"

TEST_CASE("test_check_reduce_args","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::axis_error;
    using gtensor::detail::check_reduce_args;

    //single reduce axis
    REQUIRE_NOTHROW(check_reduce_args(shape_type{0},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{10},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},dim_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},dim_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},dim_type{2}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{2}));

    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},dim_type{0}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},dim_type{1}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{1,0},dim_type{2}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},dim_type{3}), axis_error);

    //container of axes
    REQUIRE_NOTHROW(check_reduce_args(shape_type{},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{0},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{0},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{10},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{10},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{0,1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{1,0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{2}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{0,1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{1,2}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{1,2,0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{0,1,2}));

    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},std::vector<int>{0}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},std::vector<int>{0,0}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},std::vector<int>{1}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},std::vector<int>{1,0}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},std::vector<int>{1}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},std::vector<int>{0,0}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{10},std::vector<int>{1}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{10},std::vector<int>{0,1}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{3}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{0,0}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{0,1,0}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{1,2,0,1}), axis_error);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{1,2,3}), axis_error);
}

TEST_CASE("test_make_reduce_shape","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_reduce_shape;
    using helpers_for_testing::apply_by_element;
    //0pshape,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //single axis
        //keep_dims is false
        std::make_tuple(shape_type{0},dim_type{0},false,shape_type{}),
        std::make_tuple(shape_type{1},dim_type{0},false,shape_type{}),
        std::make_tuple(shape_type{10},dim_type{0},false,shape_type{}),
        std::make_tuple(shape_type{2,3,0},dim_type{0},false,shape_type{3,0}),
        std::make_tuple(shape_type{2,3,0},dim_type{1},false,shape_type{2,0}),
        std::make_tuple(shape_type{2,3,0},dim_type{2},false,shape_type{2,3}),
        std::make_tuple(shape_type{2,3,4},dim_type{0},false,shape_type{3,4}),
        std::make_tuple(shape_type{2,3,4},dim_type{1},false,shape_type{2,4}),
        std::make_tuple(shape_type{2,3,4},dim_type{2},false,shape_type{2,3}),
        //keep_dims is true
        std::make_tuple(shape_type{0},dim_type{0},true,shape_type{1}),
        std::make_tuple(shape_type{1},dim_type{0},true,shape_type{1}),
        std::make_tuple(shape_type{10},dim_type{0},true,shape_type{1}),
        std::make_tuple(shape_type{2,3,0},dim_type{0},true,shape_type{1,3,0}),
        std::make_tuple(shape_type{2,3,0},dim_type{1},true,shape_type{2,1,0}),
        std::make_tuple(shape_type{2,3,0},dim_type{2},true,shape_type{2,3,1}),
        std::make_tuple(shape_type{2,3,4},dim_type{0},true,shape_type{1,3,4}),
        std::make_tuple(shape_type{2,3,4},dim_type{1},true,shape_type{2,1,4}),
        std::make_tuple(shape_type{2,3,4},dim_type{2},true,shape_type{2,3,1}),
        //container of axees
        //keep_dims is false, empty container
        std::make_tuple(shape_type{},std::vector<int>{},false,shape_type{}),
        std::make_tuple(shape_type{0},std::vector<int>{},false,shape_type{0}),
        std::make_tuple(shape_type{1},std::vector<int>{},false,shape_type{1}),
        std::make_tuple(shape_type{10},std::vector<int>{},false,shape_type{10}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{},false,shape_type{2,3,0}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{},false,shape_type{2,3,4}),
        //keep_dims is false, not empty container
        std::make_tuple(shape_type{0},std::vector<int>{0},false,shape_type{}),
        std::make_tuple(shape_type{10},std::vector<int>{0},false,shape_type{}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0},false,shape_type{3,0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{1,0},false,shape_type{0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0,1},false,shape_type{0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{2},false,shape_type{2,3}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{1,2},false,shape_type{2}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0,1,2},false,shape_type{}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{0,1,2},false,shape_type{}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{0},false,shape_type{3,4}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{2,0},false,shape_type{3}),
        //keep_dims is true, empty container
        std::make_tuple(shape_type{},std::vector<int>{},true,shape_type{}),
        std::make_tuple(shape_type{0},std::vector<int>{},true,shape_type{0}),
        std::make_tuple(shape_type{1},std::vector<int>{},true,shape_type{1}),
        std::make_tuple(shape_type{10},std::vector<int>{},true,shape_type{10}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{},true,shape_type{2,3,0}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{},true,shape_type{2,3,4}),
        //keep_dims is true, not empty container
        std::make_tuple(shape_type{0},std::vector<int>{0},true,shape_type{1}),
        std::make_tuple(shape_type{10},std::vector<int>{0},true,shape_type{1}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0},true,shape_type{1,3,0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{1,0},true,shape_type{1,1,0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0,1},true,shape_type{1,1,0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{2},true,shape_type{2,3,1}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{1,2},true,shape_type{2,1,1}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0,1,2},true,shape_type{1,1,1}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{0,1,2},true,shape_type{1,1,1}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{0},true,shape_type{1,3,4}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{2,0},true,shape_type{1,3,1})
    );
    auto test = [](const auto& t){
        auto pshape = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_reduce_shape(pshape,axes,keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_check_slide_args","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;
    using gtensor::axis_error;
    using gtensor::value_error;
    using gtensor::detail::check_slide_args;

    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0},dim_type{0},index_type{0},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{1},shape_type{1},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{5},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{10},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{1,0},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{0},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{1},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{1},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{1},index_type{3},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{0},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{3},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{3},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{4},index_type{1}));
    //empty tensor, window_size and window_step doesnt matter
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0},dim_type{0},index_type{0},index_type{0}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0},dim_type{0},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0,2,3},dim_type{1},index_type{0},index_type{0}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0,2,3},dim_type{1},index_type{3},index_type{1}));

    //invalid axis
    REQUIRE_THROWS_AS(check_slide_args(index_type{0},shape_type{0},dim_type{1},index_type{1},index_type{1}), axis_error);
    REQUIRE_THROWS_AS(check_slide_args(index_type{0},shape_type{0,2,3},dim_type{3},index_type{1},index_type{1}), axis_error);
    REQUIRE_THROWS_AS(check_slide_args(index_type{10},shape_type{10},dim_type{1},index_type{1},index_type{1}), axis_error);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{3},index_type{1},index_type{1}), axis_error);
    //window_size greater than axis size
    REQUIRE_THROWS_AS(check_slide_args(index_type{1},shape_type{},dim_type{0},index_type{1},index_type{1}), value_error);
    REQUIRE_THROWS_AS(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{11},index_type{1}), value_error);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{0},index_type{3},index_type{1}), value_error);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{4},index_type{1}), value_error);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{5},index_type{1}), value_error);
    //zero window_step
    REQUIRE_THROWS_AS(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{3},index_type{0}), value_error);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{1},index_type{0}), value_error);
}

TEST_CASE("test_make_slide_shape","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_slide_shape;
    //0psize,1pshape,2axis,3window_size,4window_step,5expected
    using test_type = std::tuple<index_type,shape_type,dim_type,index_type,index_type,shape_type>;
    auto test_data = GENERATE(
        test_type{index_type{0},shape_type{0},dim_type{0},index_type{1},index_type{1},shape_type{0}},
        test_type{index_type{0},shape_type{0},dim_type{0},index_type{2},index_type{1},shape_type{0}},
        test_type{index_type{0},shape_type{0},dim_type{0},index_type{1},index_type{2},shape_type{0}},
        test_type{index_type{0},shape_type{20,30,0},dim_type{0},index_type{5},index_type{2},shape_type{20,30,0}},
        test_type{index_type{0},shape_type{20,30,0},dim_type{1},index_type{5},index_type{2},shape_type{20,30,0}},
        test_type{index_type{0},shape_type{20,30,0},dim_type{2},index_type{5},index_type{2},shape_type{20,30,0}},
        test_type{index_type{1},shape_type{1},dim_type{0},index_type{1},index_type{1},shape_type{1}},
        test_type{index_type{1},shape_type{1},dim_type{0},index_type{1},index_type{2},shape_type{1}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{1},index_type{1},shape_type{10}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{1},index_type{2},shape_type{5}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{1},index_type{5},shape_type{2}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{2},index_type{1},shape_type{9}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{2},index_type{2},shape_type{5}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{2},index_type{5},shape_type{2}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{5},index_type{1},shape_type{6}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{5},index_type{2},shape_type{3}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{5},index_type{5},shape_type{2}},
        test_type{index_type{6000},shape_type{5,30,40},dim_type{0},index_type{3},index_type{2},shape_type{2,30,40}},
        test_type{index_type{6000},shape_type{5,30,40},dim_type{1},index_type{5},index_type{1},shape_type{5,26,40}},
        test_type{index_type{6000},shape_type{5,30,40},dim_type{2},index_type{10},index_type{3},shape_type{5,30,11}}
    );
    auto psize = std::get<0>(test_data);
    auto pshape = std::get<1>(test_data);
    auto axis = std::get<2>(test_data);
    auto window_size = std::get<3>(test_data);
    auto window_step = std::get<4>(test_data);
    auto expected = std::get<5>(test_data);
    auto result = make_slide_shape(psize,pshape,axis,window_size,window_step);
    REQUIRE(result == expected);
}

TEST_CASE("test_make_leading_axes","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using axes_type = config_type::template shape<dim_type>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_leading_axes;
    using helpers_for_testing::apply_by_element;
    //0sorted_axes,1order,2expected
    auto test_data = std::make_tuple(
        //c_order
        //axes scalar
        std::make_tuple(dim_type{0},c_order{},std::make_pair(dim_type{0},dim_type{0})),
        std::make_tuple(dim_type{1},c_order{},std::make_pair(dim_type{1},dim_type{1})),
        std::make_tuple(dim_type{2},c_order{},std::make_pair(dim_type{2},dim_type{2})),
        //axes container
        std::make_tuple(axes_type{0},c_order{},std::make_pair(dim_type{0},dim_type{0})),
        std::make_tuple(axes_type{1},c_order{},std::make_pair(dim_type{1},dim_type{1})),
        std::make_tuple(axes_type{5},c_order{},std::make_pair(dim_type{5},dim_type{5})),
        std::make_tuple(axes_type{1,2},c_order{},std::make_pair(dim_type{2},dim_type{1})),
        std::make_tuple(axes_type{0,2},c_order{},std::make_pair(dim_type{2},dim_type{2})),
        std::make_tuple(axes_type{0,2,3},c_order{},std::make_pair(dim_type{3},dim_type{2})),
        std::make_tuple(axes_type{0,1,5},c_order{},std::make_pair(dim_type{5},dim_type{5})),
        std::make_tuple(axes_type{0,3,5,6,7},c_order{},std::make_pair(dim_type{7},dim_type{5})),
        std::make_tuple(axes_type{1,2,3,6,7},c_order{},std::make_pair(dim_type{7},dim_type{6})),
        //f_order
        //axes scalar
        std::make_tuple(dim_type{0},f_order{},std::make_pair(dim_type{0},dim_type{0})),
        std::make_tuple(dim_type{1},f_order{},std::make_pair(dim_type{1},dim_type{1})),
        std::make_tuple(dim_type{2},f_order{},std::make_pair(dim_type{2},dim_type{2})),
        //axes container
        std::make_tuple(axes_type{0},f_order{},std::make_pair(dim_type{0},dim_type{0})),
        std::make_tuple(axes_type{1},f_order{},std::make_pair(dim_type{1},dim_type{1})),
        std::make_tuple(axes_type{1,2},f_order{},std::make_pair(dim_type{1},dim_type{2})),
        std::make_tuple(axes_type{0,2},f_order{},std::make_pair(dim_type{0},dim_type{0})),
        std::make_tuple(axes_type{0,2,3},f_order{},std::make_pair(dim_type{0},dim_type{0})),
        std::make_tuple(axes_type{0,1,5},f_order{},std::make_pair(dim_type{0},dim_type{1})),
        std::make_tuple(axes_type{3,4,5,7,8},f_order{},std::make_pair(dim_type{3},dim_type{5})),
        std::make_tuple(axes_type{1,2,4,7,8},f_order{},std::make_pair(dim_type{1},dim_type{2})),
        std::make_tuple(axes_type{2,4,7,8},f_order{},std::make_pair(dim_type{2},dim_type{2}))
    );
    auto test = [](const auto& t){
        auto sorted_axes = std::get<0>(t);
        auto order = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_leading_axes(sorted_axes,order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_inner_size","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_inner_size;
    using helpers_for_testing::apply_by_element;
    //0strides,1leading_axes,2expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(shape_type{1},std::make_pair(dim_type{0},dim_type{0}),1),
        std::make_tuple(shape_type{20,4,1},std::make_pair(dim_type{1},dim_type{1}),4),
        std::make_tuple(shape_type{25,5,1},std::make_pair(dim_type{2},dim_type{1}),1),
        std::make_tuple(shape_type{120,60,12,3,1},std::make_pair(dim_type{2},dim_type{1}),12),
        //f_order
        std::make_tuple(shape_type{1},std::make_pair(dim_type{0},dim_type{0}),1),
        std::make_tuple(shape_type{1,4,20},std::make_pair(dim_type{1},dim_type{1}),4),
        std::make_tuple(shape_type{1,5,25},std::make_pair(dim_type{2},dim_type{1}),25),
        std::make_tuple(shape_type{1,3,12,60,120},std::make_pair(dim_type{2},dim_type{1}),12)
    );
    auto test = [](const auto& t){
        auto strides = std::get<0>(t);
        auto leading_axes = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_inner_size(strides,leading_axes);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_outer_size","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_outer_size;
    using helpers_for_testing::apply_by_element;
    //0shape,1leading_axes,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{4},std::make_pair(dim_type{0},dim_type{0}),4),
        std::make_tuple(shape_type{2,3,3},std::make_pair(dim_type{1},dim_type{0}),6),
        std::make_tuple(shape_type{2,5,4},std::make_pair(dim_type{1},dim_type{1}),5),
        std::make_tuple(shape_type{5,5,5},std::make_pair(dim_type{2},dim_type{1}),25),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{2},dim_type{1}),20),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{2},dim_type{0}),40),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{1},dim_type{0}),10),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{1},dim_type{1}),5)
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto leading_axes = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_outer_size(shape,leading_axes);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_traverse_index_shape","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_traverse_index_shape;
    using helpers_for_testing::apply_by_element;
    //0shape,1leading_axes,2order,3expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(shape_type{4},std::make_pair(dim_type{0},dim_type{0}),c_order{},shape_type{}),
        std::make_tuple(shape_type{2,5,4},std::make_pair(dim_type{1},dim_type{1}),c_order{},shape_type{2}),
        std::make_tuple(shape_type{5,5,5},std::make_pair(dim_type{2},dim_type{1}),c_order{},shape_type{5}),
        std::make_tuple(shape_type{5,5,5},std::make_pair(dim_type{2},dim_type{2}),c_order{},shape_type{5,5}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{2},dim_type{1}),c_order{},shape_type{2}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{3},dim_type{1}),c_order{},shape_type{2}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{2},dim_type{0}),c_order{},shape_type{}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{3},dim_type{0}),c_order{},shape_type{}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{1},dim_type{0}),c_order{},shape_type{}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{1},dim_type{1}),c_order{},shape_type{2}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{2},dim_type{2}),c_order{},shape_type{2,5}),
        //f_order
        std::make_tuple(shape_type{4},std::make_pair(dim_type{0},dim_type{0}),f_order{},shape_type{}),
        std::make_tuple(shape_type{2,5,4},std::make_pair(dim_type{1},dim_type{1}),f_order{},shape_type{4}),
        std::make_tuple(shape_type{5,5,5},std::make_pair(dim_type{1},dim_type{2}),f_order{},shape_type{}),
        std::make_tuple(shape_type{5,5,5},std::make_pair(dim_type{2},dim_type{2}),f_order{},shape_type{}),
        std::make_tuple(shape_type{5,5,5},std::make_pair(dim_type{0},dim_type{1}),f_order{},shape_type{5}),
        std::make_tuple(shape_type{5,5,5},std::make_pair(dim_type{0},dim_type{0}),f_order{},shape_type{5,5}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{1},dim_type{2}),f_order{},shape_type{3}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{0},dim_type{2}),f_order{},shape_type{3}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{0},dim_type{1}),f_order{},shape_type{4,3}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{1},dim_type{1}),f_order{},shape_type{4,3}),
        std::make_tuple(shape_type{2,5,4,3},std::make_pair(dim_type{2},dim_type{2}),f_order{},shape_type{3})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto leading_axes = std::get<1>(t);
        auto order = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_traverse_index_shape(shape,leading_axes,order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_reduce_axes_map","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using axes_type = config_type::template shape<dim_type>;
    using gtensor::detail::make_reduce_axes_map;
    using helpers_for_testing::apply_by_element;
    //0dim,1sorted_axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //axes scalar
        std::make_tuple(dim_type{1},dim_type{0},false,axes_type{}),
        std::make_tuple(dim_type{2},dim_type{0},false,axes_type{1}),
        std::make_tuple(dim_type{2},dim_type{1},false,axes_type{0}),
        std::make_tuple(dim_type{4},dim_type{0},false,axes_type{1,2,3}),
        std::make_tuple(dim_type{4},dim_type{1},false,axes_type{0,2,3}),
        std::make_tuple(dim_type{4},dim_type{2},false,axes_type{0,1,3}),
        std::make_tuple(dim_type{4},dim_type{3},false,axes_type{0,1,2}),
        //axes container
        std::make_tuple(dim_type{1},axes_type{0},false,axes_type{}),
        std::make_tuple(dim_type{2},axes_type{0},false,axes_type{1}),
        std::make_tuple(dim_type{2},axes_type{1},false,axes_type{0}),
        std::make_tuple(dim_type{2},axes_type{0,1},false,axes_type{}),
        std::make_tuple(dim_type{5},axes_type{1,3},false,axes_type{0,2,4}),
        std::make_tuple(dim_type{5},axes_type{0,1,2},false,axes_type{3,4}),
        std::make_tuple(dim_type{5},axes_type{0,4},false,axes_type{1,2,3}),
        std::make_tuple(dim_type{5},axes_type{0,2,4},false,axes_type{1,3}),
        //keep_dims
        std::make_tuple(dim_type{3},dim_type{0},true,axes_type{0,1,2}),
        std::make_tuple(dim_type{5},axes_type{0,2,4},true,axes_type{0,1,2,3,4})

    );
    auto test = [](const auto& t){
        auto dim = std::get<0>(t);
        auto sorted_axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_reduce_axes_map<config_type>(dim,sorted_axes,keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_traverse_index_strides","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using axes_type = config_type::template shape<dim_type>;
    using gtensor::detail::make_traverse_index_strides;
    using helpers_for_testing::apply_by_element;
    //0traverse_shape,1res_strides,2leading_axes,3axes_map,4order,5expected
    auto test_data = std::make_tuple(
        //c_order
        //axes scalar
        std::make_tuple(shape_type{},shape_type{},std::make_pair(dim_type{0},dim_type{0}),axes_type{},c_order{},axes_type{}),
        std::make_tuple(shape_type{},shape_type{1},std::make_pair(dim_type{0},dim_type{0}),axes_type{1},c_order{},axes_type{}), //(3,4), reduce axes 0
        std::make_tuple(shape_type{3},shape_type{1},std::make_pair(dim_type{1},dim_type{1}),axes_type{0},c_order{},axes_type{1}),   //(3,4), reduce axes 1
        std::make_tuple(shape_type{4,2},shape_type{80,20,5,1},std::make_pair(dim_type{2},dim_type{2}),axes_type{0,1,3,4},c_order{},axes_type{80,20}), //(4,2,3,4,5), reduce axes 2
        std::make_tuple(shape_type{},shape_type{60,20,5,1},std::make_pair(dim_type{0},dim_type{0}),axes_type{1,2,3,4},c_order{},axes_type{}), //(4,2,3,4,5), reduce axes 0
        std::make_tuple(shape_type{4,2,3,4},shape_type{24,12,4,1},std::make_pair(dim_type{4},dim_type{4}),axes_type{0,1,2,3},c_order{},axes_type{24,12,4,1}), //(4,2,3,4,5), reduce axes 4
        //axes container
        std::make_tuple(shape_type{4,2,3},shape_type{15,5,1},std::make_pair(dim_type{3},dim_type{3}),axes_type{0,2,4},c_order{},axes_type{15,0,5}), //(4,2,3,4,5), reduce axes 1,3
        std::make_tuple(shape_type{4},shape_type{20,5,1},std::make_pair(dim_type{2},dim_type{1}),axes_type{0,3,4},c_order{},axes_type{20}), //(4,2,3,4,5), reduce axes 1,2
        std::make_tuple(shape_type{4,2,3},shape_type{15,5,1},std::make_pair(dim_type{3},dim_type{3}),axes_type{1,2,4},c_order{},axes_type{0,15,5}), //(4,2,3,4,5), reduce axes 0,3
        std::make_tuple(shape_type{4,2,3,4},shape_type{4,1},std::make_pair(dim_type{4},dim_type{4}),axes_type{1,3},c_order{},axes_type{0,4,0,1}), //(4,2,3,4,5), reduce axes 0,2,4
        std::make_tuple(shape_type{4,2},shape_type{10,5,1},std::make_pair(dim_type{3},dim_type{2}),axes_type{0,1,4},c_order{},axes_type{10,5}), //(4,2,3,4,5), reduce axes 2,3
        std::make_tuple(shape_type{4,2,3},shape_type{5,1},std::make_pair(dim_type{3},dim_type{3}),axes_type{2,4},c_order{},axes_type{0,0,5}), //(4,2,3,4,5), reduce axes 0,1,3
        std::make_tuple(shape_type{4,2},shape_type{5,1},std::make_pair(dim_type{3},dim_type{2}),axes_type{1,4},c_order{},axes_type{0,5}), //(4,2,3,4,5), reduce axes 0,2,3
        //f_order
        //axes scalar
        std::make_tuple(shape_type{},shape_type{},std::make_pair(dim_type{0},dim_type{0}),axes_type{},f_order{},axes_type{}),
        std::make_tuple(shape_type{4},shape_type{1},std::make_pair(dim_type{0},dim_type{0}),axes_type{1},f_order{},axes_type{1}), //(3,4), reduce axes 0
        std::make_tuple(shape_type{},shape_type{1},std::make_pair(dim_type{1},dim_type{1}),axes_type{0},f_order{},axes_type{}),   //(3,4), reduce axes 1
        std::make_tuple(shape_type{4,5},shape_type{1,4,8,32},std::make_pair(dim_type{2},dim_type{2}),axes_type{0,1,3,4},f_order{},axes_type{8,32}), //(4,2,3,4,5), reduce axes 2
        std::make_tuple(shape_type{2,3,4,5},shape_type{1,2,6,24},std::make_pair(dim_type{0},dim_type{0}),axes_type{1,2,3,4},f_order{},axes_type{1,2,6,24}), //(4,2,3,4,5), reduce axes 0
        std::make_tuple(shape_type{},shape_type{1,4,8,24},std::make_pair(dim_type{4},dim_type{4}),axes_type{0,1,2,3},f_order{},axes_type{}), //(4,2,3,4,5), reduce axes 4
        //axes container
        std::make_tuple(shape_type{3,4,5},shape_type{1,4,12},std::make_pair(dim_type{1},dim_type{1}),axes_type{0,2,4},f_order{},axes_type{4,0,12}), //(4,2,3,4,5), reduce axes 1,3
        std::make_tuple(shape_type{4,5},shape_type{1,4,16},std::make_pair(dim_type{1},dim_type{2}),axes_type{0,3,4},f_order{},axes_type{4,16}), //(4,2,3,4,5), reduce axes 1,2
        std::make_tuple(shape_type{2,3,4,5},shape_type{1,2,6},std::make_pair(dim_type{0},dim_type{0}),axes_type{1,2,4},f_order{},axes_type{1,2,0,6}), //(4,2,3,4,5), reduce axes 0,3
        std::make_tuple(shape_type{2,3,4,5},shape_type{1,2},std::make_pair(dim_type{0},dim_type{0}),axes_type{1,3},f_order{},axes_type{1,0,2,0}), //(4,2,3,4,5), reduce axes 0,2,4
        std::make_tuple(shape_type{5},shape_type{1,4,8},std::make_pair(dim_type{2},dim_type{3}),axes_type{0,1,4},f_order{},axes_type{8}), //(4,2,3,4,5), reduce axes 2,3
        std::make_tuple(shape_type{3,4,5},shape_type{1,3},std::make_pair(dim_type{0},dim_type{1}),axes_type{2,4},f_order{},axes_type{1,0,3}), //(4,2,3,4,5), reduce axes 0,1,3
        std::make_tuple(shape_type{2,3,4,5},shape_type{1,2},std::make_pair(dim_type{0},dim_type{0}),axes_type{1,4},f_order{},axes_type{1,0,0,2}) //(4,2,3,4,5), reduce axes 0,2,3
    );
    auto test = [](const auto& t){
        auto traverse_shape = std::get<0>(t);
        auto res_strides = std::get<1>(t);
        auto leading_axes = std::get<2>(t);
        auto axes_map = std::get<3>(t);
        auto order = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = make_traverse_index_strides(traverse_shape,res_strides,leading_axes,axes_map,order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

