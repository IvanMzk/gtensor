#include <algorithm>
#include "catch.hpp"
#include "builder.hpp"
#include "reduce.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

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

namespace test_reduce_binary_{
    struct max{
        template<typename T, typename U>
        auto operator()(const T& t, const U& u){
            return t>u ? t:u;
        }
    };
    struct min{
        template<typename T, typename U>
        auto operator()(const T& t, const U& u){
            return t<u ? t:u;
        }
    };
}

TEMPLATE_TEST_CASE("test_reduce_binary","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::reduce_binary;
    using gtensor::detail::no_value;
    using test_reduce_binary_::max;
    using test_reduce_binary_::min;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{
        {{{{7,4,6,5,7},{3,1,3,3,8},{3,5,6,7,6},{5,7,1,1,6}},{{6,4,0,3,8},{5,3,3,8,7},{0,1,7,2,3},{5,5,0,2,5}},{{8,7,7,4,5},{1,8,6,8,4},{2,7,1,6,2},{6,5,6,0,3}}},
        {{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}}},
        {{{{1,7,2,7,8},{1,7,1,8,1},{2,3,4,2,0},{7,5,8,5,0}},{{5,5,1,3,8},{0,8,0,0,2},{5,1,2,3,0},{6,7,3,7,4}},{{8,0,7,0,0},{2,4,1,5,8},{5,6,8,4,8},{4,1,3,2,7}}},
        {{{0,6,2,7,3},{6,4,2,6,4},{7,0,3,3,1},{2,1,3,0,4}},{{7,4,4,7,6},{3,3,6,7,4},{1,7,4,0,1},{2,3,0,6,8}},{{2,4,1,6,0},{3,5,2,6,7},{5,7,5,4,4},{7,8,0,2,2}}}},
        {{{{0,7,1,1,0},{2,7,5,3,3},{6,5,4,8,6},{4,8,0,6,4}},{{0,0,5,8,0},{8,1,6,4,7},{2,5,4,6,3},{0,4,0,2,7}},{{6,0,3,6,4},{1,5,3,8,0},{8,7,2,4,0},{8,3,2,3,6}}},
        {{{2,8,5,4,4},{0,0,3,8,5},{4,1,4,2,1},{4,1,8,1,1}},{{7,2,8,8,3},{3,4,3,3,6},{1,6,2,7,7},{0,5,4,6,1}},{{1,4,0,7,6},{8,7,6,8,2},{6,4,0,5,8},{6,4,2,4,0}}}},
        {{{{0,5,0,8,6},{5,5,3,8,1},{8,3,7,8,5},{1,4,3,4,4}},{{4,0,0,6,8},{4,8,0,1,7},{6,2,6,4,2},{4,7,5,8,1}},{{3,3,1,5,5},{2,4,6,0,5},{3,1,7,6,5},{6,2,8,1,2}}},
        {{{4,7,5,2,1},{6,5,3,1,5},{8,8,5,5,4},{3,3,4,1,5}},{{7,8,2,8,1},{6,0,2,4,5},{8,4,5,0,3},{7,2,5,0,0}},{{2,2,2,7,8},{1,0,7,5,8},{0,2,5,4,4},{1,3,5,8,4}}}}
    };  //(4,2,3,4,5)

    //0tensor,1axes,2functor,3keep_dims,4initial,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(test_ten,std::vector<int>{4},std::plus<void>{},false,no_value{},tensor_type{{{{29,18,27,20},{21,26,13,17},{31,27,18,20}},{{21,11,16,27},{19,17,5,28},{24,16,27,30}}},{{{25,18,11,25},{22,10,11,27},{15,20,31,17}},{{18,22,14,10},{28,23,13,19},{13,23,25,19}}},{{{9,20,29,22},{13,26,20,13},{19,17,21,22}},{{23,16,12,15},{28,19,23,16},{18,31,23,16}}},{{{19,22,31,16},{18,20,20,25},{17,17,22,19}},{{19,20,30,16},{26,17,20,14},{21,21,15,21}}}}),
        std::make_tuple(test_ten,std::vector<int>{3,4},std::plus<void>{},false,no_value{},tensor_type{{{94,77,96},{75,69,97}},{{79,70,83},{64,83,80}},{{80,72,79},{66,86,88}},{{88,83,75},{85,77,78}}}),
        std::make_tuple(test_ten,std::vector<int>{2,3,4},std::plus<void>{},false,no_value{},tensor_type{{267,241},{232,227},{231,240},{246,240}}),
        std::make_tuple(test_ten,std::vector<int>{0,4},std::plus<void>{},false,no_value{},tensor_type{{{82,78,98,83},{74,82,64,82},{82,81,92,78}},{{81,69,72,68},{101,76,61,77},{76,91,90,86}}}),
        std::make_tuple(test_ten,std::vector<int>{2,4},std::plus<void>{},false,no_value{},tensor_type{{{81,71,58,57},{64,44,48,85}},{{62,48,53,69},{59,68,52,48}},{{41,63,70,57},{69,66,58,47}},{{54,59,73,60},{66,58,65,51}}}),
        std::make_tuple(test_ten,std::vector<int>{0,2,4},std::plus<void>{},false,no_value{},tensor_type{{238,241,254,243},{258,236,223,231}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,4},std::plus<void>{},false,no_value{},tensor_type{{163,147,170,151},{175,158,125,159},{158,172,182,164}}),
        std::make_tuple(test_ten,std::vector<int>{0,3,4},std::plus<void>{},false,no_value{},tensor_type{{341,302,333},{290,315,343}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,3,4},std::plus<void>{},false,no_value{},tensor_type{631,617,676}),
        std::make_tuple(test_ten,std::vector<int>{0},std::plus<void>{},false,no_value{},tensor_type{{{{8,23,9,21,21},{11,20,12,22,13},{19,16,21,25,17},{17,24,12,16,14}},{{15,9,6,20,24},{17,20,9,13,23},{13,9,19,15,8},{15,23,8,19,17}},{{25,10,18,15,14},{6,21,16,21,17},{18,21,18,20,15},{24,11,19,6,18}}},{{{8,23,19,18,13},{12,9,11,22,15},{27,11,17,10,7},{9,12,22,7,18}},{{22,19,20,30,10},{18,11,12,18,17},{12,18,11,8,12},{15,16,12,18,16}},{{12,16,4,23,21},{14,15,23,19,20},{14,21,16,16,23},{19,23,11,22,11}}}}),
        std::make_tuple(test_ten,std::vector<int>{2},std::plus<void>{},false,no_value{},tensor_type{{{{21,15,13,12,20},{9,12,12,19,19},{5,13,14,15,11},{16,17,7,3,14}},{{10,13,14,15,12},{8,7,12,11,6},{13,11,11,4,9},{11,21,14,19,20}}},{{{14,12,10,10,16},{3,19,2,13,11},{12,10,14,9,8},{17,13,14,14,11}},{{9,14,7,20,9},{12,12,10,19,15},{13,14,12,7,6},{11,12,3,8,14}}},{{{6,7,9,15,4},{11,13,14,15,10},{16,17,10,18,9},{12,15,2,11,17}},{{10,14,13,19,13},{11,11,12,19,13},{11,11,6,14,16},{10,10,14,11,2}}},{{{7,8,1,19,19},{11,17,9,9,13},{17,6,20,18,12},{11,13,16,13,7}},{{13,17,9,17,10},{13,5,12,10,18},{16,14,15,9,11},{11,8,14,9,9}}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1},std::plus<void>{},false,no_value{},tensor_type{{{16,46,28,39,34},{23,29,23,44,28},{46,27,38,35,24},{26,36,34,23,32}},{{37,28,26,50,34},{35,31,21,31,40},{25,27,30,23,20},{30,39,20,37,33}},{{37,26,22,38,35},{20,36,39,40,37},{32,42,34,36,38},{43,34,30,28,29}}}),
        std::make_tuple(test_ten,std::vector<int>{1,2},std::plus<void>{},false,no_value{},tensor_type{{{31,28,27,27,32},{17,19,24,30,25},{18,24,25,19,20},{27,38,21,22,34}},{{23,26,17,30,25},{15,31,12,32,26},{25,24,26,16,14},{28,25,17,22,25}},{{16,21,22,34,17},{22,24,26,34,23},{27,28,16,32,25},{22,25,16,22,19}},{{20,25,10,36,29},{24,22,21,19,31},{33,20,35,27,23},{22,21,30,22,16}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,2},std::plus<void>{},false,no_value{},tensor_type{{90,100,76,127,103},{78,96,83,115,105},{103,96,102,94,82},{99,109,84,88,94}}),
        std::make_tuple(test_ten,std::vector<int>{1,3},std::plus<void>{},false,no_value{},tensor_type{{{28,28,38,33,42},{31,29,20,33,33},{34,52,39,32,36}},{{26,33,25,38,21},{29,38,20,33,33},{36,35,27,29,36}},{{22,37,30,33,24},{21,27,32,44,34},{44,34,18,45,26}},{{35,40,30,37,31},{46,31,25,31,27},{18,17,41,36,41}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,3},std::plus<void>{},false,no_value{},tensor_type{{111,138,123,141,118},{127,125,97,141,127},{132,138,125,142,139}}),
        std::make_tuple(test_ten,std::vector<int>{1,2,3},std::plus<void>{},false,no_value{},tensor_type{{93,109,97,98,111},{91,106,72,100,90},{87,98,80,122,84},{99,88,96,104,99}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},std::plus<void>{},false,no_value{},tensor_type(1924)),
        //initial
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},std::plus<void>{},false,-24,tensor_type(1900)),
        std::make_tuple(test_ten,std::vector<int>{0},std::plus<void>{},false,-5,tensor_type{{{{3,18,4,16,16},{6,15,7,17,8},{14,11,16,20,12},{12,19,7,11,9}},{{10,4,1,15,19},{12,15,4,8,18},{8,4,14,10,3},{10,18,3,14,12}},{{20,5,13,10,9},{1,16,11,16,12},{13,16,13,15,10},{19,6,14,1,13}}},{{{3,18,14,13,8},{7,4,6,17,10},{22,6,12,5,2},{4,7,17,2,13}},{{17,14,15,25,5},{13,6,7,13,12},{7,13,6,3,7},{10,11,7,13,11}},{{7,11,-1,18,16},{9,10,18,14,15},{9,16,11,11,18},{14,18,6,17,6}}}}),
        std::make_tuple(test_ten,std::vector<int>{4},std::plus<void>{},false,5,tensor_type{{{{34,23,32,25},{26,31,18,22},{36,32,23,25}},{{26,16,21,32},{24,22,10,33},{29,21,32,35}}},{{{30,23,16,30},{27,15,16,32},{20,25,36,22}},{{23,27,19,15},{33,28,18,24},{18,28,30,24}}},{{{14,25,34,27},{18,31,25,18},{24,22,26,27}},{{28,21,17,20},{33,24,28,21},{23,36,28,21}}},{{{24,27,36,21},{23,25,25,30},{22,22,27,24}},{{24,25,35,21},{31,22,25,19},{26,26,20,26}}}}),
        std::make_tuple(test_ten,std::vector<int>{1,2,4},std::plus<void>{},false,-5,tensor_type{{140,110,101,137},{116,111,100,112},{105,124,123,99},{115,112,133,106}}),
        //keep_dims
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},std::plus<void>{},true,no_value{},tensor_type{{{{{1924}}}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},std::plus<void>{},true,-24,tensor_type{{{{{1900}}}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,3},std::plus<void>{},true,-5,tensor_type{{{{{106,133,118,136,113}},{{122,120,92,136,122}},{{127,133,120,137,134}}}}}),
        std::make_tuple(test_ten,std::vector<int>{1,3,4},std::plus<void>{},true,-5,tensor_type{{{{{164}},{{141}},{{188}}}},{{{{138}},{{148}},{{158}}}},{{{{141}},{{153}},{{162}}}},{{{{168}},{{155}},{{148}}}}}),
        std::make_tuple(test_ten,0,std::plus<void>{},true,-5,tensor_type{{{{{3,18,4,16,16},{6,15,7,17,8},{14,11,16,20,12},{12,19,7,11,9}},{{10,4,1,15,19},{12,15,4,8,18},{8,4,14,10,3},{10,18,3,14,12}},{{20,5,13,10,9},{1,16,11,16,12},{13,16,13,15,10},{19,6,14,1,13}}},{{{3,18,14,13,8},{7,4,6,17,10},{22,6,12,5,2},{4,7,17,2,13}},{{17,14,15,25,5},{13,6,7,13,12},{7,13,6,3,7},{10,11,7,13,11}},{{7,11,-1,18,16},{9,10,18,14,15},{9,16,11,11,18},{14,18,6,17,6}}}}}),
        std::make_tuple(test_ten,std::vector<int>{4},std::plus<void>{},true,-5,tensor_type{{{{{24},{13},{22},{15}},{{16},{21},{8},{12}},{{26},{22},{13},{15}}},{{{16},{6},{11},{22}},{{14},{12},{0},{23}},{{19},{11},{22},{25}}}},{{{{20},{13},{6},{20}},{{17},{5},{6},{22}},{{10},{15},{26},{12}}},{{{13},{17},{9},{5}},{{23},{18},{8},{14}},{{8},{18},{20},{14}}}},{{{{4},{15},{24},{17}},{{8},{21},{15},{8}},{{14},{12},{16},{17}}},{{{18},{11},{7},{10}},{{23},{14},{18},{11}},{{13},{26},{18},{11}}}},{{{{14},{17},{26},{11}},{{13},{15},{15},{20}},{{12},{12},{17},{14}}},{{{14},{15},{25},{11}},{{21},{12},{15},{9}},{{16},{16},{10},{16}}}}}),
        //functor
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},max{},false,no_value{},tensor_type(8)),
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},min{},false,no_value{},tensor_type(0)),
        std::make_tuple(test_ten,std::vector<int>{1,3},max{},false,no_value{},tensor_type{{{8,7,7,7,8},{6,6,7,8,8},{8,8,8,8,7}},{{7,7,8,8,8},{7,8,6,7,8},{8,8,8,6,8}},{{6,8,8,8,6},{8,6,8,8,7},{8,7,6,8,8}},{{8,8,7,8,6},{8,8,6,8,8},{6,4,8,8,8}}}),
        std::make_tuple(test_ten,std::vector<int>{1,3},max{},false,7,tensor_type{{{8,7,7,7,8},{7,7,7,8,8},{8,8,8,8,7}},{{7,7,8,8,8},{7,8,7,7,8},{8,8,8,7,8}},{{7,8,8,8,7},{8,7,8,8,7},{8,7,7,8,8}},{{8,8,7,8,7},{8,8,7,8,8},{7,7,8,8,8}}}),
        std::make_tuple(test_ten,std::vector<int>{0,2},min{},false,no_value{},tensor_type{{{0,0,0,0,0},{0,1,0,0,0},{0,1,1,2,0},{0,1,0,0,0}},{{0,2,0,2,0},{0,0,1,0,1},{0,0,0,0,1},{0,1,0,0,0}}}),
        std::make_tuple(test_ten,std::vector<int>{0,2},min{},false,1,tensor_type{{{0,0,0,0,0},{0,1,0,0,0},{0,1,1,1,0},{0,1,0,0,0}},{{0,1,0,1,0},{0,0,1,0,1},{0,0,0,0,1},{0,1,0,0,0}}}),
        //unsorted, negative axes
        std::make_tuple(test_ten,std::vector<int>{3,0,1},std::plus<void>{},true,-5,tensor_type{{{{{106,133,118,136,113}},{{122,120,92,136,122}},{{127,133,120,137,134}}}}}),
        std::make_tuple(test_ten,std::vector<int>{1,2,0,3,-1},std::plus<void>{},true,-24,tensor_type{{{{{1900}}}}}),
        std::make_tuple(test_ten,std::vector<int>{2,-1,-4,},std::plus<void>{},false,-5,tensor_type{{140,110,101,137},{116,111,100,112},{105,124,123,99},{115,112,133,106}}),
        //input 1d
        std::make_tuple(tensor_type{1,2,3,4,5},0,std::plus<void>{},false,no_value{},tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},std::plus<void>{},false,no_value{},tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,std::plus<void>{},false,4,tensor_type(19)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},std::plus<void>{},false,4,tensor_type(19)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,std::plus<void>{},true,no_value{},tensor_type{15}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},std::plus<void>{},true,no_value{},tensor_type{15}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,std::plus<void>{},true,-4,tensor_type{11}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},std::plus<void>{},true,-4,tensor_type{11}),
        //input unit dims
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},0,std::plus<void>{},false,0,tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{1},std::plus<void>{},false,0,tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},2,std::plus<void>{},true,0,tensor_type{{{15}}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{2},std::plus<void>{},false,0,tensor_type{{15}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{0,1},std::plus<void>{},false,0,tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{1,2},std::plus<void>{},true,0,tensor_type{{{15}}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{0,1,2},std::plus<void>{},false,0,tensor_type(15)),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},0,std::plus<void>{},false,0,tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},1,std::plus<void>{},false,0,tensor_type{{15}}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},2,std::plus<void>{},true,0,tensor_type{{{1},{2},{3},{4},{5}}}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},std::vector<int>{0,1},std::plus<void>{},false,0,tensor_type{15}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},std::vector<int>{0,2},std::plus<void>{},false,0,tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},std::vector<int>{0,2},std::plus<void>{},true,0,tensor_type{{{1},{2},{3},{4},{5}}}),
        //empty axes container
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},std::plus<void>{},true,no_value{},tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},std::plus<void>{},true,2,tensor_type{{3,4,5},{6,7,8}}),
        //reduce zero size axes
        std::make_tuple(tensor_type{},std::vector<int>{0},std::plus<void>{},false,0,tensor_type(0)),
        std::make_tuple(tensor_type{},std::vector<int>{0},std::plus<void>{},false,3,tensor_type(3)),
        std::make_tuple(tensor_type{},std::vector<int>{0},std::plus<void>{},true,2,tensor_type{2}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0},std::plus<void>{},false,4,tensor_type{{4,4,4},{4,4,4}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0},std::plus<void>{},true,4,tensor_type{{{4,4,4},{4,4,4}}}),
        std::make_tuple(tensor_type{}.reshape(0,2,0,3),std::vector<int>{0,2},std::plus<void>{},false,0,tensor_type{{0,0,0},{0,0,0}}),
        //empty result
        std::make_tuple(tensor_type{}.reshape(2,0,3),std::vector<int>{0},std::plus<void>{},false,no_value{},tensor_type{}.reshape(0,3)),
        std::make_tuple(tensor_type{}.reshape(2,0,3),std::vector<int>{0},std::plus<void>{},true,0,tensor_type{}.reshape(1,0,3)),
        std::make_tuple(tensor_type{}.reshape(0,2,0,3),std::vector<int>{0,1},std::plus<void>{},false,0,tensor_type{}.reshape(0,3)),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten,std::vector<int>{0,1,2},std::plus<void>{},false,0,tensor_type{{360,400,304,508,412},{312,384,332,460,420},{412,384,408,376,328},{396,436,336,352,376}}),
        std::make_tuple((test_ten-1)*(test_ten+1),std::vector<int>{1,2,3},std::plus<void>{},false,0,tensor_type{{491,613,533,540,639},{465,590,322,550,530},{497,532,368,728,438},{537,446,490,628,513}}),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,1)+test_ten(2,0,2),std::vector<int>{0,1,2},std::plus<void>{},false,0,tensor_type{{430,324,312,539,399},{266,388,331,579,325},{471,472,346,322,210},{487,429,240,312,486}}),
        std::make_tuple((test_ten+test_ten(0))*(test_ten(1,1)-test_ten(2,0,2)),std::vector<int>{1,2,3},std::plus<void>{},false,0,tensor_type{{-422,194,-6,-98,176},{-388,206,51,-79,284},{-329,169,26,-155,179},{-383,129,6,-87,220}})
    );

    //seq execution
    SECTION("exec_pol<1>")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto initial = std::get<4>(t);
            auto expected = std::get<5>(t);

            auto result = reduce_binary(multithreading::exec_pol<1>{},ten,axes,functor,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }

    //4 par tasks execution
    SECTION("exec_pol<4>")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto initial = std::get<4>(t);
            auto expected = std::get<5>(t);

            auto result = reduce_binary(multithreading::exec_pol<4>{},ten,axes,functor,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }

    //auto par tasks execution
    SECTION("exec_pol<0>")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto initial = std::get<4>(t);
            auto expected = std::get<5>(t);

            auto result = reduce_binary(multithreading::exec_pol<0>{},ten,axes,functor,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_reduce_binary_exception","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::reduce_binary;
    using gtensor::detail::no_value;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axes,2functor,3keep_dims,4initial
    auto test_data = std::make_tuple(
        //reduce zero size axes without initial
        std::make_tuple(tensor_type{},0,std::plus<void>{},false,no_value{}),
        std::make_tuple(tensor_type{},std::vector<int>{0},std::plus<void>{},true,no_value{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0},std::plus<void>{},false,no_value{}),
        std::make_tuple(tensor_type{}.reshape(0,2,0,3),std::vector<int>{0,2},std::plus<void>{},false,no_value{}),
        //axes out of range
        std::make_tuple(tensor_type{},1,std::plus<void>{},false,0),
        std::make_tuple(tensor_type{1,2,3,4,5},1,std::plus<void>{},false,0),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{1},std::plus<void>{},false,0),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},2,std::plus<void>{},false,0),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{0,1,2},std::plus<void>{},false,0),
        //repeating axes
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{0,0},std::plus<void>{},false,0),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{2,-1},std::plus<void>{},false,0)
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto keep_dims = std::get<3>(t);
        auto initial = std::get<4>(t);
        REQUIRE_THROWS_AS(reduce_binary(multithreading::exec_pol<1>{},ten,axes,functor,keep_dims,initial),value_error);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_reduce_binary_flatten","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using test_reduce_binary_::max;
    using test_reduce_binary_::min;
    using gtensor::detail::no_value;
    using gtensor::reduce_binary_flatten;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{
        {{{{7,4,6,5,7},{3,1,3,3,8},{3,5,6,7,6},{5,7,1,1,6}},{{6,4,0,3,8},{5,3,3,8,7},{0,1,7,2,3},{5,5,0,2,5}},{{8,7,7,4,5},{1,8,6,8,4},{2,7,1,6,2},{6,5,6,0,3}}},
        {{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}}},
        {{{{1,7,2,7,8},{1,7,1,8,1},{2,3,4,2,0},{7,5,8,5,0}},{{5,5,1,3,8},{0,8,0,0,2},{5,1,2,3,0},{6,7,3,7,4}},{{8,0,7,0,0},{2,4,1,5,8},{5,6,8,4,8},{4,1,3,2,7}}},
        {{{0,6,2,7,3},{6,4,2,6,4},{7,0,3,3,1},{2,1,3,0,4}},{{7,4,4,7,6},{3,3,6,7,4},{1,7,4,0,1},{2,3,0,6,8}},{{2,4,1,6,0},{3,5,2,6,7},{5,7,5,4,4},{7,8,0,2,2}}}},
        {{{{0,7,1,1,0},{2,7,5,3,3},{6,5,4,8,6},{4,8,0,6,4}},{{0,0,5,8,0},{8,1,6,4,7},{2,5,4,6,3},{0,4,0,2,7}},{{6,0,3,6,4},{1,5,3,8,0},{8,7,2,4,0},{8,3,2,3,6}}},
        {{{2,8,5,4,4},{0,0,3,8,5},{4,1,4,2,1},{4,1,8,1,1}},{{7,2,8,8,3},{3,4,3,3,6},{1,6,2,7,7},{0,5,4,6,1}},{{1,4,0,7,6},{8,7,6,8,2},{6,4,0,5,8},{6,4,2,4,0}}}},
        {{{{0,5,0,8,6},{5,5,3,8,1},{8,3,7,8,5},{1,4,3,4,4}},{{4,0,0,6,8},{4,8,0,1,7},{6,2,6,4,2},{4,7,5,8,1}},{{3,3,1,5,5},{2,4,6,0,5},{3,1,7,6,5},{6,2,8,1,2}}},
        {{{4,7,5,2,1},{6,5,3,1,5},{8,8,5,5,4},{3,3,4,1,5}},{{7,8,2,8,1},{6,0,2,4,5},{8,4,5,0,3},{7,2,5,0,0}},{{2,2,2,7,8},{1,0,7,5,8},{0,2,5,4,4},{1,3,5,8,4}}}}
    };  //(4,2,3,4,5)

    //0tensor,1functor,2keep_dims,3initial,4expected
    auto test_data = std::make_tuple(
        //keep_dims is false
        std::make_tuple(tensor_type{}, std::plus<void>{}, false, value_type{0}, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, std::multiplies<void>{}, false, value_type{1}, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::plus<void>{}, false, value_type{0}, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::multiplies<void>{}, false, value_type{1}, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::plus<void>{}, false, no_value{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::plus<void>{}, false, no_value{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::plus<void>{}, false, no_value{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::plus<void>{}, false, no_value{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::multiplies<void>{}, false, no_value{}, tensor_type(720)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, false, no_value{}, tensor_type(9)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, false, no_value{}, tensor_type(0)),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::plus<void>{}, false, no_value{}, tensor_type(28)),
        //keep_dims is true
        std::make_tuple(tensor_type{}, std::plus<void>{}, true, value_type{0}, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}, std::multiplies<void>{}, true, value_type{1}, tensor_type{value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(2,1,0), std::plus<void>{}, true, value_type{0}, tensor_type{{{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), std::multiplies<void>{}, true, value_type{1}, tensor_type{{{value_type{1}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::plus<void>{}, true, no_value{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::plus<void>{}, true, no_value{}, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::plus<void>{}, true, no_value{}, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::plus<void>{}, true, no_value{}, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::multiplies<void>{}, true, no_value{}, tensor_type{{720}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, true, no_value{}, tensor_type{{9}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, true, no_value{}, tensor_type{{0}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::plus<void>{}, true, no_value{}, tensor_type{{{28}}}),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten,std::plus<void>{},false,0,tensor_type(7696)),
        std::make_tuple((test_ten-1)*(test_ten+1),std::plus<void>{},false,0,tensor_type(10450)),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,1)+test_ten(2,0,2),std::plus<void>{},false,0,tensor_type(7668)),
        std::make_tuple((test_ten+test_ten(0))*(test_ten(1,1)-test_ten(2,0,2)),std::plus<void>{},false,0,tensor_type(-307))
    );
    SECTION("exec_pol<1>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto functor = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = reduce_binary_flatten(multithreading::exec_pol<1>{},tensor, functor, keep_dims, initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("exec_pol<4>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto functor = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = reduce_binary_flatten(multithreading::exec_pol<4>{},tensor, functor, keep_dims, initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("exec_pol<0>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto functor = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = reduce_binary_flatten(multithreading::exec_pol<0>{},tensor, functor, keep_dims, initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

namespace test_reduce_{

struct max
{
    template<typename It>
    auto operator()(It first, It last){
        if (first==last){throw gtensor::value_error{"empty range"};}
        const auto& init = *first;
        return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::max(u,v);});
    }
};
struct min
{
    template<typename It>
    auto operator()(It first, It last){
        if (first==last){throw gtensor::value_error{"empty range"};}
        const auto& init = *first;
        return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::min(u,v);});
    }
};
struct min_or_zero
{
    template<typename It>
    auto operator()(It first, It last){
        auto res = min{}(first,last);
        return res < 0 ? 0 : res;
    }
};

struct sum
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        const auto& init = *first;
        return std::accumulate(++first,last,init,std::plus{});
    }
};
struct sum_of_squares
{
    template<typename It>
    auto operator()(It first, It last){
        const auto res = sum{}(first,last);
        return res*res;
    }
};
struct sum_random_access
{
    template<typename It>
    auto operator()(It first, It last){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        const auto n = last-first;
        difference_type i{0};
        value_type res = first[i];
        for (++i;i!=n; ++i){
            res+=first[i];
        }
        return res;
    }
};
struct sum_random_access_reverse
{
    template<typename It>
    auto operator()(It first, It last){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        const auto n = last-first;
        difference_type i{-1};
        value_type res = last[i];
        for (--i;i!=-n-1; --i){
            res+=last[i];
        }
        return res;
    }
};

struct sum_init
{
    template<typename It>
    auto operator()(It first, It last, const typename std::iterator_traits<It>::value_type& init){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        return std::accumulate(first,last,init,std::plus{});
    }
};
struct prod
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{1};}
        value_type prod{1};
        while(last!=first){
            prod*=*--last;
        }
        return prod;
    }
};

struct cumsum{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        auto cumsum_ = *first;
        *dfirst = cumsum_;
        for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
            cumsum_+=*first;
            *dfirst = cumsum_;
        }
    }
};

struct cumprod_reverse{
    template<typename It, typename DstIt>
    void operator()(It, It last, DstIt dfirst, DstIt dlast){
        auto cumprod_ = *--last;
        *--dlast = cumprod_;
        while(dlast!=dfirst){
            cumprod_*=*--last;
            *--dlast = cumprod_;
        }
    }
};

struct moving_avarage{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It, DstIt dfirst, DstIt dlast, const IdxT& window_size, const IdxT& window_step, const typename std::iterator_traits<It>::value_type& denom){
        using index_type = IdxT;
        using value_type = typename std::iterator_traits<It>::value_type;
        value_type sum{0};
        auto it = first;
        for (index_type i{0}; i!=window_size; ++i, ++it){
            sum+=*it;
        }
        for(;dfirst!=dlast;++dfirst){
            *dfirst = sum/denom;
            for (index_type i{0}; i!=window_step; ++i,++first){
                sum-=*first;
            }
            for (index_type i{0}; i!=window_step; ++i, ++it){
                sum+=*it;
            }
        }
    }
};

struct diff_1{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        for (;dfirst!=dlast;++dfirst){
            auto prev = *first;
            *dfirst = *(++first) - prev;
        }
    }
};

struct diff_2{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        for (;dfirst!=dlast;++dfirst){
            auto v0 = *first;
            auto v1 = *(++first);
            auto v2 = *(++first);
            *dfirst = v2-v1-v1+v0;
            --first;
        }
    }
};

struct sort{
    template<typename It>
    void operator()(It first, It last){
        std::sort(first,last);
    }
};

//take central element, order matters
struct center
{
    template<typename It>
    auto operator()(It first, It last){
        const auto n = last-first;
        const auto i=n/2;
        auto center_it = first+i;
        const auto res = *center_it;
        if (n%2==0){
            return (res+*--center_it)/2;
        }
        return res;
    }
};

}   //end of namespace test_reduce_

TEMPLATE_TEST_CASE("test_reduce_range","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum;
    using test_reduce_::sum_of_squares;
    using test_reduce_::sum_random_access;
    using test_reduce_::sum_random_access_reverse;
    using test_reduce_::prod;
    using test_reduce_::max;
    using test_reduce_::min;
    using test_reduce_::min_or_zero;
    using test_reduce_::center;
    using gtensor::reduce_range;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{
        {{{{7,4,6,5,7},{3,1,3,3,8},{3,5,6,7,6},{5,7,1,1,6}},{{6,4,0,3,8},{5,3,3,8,7},{0,1,7,2,3},{5,5,0,2,5}},{{8,7,7,4,5},{1,8,6,8,4},{2,7,1,6,2},{6,5,6,0,3}}},
        {{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}}},
        {{{{1,7,2,7,8},{1,7,1,8,1},{2,3,4,2,0},{7,5,8,5,0}},{{5,5,1,3,8},{0,8,0,0,2},{5,1,2,3,0},{6,7,3,7,4}},{{8,0,7,0,0},{2,4,1,5,8},{5,6,8,4,8},{4,1,3,2,7}}},
        {{{0,6,2,7,3},{6,4,2,6,4},{7,0,3,3,1},{2,1,3,0,4}},{{7,4,4,7,6},{3,3,6,7,4},{1,7,4,0,1},{2,3,0,6,8}},{{2,4,1,6,0},{3,5,2,6,7},{5,7,5,4,4},{7,8,0,2,2}}}},
        {{{{0,7,1,1,0},{2,7,5,3,3},{6,5,4,8,6},{4,8,0,6,4}},{{0,0,5,8,0},{8,1,6,4,7},{2,5,4,6,3},{0,4,0,2,7}},{{6,0,3,6,4},{1,5,3,8,0},{8,7,2,4,0},{8,3,2,3,6}}},
        {{{2,8,5,4,4},{0,0,3,8,5},{4,1,4,2,1},{4,1,8,1,1}},{{7,2,8,8,3},{3,4,3,3,6},{1,6,2,7,7},{0,5,4,6,1}},{{1,4,0,7,6},{8,7,6,8,2},{6,4,0,5,8},{6,4,2,4,0}}}},
        {{{{0,5,0,8,6},{5,5,3,8,1},{8,3,7,8,5},{1,4,3,4,4}},{{4,0,0,6,8},{4,8,0,1,7},{6,2,6,4,2},{4,7,5,8,1}},{{3,3,1,5,5},{2,4,6,0,5},{3,1,7,6,5},{6,2,8,1,2}}},
        {{{4,7,5,2,1},{6,5,3,1,5},{8,8,5,5,4},{3,3,4,1,5}},{{7,8,2,8,1},{6,0,2,4,5},{8,4,5,0,3},{7,2,5,0,0}},{{2,2,2,7,8},{1,0,7,5,8},{0,2,5,4,4},{1,3,5,8,4}}}}
    };  //(4,2,3,4,5)

    //0tensor,1axes,2functor,3keep_dims,4any_order,5expected
    auto test_data = std::make_tuple(
        //single axis
        //keep_dims is false
        std::make_tuple(tensor_type{}, dim_type{0}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, dim_type{0}, prod{}, false, true, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, false, true, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, false, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, false, true, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, false, true, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, false, true, tensor_type{{value_type{0},value_type{0},value_type{0}},{value_type{0},value_type{0},value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, prod{}, false, true, tensor_type{{value_type{1},value_type{1},value_type{1}},{value_type{1},value_type{1},value_type{1}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, false, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, false, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, false, true, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, false, true, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, false, true, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, false, true, tensor_type{4,6,7,9}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, false, true, tensor_type{1,0}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, false, true, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum_random_access{}, false, true, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum_random_access_reverse{}, false, true, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, false, true, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-2}, sum{}, false, true, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, false, true, tensor_type{{1,5},{9,13}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-1}, sum{}, false, true, tensor_type{{1,5},{9,13}}),
        //keep_dims is true
        std::make_tuple(tensor_type{}, dim_type{0}, sum{}, true, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, true, true, tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, true, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, true, true, tensor_type{}.reshape(1,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, true, true, tensor_type{}.reshape(2,1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, true, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}},{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, prod{}, true, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}},{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, true, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, true, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, true, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, true, true, tensor_type{{5,7,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, true, true, tensor_type{{6},{15}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, true, true, tensor_type{{6},{120}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, true, true, tensor_type{{4,6,7,9}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, true, true, tensor_type{{1},{0}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, true, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum{}, true, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, true, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-2}, sum{}, true, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, true, true, tensor_type{{{1},{5}},{{9},{13}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-1}, sum{}, true, true, tensor_type{{{1},{5}},{{9},{13}}}),
        //axes is container
        //keep_dims is false
        //empty axes
        std::make_tuple(tensor_type{}, std::vector<dim_type>{}, sum{}, false, true, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{}, sum{}, false, true, tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum_of_squares{}, false, true, tensor_type{1,4,9,16,25,36}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum{}, false, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum_of_squares{}, false, true, tensor_type{{1},{4},{9},{16},{25},{36}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum{}, false, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum_of_squares{}, false, true, tensor_type{{1,4,9,16,25,36}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{}, prod{}, false, true, tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{1,-2,3},{-4,5,6}}, std::vector<dim_type>{}, min_or_zero{}, false, true, tensor_type{{1,0,3},{0,5,6}}),
        //not empty axes
        std::make_tuple(tensor_type{}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0}, sum{}, false, true, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1}, sum{}, false, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0,1}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1,0}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0}, sum{}, false, true, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{1}, sum{}, false, true, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, sum{}, false, true, tensor_type{{value_type{0},value_type{0},value_type{0}},{value_type{0},value_type{0},value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, sum{}, false, true, tensor_type{value_type{0},value_type{0},value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, prod{}, false, true, tensor_type{{value_type{1},value_type{1},value_type{1}},{value_type{1},value_type{1},value_type{1}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, prod{}, false, true, tensor_type{value_type{1},value_type{1},value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0,1}, sum{}, false, true, tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1,0}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0,1}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{1}, sum{}, false, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0,1}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, sum{}, false, true, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1,0}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, prod{}, false, true, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, prod{}, false, true, tensor_type{4,10,18}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0,1}, prod{}, false, true, tensor_type(720)),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, sum{}, false, true, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, prod{}, false, true, tensor_type{{0,3},{24,35}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2}, sum{}, false, true, tensor_type{{1,5},{9,13}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum_random_access{}, false, true, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum_random_access_reverse{}, false, true, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2,0}, prod{}, false, true, tensor_type{0,252}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-2,-1}, sum{}, false, true, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-1,-3}, prod{}, false, true, tensor_type{0,252}),
        //keep_dims is true
        //empty axes
        std::make_tuple(tensor_type{}, std::vector<dim_type>{}, sum{}, true, true, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{}, sum{}, true, true, tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum{}, true, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum_of_squares{}, true, true, tensor_type{1,4,9,16,25,36}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum{}, true, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum_of_squares{}, true, true, tensor_type{{1},{4},{9},{16},{25},{36}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum{}, true, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum_of_squares{}, true, true, tensor_type{{1,4,9,16,25,36}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{}, prod{}, true, true, tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{1,-2,3},{-4,5,6}}, std::vector<dim_type>{}, min_or_zero{}, true, true, tensor_type{{1,0,3},{0,5,6}}),
        //not empty axes
        std::make_tuple(tensor_type{}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0}, sum{}, true, true, tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0,1}, sum{}, true, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1,0}, sum{}, true, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0}, sum{}, true, true, tensor_type{}.reshape(1,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{1}, sum{}, true, true, tensor_type{}.reshape(2,1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, sum{}, true, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}},{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, sum{}, true, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, prod{}, true, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}},{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, prod{}, true, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0,1}, sum{}, true, true, tensor_type{}.reshape(1,1,0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1,0}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0,1}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0,1}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{{5,7,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{6},{15}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1,0}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, prod{}, true, true, tensor_type{{6},{120}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, prod{}, true, true, tensor_type{{4,10,18}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0,1}, prod{}, true, true, tensor_type{{720}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, prod{}, true, true, tensor_type{{{0,3}},{{24,35}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2}, sum{}, true, true, tensor_type{{{1},{5}},{{9},{13}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum{}, true, true, tensor_type{{{6}},{{22}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2,0}, prod{}, true, true, tensor_type{{{0},{252}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-2,-1}, sum{}, true, true, tensor_type{{{6}},{{22}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-1,-3}, prod{}, true, true, tensor_type{{{0},{252}}}),
        //any_order false, c_order traverse along axes
        std::make_tuple(tensor_type{{{7,4,8,8},{3,4,5,6},{4,0,0,0}},{{0,1,4,7},{0,1,2,7},{2,8,3,4}}}, std::vector<dim_type>{0,1}, center{}, false, false, tensor_type{2.0,0.5,2.0,3.5}),
        std::make_tuple(tensor_type{{{7,4,8,8},{3,4,5,6},{4,0,0,0}},{{0,1,4,7},{0,1,2,7},{2,8,3,4}}}, std::vector<dim_type>{1,2}, center{}, false, false, tensor_type{4.5,1.5}),
        std::make_tuple(tensor_type{{{7,4,8,8},{3,4,5,6},{4,0,0,0}},{{0,1,4,7},{0,1,2,7},{2,8,3,4}}}, std::vector<dim_type>{0,1,2}, center{}, false, false, tensor_type(0)),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten,std::vector<int>{0,1,2},sum{},false,false,tensor_type{{360,400,304,508,412},{312,384,332,460,420},{412,384,408,376,328},{396,436,336,352,376}}),
        std::make_tuple((test_ten-1)*(test_ten+1),std::vector<int>{1,2,3},sum{},false,false,tensor_type{{491,613,533,540,639},{465,590,322,550,530},{497,532,368,728,438},{537,446,490,628,513}}),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,1)+test_ten(2,0,2),std::vector<int>{0,1,2},sum{},false,false,tensor_type{{430,324,312,539,399},{266,388,331,579,325},{471,472,346,322,210},{487,429,240,312,486}}),
        std::make_tuple((test_ten+test_ten(0))*(test_ten(1,1)-test_ten(2,0,2)),std::vector<int>{1,2,3},sum{},false,false,tensor_type{{-422,194,-6,-98,176},{-388,206,51,-79,284},{-329,169,26,-155,179},{-383,129,6,-87,220}})
    );
    SECTION("exec_pol<1>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto any_order = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto result = reduce_range(multithreading::exec_pol<1>{},tensor, axes, functor, keep_dims, any_order);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("exec_pol<4>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto any_order = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto result = reduce_range(multithreading::exec_pol<4>{},tensor, axes, functor, keep_dims, any_order);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("exec_pol<0>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto any_order = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto result = reduce_range(multithreading::exec_pol<0>{},tensor, axes, functor, keep_dims, any_order);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_reduce_range_custom_arg","[test_reduce]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum_init;
    using gtensor::reduce_range;
    using helpers_for_testing::apply_by_element;
    //0tensor,1axes,2functor,3keep_dims,4any_order,5init,6expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, sum_init{}, false, true, value_type{0}, tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, sum_init{}, true, true, value_type{-1}, tensor_type{14}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum_init{}, false, true, value_type{-1}, tensor_type{4,6,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum_init{}, false, true, value_type{1}, tensor_type{7,16})
    );
    SECTION("exec_pol<1>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto any_order = std::get<4>(t);
            auto init = std::get<5>(t);
            auto expected = std::get<6>(t);
            auto result = reduce_range(multithreading::exec_pol<1>{}, tensor, axes, functor, keep_dims, any_order, init);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("exec_pol<4>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto any_order = std::get<4>(t);
            auto init = std::get<5>(t);
            auto expected = std::get<6>(t);
            auto result = reduce_range(multithreading::exec_pol<4>{}, tensor, axes, functor, keep_dims, any_order, init);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("exec_pol<0>")
    {
        auto test = [](const auto& t){
            auto tensor = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto any_order = std::get<4>(t);
            auto init = std::get<5>(t);
            auto expected = std::get<6>(t);
            auto result = reduce_range(multithreading::exec_pol<0>{}, tensor, axes, functor, keep_dims, any_order, init);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_reduce_range_ecxeption","[test_reduce]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum;
    using gtensor::axis_error;
    using gtensor::reduce_range;
    using helpers_for_testing::apply_by_element;


    //0tensor,1axes,2functor,3keep_dim,4any_order
    auto test_data = std::make_tuple(
        //single axis
        std::make_tuple(tensor_type(0), dim_type{0}, sum{}, false, true),
        std::make_tuple(tensor_type{}, dim_type{1}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{1}, sum{}, false, true),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{2}, sum{}, false, true),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{2}, sum{}, false, true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{4}, sum{}, false, true),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{3}, sum{}, false, true),
        //axes container
        std::make_tuple(tensor_type(0), std::vector<dim_type>{0}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{0,0}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1,1}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{0,1}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1,0}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0,0}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{1,1}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0,1}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{1,0}, sum{}, false, true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{3}, sum{}, false, true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{0,1,0}, sum{}, false, true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{1,1}, sum{}, false, true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{0,1,2,0}, sum{}, false, true)
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto keep_dim = std::get<3>(t);
        auto any_order = std::get<4>(t);
        REQUIRE_THROWS_AS(reduce_range(multithreading::exec_pol<1>{}, tensor, axes, functor, keep_dim, any_order), axis_error);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_reduce_range_flatten","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using test_reduce_::sum;
    using test_reduce_::prod;
    using test_reduce_::max;
    using test_reduce_::min;
    using test_reduce_::center;
    using gtensor::reduce_range_flatten;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{
        {{{{7,4,6,5,7},{3,1,3,3,8},{3,5,6,7,6},{5,7,1,1,6}},{{6,4,0,3,8},{5,3,3,8,7},{0,1,7,2,3},{5,5,0,2,5}},{{8,7,7,4,5},{1,8,6,8,4},{2,7,1,6,2},{6,5,6,0,3}}},
        {{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}}},
        {{{{1,7,2,7,8},{1,7,1,8,1},{2,3,4,2,0},{7,5,8,5,0}},{{5,5,1,3,8},{0,8,0,0,2},{5,1,2,3,0},{6,7,3,7,4}},{{8,0,7,0,0},{2,4,1,5,8},{5,6,8,4,8},{4,1,3,2,7}}},
        {{{0,6,2,7,3},{6,4,2,6,4},{7,0,3,3,1},{2,1,3,0,4}},{{7,4,4,7,6},{3,3,6,7,4},{1,7,4,0,1},{2,3,0,6,8}},{{2,4,1,6,0},{3,5,2,6,7},{5,7,5,4,4},{7,8,0,2,2}}}},
        {{{{0,7,1,1,0},{2,7,5,3,3},{6,5,4,8,6},{4,8,0,6,4}},{{0,0,5,8,0},{8,1,6,4,7},{2,5,4,6,3},{0,4,0,2,7}},{{6,0,3,6,4},{1,5,3,8,0},{8,7,2,4,0},{8,3,2,3,6}}},
        {{{2,8,5,4,4},{0,0,3,8,5},{4,1,4,2,1},{4,1,8,1,1}},{{7,2,8,8,3},{3,4,3,3,6},{1,6,2,7,7},{0,5,4,6,1}},{{1,4,0,7,6},{8,7,6,8,2},{6,4,0,5,8},{6,4,2,4,0}}}},
        {{{{0,5,0,8,6},{5,5,3,8,1},{8,3,7,8,5},{1,4,3,4,4}},{{4,0,0,6,8},{4,8,0,1,7},{6,2,6,4,2},{4,7,5,8,1}},{{3,3,1,5,5},{2,4,6,0,5},{3,1,7,6,5},{6,2,8,1,2}}},
        {{{4,7,5,2,1},{6,5,3,1,5},{8,8,5,5,4},{3,3,4,1,5}},{{7,8,2,8,1},{6,0,2,4,5},{8,4,5,0,3},{7,2,5,0,0}},{{2,2,2,7,8},{1,0,7,5,8},{0,2,5,4,4},{1,3,5,8,4}}}}
    };  //(4,2,3,4,5)

    //0tensor,1functor,2keep_dims,3any_order,4expected
    auto test_data = std::make_tuple(
        //keep_dims is false
        std::make_tuple(tensor_type{}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, prod{}, false, true, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(1,0), sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), prod{}, false, true, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, prod{}, false, true, tensor_type(720)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, false, true, tensor_type(9)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, false, true, tensor_type(0)),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, sum{}, false, true, tensor_type(28)),
        //keep_dims is true
        std::make_tuple(tensor_type{}, sum{}, true, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}, prod{}, true, true, tensor_type{value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(2,1,0), sum{}, true, true, tensor_type{{{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), prod{}, true, true, tensor_type{{{value_type{1}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, sum{}, true, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, prod{}, true, true, tensor_type{{720}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, true, true, tensor_type{{9}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, true, true, tensor_type{{0}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, sum{}, true, true, tensor_type{{{28}}}),
        //any_order
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, center{}, false, false, tensor_type(6.5)),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten,sum{},false,false,tensor_type(7696)),
        std::make_tuple((test_ten-1)*(test_ten+1),sum{},false,false,tensor_type(10450)),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,1)+test_ten(2,0,2),sum{},false,false,tensor_type(7668)),
        std::make_tuple((test_ten+test_ten(0))*(test_ten(1,1)-test_ten(2,0,2)),sum{},false,false,tensor_type(-307))
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto functor = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto any_arder = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto result = reduce_range_flatten(tensor, functor, keep_dims, any_arder);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_reduce","[test_reduce]",
    (gtensor::reduce_bin<1>),
    (gtensor::reduce_bin<4>),
    (gtensor::reduce_bin<0>),
    (gtensor::reduce_rng<1>),
    (gtensor::reduce_rng<4>),
    (gtensor::reduce_rng<0>),
    (gtensor::reduce_auto<1>),
    (gtensor::reduce_auto<4>),
    (gtensor::reduce_auto<0>)
)
{
    using policy = TestType;
    using value_type = std::size_t;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = tensor_type::shape_type;
    using test_reduce_::sum;
    using gtensor::reduce;
    using helpers_for_testing::generate_lehmer;

    tensor_type t(shape_type{32,16,8,64,4,16}); //1<<24
    generate_lehmer(t.begin(),t.end(),123);
    std::for_each(t.begin(),t.end(),[](auto& e){e%=2;});

    REQUIRE(
        reduce(policy{},t,std::vector<int>{0,2,3,5},std::plus<void>{},sum{},false,true,value_type{0}) ==
        tensor_type{{130985,131241,131122,131418},{131063,130985,130771,131109},{130880,130776,131173,130602},{130953,131504,130845,130713},{130533,131118,131072,130999},{131537,131601,131137,130747},{130771,131109,131092,131087},{131009,131288,131239,131240},{131045,130487,131331,131042},{130738,130992,131102,131046},{131303,130886,131084,131374},{130716,131235,131133,130959},{130922,131557,131289,131151},{130930,130964,131054,130756},{131444,131149,131506,130919},{131779,130963,131140,130513}}
    );

    REQUIRE(
        reduce(policy{},t,std::vector<int>{0,1,3,5},std::plus<void>{},sum{},false,true,value_type{0}) ==
        tensor_type{{262550,262341,261387,262795},{262126,262338,263037,261910},{262039,261752,262161,261542},{261610,261842,262138,262175},{262190,262597,262274,262151},{262391,262182,262827,261800},{261970,262571,261781,261997},{261732,262232,262485,261305}}
    );

    REQUIRE(
        reduce(policy{},t,std::vector<int>{0,1,2,3,5},std::plus<void>{},sum{},false,true,value_type{0}) ==
        tensor_type{2096608,2097855,2098090,2095675}
    );

    REQUIRE(
        reduce(policy{},t,std::vector<int>{1,2,3,4,5},std::plus<void>{},sum{},false,true,value_type{0}) ==
        tensor_type{262294,262306,261408,262194,261907,262785,262000,262093,262364,261515,261966,262240,262489,262095,262097,262023,262345,261632,262444,262166,262217,262027,262144,262242,262540,261711,262199,261935,262167,262702,261858,262123}
    );


    REQUIRE(
        reduce(policy{},t.reshape(-1,8),0,std::plus<void>{},sum{},false,true,value_type{0}) ==
        tensor_type{1048319,1046857,1048302,1049964,1049955,1047962,1048209,1048660}
    );

    REQUIRE(
        reduce(policy{},t,std::vector<int>{0,1,2,3,4,5},std::plus<void>{},sum{},false,true,value_type{0}) ==
        tensor_type(8388228)
    );

    REQUIRE(
        reduce_flatten(policy{},t,std::plus<void>{},sum{},false,true,value_type{0}) ==
        tensor_type(8388228)
    );
}

TEST_CASE("test_slide","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using test_reduce_::cumprod_reverse;
    using test_reduce_::cumsum;
    using test_reduce_::diff_1;
    using test_reduce_::diff_2;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}};

    //0tensor,1axis,2functor,3window_size,4window_step,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, cumsum{}, index_type{0}, index_type{1}, tensor_type{}),
        std::make_tuple(tensor_type{}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{1}, cumsum{}, index_type{5}, index_type{1}, tensor_type{}.reshape(0,2,3)),
        std::make_tuple(tensor_type{1}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{120,120,60,20,5}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{1,2,3},{5,7,9},{12,15,18}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{1,3,6},{4,9,15},{7,15,24}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{28,80,162},{28,40,54},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{6,6,3},{120,30,6},{504,72,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{-2}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{28,80,162},{28,40,54},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{-1}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{6,6,3},{120,30,6},{504,72,9}}),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{{8,8,28,20,20},{0,0,12,28,4},{32,8,20,0,4},{0,28,28,20,32}},{{12,28,52,48,20},{24,16,16,44,12},{40,12,20,4,8},{24,52,40,44,60}},{{40,52,56,60,48},{32,28,48,44,24},{52,44,44,16,36},{44,84,56,76,80}}}),
        std::make_tuple((test_ten+1)*(test_ten-1), dim_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{{3,3,48,24,24},{2,2,56,72,24},{65,5,80,71,24},{64,53,128,95,87}},{{0,24,35,48,-1},{35,39,35,63,2},{38,39,34,63,2},{73,74,42,98,50}},{{48,35,0,8,48},{51,43,63,7,56},{59,106,98,15,104},{83,169,113,78,128}}}),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,2)+test_ten(2,3), dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{{11,13,18,19,16},{7,9,10,23,8},{23,13,14,9,8},{7,23,18,19,22}},{{21,29,35,40,27},{20,22,18,43,17},{40,25,23,19,16},{20,45,32,39,43}},{{37,46,47,57,45},{29,34,33,59,27},{58,44,38,31,30},{32,69,47,61,62}}}),
        std::make_tuple((test_ten-test_ten(0))*(test_ten(1,2)+test_ten(2,3)), dim_type{2}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}},{{-7,20,16,34,4},{42,78,70,43,49},{-42,-51,-71,-62,-62},{42,33,17,26,20}},{{35,71,47,29,41},{14,41,61,-2,10},{-35,19,23,50,86},{35,44,32,59,41}}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = slide<value_type>(tensor, axis, functor, window_size, window_step);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_slide_flatten","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type,order>;
    using index_type = typename tensor_type::index_type;
    using test_reduce_::cumprod_reverse;
    using test_reduce_::cumsum;
    using test_reduce_::diff_1;
    using test_reduce_::diff_2;
    using gtensor::slide_flatten;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}};

    //0tensor,1functor,2window_size,3window_step,4expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{120,120,60,20,5}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15,21,28,36,45}),
        std::make_tuple(tensor_type{{1,3,2},{5,7,4},{6,7,8}}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{{1,3,2},{5,7,4},{6,7,8}}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten, cumsum{}, index_type{1}, index_type{1}, tensor_type{8,16,44,64,84,84,84,96,124,128,160,168,188,188,192,192,220,248,268,300,304,324,348,376,376,400,416,420,436,444,452,456,456,460,464,488,512,524,548,576,604,628,632,644,672,680,692,724,724,736,748,780,804,816,844,864,896,912,944,964}),
        std::make_tuple((test_ten+1)*(test_ten-1), cumsum{}, index_type{1}, index_type{1}, tensor_type{3,6,54,78,102,101,100,108,156,156,219,222,246,245,245,244,292,340,364,427,427,451,486,534,533,568,583,583,598,601,604,604,603,603,603,638,673,681,716,764,812,847,847,855,903,906,914,977,976,984,992,1055,1090,1098,1146,1170,1233,1248,1311,1335}),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,2)+test_ten(2,3), cumsum{}, index_type{1}, index_type{1}, tensor_type{11,24,42,61,77,84,93,103,126,134,157,170,184,193,201,208,231,249,268,290,300,316,333,354,365,378,391,399,419,428,445,457,466,476,484,497,519,533,553,574,590,607,619,636,654,663,675,690,706,716,734,753,768,780,794,806,830,845,867,886}),
        std::make_tuple((test_ten-test_ten(0))*(test_ten(1,2)+test_ten(2,3)), cumsum{}, index_type{1}, index_type{1}, tensor_type{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,20,16,34,4,46,82,74,47,53,11,2,-18,-9,-9,33,24,8,17,11,46,82,58,40,52,66,93,113,50,62,27,81,85,112,148,183,192,180,207,189})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto functor = std::get<1>(t);
        auto window_size = std::get<2>(t);
        auto window_step = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto result = slide_flatten<value_type>(tensor, functor, window_size, window_step);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_slide_custom_arg","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using test_reduce_::moving_avarage;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3window_size,4window_step,5denom,6expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{1}, value_type{3}, tensor_type{2,3,4,5,6,7,8,9}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{2}, value_type{3}, tensor_type{2,4,6,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{2,3,4},{5,6,7}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{1,2},{4,5},{7,8}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        auto denom = std::get<5>(t);
        auto expected = std::get<6>(t);
        auto result = slide<value_type>(tensor, axis, functor, window_size, window_step, window_size, window_step, denom);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_slide_exception","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using gtensor::value_error;
    using test_reduce_::cumsum;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3window_size,4window_step
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(0), dim_type{0}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{3}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{1}, dim_type{0}, cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum{}, index_type{6}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{2}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, cumsum{}, index_type{3}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, cumsum{}, index_type{4}, index_type{1})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        REQUIRE_THROWS_AS(slide<value_type>(tensor, axis, functor, window_size, window_step), value_error);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_slide_flatten_exception","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using gtensor::value_error;
    using test_reduce_::cumsum;
    using gtensor::slide_flatten;
    using helpers_for_testing::apply_by_element;

    //0tensor,1functor,2window_size,3window_step
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(0), cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1}, cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumsum{}, index_type{6}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, cumsum{}, index_type{7}, index_type{1})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto functor = std::get<1>(t);
        auto window_size = std::get<2>(t);
        auto window_step = std::get<3>(t);
        REQUIRE_THROWS_AS(slide_flatten<value_type>(tensor, functor, window_size, window_step), value_error);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_transform","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sort;
    using gtensor::transform;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, sort{}, tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,3,2,1,0}, dim_type{0}, sort{}, tensor_type{0,1,1,2,2,3,3}),
        std::make_tuple(tensor_type{{2,1,3},{3,0,1}}, dim_type{0}, sort{}, tensor_type{{2,0,1},{3,1,3}}),
        std::make_tuple(tensor_type{{2,1,3},{3,0,1}}, dim_type{1}, sort{}, tensor_type{{1,2,3},{0,1,3}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{0}, sort{}, tensor_type{{{0,1,1},{3,0,1}},{{2,2,3},{3,0,1}}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{1}, sort{}, tensor_type{{{2,0,1},{3,1,3}},{{0,0,1},{3,2,1}}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{2}, sort{}, tensor_type{{{1,2,3},{0,1,3}},{{0,1,2},{0,1,3}}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto expected = std::get<3>(t);
        transform(tensor, axis, functor);
        REQUIRE(tensor == expected);
    };
    apply_by_element(test, test_data);
}

