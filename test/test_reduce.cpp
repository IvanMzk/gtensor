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
    //0dim,1sorted_axes,2expected
    auto test_data = std::make_tuple(
        //axes scalar
        std::make_tuple(dim_type{1},dim_type{0},axes_type{}),
        std::make_tuple(dim_type{2},dim_type{0},axes_type{1}),
        std::make_tuple(dim_type{2},dim_type{1},axes_type{0}),
        std::make_tuple(dim_type{4},dim_type{0},axes_type{1,2,3}),
        std::make_tuple(dim_type{4},dim_type{1},axes_type{0,2,3}),
        std::make_tuple(dim_type{4},dim_type{2},axes_type{0,1,3}),
        std::make_tuple(dim_type{4},dim_type{3},axes_type{0,1,2}),
        //axes container
        std::make_tuple(dim_type{1},axes_type{0},axes_type{}),
        std::make_tuple(dim_type{2},axes_type{0},axes_type{1}),
        std::make_tuple(dim_type{2},axes_type{1},axes_type{0}),
        std::make_tuple(dim_type{2},axes_type{0,1},axes_type{}),
        std::make_tuple(dim_type{5},axes_type{1,3},axes_type{0,2,4}),
        std::make_tuple(dim_type{5},axes_type{0,1,2},axes_type{3,4}),
        std::make_tuple(dim_type{5},axes_type{0,4},axes_type{1,2,3}),
        std::make_tuple(dim_type{5},axes_type{0,2,4},axes_type{1,3})
    );
    auto test = [](const auto& t){
        auto dim = std::get<0>(t);
        auto sorted_axes = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_reduce_axes_map<config_type>(dim,sorted_axes);
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



// namespace test_reduce_{

// struct max
// {
//     template<typename It>
//     auto operator()(It first, It last){
//         if (first==last){throw gtensor::value_error{"empty range"};}
//         const auto& init = *first;
//         return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::max(u,v);});
//     }
// };
// struct min
// {
//     template<typename It>
//     auto operator()(It first, It last){
//         if (first==last){throw gtensor::value_error{"empty range"};}
//         const auto& init = *first;
//         return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::min(u,v);});
//     }
// };
// struct min_or_zero
// {
//     template<typename It>
//     auto operator()(It first, It last){
//         auto res = min{}(first,last);
//         return res < 0 ? 0 : res;
//     }
// };

// struct sum
// {
//     template<typename It>
//     auto operator()(It first, It last){
//         using value_type = typename std::iterator_traits<It>::value_type;
//         if (first==last){return value_type{0};}
//         const auto& init = *first;
//         return std::accumulate(++first,last,init,std::plus{});
//     }
// };
// struct sum_of_squares
// {
//     template<typename It>
//     auto operator()(It first, It last){
//         const auto res = sum{}(first,last);
//         return res*res;
//     }
// };
// struct sum_random_access
// {
//     template<typename It>
//     auto operator()(It first, It last){
//         using difference_type = typename std::iterator_traits<It>::difference_type;
//         using value_type = typename std::iterator_traits<It>::value_type;
//         if (first==last){return value_type{0};}
//         const auto n = last-first;
//         difference_type i{0};
//         value_type res = first[i];
//         for (++i;i!=n; ++i){
//             res+=first[i];
//         }
//         return res;
//     }
// };
// struct sum_random_access_reverse
// {
//     template<typename It>
//     auto operator()(It first, It last){
//         using difference_type = typename std::iterator_traits<It>::difference_type;
//         using value_type = typename std::iterator_traits<It>::value_type;
//         if (first==last){return value_type{0};}
//         const auto n = last-first;
//         difference_type i{-1};
//         value_type res = last[i];
//         for (--i;i!=-n-1; --i){
//             res+=last[i];
//         }
//         return res;
//     }
// };

// struct sum_init
// {
//     template<typename It>
//     auto operator()(It first, It last, const typename std::iterator_traits<It>::value_type& init){
//         using value_type = typename std::iterator_traits<It>::value_type;
//         if (first==last){return value_type{0};}
//         return std::accumulate(first,last,init,std::plus{});
//     }
// };
// struct prod
// {
//     template<typename It>
//     auto operator()(It first, It last){
//         using value_type = typename std::iterator_traits<It>::value_type;
//         if (first==last){return value_type{1};}
//         value_type prod{1};
//         while(last!=first){
//             prod*=*--last;
//         }
//         return prod;
//     }
// };

// struct cumsum{
//     template<typename It, typename DstIt>
//     void operator()(It first, It, DstIt dfirst, DstIt dlast){
//         auto cumsum_ = *first;
//         *dfirst = cumsum_;
//         for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
//             cumsum_+=*first;
//             *dfirst = cumsum_;
//         }
//     }
// };

// struct cumprod_reverse{
//     template<typename It, typename DstIt>
//     void operator()(It, It last, DstIt dfirst, DstIt dlast){
//         auto cumprod_ = *--last;
//         *--dlast = cumprod_;
//         while(dlast!=dfirst){
//             cumprod_*=*--last;
//             *--dlast = cumprod_;
//         }
//     }
// };

// struct moving_avarage{
//     template<typename It, typename DstIt, typename IdxT>
//     void operator()(It first, It, DstIt dfirst, DstIt dlast, const IdxT& window_size, const IdxT& window_step, const typename std::iterator_traits<It>::value_type& denom){
//         using index_type = IdxT;
//         using value_type = typename std::iterator_traits<It>::value_type;
//         value_type sum{0};
//         auto it = first;
//         for (index_type i{0}; i!=window_size; ++i, ++it){
//             sum+=*it;
//         }
//         for(;dfirst!=dlast;++dfirst){
//             *dfirst = sum/denom;
//             for (index_type i{0}; i!=window_step; ++i,++first){
//                 sum-=*first;
//             }
//             for (index_type i{0}; i!=window_step; ++i, ++it){
//                 sum+=*it;
//             }
//         }
//     }
// };

// struct diff_1{
//     template<typename It, typename DstIt>
//     void operator()(It first, It, DstIt dfirst, DstIt dlast){
//         for (;dfirst!=dlast;++dfirst){
//             auto prev = *first;
//             *dfirst = *(++first) - prev;
//         }
//     }
// };

// struct diff_2{
//     template<typename It, typename DstIt>
//     void operator()(It first, It, DstIt dfirst, DstIt dlast){
//         for (;dfirst!=dlast;++dfirst){
//             auto v0 = *first;
//             auto v1 = *(++first);
//             auto v2 = *(++first);
//             *dfirst = v2-v1-v1+v0;
//             --first;
//         }
//     }
// };

// struct sort{
//     template<typename It>
//     void operator()(It first, It last){
//         std::sort(first,last);
//     }
// };

// }   //end of namespace test_reduce_

// TEST_CASE("test_reduce","[test_reduce]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using test_reduce_::sum;
//     using test_reduce_::sum_of_squares;
//     using test_reduce_::sum_random_access;
//     using test_reduce_::sum_random_access_reverse;
//     using test_reduce_::prod;
//     using test_reduce_::max;
//     using test_reduce_::min;
//     using test_reduce_::min_or_zero;
//     using gtensor::reduce;
//     using helpers_for_testing::apply_by_element;
//     //0tensor,1axes,2functor,3keep_dims,4expected
//     auto test_data = std::make_tuple(
//         //single axis
//         //keep_dims is false
//         std::make_tuple(tensor_type{}, dim_type{0}, sum{}, false, tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{}, dim_type{0}, prod{}, false, tensor_type(value_type{1})),
//         std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, false, tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, false, tensor_type{value_type{0}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, false, tensor_type{}.reshape(3,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, false, tensor_type{}.reshape(2,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, false, tensor_type{{value_type{0},value_type{0},value_type{0}},{value_type{0},value_type{0},value_type{0}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, prod{}, false, tensor_type{{value_type{1},value_type{1},value_type{1}},{value_type{1},value_type{1},value_type{1}}}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, false, tensor_type{21}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, false, tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, false, tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, false, tensor_type{21}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, false, tensor_type{5,7,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, false, tensor_type{6,15}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, false, tensor_type{6,120}),
//         std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, false, tensor_type{4,6,7,9}),
//         std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, false, tensor_type{1,0}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, false, tensor_type{{4,6},{8,10}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum_random_access{}, false, tensor_type{{4,6},{8,10}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum_random_access_reverse{}, false, tensor_type{{4,6},{8,10}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, false, tensor_type{{2,4},{10,12}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-2}, sum{}, false, tensor_type{{2,4},{10,12}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, false, tensor_type{{1,5},{9,13}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-1}, sum{}, false, tensor_type{{1,5},{9,13}}),
//         //keep_dims is true
//         std::make_tuple(tensor_type{}, dim_type{0}, sum{}, true, tensor_type{value_type{0}}),
//         std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, true, tensor_type{}.reshape(1,0)),
//         std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, true, tensor_type{{value_type{0}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, true, tensor_type{}.reshape(1,3,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, true, tensor_type{}.reshape(2,1,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}},{{value_type{0}},{value_type{0}},{value_type{0}}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, prod{}, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}},{{value_type{1}},{value_type{1}},{value_type{1}}}}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, true, tensor_type{21}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, true, tensor_type{{1,2,3,4,5,6}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, true, tensor_type{{5,7,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, true, tensor_type{{6},{15}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, true, tensor_type{{6},{120}}),
//         std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, true, tensor_type{{4,6,7,9}}),
//         std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, true, tensor_type{{1},{0}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, true, tensor_type{{{4,6},{8,10}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum{}, true, tensor_type{{{4,6},{8,10}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, true, tensor_type{{{2,4}},{{10,12}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-2}, sum{}, true, tensor_type{{{2,4}},{{10,12}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, true, tensor_type{{{1},{5}},{{9},{13}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-1}, sum{}, true, tensor_type{{{1},{5}},{{9},{13}}}),
//         //axes is container
//         //keep_dims is false
//         //empty axes
//         std::make_tuple(tensor_type{}, std::vector<dim_type>{}, sum{}, false, tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{}, sum{}, false, tensor_type{}.reshape(2,3,0)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum{}, false, tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum_of_squares{}, false, tensor_type{1,4,9,16,25,36}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum{}, false, tensor_type{{1},{2},{3},{4},{5},{6}}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum_of_squares{}, false, tensor_type{{1},{4},{9},{16},{25},{36}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum{}, false, tensor_type{{1,2,3,4,5,6}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum_of_squares{}, false, tensor_type{{1,4,9,16,25,36}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{}, prod{}, false, tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(tensor_type{{1,-2,3},{-4,5,6}}, std::vector<dim_type>{}, min_or_zero{}, false, tensor_type{{1,0,3},{0,5,6}}),
//         //not empty axes
//         std::make_tuple(tensor_type{}, std::vector<dim_type>{0}, sum{}, false, tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0}, sum{}, false, tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1}, sum{}, false, tensor_type{value_type{0}}),
//         std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0,1}, sum{}, false, tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1,0}, sum{}, false, tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0}, sum{}, false, tensor_type{}.reshape(3,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{1}, sum{}, false, tensor_type{}.reshape(2,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, sum{}, false, tensor_type{{value_type{0},value_type{0},value_type{0}},{value_type{0},value_type{0},value_type{0}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, sum{}, false, tensor_type{value_type{0},value_type{0},value_type{0}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, prod{}, false, tensor_type{{value_type{1},value_type{1},value_type{1}},{value_type{1},value_type{1},value_type{1}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, prod{}, false, tensor_type{value_type{1},value_type{1},value_type{1}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0,1}, sum{}, false, tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0}, sum{}, false, tensor_type{21}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1}, sum{}, false, tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1,0}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0,1}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0}, sum{}, false, tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{1}, sum{}, false, tensor_type{21}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0,1}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, sum{}, false, tensor_type{5,7,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, sum{}, false, tensor_type{6,15}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1,0}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, prod{}, false, tensor_type{6,120}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, prod{}, false, tensor_type{4,10,18}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0,1}, prod{}, false, tensor_type(720)),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{0}, sum{}, false, tensor_type{{4,6},{8,10}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, sum{}, false, tensor_type{{2,4},{10,12}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, prod{}, false, tensor_type{{0,3},{24,35}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2}, sum{}, false, tensor_type{{1,5},{9,13}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum_random_access{}, false, tensor_type{6,22}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum_random_access_reverse{}, false, tensor_type{6,22}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2,0}, prod{}, false, tensor_type{0,252}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-2,-1}, sum{}, false, tensor_type{6,22}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-1,-3}, prod{}, false, tensor_type{0,252}),
//         //keep_dims is true
//         //empty axes
//         std::make_tuple(tensor_type{}, std::vector<dim_type>{}, sum{}, true, tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{}, sum{}, true, tensor_type{}.reshape(2,3,0)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum{}, true, tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum_of_squares{}, true, tensor_type{1,4,9,16,25,36}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum{}, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum_of_squares{}, true, tensor_type{{1},{4},{9},{16},{25},{36}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum{}, true, tensor_type{{1,2,3,4,5,6}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum_of_squares{}, true, tensor_type{{1,4,9,16,25,36}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{}, prod{}, true, tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(tensor_type{{1,-2,3},{-4,5,6}}, std::vector<dim_type>{}, min_or_zero{}, true, tensor_type{{1,0,3},{0,5,6}}),
//         //not empty axes
//         std::make_tuple(tensor_type{}, std::vector<dim_type>{0}, sum{}, true, tensor_type{value_type{0}}),
//         std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0}, sum{}, true, tensor_type{}.reshape(1,0)),
//         std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1}, sum{}, true, tensor_type{{value_type{0}}}),
//         std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0,1}, sum{}, true, tensor_type{{value_type{0}}}),
//         std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1,0}, sum{}, true, tensor_type{{value_type{0}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0}, sum{}, true, tensor_type{}.reshape(1,3,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{1}, sum{}, true, tensor_type{}.reshape(2,1,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, sum{}, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}},{{value_type{0}},{value_type{0}},{value_type{0}}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, sum{}, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, prod{}, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}},{{value_type{1}},{value_type{1}},{value_type{1}}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, prod{}, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}}}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0,1}, sum{}, true, tensor_type{}.reshape(1,1,0)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0}, sum{}, true, tensor_type{21}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1}, sum{}, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1,0}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0,1}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0}, sum{}, true, tensor_type{{1,2,3,4,5,6}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{1}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0,1}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, sum{}, true, tensor_type{{5,7,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, sum{}, true, tensor_type{{6},{15}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1,0}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, prod{}, true, tensor_type{{6},{120}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, prod{}, true, tensor_type{{4,10,18}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0,1}, prod{}, true, tensor_type{{720}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{0}, sum{}, true, tensor_type{{{4,6},{8,10}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, sum{}, true, tensor_type{{{2,4}},{{10,12}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, prod{}, true, tensor_type{{{0,3}},{{24,35}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2}, sum{}, true, tensor_type{{{1},{5}},{{9},{13}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum{}, true, tensor_type{{{6}},{{22}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2,0}, prod{}, true, tensor_type{{{0},{252}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-2,-1}, sum{}, true, tensor_type{{{6}},{{22}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-1,-3}, prod{}, true, tensor_type{{{0},{252}}}),
//         //axes in initializer_list
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::initializer_list<dim_type>{-2,-1}, sum{}, false, tensor_type{6,22}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::initializer_list<dim_type>{-1,-3}, prod{}, false, tensor_type{0,252}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::initializer_list<dim_type>{1,2}, sum{}, true, tensor_type{{{6}},{{22}}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::initializer_list<dim_type>{2,0}, prod{}, true, tensor_type{{{0},{252}}})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto functor = std::get<2>(t);
//         auto keep_dims = std::get<3>(t);
//         auto expected = std::get<4>(t);
//         auto result = reduce(tensor, axes, functor, keep_dims);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_reduce_custom_arg","[test_reduce]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using test_reduce_::sum_init;
//     using gtensor::reduce;
//     using helpers_for_testing::apply_by_element;
//     //0tensor,1axes,2functor,3keep_dims,4init,5expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, sum_init{}, false, value_type{0}, tensor_type(15)),
//         std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, sum_init{}, true, value_type{-1}, tensor_type{14}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum_init{}, false, value_type{-1}, tensor_type{4,6,8}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum_init{}, false, value_type{1}, tensor_type{7,16})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto functor = std::get<2>(t);
//         auto keep_dims = std::get<3>(t);
//         auto init = std::get<4>(t);
//         auto expected = std::get<5>(t);
//         auto result = reduce(tensor, axes, functor, keep_dims, init);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_reduce_ecxeption","[test_reduce]")
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using test_reduce_::sum;
//     using gtensor::axis_error;
//     using gtensor::reduce;
//     using helpers_for_testing::apply_by_element;


//     //0tensor,1axes,2functor,3keep_dim
//     auto test_data = std::make_tuple(
//         //single axis
//         std::make_tuple(tensor_type(0), dim_type{0}, sum{}, false),
//         std::make_tuple(tensor_type{}, dim_type{1}, sum{}, false),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{1}, sum{}, false),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{2}, sum{}, false),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{2}, sum{}, false),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{4}, sum{}, false),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{3}, sum{}, false),
//         //axes container
//         std::make_tuple(tensor_type(0), std::vector<dim_type>{0}, sum{}, false),
//         std::make_tuple(tensor_type{0}, std::vector<dim_type>{0,0}, sum{}, false),
//         std::make_tuple(tensor_type{0}, std::vector<dim_type>{1,1}, sum{}, false),
//         std::make_tuple(tensor_type{0}, std::vector<dim_type>{1}, sum{}, false),
//         std::make_tuple(tensor_type{0}, std::vector<dim_type>{0,1}, sum{}, false),
//         std::make_tuple(tensor_type{0}, std::vector<dim_type>{1,0}, sum{}, false),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0,0}, sum{}, false),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{1,1}, sum{}, false),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0,1}, sum{}, false),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{1,0}, sum{}, false),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{3}, sum{}, false),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{0,1,0}, sum{}, false),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{1,1}, sum{}, false),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{0,1,2,0}, sum{}, false)
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto functor = std::get<2>(t);
//         auto keep_dim = std::get<3>(t);
//         REQUIRE_THROWS_AS(reduce(tensor, axes, functor, keep_dim), axis_error);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_reduce_flatten","[test_reduce]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using test_reduce_::sum;
//     using test_reduce_::prod;
//     using test_reduce_::max;
//     using test_reduce_::min;
//     using gtensor::reduce_flatten;
//     using helpers_for_testing::apply_by_element;
//     //0tensor,1functor,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dims is false
//         std::make_tuple(tensor_type{}, sum{}, false, tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{}, prod{}, false, tensor_type(value_type{1})),
//         std::make_tuple(tensor_type{}.reshape(1,0), sum{}, false, tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{}.reshape(1,0), prod{}, false, tensor_type(value_type{1})),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, sum{}, false, tensor_type(21)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, prod{}, false, tensor_type(720)),
//         std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, false, tensor_type(9)),
//         std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, false, tensor_type(0)),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, sum{}, false, tensor_type(28)),
//         //keep_dims is true
//         std::make_tuple(tensor_type{}, sum{}, true, tensor_type{value_type{0}}),
//         std::make_tuple(tensor_type{}, prod{}, true, tensor_type{value_type{1}}),
//         std::make_tuple(tensor_type{}.reshape(2,1,0), sum{}, true, tensor_type{{{value_type{0}}}}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3), prod{}, true, tensor_type{{{value_type{1}}}}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6}, sum{}, true, tensor_type{21}),
//         std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5,6}}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, sum{}, true, tensor_type{{21}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, prod{}, true, tensor_type{{720}}),
//         std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, true, tensor_type{{9}}),
//         std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, true, tensor_type{{0}}),
//         std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, sum{}, true, tensor_type{{{28}}})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto functor = std::get<1>(t);
//         auto keep_dims = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         auto result = reduce_flatten(tensor, functor, keep_dims);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_reduce_big","[test_reduce]")
// {
//     using value_type = std::size_t;
//     using tensor_type = gtensor::tensor<value_type>;
//     using shape_type = tensor_type::shape_type;
//     using test_reduce_::sum;
//     using test_reduce_::prod;
//     using test_reduce_::max;
//     using test_reduce_::min;
//     using gtensor::reduce;

//     tensor_type t(shape_type{32,16,8,64,4,16}); //1<<24
//     std::for_each(t.begin(),t.end(),[init=value_type{123}](auto& e)mutable{e=init*279470273%0xfffffffb; init=e;});
//     std::for_each(t.begin(),t.end(),[](auto& e){e%=2;});

//     REQUIRE(
//         reduce(t,std::vector<int>{0,2,3,5},sum{},false) ==
//         tensor_type{{130985,131241,131122,131418},{131063,130985,130771,131109},{130880,130776,131173,130602},{130953,131504,130845,130713},{130533,131118,131072,130999},{131537,131601,131137,130747},{130771,131109,131092,131087},{131009,131288,131239,131240},{131045,130487,131331,131042},{130738,130992,131102,131046},{131303,130886,131084,131374},{130716,131235,131133,130959},{130922,131557,131289,131151},{130930,130964,131054,130756},{131444,131149,131506,130919},{131779,130963,131140,130513}}
//     );

//     REQUIRE(
//         reduce(t,std::vector<int>{0,1,3,5},sum{},false) ==
//         tensor_type{{262550,262341,261387,262795},{262126,262338,263037,261910},{262039,261752,262161,261542},{261610,261842,262138,262175},{262190,262597,262274,262151},{262391,262182,262827,261800},{261970,262571,261781,261997},{261732,262232,262485,261305}}
//     );

//     REQUIRE(
//         reduce(t,std::vector<int>{0,1,2,3,5},sum{},false) ==
//         tensor_type{2096608,2097855,2098090,2095675}
//     );

//     REQUIRE(
//         reduce(t,std::vector<int>{1,2,3,4,5},sum{},false) ==
//         tensor_type{262294,262306,261408,262194,261907,262785,262000,262093,262364,261515,261966,262240,262489,262095,262097,262023,262345,261632,262444,262166,262217,262027,262144,262242,262540,261711,262199,261935,262167,262702,261858,262123}
//     );
// }

// TEST_CASE("test_slide","[test_reduce]")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using index_type = typename tensor_type::index_type;
//     using test_reduce_::cumprod_reverse;
//     using test_reduce_::cumsum;
//     using test_reduce_::diff_1;
//     using test_reduce_::diff_2;
//     using gtensor::slide;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1axis,2functor,3window_size,4window_step,5expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{}, dim_type{0}, cumsum{}, index_type{0}, index_type{1}, tensor_type{}),
//         std::make_tuple(tensor_type{}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{1}, cumsum{}, index_type{5}, index_type{1}, tensor_type{}.reshape(0,2,3)),
//         std::make_tuple(tensor_type{1}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
//         std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{120,120,60,20,5}),
//         std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
//         std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{1,2,3},{5,7,9},{12,15,18}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{1,3,6},{4,9,15},{7,15,24}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{28,80,162},{28,40,54},{7,8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{6,6,3},{120,30,6},{504,72,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{-2}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{28,80,162},{28,40,54},{7,8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{-1}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{6,6,3},{120,30,6},{504,72,9}})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto functor = std::get<2>(t);
//         auto window_size = std::get<3>(t);
//         auto window_step = std::get<4>(t);
//         auto expected = std::get<5>(t);
//         auto result = slide<value_type>(tensor, axis, functor, window_size, window_step);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEMPLATE_TEST_CASE("test_slide_flatten","[test_reduce]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using order = TestType;
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type,order>;
//     using index_type = typename tensor_type::index_type;
//     using test_reduce_::cumprod_reverse;
//     using test_reduce_::cumsum;
//     using test_reduce_::diff_1;
//     using test_reduce_::diff_2;
//     using gtensor::slide_flatten;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1functor,2window_size,3window_step,4expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
//         std::make_tuple(tensor_type{1,2,3,4,5}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{120,120,60,20,5}),
//         std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
//         std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15,21,28,36,45}),
//         std::make_tuple(tensor_type{{1,3,2},{5,7,4},{6,7,8}}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
//         std::make_tuple(tensor_type{{1,3,2},{5,7,4},{6,7,8}}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto functor = std::get<1>(t);
//         auto window_size = std::get<2>(t);
//         auto window_step = std::get<3>(t);
//         auto expected = std::get<4>(t);
//         auto result = slide_flatten<value_type>(tensor, functor, window_size, window_step);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_slide_custom_arg","[test_reduce]")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using index_type = typename tensor_type::index_type;
//     using test_reduce_::moving_avarage;
//     using gtensor::slide;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1axis,2functor,3window_size,4window_step,5denom,6expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{1}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{1}, value_type{3}, tensor_type{2,3,4,5,6,7,8,9}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{2}, value_type{3}, tensor_type{2,4,6,8}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{2,3,4},{5,6,7}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{1,2},{4,5},{7,8}})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto functor = std::get<2>(t);
//         auto window_size = std::get<3>(t);
//         auto window_step = std::get<4>(t);
//         auto denom = std::get<5>(t);
//         auto expected = std::get<6>(t);
//         auto result = slide<value_type>(tensor, axis, functor, window_size, window_step, window_size, window_step, denom);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_slide_exception","[test_reduce]")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using index_type = typename tensor_type::index_type;
//     using gtensor::value_error;
//     using test_reduce_::cumsum;
//     using gtensor::slide;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1axis,2functor,3window_size,4window_step
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(0), dim_type{0}, cumsum{}, index_type{1}, index_type{1}),
//         std::make_tuple(tensor_type{}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{3}, cumsum{}, index_type{1}, index_type{1}),
//         std::make_tuple(tensor_type{1}, dim_type{0}, cumsum{}, index_type{2}, index_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum{}, index_type{6}, index_type{1}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{2}, cumsum{}, index_type{1}, index_type{1}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, cumsum{}, index_type{3}, index_type{1}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, cumsum{}, index_type{4}, index_type{1})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto functor = std::get<2>(t);
//         auto window_size = std::get<3>(t);
//         auto window_step = std::get<4>(t);
//         REQUIRE_THROWS_AS(slide<value_type>(tensor, axis, functor, window_size, window_step), value_error);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_slide_flatten_exception","[test_reduce]")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using index_type = typename tensor_type::index_type;
//     using gtensor::value_error;
//     using test_reduce_::cumsum;
//     using gtensor::slide_flatten;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1functor,2window_size,3window_step
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(0), cumsum{}, index_type{2}, index_type{1}),
//         std::make_tuple(tensor_type{1}, cumsum{}, index_type{2}, index_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5}, cumsum{}, index_type{6}, index_type{1}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, cumsum{}, index_type{7}, index_type{1})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto functor = std::get<1>(t);
//         auto window_size = std::get<2>(t);
//         auto window_step = std::get<3>(t);
//         REQUIRE_THROWS_AS(slide_flatten<value_type>(tensor, functor, window_size, window_step), value_error);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_transform","[test_reduce]")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using test_reduce_::sort;
//     using gtensor::transform;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1axis,2functor,3expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{}, dim_type{0}, sort{}, tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3,3,2,1,0}, dim_type{0}, sort{}, tensor_type{0,1,1,2,2,3,3}),
//         std::make_tuple(tensor_type{{2,1,3},{3,0,1}}, dim_type{0}, sort{}, tensor_type{{2,0,1},{3,1,3}}),
//         std::make_tuple(tensor_type{{2,1,3},{3,0,1}}, dim_type{1}, sort{}, tensor_type{{1,2,3},{0,1,3}}),
//         std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{0}, sort{}, tensor_type{{{0,1,1},{3,0,1}},{{2,2,3},{3,0,1}}}),
//         std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{1}, sort{}, tensor_type{{{2,0,1},{3,1,3}},{{0,0,1},{3,2,1}}}),
//         std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{2}, sort{}, tensor_type{{{1,2,3},{0,1,3}},{{0,1,2},{0,1,3}}})
//     );
//     auto test = [](const auto& t){
//         auto tensor = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto functor = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         transform(tensor, axis, functor);
//         REQUIRE(tensor == expected);
//     };
//     apply_by_element(test, test_data);
// }

