#include <list>
#include <iostream>
#include "catch.hpp"
#include "descriptor.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"


TEMPLATE_TEST_CASE("test_make_broadcast_shape","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using result_shape_type = shape_type;
    using helpers_for_testing::apply_by_element;
    using gtensor::detail::make_broadcast_shape;
    //0shapes,1expected broadcast shape
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(shape_type{}), result_shape_type{}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{}), result_shape_type{}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{1}, shape_type{}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{2,3,4}, shape_type{}), result_shape_type{2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}), result_shape_type{1,2,3}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{1}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{1}, shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{5}, shape_type{5}), result_shape_type{5}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{0}), result_shape_type{1,0}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{1,1}), result_shape_type{1,0}),
        std::make_tuple(std::make_tuple(shape_type{2,1}, shape_type{0}), result_shape_type{2,0}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1}), result_shape_type{1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1}, shape_type{1,1,1}, shape_type{1,1}), result_shape_type{1,1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1,1}, shape_type{1,1}), result_shape_type{1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,5}, shape_type{5,1}), result_shape_type{5,5}),
        std::make_tuple(std::make_tuple(shape_type{1,5}, shape_type{5,1}, shape_type{1,5}, shape_type{1,1}), result_shape_type{5,5}),
        std::make_tuple(std::make_tuple(shape_type{1,2,0}, shape_type{3,1,1}), result_shape_type{3,2,0}),
        std::make_tuple(std::make_tuple(shape_type{2,3,4}, shape_type{3,4}), result_shape_type{2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,3,4}, shape_type{3,4}, shape_type{1,1,1,1}, shape_type{5,1,1,1}), result_shape_type{5,2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,1,4}, shape_type{3,1}, shape_type{3,4}), result_shape_type{2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{0,1,4}, shape_type{3,1}, shape_type{0,3,4}, shape_type{1,3,4}), result_shape_type{0,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4}),
        std::make_tuple(std::make_tuple(shape_type{2,1}, shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4})
    );
    auto test = [](const auto& t){
        auto shapes = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto apply_shapes = [](const auto&...shapes_){
            return make_broadcast_shape<result_shape_type>(shapes_...);
        };
        auto result = std::apply(apply_shapes, shapes);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_make_broadcast_shape_exception","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::broadcast_exception;
    using gtensor::detail::make_broadcast_shape;
    using helpers_for_testing::apply_by_element;
    //0shapes
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{0}, shape_type{2}),
        std::make_tuple(shape_type{2}, shape_type{0}),
        std::make_tuple(shape_type{3}, shape_type{2}),
        std::make_tuple(shape_type{2}, shape_type{3}),
        std::make_tuple(shape_type{1,2}, shape_type{0}),
        std::make_tuple(shape_type{1,2}, shape_type{3}),
        std::make_tuple(shape_type{1,2}, shape_type{4,3}),
        std::make_tuple(shape_type{3,2}, shape_type{4,2}),
        std::make_tuple(shape_type{5,1,2}, shape_type{4,4,2}),
        std::make_tuple(shape_type{3}, shape_type{0}, shape_type{3}),
        std::make_tuple(shape_type{3}, shape_type{3}, shape_type{2}),
        std::make_tuple(shape_type{1,2}, shape_type{3}, shape_type{1}),
        std::make_tuple(shape_type{1,2}, shape_type{1,1}, shape_type{4,4}),
        std::make_tuple(shape_type{5,1,0}, shape_type{2,1}, shape_type{5,2,2}),
        std::make_tuple(shape_type{5,1,2}, shape_type{2,2}, shape_type{4,4,2})
    );
    auto test = [](const auto& shapes){
        auto apply_shapes = [](const auto&...shapes_){
            return make_broadcast_shape<shape_type>(shapes_...);
        };
        REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), broadcast_exception);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_make_strides","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_strides;
    using helpers_for_testing::apply_by_element;
    //0shape,1order,2expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(shape_type{}, c_order{}, shape_type{}),
        std::make_tuple(shape_type{0}, c_order{}, shape_type{1}),
        std::make_tuple(shape_type{1}, c_order{}, shape_type{1}),
        std::make_tuple(shape_type{5}, c_order{}, shape_type{1}),
        std::make_tuple(shape_type{0,0}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{1,0}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{0,1}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{5,0}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{0,5}, c_order{}, shape_type{5,1}),
        std::make_tuple(shape_type{1,1}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{5,1}, c_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{1,5}, c_order{}, shape_type{5,1}),
        std::make_tuple(shape_type{2,3,4}, c_order{}, shape_type{12,4,1}),
        std::make_tuple(shape_type{0,0,0}, c_order{}, shape_type{1,1,1}),
        std::make_tuple(shape_type{2,2,0,2}, c_order{}, shape_type{4,2,2,1}),
        std::make_tuple(shape_type{4,3,2,0}, c_order{}, shape_type{6,2,1,1}),
        std::make_tuple(shape_type{0,3,2,1}, c_order{}, shape_type{6,2,1,1}),
        //f_order
        std::make_tuple(shape_type{}, f_order{}, shape_type{}),
        std::make_tuple(shape_type{0}, f_order{}, shape_type{1}),
        std::make_tuple(shape_type{1}, f_order{}, shape_type{1}),
        std::make_tuple(shape_type{5}, f_order{}, shape_type{1}),
        std::make_tuple(shape_type{0,0}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{1,0}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{0,1}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{5,0}, f_order{}, shape_type{1,5}),
        std::make_tuple(shape_type{0,5}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{1,1}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{5,1}, f_order{}, shape_type{1,5}),
        std::make_tuple(shape_type{1,5}, f_order{}, shape_type{1,1}),
        std::make_tuple(shape_type{2,3,4}, f_order{}, shape_type{1,2,6}),
        std::make_tuple(shape_type{0,0,0}, f_order{}, shape_type{1,1,1}),
        std::make_tuple(shape_type{2,2,0,2}, f_order{}, shape_type{1,2,4,4}),
        std::make_tuple(shape_type{4,3,2,0}, f_order{}, shape_type{1,4,12,24}),
        std::make_tuple(shape_type{0,3,2,1}, f_order{}, shape_type{1,1,3,6})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto order = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_strides(shape, order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_make_strides_div","[test_descriptor]",
    gtensor::config::mode_div_libdivide,
    gtensor::config::mode_div_native
)
{
    using config_type = gtensor::config::extend_config_t<test_config::config_div_mode_selector_t<TestType>, int>;
    using shape_type = typename config_type::shape_type;
    using strides_div_type = gtensor::detail::strides_div_t<config_type>;
    using divider_type = typename strides_div_type::value_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_strides_div;
    using helpers_for_testing::apply_by_element;
    //0shape,1order,2expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(shape_type{}, c_order{}, strides_div_type{}),
        std::make_tuple(shape_type{0}, c_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{1}, c_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{5}, c_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{0,0}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{1,0}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{5,0}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,1}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,5}, c_order{}, strides_div_type{divider_type(5),divider_type(1)}),
        std::make_tuple(shape_type{1,1}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{5,1}, c_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{1,5}, c_order{}, strides_div_type{divider_type(5),divider_type(1)}),
        std::make_tuple(shape_type{0,0,0}, c_order{}, strides_div_type{divider_type(1),divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{2,3,4}, c_order{}, strides_div_type{divider_type(12),divider_type(4),divider_type(1)}),
        std::make_tuple(shape_type{2,2,0,2}, c_order{}, strides_div_type{divider_type(4),divider_type(2),divider_type(2),divider_type(1)}),
        std::make_tuple(shape_type{4,3,2,0}, c_order{}, strides_div_type{divider_type(6),divider_type(2),divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,3,2,1}, c_order{}, strides_div_type{divider_type(6),divider_type(2),divider_type(1),divider_type(1)}),
        //f_order
        std::make_tuple(shape_type{}, f_order{}, strides_div_type{}),
        std::make_tuple(shape_type{0}, f_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{1}, f_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{5}, f_order{}, strides_div_type{divider_type(1)}),
        std::make_tuple(shape_type{0,0}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{1,0}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{5,0}, f_order{}, strides_div_type{divider_type(1),divider_type(5)}),
        std::make_tuple(shape_type{0,1}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,5}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{1,1}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{5,1}, f_order{}, strides_div_type{divider_type(1),divider_type(5)}),
        std::make_tuple(shape_type{1,5}, f_order{}, strides_div_type{divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{0,0,0}, f_order{}, strides_div_type{divider_type(1),divider_type(1),divider_type(1)}),
        std::make_tuple(shape_type{2,3,4}, f_order{}, strides_div_type{divider_type(1),divider_type(2),divider_type(6)}),
        std::make_tuple(shape_type{2,2,0,2}, f_order{}, strides_div_type{divider_type(1),divider_type(2),divider_type(4),divider_type(4)}),
        std::make_tuple(shape_type{4,3,2,0}, f_order{}, strides_div_type{divider_type(1),divider_type(4),divider_type(12),divider_type(24)}),
        std::make_tuple(shape_type{0,3,2,1}, f_order{}, strides_div_type{divider_type(1),divider_type(1),divider_type(3),divider_type(6)})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto order = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_strides_div<config_type>(shape, order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_make_reset_strides","[test_descriptor]",std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::detail::make_reset_strides;
    //0shape,1strides,2expected reset strides
    using test_type = typename std::tuple<shape_type,shape_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},shape_type{},shape_type{}},
        test_type{shape_type{0},shape_type{1},shape_type{0}},
        test_type{shape_type{1},shape_type{1},shape_type{0}},
        test_type{shape_type{5},shape_type{1},shape_type{4}},
        test_type{shape_type{0,0},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{1,0},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{0,1},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{5,0},shape_type{1,1},shape_type{4,0}},
        test_type{shape_type{0,5},shape_type{5,1},shape_type{0,4}},
        test_type{shape_type{1,1},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{5,1},shape_type{1,1},shape_type{4,0}},
        test_type{shape_type{1,5},shape_type{5,1},shape_type{0,4}},
        test_type{shape_type{0,0,0},shape_type{1,1,1},shape_type{0,0,0}},
        test_type{shape_type{2,3,4},shape_type{12,4,1},{12,8,3}},
        test_type{shape_type{2,2,0,2},shape_type{4,2,2,1},shape_type{4,2,0,1}},
        test_type{shape_type{4,3,2,0},shape_type{6,2,1,1},shape_type{18,4,1,0}},
        test_type{shape_type{0,3,2,1},shape_type{6,2,1,1},shape_type{0,4,1,0}}
    );
    auto shape = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto reset_strides_expected = std::get<2>(test_data);
    auto reset_strides_result = make_reset_strides(shape,strides);
    REQUIRE(reset_strides_result == reset_strides_expected);
}

TEMPLATE_TEST_CASE("test_make_adapted_strides","[test_descriptor]",std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::detail::make_adapted_strides;
    //0shape,1strides,2expected
    using test_type = typename std::tuple<shape_type,shape_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},shape_type{},shape_type{}},
        test_type{shape_type{0},shape_type{1},shape_type{1}},
        test_type{shape_type{1},shape_type{1},shape_type{0}},
        test_type{shape_type{5},shape_type{1},shape_type{1}},
        test_type{shape_type{0,0},shape_type{1,1},shape_type{1,1}},
        test_type{shape_type{1,0},shape_type{1,1},shape_type{0,1}},
        test_type{shape_type{0,1},shape_type{1,1},shape_type{1,0}},
        test_type{shape_type{5,0},shape_type{1,1},shape_type{1,1}},
        test_type{shape_type{0,5},shape_type{5,1},shape_type{5,1}},
        test_type{shape_type{1,1},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{5,1},shape_type{1,1},shape_type{1,0}},
        test_type{shape_type{1,5},shape_type{5,1},shape_type{0,1}},
        test_type{shape_type{0,0,0},shape_type{1,1,1},shape_type{1,1,1}},
        test_type{shape_type{2,1,4},shape_type{4,4,1},{4,0,1}},
        test_type{shape_type{2,3,4},shape_type{12,4,1},{12,4,1}},
        test_type{shape_type{2,2,0,2},shape_type{4,2,2,1},shape_type{4,2,2,1}},
        test_type{shape_type{4,3,2,0},shape_type{6,2,1,1},shape_type{6,2,1,1}},
        test_type{shape_type{0,3,2,1},shape_type{6,2,1,1},shape_type{6,2,1,0}}
    );
    auto shape = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = make_adapted_strides(shape,strides);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_make_size","[test_descriptor]",std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using index_type = typename TestType::value_type;
    using gtensor::detail::make_size;
    //shape,expected
    using test_type = typename std::tuple<shape_type,index_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},index_type{1}},
        test_type{shape_type{0},index_type{0}},
        test_type{shape_type{1},index_type{1}},
        test_type{shape_type{5},index_type{5}},
        test_type{shape_type{0,0},index_type{0}},
        test_type{shape_type{1,0},index_type{0}},
        test_type{shape_type{0,1},index_type{0}},
        test_type{shape_type{0,5},index_type{0}},
        test_type{shape_type{5,0},index_type{0}},
        test_type{shape_type{1,1},index_type{1}},
        test_type{shape_type{5,1},index_type{5}},
        test_type{shape_type{1,5},index_type{5}},
        test_type{shape_type{0,0,0},index_type{0}},
        test_type{shape_type{2,3,4},index_type{24}},
        test_type{shape_type{2,2,0,2},index_type{0}},
        test_type{shape_type{4,3,2,0},index_type{0}},
        test_type{shape_type{0,3,2,1},index_type{0}}
    );
    auto shape = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    auto result = make_size(shape);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_flat_to_flat", "[test_descriptor]",
    gtensor::config::mode_div_native,
    gtensor::config::mode_div_libdivide
)
{
    using config_type = gtensor::config::extend_config_t<test_config::config_div_mode_selector_t<TestType>, int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_dividers;
    using gtensor::detail::flat_to_flat;
    using helpers_for_testing::apply_by_element;
    //0flat_idx,1strides,2cstrides,3offset,4order,5expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(index_type{0}, shape_type{1}, shape_type{1}, index_type{0}, c_order{}, index_type{0}),
        std::make_tuple(index_type{0}, shape_type{1}, shape_type{1}, index_type{1}, c_order{}, index_type{1}),
        std::make_tuple(index_type{5}, shape_type{1}, shape_type{1}, index_type{0}, c_order{}, index_type{5}),
        std::make_tuple(index_type{5}, shape_type{1}, shape_type{1}, index_type{1}, c_order{}, index_type{6}),
        std::make_tuple(index_type{5}, shape_type{1,1}, shape_type{1,1}, index_type{0}, c_order{}, index_type{5}),
        std::make_tuple(index_type{5}, shape_type{1,1}, shape_type{1,1}, index_type{10}, c_order{}, index_type{15}),
        std::make_tuple(index_type{0}, shape_type{3,1}, shape_type{2,1}, index_type{0}, c_order{}, index_type{0}),
        std::make_tuple(index_type{5}, shape_type{3,1}, shape_type{2,1}, index_type{0}, c_order{}, index_type{4}),
        std::make_tuple(index_type{5}, shape_type{3,1}, shape_type{2,1}, index_type{10}, c_order{}, index_type{14}),
        std::make_tuple(index_type{34}, shape_type{12,3,1}, shape_type{6,3,1}, index_type{0}, c_order{}, index_type{22}),
        std::make_tuple(index_type{34}, shape_type{12,3,1}, shape_type{6,3,1}, index_type{3}, c_order{}, index_type{25}),
        //f_order
        std::make_tuple(index_type{0}, shape_type{1}, shape_type{1}, index_type{0}, f_order{}, index_type{0}),
        std::make_tuple(index_type{0}, shape_type{1}, shape_type{1}, index_type{1}, f_order{}, index_type{1}),
        std::make_tuple(index_type{5}, shape_type{1}, shape_type{1}, index_type{0}, f_order{}, index_type{5}),
        std::make_tuple(index_type{5}, shape_type{1}, shape_type{1}, index_type{1}, f_order{}, index_type{6}),
        std::make_tuple(index_type{5}, shape_type{1,1}, shape_type{1,1}, index_type{0}, f_order{}, index_type{5}),
        std::make_tuple(index_type{5}, shape_type{1,1}, shape_type{1,1}, index_type{10}, f_order{}, index_type{15}),
        std::make_tuple(index_type{0}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{0}),
        std::make_tuple(index_type{1}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{3}),
        std::make_tuple(index_type{2}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{1}),
        std::make_tuple(index_type{3}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{4}),
        std::make_tuple(index_type{4}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{2}),
        std::make_tuple(index_type{5}, shape_type{1,2}, shape_type{3,1}, index_type{0}, f_order{}, index_type{5}),
        std::make_tuple(index_type{0}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{2}),
        std::make_tuple(index_type{1}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{3}),
        std::make_tuple(index_type{2}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{4}),
        std::make_tuple(index_type{3}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{5}),
        std::make_tuple(index_type{6}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{14}),
        std::make_tuple(index_type{7}, shape_type{1,2,6}, shape_type{1,2,12}, index_type{2}, f_order{}, index_type{15})
    );
    auto test = [](const auto& t){
        auto flat_idx = std::get<0>(t);
        auto strides = std::get<1>(t);
        auto strides_div = make_dividers<config_type>(strides);
        auto cstrides = std::get<2>(t);
        auto offset = std::get<3>(t);
        auto order = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = flat_to_flat(strides_div, cstrides, offset, flat_idx, order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_shape_of_type","[test_descriptor]"){
    using gtensor::detail::make_shape_of_type;
    using shape_type = std::vector<int>;
    auto s = shape_type{1,2,3};
    auto l = std::list<int>{1,2,3};
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(shape_type{1,2,3})), shape_type&&>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(s)), shape_type&>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(std::list{1,2,3})), shape_type>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(l)), shape_type>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>({1,2,3})), shape_type>);

    REQUIRE(make_shape_of_type<shape_type>(shape_type{1,2,3}) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(s) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(std::list{1,2,3}) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(l) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>({1,2,3}) == shape_type{1,2,3});
}

TEST_CASE("test_basic_descriptor","[test_descriptor]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using gtensor::basic_descriptor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::change_order_t;
    using gtensor::detail::make_dividers;
    using gtensor::detail::make_strides_div;
    using helpers_for_testing::apply_by_element;
    //0order,1shape,2expected_strides,3expected_adapted_strides,4expected_reset_strides,5expected_size,6expected_dim
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(c_order{},shape_type{},shape_type{},shape_type{},shape_type{},index_type{1},dim_type{0}),
        std::make_tuple(c_order{},shape_type{0},shape_type{1},shape_type{1},shape_type{0},index_type{0},dim_type{1}),
        std::make_tuple(c_order{},shape_type{1},shape_type{1},shape_type{0},shape_type{0},index_type{1},dim_type{1}),
        std::make_tuple(c_order{},shape_type{5},shape_type{1},shape_type{1},shape_type{4},index_type{5},dim_type{1}),
        std::make_tuple(c_order{},shape_type{0,0},shape_type{1,1},shape_type{1,1},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{1,0},shape_type{1,1},shape_type{0,1},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{0,1},shape_type{1,1},shape_type{1,0},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{5,0},shape_type{1,1},shape_type{1,1},shape_type{4,0},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{0,5},shape_type{5,1},shape_type{5,1},shape_type{0,4},index_type{0},dim_type{2}),
        std::make_tuple(c_order{},shape_type{1,1},shape_type{1,1},shape_type{0,0},shape_type{0,0},index_type{1},dim_type{2}),
        std::make_tuple(c_order{},shape_type{1,5},shape_type{5,1},shape_type{0,1},shape_type{0,4},index_type{5},dim_type{2}),
        std::make_tuple(c_order{},shape_type{5,1},shape_type{1,1},shape_type{1,0},shape_type{4,0},index_type{5},dim_type{2}),
        std::make_tuple(c_order{},shape_type{5,4,3},shape_type{12,3,1},shape_type{12,3,1},shape_type{48,9,2},index_type{60},dim_type{3}),
        std::make_tuple(c_order{},shape_type{2,2,0,2},shape_type{4,2,2,1},shape_type{4,2,2,1},shape_type{4,2,0,1},index_type{0},dim_type{4}),
        std::make_tuple(c_order{},shape_type{4,3,2,0},shape_type{6,2,1,1},shape_type{6,2,1,1},shape_type{18,4,1,0},index_type{0},dim_type{4}),
        std::make_tuple(c_order{},shape_type{0,3,2,1},shape_type{6,2,1,1},shape_type{6,2,1,0},shape_type{0,4,1,0},index_type{0},dim_type{4}),
        //f_order
        std::make_tuple(f_order{},shape_type{},shape_type{},shape_type{},shape_type{},index_type{1},dim_type{0}),
        std::make_tuple(f_order{},shape_type{0},shape_type{1},shape_type{1},shape_type{0},index_type{0},dim_type{1}),
        std::make_tuple(f_order{},shape_type{1},shape_type{1},shape_type{0},shape_type{0},index_type{1},dim_type{1}),
        std::make_tuple(f_order{},shape_type{5},shape_type{1},shape_type{1},shape_type{4},index_type{5},dim_type{1}),
        std::make_tuple(f_order{},shape_type{0,0},shape_type{1,1},shape_type{1,1},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{1,0},shape_type{1,1},shape_type{0,1},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{0,1},shape_type{1,1},shape_type{1,0},shape_type{0,0},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{5,0},shape_type{1,5},shape_type{1,5},shape_type{4,0},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{0,5},shape_type{1,1},shape_type{1,1},shape_type{0,4},index_type{0},dim_type{2}),
        std::make_tuple(f_order{},shape_type{1,1},shape_type{1,1},shape_type{0,0},shape_type{0,0},index_type{1},dim_type{2}),
        std::make_tuple(f_order{},shape_type{1,5},shape_type{1,1},shape_type{0,1},shape_type{0,4},index_type{5},dim_type{2}),
        std::make_tuple(f_order{},shape_type{5,1},shape_type{1,5},shape_type{1,0},shape_type{4,0},index_type{5},dim_type{2}),
        std::make_tuple(f_order{},shape_type{5,4,3},shape_type{1,5,20},shape_type{1,5,20},shape_type{4,15,40},index_type{60},dim_type{3}),
        std::make_tuple(f_order{},shape_type{2,2,0,2},shape_type{1,2,4,4},shape_type{1,2,4,4},shape_type{1,2,0,4},index_type{0},dim_type{4}),
        std::make_tuple(f_order{},shape_type{4,3,2,0},shape_type{1,4,12,24},shape_type{1,4,12,24},shape_type{3,8,12,0},index_type{0},dim_type{4}),
        std::make_tuple(f_order{},shape_type{0,3,2,1},shape_type{1,1,3,6},shape_type{1,1,3,0},shape_type{0,2,3,0},index_type{0},dim_type{4})
    );
    auto test = [](const auto& t){
        auto order = std::get<0>(t);
        using order_type = decltype(order);
        auto shape = std::get<1>(t);
        auto expected_shape = shape;
        auto expected_strides = std::get<2>(t);
        auto expected_strides_div = make_dividers<config_type>(expected_strides);
        auto expected_strides_div_c_order = make_strides_div<config_type>(shape, c_order{});
        auto expected_strides_div_f_order = make_strides_div<config_type>(shape, f_order{});
        auto expected_cstrides = expected_strides;
        auto expected_adapted_strides = std::get<3>(t);
        auto expected_reset_strides = std::get<4>(t);
        auto expected_size = std::get<5>(t);
        auto expected_dim = std::get<6>(t);
        auto expected_offset = index_type{0};
        using descriptor_type = basic_descriptor<config_type, order_type>;
        auto descriptor = descriptor_type{shape};

        auto result_shape = descriptor.shape();
        auto result_strides = descriptor.strides();
        auto result_strides_div = descriptor.strides_div();
        auto result_strides_div_c_order = descriptor.strides_div(c_order{});
        auto result_strides_div_f_order = descriptor.strides_div(f_order{});
        auto result_adapted_strides = descriptor.adapted_strides();
        auto result_reset_strides = descriptor.reset_strides();
        auto result_cstrides = descriptor.cstrides();
        auto result_size = descriptor.size();
        auto result_dim = descriptor.dim();
        auto result_offset = descriptor.offset();

        REQUIRE(result_shape == expected_shape);
        REQUIRE(result_strides == expected_strides);
        REQUIRE(result_strides_div == expected_strides_div);
        REQUIRE(result_strides_div_c_order == expected_strides_div_c_order);
        REQUIRE(result_strides_div_f_order == expected_strides_div_f_order);
        REQUIRE(result_adapted_strides == expected_adapted_strides);
        REQUIRE(result_reset_strides == expected_reset_strides);
        REQUIRE(result_cstrides == expected_cstrides);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_offset == expected_offset);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_basic_descriptor_convert", "[test_descriptor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using descriptor_type = gtensor::basic_descriptor<config_type, order>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<shape_type, index_type, index_type>;
    //0descriptor,1idx,2expected
    auto test_data = GENERATE(
        test_type{shape_type{15},0,0},
        test_type{shape_type{1,15},7,7},
        test_type{shape_type{3,3,5},22,22}
    );
    auto shape = std::get<0>(test_data);
    auto idx = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto descriptor = descriptor_type{shape};
    SECTION("test_call_operator")
    {
        auto result = descriptor(idx);
        REQUIRE(result == expected);
    }
    SECTION("test_convert")
    {
        auto result = descriptor.convert(idx);
        REQUIRE(result == expected);
    }
}

TEST_CASE("test_basic_descriptor_changing_order_convert", "[test_descriptor]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    //using descriptor_type = gtensor::basic_descriptor<config_type, order>;
    using shape_type = typename config_type::shape_type;
    using gtensor::basic_descriptor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;
    //0order,1shape,2idx,3expected
    auto test_data = std::make_tuple(
        //convert from f_order to c_order
        std::make_tuple(c_order{},shape_type{15},0,0),
        std::make_tuple(c_order{},shape_type{15},3,3),
        std::make_tuple(c_order{},shape_type{15},14,14),
        std::make_tuple(c_order{},shape_type{1,15},1,1),
        std::make_tuple(c_order{},shape_type{1,15},7,7),
        std::make_tuple(c_order{},shape_type{15,1},1,1),
        std::make_tuple(c_order{},shape_type{15,1},7,7),
        std::make_tuple(c_order{},shape_type{3,3,5},0,0),
        std::make_tuple(c_order{},shape_type{3,3,5},1,15),
        std::make_tuple(c_order{},shape_type{3,3,5},2,30),
        std::make_tuple(c_order{},shape_type{3,3,5},3,5),
        std::make_tuple(c_order{},shape_type{3,3,5},11,31),
        std::make_tuple(c_order{},shape_type{3,3,5},44,44),
        //convert from f_order to c_order
        std::make_tuple(f_order{},shape_type{15},0,0),
        std::make_tuple(f_order{},shape_type{15},3,3),
        std::make_tuple(f_order{},shape_type{15},14,14),
        std::make_tuple(f_order{},shape_type{1,15},1,1),
        std::make_tuple(f_order{},shape_type{1,15},7,7),
        std::make_tuple(f_order{},shape_type{15,1},1,1),
        std::make_tuple(f_order{},shape_type{15,1},7,7),
        std::make_tuple(f_order{},shape_type{3,3,5},0,0),
        std::make_tuple(f_order{},shape_type{3,3,5},1,9),
        std::make_tuple(f_order{},shape_type{3,3,5},2,18),
        std::make_tuple(f_order{},shape_type{3,3,5},11,15),
        std::make_tuple(f_order{},shape_type{3,3,5},23,31),
        std::make_tuple(c_order{},shape_type{3,3,5},44,44)
    );
    auto make_descriptor = [](const auto& t){
        auto order = std::get<0>(t);
        auto shape = std::get<1>(t);
        using order_type = decltype(order);
        using descriptor_type = basic_descriptor<config_type, order_type>;
        return descriptor_type{shape};
    };
    SECTION("test_call_operator")
    {
        auto test = [&make_descriptor](const auto& t){
            auto idx = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto descriptor = make_descriptor(t);
            auto result = descriptor(idx,0);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_convert")
    {
        auto test = [&make_descriptor](const auto& t){
            auto idx = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto descriptor = make_descriptor(t);
            auto result = descriptor.convert(idx,0);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_descriptor_with_offset_c_layout","[test_descriptor]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using gtensor::descriptor_with_offset;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;
    //0order,1shape,2offset,3expected_strides,4expected_size,5expected_dim
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(c_order{}, shape_type{},index_type{0},shape_type{},index_type{1},dim_type{0}),
        std::make_tuple(c_order{}, shape_type{1},index_type{0},shape_type{1},index_type{1},dim_type{1}),
        std::make_tuple(c_order{}, shape_type{5},index_type{10},shape_type{1},index_type{5},dim_type{1}),
        std::make_tuple(c_order{}, shape_type{1,1},index_type{0},shape_type{1,1},index_type{1},dim_type{2}),
        std::make_tuple(c_order{}, shape_type{1,5},index_type{0},shape_type{5,1},index_type{5},dim_type{2}),
        std::make_tuple(c_order{}, shape_type{5,1},index_type{4},shape_type{1,1},index_type{5},dim_type{2}),
        std::make_tuple(c_order{}, shape_type{5,4,3},index_type{100},shape_type{12,3,1},index_type{60},dim_type{3}),
        //f_order
        std::make_tuple(f_order{}, shape_type{},index_type{0},shape_type{},index_type{1},dim_type{0}),
        std::make_tuple(f_order{}, shape_type{1},index_type{0},shape_type{1},index_type{1},dim_type{1}),
        std::make_tuple(f_order{}, shape_type{5},index_type{10},shape_type{1},index_type{5},dim_type{1}),
        std::make_tuple(f_order{}, shape_type{1,1},index_type{0},shape_type{1,1},index_type{1},dim_type{2}),
        std::make_tuple(f_order{}, shape_type{1,5},index_type{0},shape_type{1,1},index_type{5},dim_type{2}),
        std::make_tuple(f_order{}, shape_type{5,1},index_type{4},shape_type{1,5},index_type{5},dim_type{2}),
        std::make_tuple(f_order{}, shape_type{5,4,3},index_type{100},shape_type{1,5,20},index_type{60},dim_type{3})
    );
    auto test = [](const auto& t){
        auto order = std::get<0>(t);
        auto shape = std::get<1>(t);
        auto expected_shape = shape;
        auto offset = std::get<2>(t);
        auto expected_offset = offset;
        auto expected_strides = std::get<3>(t);
        auto expected_size = std::get<4>(t);
        auto expected_dim = std::get<5>(t);
        using order_type = decltype(order);
        using descriptor_type = descriptor_with_offset<config_type,order_type>;
        auto descriptor = descriptor_type{shape, offset};

        auto result_shape = descriptor.shape();
        auto result_strides = descriptor.strides();
        auto result_size = descriptor.size();
        auto result_dim = descriptor.dim();
        auto result_offset = descriptor.offset();

        REQUIRE(result_shape == expected_shape);
        REQUIRE(result_strides == expected_strides);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_offset == expected_offset);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_descriptor_with_offset_convert", "[test_descriptor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using descriptor_type = gtensor::descriptor_with_offset<config_type, order>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<shape_type, index_type, index_type, index_type>;
    //0shape,1offset,2idx,3expected
    auto test_data = GENERATE(
        test_type{shape_type{15},0,0,0},
        test_type{shape_type{1,15},0,7,7},
        test_type{shape_type{15,1},2,7,9},
        test_type{shape_type{3,3,5},5,22,27}
    );
    auto shape = std::get<0>(test_data);
    auto offset = std::get<1>(test_data);
    auto idx = std::get<2>(test_data);
    auto expected = std::get<3>(test_data);
    auto descriptor = descriptor_type{shape,offset};
    SECTION("test_call_operator")
    {
        auto result = descriptor(idx);
        REQUIRE(result == expected);
    }
    SECTION("test_convert")
    {
        auto result = descriptor.convert(idx);
        REQUIRE(result == expected);
    }
}

TEST_CASE("test_converting_descriptor", "[test_descriptor]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using gtensor::converting_descriptor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_dividers;
    using helpers_for_testing::apply_by_element;
    //0order,1shape,2cstrides,3offset,4expected_strides,5expected_dim,6expected_size
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(c_order{},shape_type{15},shape_type{-1},index_type{14},shape_type{1},dim_type{1},index_type{15}),
        std::make_tuple(c_order{},shape_type{3,1,7},shape_type{7,7,1},index_type{0},shape_type{7,7,1},dim_type{3},index_type{21}),
        std::make_tuple(c_order{},shape_type{3,1,7},shape_type{7,7,-1},index_type{6},shape_type{7,7,1},dim_type{3},index_type{21}),
        //forder
        std::make_tuple(f_order{},shape_type{15},shape_type{-1},index_type{14},shape_type{1},dim_type{1},index_type{15}),
        std::make_tuple(f_order{},shape_type{3,1,7},shape_type{7,7,1},index_type{0},shape_type{1,3,3},dim_type{3},index_type{21}),
        std::make_tuple(f_order{},shape_type{3,1,7},shape_type{7,7,-1},index_type{6},shape_type{1,3,3},dim_type{3},index_type{21})
    );
    auto test = [](const auto& t){
        auto order = std::get<0>(t);
        auto shape = std::get<1>(t);
        auto expected_shape = shape;
        auto cstrides = std::get<2>(t);
        auto expected_cstrides = cstrides;
        auto offset = std::get<3>(t);
        auto expected_offset = offset;
        auto expected_strides = std::get<4>(t);
        auto expected_dim = std::get<5>(t);
        auto expected_size = std::get<6>(t);
        using order_type = decltype(order);
        using descriptor_type = converting_descriptor<config_type, order_type>;
        auto descriptor = descriptor_type{shape,cstrides,offset};

        auto result_shape = descriptor.shape();
        auto result_strides = descriptor.strides();
        auto result_cstrides = descriptor.cstrides();
        auto result_dim = descriptor.dim();
        auto result_size = descriptor.size();
        auto result_offset = descriptor.offset();

        REQUIRE(result_shape == expected_shape);
        REQUIRE(result_strides == expected_strides);
        REQUIRE(result_cstrides == expected_cstrides);
        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_offset == expected_offset);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_converting_descriptor_convert", "[test_descriptor]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::converting_descriptor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;
    //0order,1shape,2cstrides,3offset,4idx,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(c_order{},shape_type{15},shape_type{1},index_type{0},index_type{0},index_type{0}),
        std::make_tuple(c_order{},shape_type{15},shape_type{1},index_type{0},index_type{7},index_type{7}),
        std::make_tuple(c_order{},shape_type{3,3,5},shape_type{15,5,1},index_type{5},index_type{22},index_type{27}),
        std::make_tuple(c_order{},shape_type{3,2,5},shape_type{-20,10,1},index_type{40},index_type{22},index_type{2}),
        //f_order
        std::make_tuple(f_order{},shape_type{15},shape_type{1},index_type{0},index_type{0},index_type{0}),
        std::make_tuple(f_order{},shape_type{15},shape_type{1},index_type{3},index_type{0},index_type{3}),
        std::make_tuple(f_order{},shape_type{15},shape_type{1},index_type{0},index_type{7},index_type{7}),
        std::make_tuple(f_order{},shape_type{15},shape_type{1},index_type{3},index_type{7},index_type{10}),
        std::make_tuple(f_order{},shape_type{4,3,2},shape_type{6,2,1},index_type{0},index_type{22},index_type{17}),
        std::make_tuple(f_order{},shape_type{2,2,2},shape_type{1,2,12},index_type{2},index_type{5},index_type{15})
    );
    auto make_descriptor = [](const auto& t){
        auto order = std::get<0>(t);
        auto shape = std::get<1>(t);
        auto cstrides = std::get<2>(t);
        auto offset = std::get<3>(t);
        using order_type = decltype(order);
        using descriptor_type = converting_descriptor<config_type, order_type>;
        return descriptor_type{shape,cstrides,offset};
    };
    SECTION("test_call_operator")
    {
        auto test = [&make_descriptor](const auto& t){
            auto idx = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto descriptor = make_descriptor(t);
            auto result = descriptor(idx);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_convert")
    {
        auto test = [&make_descriptor](const auto& t){
            auto idx = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto descriptor = make_descriptor(t);
            auto result = descriptor.convert(idx);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

namespace test_mapping_descriptor_{

struct test_config : gtensor::config::default_config{
    template<typename T> using storage = std::vector<T>;
    template<typename T> using index_map = std::vector<T>;
};

}

TEST_CASE("test_mapping_descriptor","[test_descriptor]"){
    using config_type = gtensor::config::extend_config_t<test_mapping_descriptor_::test_config,int>;
    using index_map_type = typename config_type::index_map_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using gtensor::mapping_descriptor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_dividers;
    using helpers_for_testing::apply_by_element;
    //0order,1shape,2index_map,3expected_strides,4expected_size,5expected_dim
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(c_order{},shape_type{},index_map_type{},shape_type{},index_type{1},dim_type{0}),
        std::make_tuple(c_order{},shape_type{1},index_map_type{0},shape_type{1},index_type{1},dim_type{1}),
        std::make_tuple(c_order{},shape_type{5},index_map_type{0,1,2,3,4,5},shape_type{1},index_type{5},dim_type{1}),
        std::make_tuple(c_order{},shape_type{1,1},index_map_type{0},shape_type{1,1},index_type{1},dim_type{2}),
        std::make_tuple(c_order{},shape_type{1,5},index_map_type{0,1,2,3,4,5},shape_type{5,1},index_type{5},dim_type{2}),
        std::make_tuple(c_order{},shape_type{5,1},index_map_type{0,1,2,3,4,5},shape_type{1,1},index_type{5},dim_type{2}),
        std::make_tuple(c_order{},shape_type{2,2,2},index_map_type{0,1,2,3,4,5,6,7},shape_type{4,2,1},index_type{8},dim_type{3}),
        //f_order
        std::make_tuple(f_order{},shape_type{},index_map_type{},shape_type{},index_type{1},dim_type{0}),
        std::make_tuple(f_order{},shape_type{1},index_map_type{0},shape_type{1},index_type{1},dim_type{1}),
        std::make_tuple(f_order{},shape_type{5},index_map_type{0,1,2,3,4,5},shape_type{1},index_type{5},dim_type{1}),
        std::make_tuple(f_order{},shape_type{1,1},index_map_type{0},shape_type{1,1},index_type{1},dim_type{2}),
        std::make_tuple(f_order{},shape_type{1,5},index_map_type{0,1,2,3,4,5},shape_type{1,1},index_type{5},dim_type{2}),
        std::make_tuple(f_order{},shape_type{5,1},index_map_type{0,1,2,3,4,5},shape_type{1,5},index_type{5},dim_type{2}),
        std::make_tuple(f_order{},shape_type{2,2,2},index_map_type{0,1,2,3,4,5,6,7},shape_type{1,2,4},index_type{8},dim_type{3})
    );
    auto test = [](const auto& t){
        auto order = std::get<0>(t);
        auto shape = std::get<1>(t);
        auto expected_shape = shape;
        auto index_map = std::get<2>(t);
        auto expected_strides = std::get<3>(t);
        auto expected_size = std::get<4>(t);
        auto expected_dim = std::get<5>(t);
        auto expected_offset = index_type{0};
        using order_type = decltype(order);
        using descriptor_type = mapping_descriptor<config_type, order_type>;
        auto descriptor = descriptor_type{shape, index_map};

        auto result_shape = descriptor.shape();
        auto result_strides = descriptor.strides();
        auto result_size = descriptor.size();
        auto result_dim = descriptor.dim();
        auto result_offset = descriptor.offset();

        REQUIRE(result_shape == expected_shape);
        REQUIRE(result_strides == expected_strides);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_offset == expected_offset);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_mapping_descriptor_convert","[test_descriptor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using config_type = gtensor::config::extend_config_t<test_mapping_descriptor_::test_config,int>;
    using descriptor_type = gtensor::mapping_descriptor<config_type, order>;
    using index_map_type = typename config_type::index_map_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<shape_type, index_map_type, index_type, index_type>;
    //0shape,1index_map,2idx,3expected
    auto test_data = GENERATE(
        test_type(shape_type{1},index_map_type{0},0,0),
        test_type(shape_type{5},index_map_type{1,2,0,3,4},0,1),
        test_type(shape_type{5},index_map_type{1,2,0,3,4},2,0),
        test_type(shape_type{2,2,2},index_map_type{11,12,13,17,18,19,1,2},0,11),
        test_type(shape_type{2,2,2},index_map_type{11,12,13,17,18,19,1,2},7,2)
    );
    auto shape = std::get<0>(test_data);
    auto index_map = std::get<1>(test_data);
    auto idx = std::get<2>(test_data);
    auto expected = std::get<3>(test_data);
    auto descriptor = descriptor_type{shape, index_map};

    auto result = descriptor.convert(idx);
    REQUIRE(result == expected);
}

