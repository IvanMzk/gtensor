#include "catch.hpp"
#include "gtensor.hpp"
#include "combine.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_check_stack_shapes","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::check_stack_shapes;
    using helpers_for_testing::apply_by_element;
    //0shapes
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{}),
        std::make_tuple(shape_type{1}),
        std::make_tuple(shape_type{1,2,3}),
        std::make_tuple(shape_type{}, shape_type{}),
        std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}),
        std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})
    );
    auto test = [](const auto& t){
        auto shapes = t;
        auto apply_shapes = [](const auto&...shapes_){
            check_stack_shapes(shapes_...);
        };
        REQUIRE_NOTHROW(std::apply(apply_shapes, shapes));
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_check_stack_shapes_exception","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using shape_type = typename config_type::shape_type;
    using gtensor::combine_exception;
    using gtensor::detail::check_stack_shapes;
    using helpers_for_testing::apply_by_element;
    //0shapes
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{}, shape_type{1}),
        std::make_tuple(shape_type{1}, shape_type{}),
        std::make_tuple(shape_type{1,2,3}, shape_type{}),
        std::make_tuple(shape_type{}, shape_type{1,2,3}),
        std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,1,2,3}),
        std::make_tuple(shape_type{1,2,3}, shape_type{2,2,3}, shape_type{1,2,3})
    );
    auto test = [](const auto& t){
        auto shapes = t;
        auto apply_shapes = [](const auto&...shapes_){
            check_stack_shapes(shapes_...);
        };
        REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), combine_exception);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_make_stack_shape","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_stack_shape;
    using test_type = std::tuple<shape_type,size_type,index_type,shape_type>;
    //0shape,1direction,2tensors_number,3expected
    auto test_data = GENERATE(
        test_type{shape_type{5},size_type{0},index_type{1},shape_type{1,5}},
        test_type{shape_type{5},size_type{1},index_type{1},shape_type{5,1}},
        test_type{shape_type{3,4},size_type{0},index_type{7},shape_type{7,3,4}},
        test_type{shape_type{3,4},size_type{1},index_type{7},shape_type{3,7,4}},
        test_type{shape_type{3,4},size_type{2},index_type{7},shape_type{3,4,7}},
        test_type{shape_type{3,4,5},size_type{0},index_type{7},shape_type{7,3,4,5}},
        test_type{shape_type{3,4,5},size_type{1},index_type{7},shape_type{3,7,4,5}},
        test_type{shape_type{3,4,5},size_type{2},index_type{7},shape_type{3,4,7,5}},
        test_type{shape_type{3,4,5},size_type{3},index_type{7},shape_type{3,4,5,7}}
    );

    auto shape = std::get<0>(test_data);
    auto direction = std::get<1>(test_data);
    auto tensors_number = std::get<2>(test_data);
    auto expected = std::get<3>(test_data);
    auto result = make_stack_shape(shape,direction,tensors_number);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_stack","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0direction,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1},{2},{3}}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1,2,3}}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{1,2,3,4},tensor_type{5,6,7,8},tensor_type{9,10,11,12}), tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_type{1,2,3,4},tensor_type{5,6,7,8},tensor_type{9,10,11,12}), tensor_type{{1,5,9},{2,6,10},{3,7,11},{4,8,12}}),
        std::make_tuple(
            size_type{0},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}
        ),
        std::make_tuple(
            size_type{1},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{7,8,9},{13,14,15},{19,20,21}},{{4,5,6},{10,11,12},{16,17,18},{22,23,24}}}
        ),
        std::make_tuple(
            size_type{2},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,7,13,19},{2,8,14,20},{3,9,15,21}},{{4,10,16,22},{5,11,17,23},{6,12,18,24}}}
        )
    );
    auto test = [](const auto& t){
        auto direction = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&direction](const auto&...tensors_){
            return stack(direction, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_stack_common_type","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using tensor_int32_type = gtensor::tensor<int, config_type>;
    using tensor_int64_type = gtensor::tensor<std::int64_t, config_type>;
    using tensor_double_type = gtensor::tensor<double, config_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0direction,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{0}, std::make_tuple(tensor_int32_type{1},tensor_int32_type{2},tensor_int64_type{3}), tensor_int64_type{{1},{2},{3}}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_int32_type{1},tensor_double_type{2},tensor_int64_type{3}), tensor_double_type{{1,2,3}})
    );
    auto test = [](const auto& t){
        auto direction = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&direction](const auto&...tensors_){
            return stack(direction, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test, test_data);
}