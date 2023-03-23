#include "catch.hpp"
#include "gtensor.hpp"
#include "combine.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_check_stack_args","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::check_stack_args;
    using helpers_for_testing::apply_by_element;
    //0direction,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{3}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{}, shape_type{})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{}, shape_type{})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}))
    );
    auto test = [](const auto& t){
        auto direction = std::get<0>(t);
        auto shapes = std::get<1>(t);
        auto apply_shapes = [&direction](const auto&...shapes_){
            check_stack_args(direction, shapes_...);
        };
        REQUIRE_NOTHROW(std::apply(apply_shapes, shapes));
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_check_concatenate_args","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::check_concatenate_variadic_args;
    using helpers_for_testing::apply_by_element;
    //0direction,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{}, shape_type{})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{}, shape_type{})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{2,2}, shape_type{1,2})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{2,2}, shape_type{2,1})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{5,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,20,3}, shape_type{1,10,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{1,2,3}, shape_type{5,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,22,3}, shape_type{1,12,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,13}, shape_type{1,2,33}))

    );
    auto test = [](const auto& t){
        auto direction = std::get<0>(t);
        auto shapes = std::get<1>(t);
        auto apply_shapes = [&direction](const auto&...shapes_){
            check_concatenate_variadic_args(direction, shapes_...);
        };
        REQUIRE_NOTHROW(std::apply(apply_shapes, shapes));
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_check_vstack_args","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::check_vstack_variadic_args;
    using gtensor::detail::check_vstack_container_args;
    using helpers_for_testing::apply_by_element;
    //0shapes
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(shape_type{})),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{})),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{}, shape_type{})),
        std::make_tuple(std::make_tuple(shape_type{1})),
        std::make_tuple(std::make_tuple(shape_type{5}, shape_type{5}, shape_type{5})),
        std::make_tuple(std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(std::make_tuple(shape_type{2,3}, shape_type{3})),
        std::make_tuple(std::make_tuple(shape_type{2,3}, shape_type{3}, shape_type{3}, shape_type{1,3})),
        std::make_tuple(std::make_tuple(shape_type{3}, shape_type{2,3}, shape_type{3}, shape_type{1,3})),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(std::make_tuple(shape_type{10,2,3}, shape_type{5,2,3})),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(std::make_tuple(shape_type{10,2,3}, shape_type{1,2,3}, shape_type{5,2,3}))
    );

    SECTION("test_check_vstack_variadic_args"){
        auto test = [](const auto& t){
            auto shapes = std::get<0>(t);
            auto apply_shapes = [](const auto&...shapes_){
                check_vstack_variadic_args(size_type{0}, shapes_...);
            };
            REQUIRE_NOTHROW(std::apply(apply_shapes, shapes));
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_vstack_container_args"){
        using value_type = double;
        using tensor_type = gtensor::tensor<value_type, config_type>;
        using container_type = std::vector<tensor_type>;
        auto test = [](const auto& t){
            auto shapes = std::get<0>(t);
            auto apply_shapes = [](const auto&...shapes_){
                return container_type{tensor_type(shapes_, value_type{})...};
            };
            auto container = std::apply(apply_shapes, shapes);
            REQUIRE_NOTHROW(check_vstack_container_args(size_type{0}, container));
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_check_stack_args_exception","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::combine_exception;
    using gtensor::detail::check_stack_args;
    using helpers_for_testing::apply_by_element;
    //0direction,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1})),
        std::make_tuple(size_type{4}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1}, shape_type{1})),
        std::make_tuple(size_type{4}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{}, shape_type{1})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1}, shape_type{})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{2,2,3}, shape_type{1,2,3}))
    );
    auto test = [](const auto& t){
        auto direction = std::get<0>(t);
        auto shapes = std::get<1>(t);
        auto apply_shapes = [&direction](const auto&...shapes_){
            check_stack_args(direction, shapes_...);
        };
        REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), combine_exception);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_check_concatenate_args_exception","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::combine_exception;
    using gtensor::detail::check_concatenate_variadic_args;
    using helpers_for_testing::apply_by_element;
    //0direction,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1})),
        std::make_tuple(size_type{3}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1}, shape_type{1})),
        std::make_tuple(size_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{}, shape_type{1})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1}, shape_type{})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,20,3}, shape_type{1,20,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{10,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{10,2,3}, shape_type{1,2,3})),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,12,3}, shape_type{1,2,3}))

    );
    auto test = [](const auto& t){
        auto direction = std::get<0>(t);
        auto shapes = std::get<1>(t);
        auto apply_shapes = [&direction](const auto&...shapes_){
            check_concatenate_variadic_args(direction, shapes_...);
        };
        REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), combine_exception);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_check_vstack_args_exception","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::combine_exception;
    using gtensor::detail::check_vstack_variadic_args;
    using gtensor::detail::check_vstack_container_args;
    using helpers_for_testing::apply_by_element;
    //0shapes
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{1})),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{})),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{2})),
        std::make_tuple(std::make_tuple(shape_type{2,3}, shape_type{4})),
        std::make_tuple(std::make_tuple(shape_type{2,3}, shape_type{3}, shape_type{3}, shape_type{1,2})),
        std::make_tuple(std::make_tuple(shape_type{1,1,3}, shape_type{3})),
        std::make_tuple(std::make_tuple(shape_type{1,1,3}, shape_type{1,3})),
        std::make_tuple(std::make_tuple(shape_type{2,3,4}, shape_type{3,4})),
        std::make_tuple(std::make_tuple(shape_type{1,3,3}, shape_type{1,2,3})),
        std::make_tuple(std::make_tuple(shape_type{10,2,4}, shape_type{5,2,3})),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}, shape_type{1,3,3}, shape_type{1,2,3}))
    );

    SECTION("test_check_vstack_variadic_args_exception"){
        auto test = [](const auto& t){
            auto shapes = std::get<0>(t);
            auto apply_shapes = [](const auto&...shapes_){
                check_vstack_variadic_args(size_type{0}, shapes_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), combine_exception);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_vstack_container_args_exception"){
        using value_type = double;
        using tensor_type = gtensor::tensor<value_type, config_type>;
        using container_type = std::vector<tensor_type>;
        auto test = [](const auto& t){
            auto shapes = std::get<0>(t);
            auto apply_shapes = [](const auto&...shapes_){
                return container_type{tensor_type(shapes_, value_type{})...};
            };
            auto container = std::apply(apply_shapes, shapes);
            REQUIRE_THROWS_AS(check_vstack_container_args(size_type{0}, container), combine_exception);
        };
        apply_by_element(test, test_data);
    }
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
    auto result = make_stack_shape(direction,shape,tensors_number);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_make_concatenate_shape","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::make_concatenate_shape;
    using helpers_for_testing::apply_by_element;
    //0direction,1shapes,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{}), shape_type{}),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{}), shape_type{}),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{}, shape_type{}), shape_type{}),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{}, shape_type{}), shape_type{}),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{2,2,3}),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{5,2,3}), shape_type{15,2,3}),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,4,3}),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,20,3}, shape_type{1,10,3}), shape_type{1,30,3}),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,2,6}),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,30}, shape_type{1,2,3}), shape_type{1,2,33}),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{3,2,3}),
        std::make_tuple(size_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{1,2,3}, shape_type{5,2,3}), shape_type{16,2,3}),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,6,3}),
        std::make_tuple(size_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,22,3}, shape_type{1,12,3}), shape_type{1,36,3}),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,2,9}),
        std::make_tuple(size_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,13}, shape_type{1,2,33}), shape_type{1,2,49})
    );
    auto test = [](const auto& t){
        auto direction = std::get<0>(t);
        auto shapes = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_shapes = [&direction](const auto&...shapes_){
            return make_concatenate_shape(direction, shapes_...);
        };
        auto result = std::apply(apply_shapes, shapes);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_make_vstack_shape","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::make_vstack_shape;
    using helpers_for_testing::apply_by_element;
    //0shapes,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(shape_type{}), shape_type{}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{}), shape_type{}),
        std::make_tuple(std::make_tuple(shape_type{}, shape_type{}, shape_type{}), shape_type{}),
        std::make_tuple(std::make_tuple(shape_type{1}), shape_type{1,1}),
        std::make_tuple(std::make_tuple(shape_type{5}, shape_type{5}, shape_type{5}), shape_type{3,5}),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(std::make_tuple(shape_type{2,3}, shape_type{3}), shape_type{3,3}),
        std::make_tuple(std::make_tuple(shape_type{2,3}, shape_type{3}, shape_type{3}, shape_type{1,3}), shape_type{5,3}),
        std::make_tuple(std::make_tuple(shape_type{3}, shape_type{2,3}, shape_type{3}, shape_type{1,3}), shape_type{5,3}),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{2,2,3}),
        std::make_tuple(std::make_tuple(shape_type{10,2,3}, shape_type{5,2,3}), shape_type{15,2,3}),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{3,2,3}),
        std::make_tuple(std::make_tuple(shape_type{10,2,3}, shape_type{1,2,3}, shape_type{5,2,3}), shape_type{16,2,3})
    );
    auto test = [](const auto& t){
        auto shapes = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto apply_shapes = [](const auto&...shapes_){
            return make_vstack_shape(size_type{0}, shapes_...);
        };
        auto result = std::apply(apply_shapes, shapes);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
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

TEMPLATE_TEST_CASE("test_concatenate","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0direction,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{1}), tensor_type{1}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{1,2,3}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{1},tensor_type{2,3},tensor_type{4,5,6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{1,2,3,4},tensor_type{5},tensor_type{6,7}), tensor_type{1,2,3,4,5,6,7}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(
            size_type{0},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12},{13,14,15}},tensor_type{{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15},{16,17,18},{19,20,21},{22,23,24}}
        ),
        std::make_tuple(
            size_type{1},
            std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13},{14}}),
            tensor_type{{1,2,5,7,8,9,13},{3,4,6,10,11,12,14}}
        ),
        std::make_tuple(
            size_type{0},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10},{11,12}}},tensor_type{{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}
        ),
        std::make_tuple(
            size_type{1},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10}},{{11,12}}},tensor_type{{{13,14},{15,16},{17,18}},{{19,20},{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4},{9,10},{13,14},{15,16},{17,18}},{{5,6},{7,8},{11,12},{19,20},{21,22},{23,24}}}
        ),
        std::make_tuple(
            size_type{2},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9},{10}},{{11},{12}}},tensor_type{{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}),
            tensor_type{{{1,2,9,13,14,15},{3,4,10,16,17,18}},{{5,6,11,19,20,21},{7,8,12,22,23,24}}}
        ),
        std::make_tuple(size_type{2}, std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}), tensor_type{{{1,3}},{{2,4}}}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_type{{{1,2}},{{3,4}}}, tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}), tensor_type{{{1,2},{5,6},{7,8}},{{3,4},{9,10},{11,12}}}),
        std::make_tuple(size_type{0}, std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}, tensor_type{{{13,14},{15,16},{17,18}}}),
            tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );

    SECTION("test_concatenate_variadic")
    {
        auto test_concatenate_variadic = [](const auto& t){
            auto direction = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);

            auto apply_tensors = [&direction](const auto&...tensors_){
                return concatenate(direction, tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result.equals(expected));
        };
        apply_by_element(test_concatenate_variadic, test_data);
    }

    SECTION("test_concatenate_container")
    {
        using container_type = std::vector<tensor_type>;
        auto test_concatenate_container = [](const auto& t){
            auto direction = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts...};}, tensors);
            auto result = concatenate(direction, container);
            REQUIRE(result.equals(expected));
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEMPLATE_TEST_CASE("test_concatenate_common_type","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using tensor_int32_type = gtensor::tensor<int, config_type>;
    using tensor_int64_type = gtensor::tensor<std::int64_t, config_type>;
    using tensor_double_type = gtensor::tensor<double, config_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0direction,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(size_type{0}, std::make_tuple(tensor_int32_type{{1,2},{3,4}},tensor_int64_type{{5,6}}), tensor_int64_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(size_type{1}, std::make_tuple(tensor_int64_type{{1,2},{3,4}},tensor_double_type{{5},{6}}), tensor_double_type{{1,2,5},{3,4,6}})
    );
    auto test = [](const auto& t){
        auto direction = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&direction](const auto&...tensors_){
            return concatenate(direction, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_max_block_dim","test_combine"){
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using size_type = typename tensor_type::size_type;
    using gtensor::detail::max_block_dim;
    value_type v{};

    REQUIRE(max_block_dim(std::initializer_list<tensor_type>{tensor_type({5},v)}) == size_type{1});
    REQUIRE(max_block_dim(std::initializer_list<tensor_type>{tensor_type({5},v), tensor_type({5,4},v)}) == size_type{2});
    REQUIRE(max_block_dim(std::initializer_list<tensor_type>{tensor_type({5,4},v), tensor_type({5},v)}) == size_type{2});
    REQUIRE(max_block_dim(
        std::initializer_list<std::initializer_list<tensor_type>>{{tensor_type({5,4},v), tensor_type({5},v)}, {tensor_type({5,4},v), tensor_type({3,3,3},v), tensor_type({5},v)}}
            ) == size_type{3}
    );
}

TEST_CASE("test_make_block_shape","test_combine"){
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using size_type = typename tensor_type::size_type;
    using shape_type = typename tensor_type::shape_type;
    using gtensor::detail::nested_initialiser_list_depth;
    using gtensor::detail::nested_init_list1_type;
    using gtensor::detail::nested_init_list2_type;
    using gtensor::detail::nested_init_list3_type;
    using gtensor::combine_exception;
    using gtensor::detail::make_block_shape;
    using gtensor::detail::max_block_dim;
    value_type v{};

    auto make_block_shape_caller = [](auto blocks){
        size_type res_dim = std::max(nested_initialiser_list_depth<decltype(blocks)>::value, max_block_dim(blocks));
        return make_block_shape(blocks, res_dim);
    };

    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({2},v)}) == shape_type{2});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2},v)}}) == shape_type{1,2});
    REQUIRE(make_block_shape_caller(nested_init_list3_type<tensor_type>{{{tensor_type({2},v)}}}) == shape_type{1,1,2});
    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({3,4},v)}) == shape_type{3,4});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({3,4},v)}}) == shape_type{3,4});
    REQUIRE(make_block_shape_caller(nested_init_list3_type<tensor_type>{{{tensor_type({3,4},v)}}}) == shape_type{1,3,4});

    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({2},v), tensor_type({3},v), tensor_type({4},v)}) == shape_type{9});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2},v), tensor_type({3},v), tensor_type({4},v)}}) == shape_type{1,9});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2},v)}, {tensor_type({2},v)}, {tensor_type({2},v)}}) == shape_type{3,2});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({1},v), tensor_type({2},v)}, {tensor_type({3},v)}}) == shape_type{2,3});
    REQUIRE(make_block_shape_caller(nested_init_list3_type<tensor_type>{{{tensor_type({2},v), tensor_type({3},v), tensor_type({4},v)}}}) == shape_type{1,1,9});
    REQUIRE(make_block_shape_caller(nested_init_list3_type<tensor_type>{{{tensor_type({2},v)}}, {{tensor_type({2},v)}}, {{tensor_type({2},v)}}}) == shape_type{3,1,2});
    REQUIRE(make_block_shape_caller(nested_init_list3_type<tensor_type>{{{tensor_type({2},v)}, {tensor_type({2},v)}}, {{tensor_type({2},v)}, {tensor_type({2},v)}}}) == shape_type{2,2,2});

    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({2},v), tensor_type({1,3},v)}) == shape_type{1,5});
    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({1,3},v), tensor_type({2},v), tensor_type({1,4},v)}) == shape_type{1,9});
    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({2},v), tensor_type({1,1,3},v)}) == shape_type{1,1,5});
    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({1,3},v), tensor_type({2},v), tensor_type({1,1,4},v)}) == shape_type{1,1,9});
    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({2,2},v), tensor_type({2,3},v)}) == shape_type{2,5});
    REQUIRE(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({2,2},v), tensor_type({2,3},v), tensor_type({2,4},v)}) == shape_type{2,9});

    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2},v)}, {tensor_type({2,2},v)}}) == shape_type{3,2});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2,3},v), tensor_type({2,4},v)}}) == shape_type{2,7});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({3,2},v)}, {tensor_type({4,2},v)}}) == shape_type{7,2});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2,3},v), tensor_type({2,4},v)},{tensor_type({5,7},v)}}) == shape_type{7,7});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2,3},v), tensor_type({2,4},v)},{tensor_type({5,2},v), tensor_type({5,5},v)}}) == shape_type{7,7});
    REQUIRE(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({3,5},v)}, {tensor_type({4,5},v)}, {tensor_type({2,1},v), tensor_type({2,4},v)}}) == shape_type{9,5});
    REQUIRE(make_block_shape_caller(nested_init_list3_type<tensor_type>{
        {{tensor_type({2,2},v), tensor_type({2,3},v)}, {tensor_type({3,2},v), tensor_type({3,3},v)}},
        {{tensor_type({3,3,2},v), tensor_type({3,3,3},v)}, {tensor_type({3,2,5},v)}}}) == shape_type{4,5,5});

    REQUIRE_THROWS_AS(make_block_shape_caller(nested_init_list3_type<tensor_type>{{{tensor_type({2},v)}}, {{tensor_type({2},v)}, {tensor_type({2},v)}}}), combine_exception);
    REQUIRE_THROWS_AS(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2},v)}, {tensor_type({3},v)}}), combine_exception);
    REQUIRE_THROWS_AS(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({2},v), tensor_type({2,3},v)}), combine_exception);
    REQUIRE_THROWS_AS(make_block_shape_caller(nested_init_list1_type<tensor_type>{tensor_type({2,3},v), tensor_type({2},v)}), combine_exception);
    REQUIRE_THROWS_AS(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2,3},v), tensor_type({2},v)}}), combine_exception);
    REQUIRE_THROWS_AS(make_block_shape_caller(nested_init_list2_type<tensor_type>{{tensor_type({2},v), tensor_type({2,3},v)}}), combine_exception);
    REQUIRE_THROWS_AS(make_block_shape_caller(nested_init_list3_type<tensor_type>{{{tensor_type({2},v)}}, {{tensor_type({2,2},v)}}}), combine_exception);
}

TEMPLATE_TEST_CASE("test_block","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using gtensor::detail::nested_init_list1_type;
    using gtensor::detail::nested_init_list2_type;
    using gtensor::detail::nested_init_list3_type;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;

    //0result,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{}}), tensor_type{}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{}, tensor_type{}, tensor_type{}}), tensor_type{}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{}, tensor_type{1,2,3}, tensor_type{}, tensor_type{4,5}, tensor_type{}}), tensor_type{1,2,3,4,5}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{}, tensor_type{{1,2,3}}, tensor_type{}, tensor_type{{{4,5}}}, tensor_type{}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{}}}), tensor_type{}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{},tensor_type{}},{tensor_type{},tensor_type{}}}), tensor_type{}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{},tensor_type{1,2,3}},{tensor_type{4,5,6},tensor_type{}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list3_type<tensor_type>{{{tensor_type{},tensor_type{}},{tensor_type{}}}, {{tensor_type{}}}}), tensor_type{}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{1,2,3,4,5}}), tensor_type{1,2,3,4,5}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{{1,2,3},{4,5,6}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7,8},{9,10}}}), tensor_type{{1,2,3,7,8},{4,5,6,9,10}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{1,2,3,4,5}}}), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(block(nested_init_list3_type<tensor_type>{{{tensor_type{1,2,3,4,5}}}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{{1,2,3},{4,5,6}}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list3_type<tensor_type>{{{tensor_type{{1,2,3},{4,5,6}}}}}), tensor_type{{{1,2,3},{4,5,6}}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}}), tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{1,2}}, {tensor_type{3,4}}, {tensor_type{5,6}}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{1}, tensor_type{2,3}}, {tensor_type{4,5,6}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list3_type<tensor_type>{{{tensor_type{1,2}}},{{tensor_type{3,4}}},{{tensor_type{5,6}}}}), tensor_type{{{1,2}},{{3,4}},{{5,6}}}),
        std::make_tuple(block(nested_init_list3_type<tensor_type>{{{tensor_type{1,2}},{tensor_type{3,4}}},{{tensor_type{5,6}},{tensor_type{7,8}}}}), tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        std::make_tuple(block(nested_init_list3_type<tensor_type>{{{tensor_type{1,2}, tensor_type{3,4,5}, tensor_type{6,7,8,9}}}}), tensor_type{{{1,2,3,4,5,6,7,8,9}}}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{{{1}},{{2}},{{3}}}, tensor_type{{{4}},{{5}},{{6}}}}), tensor_type{{{1,4}},{{2,5}},{{3,6}}}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{1,2}, tensor_type{{3,4,5}}}), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{{3,4,5}}, tensor_type{1,2}}), tensor_type{{3,4,5,1,2}}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{1,2}, tensor_type{{{3,4,5}}}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list1_type<tensor_type>{tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}, tensor_type{{11},{12}}}), tensor_type{{1,2,5,6,7,11},{3,4,8,9,10,12}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{1,2}}, {tensor_type{{3,4},{5,6}}}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{{1,2,3},{3,4,5}}, tensor_type{{6,7,8,9},{10,11,12,13}}}}), tensor_type{{1,2,3,6,7,8,9},{3,4,5,10,11,12,13}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{{1,2},{3,4}}}, {tensor_type{{7,8},{9,10},{11,12}}}}), tensor_type{{1,2},{3,4},{7,8},{9,10},{11,12}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}}, {tensor_type{{11,12,13,14,15}}}}), tensor_type{{1,2,5,6,7},{3,4,8,9,10},{11,12,13,14,15}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{{1,2},{3,4}}, tensor_type{{5,6},{7,8}}}, {tensor_type{{9},{10}}, tensor_type{{11,12,13},{14,15,16}}}}), tensor_type{{1,2,5,6},{3,4,7,8},{9,11,12,13},{10,14,15,16}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{{tensor_type{{1,2},{3,4}}}, {tensor_type{{5,6},{7,8}}}, {tensor_type{{9},{10}}, tensor_type{{11},{12}}}}), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,11},{10,12}}),
        std::make_tuple(block(nested_init_list2_type<tensor_type>{
            {tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}},
            {tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}),
        std::make_tuple(block(nested_init_list3_type<tensor_type>{
            {{tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}}, {tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}}},
            {{tensor_type{13,14}}, {tensor_type{15,16}}, {tensor_type{17,18}}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result.equals(expected));
    };

    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_block_exception","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using gtensor::combine_exception;
    using gtensor::detail::nested_init_list1_type;
    using gtensor::detail::nested_init_list2_type;
    using gtensor::detail::nested_init_list3_type;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;

    REQUIRE_THROWS_AS(block(nested_init_list1_type<tensor_type>{tensor_type{1,2},tensor_type{{3,4},{5,6}}}), combine_exception);
    REQUIRE_THROWS_AS(block(nested_init_list1_type<tensor_type>{tensor_type{{1},{2},{3}},tensor_type{{3,4},{5,6}}}), combine_exception);
    REQUIRE_THROWS_AS(block(nested_init_list1_type<tensor_type>{tensor_type{{3,4},{5,6}}, tensor_type{}}), combine_exception);
    REQUIRE_THROWS_AS(block(nested_init_list1_type<tensor_type>{tensor_type{},tensor_type{{3,4},{5,6}}}), combine_exception);
    REQUIRE_THROWS_AS(block(nested_init_list1_type<tensor_type>{tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}},{{5}}} }), combine_exception);
    REQUIRE_THROWS_AS(block(nested_init_list2_type<tensor_type>{{tensor_type{},tensor_type{1,2,3}},{tensor_type{}}}), combine_exception);
    REQUIRE_THROWS_AS(block(nested_init_list2_type<tensor_type>{{tensor_type{1,2}},{tensor_type{}}}), combine_exception);
    REQUIRE_THROWS_AS(block(nested_init_list2_type<tensor_type>{{tensor_type{1,2}},{tensor_type{3,4,5}}}), combine_exception);
}

TEST_CASE("test_is_index_container","[test_combine]"){
    using gtensor::tensor;
    using config_type = gtensor::config::default_config;
    using index_type = typename config_type::index_type;
    using gtensor::detail::is_index_container_v;

    REQUIRE(is_index_container_v<std::vector<int>,int>);
    REQUIRE(is_index_container_v<std::vector<index_type>,index_type>);
    REQUIRE(is_index_container_v<std::array<index_type,3>,index_type>);
    REQUIRE(is_index_container_v<tensor<index_type>,index_type>);
    REQUIRE(!is_index_container_v<std::vector<std::string>,int>);
}

TEMPLATE_TEST_CASE("test_split_split_points","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using size_type = typename tensor_type::size_type;
    using index_type = typename tensor_type::index_type;
    using gtensor::split;
    using helpers_for_testing::apply_by_element;

    //0ten,1split_points,2direction,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}, std::vector<int>{}, size_type{0}, std::vector<tensor_type>{tensor_type{1}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{}, size_type{0}, std::vector<tensor_type>{tensor_type{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{2}, size_type{0}, std::vector<tensor_type>{tensor_type{1,2}, tensor_type{3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{2,4}, size_type{0}, std::vector<tensor_type>{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::initializer_list<int>{2,4}, size_type{0}, std::vector<tensor_type>{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, gtensor::tensor<int>{2,4}, size_type{0}, std::vector<tensor_type>{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            std::initializer_list<index_type>{1,2},
            size_type{0},
            std::vector<tensor_type>{tensor_type{{1,2,3,4}}, tensor_type{{5,6,7,8}}, tensor_type{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            std::initializer_list<int>{1,2},
            size_type{1},
            std::vector<tensor_type>{tensor_type{{1},{5},{9},{13}}, tensor_type{{2},{6},{10},{14}}, tensor_type{{3,4},{7,8},{11,12},{15,16}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            std::vector<int>{1,2},
            size_type{0},
            std::vector<tensor_type>{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            gtensor::tensor<index_type>{1},
            size_type{1},
            std::vector<tensor_type>{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            std::initializer_list<std::size_t>{1,2},
            size_type{2},
            std::vector<tensor_type>{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto split_points = std::get<1>(t);
        auto direction = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = split(ten, split_points, direction);
        REQUIRE(expected.size() == result.size());
        auto result_it = result.begin();
        for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
            REQUIRE((*result_it).equals(*expected_it));
        }
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_split_equal_parts","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using size_type = typename tensor_type::size_type;
    using index_type = typename tensor_type::index_type;
    using gtensor::split;
    using helpers_for_testing::apply_by_element;

    //0ten,1parts_number,2direction,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}, index_type{1}, size_type{0}, std::vector<tensor_type>{tensor_type{1}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, index_type{1}, size_type{0}, std::vector<tensor_type>{tensor_type{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, int{5}, size_type{0}, std::vector<tensor_type>{tensor_type{1}, tensor_type{2}, tensor_type{3}, tensor_type{4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, index_type{2}, size_type{0}, std::vector<tensor_type>{tensor_type{1,2,3}, tensor_type{4,5,6}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::size_t{3}, size_type{0}, std::vector<tensor_type>{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5,6}}),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            index_type{2},
            size_type{0},
            std::vector<tensor_type>{tensor_type{{1,2,3,4},{5,6,7,8}}, tensor_type{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            index_type{2},
            size_type{1},
            std::vector<tensor_type>{tensor_type{{1,2},{5,6},{9,10},{13,14}}, tensor_type{{3,4},{7,8},{11,12},{15,16}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            index_type{3},
            size_type{0},
            std::vector<tensor_type>{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{2},
            size_type{1},
            std::vector<tensor_type>{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{3},
            size_type{2},
            std::vector<tensor_type>{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto parts_number = std::get<1>(t);
        auto direction = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = split(ten, parts_number, direction);
        REQUIRE(expected.size() == result.size());
        auto result_it = result.begin();
        for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
            REQUIRE((*result_it).equals(*expected_it));
        }
    };
    apply_by_element(test, test_data);
}
TEMPLATE_TEST_CASE("test_split_exception","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using gtensor::combine_exception;
    using gtensor::split;

    REQUIRE_THROWS_AS(split(tensor_type{1},{},1), combine_exception);
    REQUIRE_THROWS_AS(split(tensor_type{1,2,3,4,5},{},1), combine_exception);
    REQUIRE_THROWS_AS(split(tensor_type{{1,2},{3,4},{5,6}},{},2), combine_exception);
    REQUIRE_THROWS_AS(split(tensor_type{1,2,3,4,5},0,0), combine_exception);
    REQUIRE_THROWS_AS(split(tensor_type{1,2,3,4,5},2,0), combine_exception);
    REQUIRE_THROWS_AS(split(tensor_type{1,2,3,4,5},1,1), combine_exception);
    REQUIRE_THROWS_AS(split(tensor_type{{1,2},{3,4},{5,6}},0,0), combine_exception);
    REQUIRE_THROWS_AS(split(tensor_type{{1,2},{3,4},{5,6}},4,0), combine_exception);
    REQUIRE_THROWS_AS(split(tensor_type{{1,2},{3,4},{5,6}},2,2), combine_exception);
}

TEMPLATE_TEST_CASE("test_vstack","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::vstack;
    //0tensors,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1},{2},{3}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3},tensor_type{4,5,6},tensor_type{7,8,9}), tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}), tensor_type{{1,2},{3,4}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{5,6}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}},{{5}}}, tensor_type{{{6}}}), tensor_type{{{1}},{{2}},{{3}},{{4}},{{5}},{{6}}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2}},{{3,4}}}, tensor_type{{{5,6}},{{7,8}}}, tensor_type{{{9,10}}}), tensor_type{{{1,2}},{{3,4}},{{5,6}},{{7,8}},{{9,10}}}),
        std::make_tuple(
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12},{13,14,15}},tensor_type{{16,17,18}},tensor_type{19,20,21}),
            tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15},{16,17,18},{19,20,21}}
        ),
        std::make_tuple(
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10},{11,12}}},tensor_type{{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}
        ),
        std::make_tuple(std::make_tuple(tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}, tensor_type{{{13,14},{15,16},{17,18}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    SECTION("test_vstack_variadic")
    {
        auto test_vstack_variadic = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return vstack(tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result.equals(expected));
        };
        apply_by_element(test_vstack_variadic, test_data);
    }
    SECTION("test_vstack_container")
    {
        using container_type = std::vector<tensor_type>;
        auto test_vstack_container = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts...};}, tensors);
            auto result = vstack(container);
            REQUIRE(result.equals(expected));
        };
        apply_by_element(test_vstack_container, test_data);
    }
}

TEMPLATE_TEST_CASE("test_vstack_exception","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::combine_exception;
    using gtensor::vstack;
    //0tensors
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{1}, tensor_type{1,1})),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{5,6,7})),
        std::make_tuple(std::make_tuple(tensor_type{{{1}}}, tensor_type{1})),
        std::make_tuple(std::make_tuple(tensor_type{{{1}}}, tensor_type{{1}})),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, tensor_type{{1,2},{3,4}}))
    );
    SECTION("test_vstack_variadic_exception")
    {
        auto test_vstack_variadic_exception = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return vstack(tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), combine_exception);
        };
        apply_by_element(test_vstack_variadic_exception, test_data);
    }
    SECTION("test_vstack_container_exception")
    {
        using container_type = std::vector<tensor_type>;
        auto test_vstack_container_exception = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts...};}, tensors);
            REQUIRE_THROWS_AS(vstack(container), combine_exception);
        };
        apply_by_element(test_vstack_container_exception, test_data);
    }
}
