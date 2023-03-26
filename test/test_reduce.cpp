#include "catch.hpp"
#include "gtensor.hpp"
#include "reduce.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEST_CASE("test_check_reduce_args","[test_reduce]")
{
    using config_type = gtensor::config::default_config;
    using size_type = config_type::size_type;
    using shape_type = config_type::shape_type;
    using gtensor::reduce_exception;
    using gtensor::detail::check_reduce_args;

    REQUIRE_NOTHROW(check_reduce_args(shape_type{1},size_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{10},size_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},size_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},size_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},size_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},size_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},size_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},size_type{2}));

    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},size_type{0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},size_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{1,0},size_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{1,0},size_type{2}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},size_type{3}), reduce_exception);
}

TEST_CASE("test_make_reduce_shape","[test_reduce]")
{
    using config_type = gtensor::config::default_config;
    using size_type = config_type::size_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_reduce_shape;
    //0pshape,1direction,2expected
    using test_type = std::tuple<shape_type,size_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{1},size_type{0},shape_type{1}},
        test_type{shape_type{10},size_type{0},shape_type{1}},
        test_type{shape_type{2,3,4},size_type{0},shape_type{3,4}},
        test_type{shape_type{2,3,4},size_type{1},shape_type{2,4}},
        test_type{shape_type{2,3,4},size_type{2},shape_type{2,3}}
    );
    auto pshape = std::get<0>(test_data);
    auto direction = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = make_reduce_shape(pshape,direction);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_reduce","[test_reduce]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using tensor_type = gtensor::tensor<value_type,config_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::reduce_operations::max;
    using gtensor::reduce_operations::min;
    //0tensor,1direction,2functor,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}.reshape(1,0), size_type{0}, std::plus{}, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), size_type{0}, std::plus{}, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), size_type{1}, std::plus{}, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, size_type{0}, std::plus{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, size_type{0}, std::plus{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, size_type{1}, std::plus{}, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, size_type{0}, std::plus{}, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, size_type{1}, std::plus{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, size_type{0}, std::plus{}, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, size_type{1}, std::plus{}, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, size_type{1}, std::multiplies{}, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, size_type{0}, max{}, tensor_type{4,6,7,9}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, size_type{1}, min{}, tensor_type{1,0}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, size_type{0}, std::plus{}, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, size_type{1}, std::plus{}, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, size_type{2}, std::plus{}, tensor_type{{1,5},{9,13}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto direction = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = tensor.reduce(direction, functor);
        REQUIRE(result.equals(expected));
        auto result1 = reduce(tensor, direction, functor);
        REQUIRE(result1.equals(expected));
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_reduce_ecxeption","[test_reduce]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using tensor_type = gtensor::tensor<value_type,config_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::reduce_exception;

    //0tensor,1direction,2functor
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, size_type{0}, std::plus{}),
        std::make_tuple(tensor_type{}.reshape(1,0), size_type{1}, std::plus{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, size_type{1}, std::plus{}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, size_type{2}, std::plus{}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, size_type{2}, std::plus{}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, size_type{4}, std::plus{}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, size_type{3}, std::plus{})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto direction = std::get<1>(t);
        auto functor = std::get<2>(t);
        REQUIRE_THROWS_AS(tensor.reduce(direction, functor), reduce_exception);
    };
    apply_by_element(test, test_data);
}