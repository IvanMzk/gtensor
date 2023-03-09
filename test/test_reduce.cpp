#include "catch.hpp"
#include "gtensor.hpp"
#include "reduce.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

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
        std::make_tuple(tensor_type{}, size_type{0}, std::plus{}, tensor_type{}),
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
        auto result = tensor.reduce(functor, direction);
        REQUIRE(result.equals(expected));
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
        REQUIRE_THROWS_AS(tensor.reduce(functor, direction), reduce_exception);
    };
    apply_by_element(test, test_data);
}