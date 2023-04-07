#include "catch.hpp"
#include "gtensor.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

TEMPLATE_TEST_CASE("test_binary_operator+","[test_operator]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using helpers_for_testing::apply_by_element;
    //0operands,1expected
    auto test_data = std::make_tuple(
        //0d,0d
        std::make_tuple(std::make_tuple(tensor_type(value_type{}),tensor_type(value_type{})), tensor_type(value_type{}+value_type{})),
        std::make_tuple(std::make_tuple(tensor_type(value_type{}),tensor_type(value_type{}),tensor_type(value_type{})),tensor_type(value_type{}+value_type{}+value_type{})),
        std::make_tuple(std::make_tuple(tensor_type(value_type{1}),tensor_type(value_type{2})), tensor_type(value_type{1}+value_type{2})),
        std::make_tuple(std::make_tuple(tensor_type(value_type{1}),tensor_type(value_type{2}),tensor_type(value_type{3})),tensor_type(value_type{1}+value_type{2}+value_type{3})),
        //nd,0d
        std::make_tuple(std::make_tuple(tensor_type{},tensor_type(value_type{})),tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type(value_type{}),tensor_type{}),tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{},tensor_type(value_type{3})),tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type(value_type{3}),tensor_type{}),tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}.reshape(2,3,0),tensor_type(value_type{3})),tensor_type{}.reshape(2,3,0)),
        std::make_tuple(std::make_tuple(tensor_type(value_type{3}),tensor_type{}.reshape(2,3,0)),tensor_type{}.reshape(2,3,0)),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type(value_type{1})),tensor_type{2,3,4,5,6,7}),
        std::make_tuple(std::make_tuple(tensor_type(value_type{1}),tensor_type{1,2,3,4,5,6}),tensor_type{2,3,4,5,6,7}),
        std::make_tuple(std::make_tuple(tensor_type(value_type{1}),tensor_type{{1,2,3},{4,5,6}}),tensor_type{{2,3,4},{5,6,7}}),
        std::make_tuple(std::make_tuple(tensor_type(value_type{-1}),tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}),
        std::make_tuple(std::make_tuple(tensor_type(value_type{-1}),tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type(value_type{1})),tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        //nd,nd
        std::make_tuple(std::make_tuple(tensor_type{},tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{},tensor_type{},tensor_type{}),tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}.reshape(1,0),tensor_type{}.reshape(3,0),tensor_type{0,0}.reshape(2,1,1)),tensor_type{}.reshape(2,3,0)),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type{1}),tensor_type{2,3,4,5,6,7}),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type{6,5,4,3,2,1}),tensor_type{7,7,7,7,7,7}),
        std::make_tuple(std::make_tuple(tensor_type{-1}, tensor_type{{1,2,3},{4,5,6}}),tensor_type{{0,1,2},{3,4,5}}),
        std::make_tuple(std::make_tuple(tensor_type{3,2,1}, tensor_type{{1,2,3},{4,5,6}}),tensor_type{{4,4,4},{7,7,7}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, tensor_type{1}),tensor_type{{{2,3},{4,5}},{{6,7},{8,9}}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, tensor_type{{-1},{1}}),tensor_type{{{0,1},{4,5}},{{4,5},{8,9}}})
    );
    auto test = [](const auto& t){
        auto operands = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto apply_operands = [](const auto&...operands_){
            return (...+operands_);
        };
        auto result = std::apply(apply_operands, operands);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test, test_data);
}