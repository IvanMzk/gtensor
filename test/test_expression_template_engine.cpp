#include "catch.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"
#include "expression_template_operators.hpp"
#include "tensor.hpp"

namespace test_expression_template_engine_{

struct unary_square{
    template<typename T>
    auto operator()(const T& t)const{
        return t*t;
    }
};
struct binary_mul{
    template<typename T1, typename T2>
    auto operator()(const T1& t1, const T2& t2)const{
        return t1*t2;
    }
};

}

TEMPLATE_TEST_CASE("test_expression_template_walker","[test_expression_template_engine]",
    test_config::config_engine_selector_t<gtensor::config::engine_expression_template>
)
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using dim_type = typename config_type::dim_type;
    using tensor_type = gtensor::tensor<value_type>;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using gtensor::expression_template_walker;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2max_dim,3mover,4expected
    auto test_data = std::make_tuple(
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type(2)), dim_type{0}, [](auto&){}, value_type{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type(3)), dim_type{0}, [](auto& w){w.reset_back();}, value_type{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{2}), dim_type{1}, [](auto&){}, value_type{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{3}), dim_type{1}, [](auto& w){w.reset_back();}, value_type{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto&){}, value_type{1}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.reset(0);}, value_type{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.step(0);}, value_type{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2);}, value_type{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.step_back(0);}, value_type{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.reset_back();}, value_type{1}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type(1)), dim_type{0}, [](auto&){}, value_type{2}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type(3)), dim_type{0}, [](auto&){}, value_type{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto&){}, value_type{2}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.step(0);}, value_type{4}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2);}, value_type{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.step_back(0);}, value_type{4}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.reset_back();}, value_type{2}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto&){}, value_type{4}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.reset(0);}, value_type{18}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.step(0);}, value_type{10}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2);}, value_type{18}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.step_back(0);}, value_type{10}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.reset_back();}, value_type{4})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto operands = std::get<1>(t);
        auto max_dim = std::get<2>(t);
        auto mover = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto make_walker = [max_dim,f](auto&&...operands){
            return expression_template_walker<config_type,F,decltype(operands.create_walker(max_dim))...>{f,operands.create_walker(max_dim)...};
        };
        auto result_walker = std::apply(make_walker, operands);
        mover(result_walker);
        auto result = *result_walker;
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_expression_template_core","[test_expression_template_engine]",
    test_config::config_engine_selector_t<gtensor::config::engine_expression_template>
)
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::walker_iterator;
    using gtensor::expression_template_core;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2expected_dim,3expected_size,4expected_shape,5expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type(2)), dim_type{0}, index_type{1}, shape_type{}, std::vector<value_type>{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{3}), dim_type{1}, index_type{1}, shape_type{1}, std::vector<value_type>{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3,4,5}), dim_type{1}, index_type{5}, shape_type{5}, std::vector<value_type>{1,4,9,16,25}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{{1},{2},{3},{4},{5}}), dim_type{2}, index_type{5}, shape_type{5,1}, std::vector<value_type>{1,4,9,16,25}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{},tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{2},tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{},tensor_type(1)), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type(3)), dim_type{0}, index_type{1}, shape_type{}, std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{3}), dim_type{1}, index_type{1}, shape_type{1}, std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{3,4,5}), dim_type{1}, index_type{3}, shape_type{3}, std::vector<value_type>{6,8,10}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{{3},{4},{5}}), dim_type{2}, index_type{3}, shape_type{3,1}, std::vector<value_type>{6,8,10}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{1,2},tensor_type{{3},{4}}), dim_type{2}, index_type{4}, shape_type{2,2}, std::vector<value_type>{3,6,4,8})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto operands = std::get<1>(t);
        auto expected_dim = std::get<2>(t);
        auto expected_size = std::get<3>(t);
        auto expected_shape = std::get<4>(t);
        auto expected_elements = std::get<5>(t);
        auto make_core = [f](auto&&...operands){
            return expression_template_core<config_type,F,std::remove_reference_t<decltype(operands)>...>{f,std::forward<decltype(operands)>(operands)...};
        };
        auto result_core = std::apply(make_core, operands);
        auto result_dim = result_core.descriptor().dim();
        auto result_size = result_core.descriptor().size();
        auto result_shape = result_core.descriptor().shape();
        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_shape == expected_shape);
        using iterator_type = walker_iterator<config_type,decltype(result_core.create_walker(result_dim))>;
        auto result_first = iterator_type{
            result_core.create_walker(result_dim),
            result_core.descriptor().shape(),
            result_core.descriptor().strides_div(),
            index_type{0}
        };
        auto result_last = iterator_type{
            result_core.create_walker(result_dim),
            result_core.descriptor().shape(),
            result_core.descriptor().strides_div(),
            index_type{result_size}
        };
        REQUIRE(std::equal(result_first,result_last,expected_elements.begin(),expected_elements.end()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_expression_template_n_operator","[test_expression_template_engine]",
    test_config::config_engine_selector_t<gtensor::config::engine_expression_template>
)
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using tensor_type = gtensor::tensor<value_type,config_type>;
    using gtensor::expression_template_n_operator;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(unary_square{},std::make_tuple(tensor_type{}),tensor_type{}),
        std::make_tuple(unary_square{},std::make_tuple(tensor_type(2)),tensor_type(4)),
        std::make_tuple(unary_square{},std::make_tuple(tensor_type{1,2,3,4,5}),tensor_type{1,4,9,16,25}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{1,2,3,4,5},tensor_type(2)),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{2}),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{5,4,3,2,1}),tensor_type{5,8,9,8,5}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{-1,2}),tensor_type{{{-1,4},{-3,8}},{{-5,12},{-7,16}}}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{-1},{2}}),tensor_type{{{-1,-2},{6,8}},{{-5,-6},{14,16}}})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto operands = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto apply_n_operator = [f](auto&&...operands){
            return expression_template_n_operator<F>{}(f,operands...);
        };
        auto result = std::apply(apply_n_operator, operands);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test, test_data);
}