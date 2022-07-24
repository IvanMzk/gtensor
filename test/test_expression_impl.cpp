#include "catch.hpp"
#include "config.hpp"
#include <tuple>
#include <memory>
#include "impl_stensor.hpp"
#include "impl_expression.hpp"
#include "operations.hpp"



TEMPLATE_PRODUCT_TEST_CASE("test_expression_impl_broadcast","[test_expression_impl]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
    using shape_type = TestType;
    using test_type = std::tuple<shape_type, shape_type, shape_type>;
    using gtensor::detail::broadcast;
    //shape1,shape2, expected broadcast shape 
    auto test_data = GENERATE(
                                test_type(shape_type{1}, shape_type{1}, shape_type{1}),
                                test_type(shape_type{5}, shape_type{5}, shape_type{5}),
                                test_type(shape_type{1,1}, shape_type{1}, shape_type{1,1}),
                                test_type(shape_type{1}, shape_type{1,1}, shape_type{1,1}),
                                test_type(shape_type{1,1}, shape_type{1,1}, shape_type{1,1}),
                                test_type(shape_type{1,5}, shape_type{5,1}, shape_type{5,5}),
                                test_type(shape_type{2,3,4}, shape_type{3,4}, shape_type{2,3,4}),
                                test_type(shape_type{2,1,4}, shape_type{3,1}, shape_type{2,3,4}),
                                test_type(shape_type{2,4}, shape_type{3,1,4}, shape_type{3,2,4})
                            );
        
    auto shape1 = std::get<0>(test_data);
    auto shape2 = std::get<1>(test_data);
    auto expected_broadcast_shape = std::get<2>(test_data);
    REQUIRE(broadcast(shape1, shape2) == expected_broadcast_shape);
}

TEMPLATE_PRODUCT_TEST_CASE("test_expression_impl_broadcast_not_broadcastable","[test_expression_impl]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
    using shape_type = TestType;
    using test_type = std::tuple<shape_type, shape_type>;
    using gtensor::detail::broadcast;
    using gtensor::broadcast_exception;
    //shape1,shape2
    auto test_data = GENERATE(
                                test_type(shape_type{}, shape_type{}),
                                test_type(shape_type{1}, shape_type{}),
                                test_type(shape_type{}, shape_type{1}),
                                test_type(shape_type{3}, shape_type{2}),
                                test_type(shape_type{2}, shape_type{3}),
                                test_type(shape_type{1,2}, shape_type{3}),
                                test_type(shape_type{1,2}, shape_type{4,3}),
                                test_type(shape_type{3,2}, shape_type{4,2}),
                                test_type(shape_type{5,1,2}, shape_type{4,4,2})
                            );
        
    auto shape1 = std::get<0>(test_data);
    auto shape2 = std::get<1>(test_data);    
    REQUIRE_THROWS_AS(broadcast(shape1, shape2), broadcast_exception);
}

TEST_CASE("test_expression_impl_construct","[test_expression_impl]"){
    using value_type = float;
    using gtensor::binary_operations::add;
    using gtensor::expression_impl;
    using gtensor::stensor_impl;
    using gtensor::tensor_impl_base;
    using gtensor::config::default_config;
    using config_type = gtensor::config::default_config<value_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using tensor_impl_base_type = tensor_impl_base<value_type, default_config>;
    using stensor_impl_type = stensor_impl<value_type, default_config>;
    using expression_impl_type = expression_impl<value_type, default_config, add, std::shared_ptr<tensor_impl_base_type>, std::shared_ptr<tensor_impl_base_type>>;
    using test_type = std::tuple<std::shared_ptr<tensor_impl_base_type>, std::shared_ptr<tensor_impl_base_type>, shape_type, index_type, index_type>;
    //0operand,1operand,2expected_shape,3expected_dim,4expected_size
    auto test_data = GENERATE(
        test_type{new stensor_impl_type{1,2,3}, new stensor_impl_type{3,2,1}, shape_type{3}, 1, 3},
        test_type{new stensor_impl_type{1,2,3}, new stensor_impl_type{3}, shape_type{3}, 1, 3},
        test_type{new stensor_impl_type{{1,2,3},{4,5,6}}, new stensor_impl_type{1,2,3}, shape_type{2,3}, 2, 6},
        test_type{new stensor_impl_type{{1,2,3}}, new stensor_impl_type{{1},{2},{3}}, shape_type{3,3}, 2, 9}
    );
    auto operand1 = std::get<0>(test_data);
    auto operand2 = std::get<1>(test_data);
    auto expected_shape = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);
    auto expected_size = std::get<4>(test_data);
    expression_impl_type e{operand1, operand2};
    REQUIRE(e.shape() == expected_shape);
    REQUIRE(e.dim() == expected_dim);
    REQUIRE(e.size() == expected_size);
    REQUIRE(e.tensor_kind() == gtensor::detail::tensor_kinds::expression);
}
