#include "catch.hpp"
#include "config.hpp"
#include "tensor.hpp"
#include <tuple>


TEST_CASE("test_tensor_construct_from_list","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using tensor_type = tensor<value_type, default_config>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<tensor_type, shape_type, index_type, index_type>;
    //tensor,expected_shape,expected size,expected dim
    auto test_data = GENERATE(                                
                                test_type(tensor_type{1}, shape_type{1}, 1 , 1),
                                test_type(tensor_type{1,2,3}, shape_type{3}, 3 , 1),
                                test_type(tensor_type{{1}}, shape_type{1,1}, 1 , 2),
                                test_type(tensor_type{{1,2,3}}, shape_type{1,3}, 3 , 2),
                                test_type(tensor_type{{1,2,3},{4,5,6}}, shape_type{2,3}, 6 , 2),
                                test_type(tensor_type{{{1,2,3,4}}}, shape_type{1,1,4}, 4 , 3),
                                test_type(tensor_type{{{1},{2},{3},{4}}}, shape_type{1,4,1}, 4 , 3),
                                test_type(tensor_type{{{1,2,3},{2,3,4},{3,4,5},{4,5,6}}}, shape_type{1,4,3}, 12 , 3)
                            );
    
    auto t = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_size = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
    auto stor_ten = t.as_storage_tensor();
    REQUIRE(stor_ten.shape() == expected_shape);
    REQUIRE(stor_ten.size() == expected_size);
    REQUIRE(stor_ten.dim() == expected_dim);
}

TEST_CASE("test_tensor_construct_using_operator","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using tensor_type = tensor<value_type, default_config>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<tensor_type, tensor_type, shape_type, index_type, index_type>;
    //0operand1,1operand2,2expected_shape,3expected size,4expected dim
    auto test_data = GENERATE(                                
                                test_type(tensor_type{1}, tensor_type{1}, shape_type{1}, 1 , 1),
                                test_type(tensor_type{1}, tensor_type{1,2,3}, shape_type{3}, 3 , 1),
                                test_type(tensor_type{1,2,3}, tensor_type{1,2,3}, shape_type{3}, 3 , 1),
                                test_type(tensor_type{{1,2,3}}, tensor_type{1,2,3}, shape_type{1,3}, 3 , 2),
                                test_type(tensor_type{{1,2,3}}, tensor_type{{1},{2},{3}}, shape_type{3,3}, 9 , 2),
                                test_type(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{1},{2}}, shape_type{2,3}, 6 , 2),
                                test_type(tensor_type{{{1,2,3},{4,5,6}}}, tensor_type{1,2,3}, shape_type{1,2,3}, 6 , 3),
                                test_type(tensor_type{1}+tensor_type{1}, tensor_type{1} ,shape_type{1}, 1 , 1),
                                test_type(tensor_type{1,2,3}+tensor_type{1,2,3}, tensor_type{1} ,shape_type{3}, 3 , 1),
                                test_type(tensor_type{1,2,3}+tensor_type{{1},{2},{3}}, tensor_type{1,2,3} ,shape_type{3,3}, 9 , 2),
                                test_type(tensor_type{1,2,3}+tensor_type{{1},{2},{3}}, tensor_type{1,2,3}+tensor_type{1} ,shape_type{3,3}, 9 , 2)
                            );
    
    auto operand1 = std::get<0>(test_data);
    auto operand2 = std::get<1>(test_data);
    auto expected_shape = std::get<2>(test_data);
    auto expected_size = std::get<3>(test_data);
    auto expected_dim = std::get<4>(test_data);
    auto e = operand1 + operand2;
    REQUIRE(e.shape() == expected_shape);
    REQUIRE(e.size() == expected_size);
    REQUIRE(e.dim() == expected_dim);
    auto expr = e.as_expression();
    REQUIRE(expr.shape() == expected_shape);
    REQUIRE(expr.size() == expected_size);
    REQUIRE(expr.dim() == expected_dim);
}

TEST_CASE("test_tensor_construct_using_derived_operands","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using tensor_type = tensor<value_type, default_config>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;    
    using test_type = std::tuple<tensor_type, shape_type, index_type, index_type>;
    //tensor,2expected_shape,3expected size,4expected dim
    auto test_data = GENERATE(                                
                                test_type(tensor_type{{1,2,3}}.as_storage_tensor()+tensor_type{{1},{2},{3}}.as_storage_tensor(), shape_type{3,3}, 9 , 2),
                                test_type((tensor_type{1}+tensor_type{1}).as_expression()+tensor_type{1}.as_storage_tensor() ,shape_type{1}, 1 , 1),
                                test_type((tensor_type{1,2,3}+tensor_type{{1},{2},{3}}).as_expression()+(tensor_type{1,2,3}+tensor_type{1}).as_expression() ,shape_type{3,3}, 9 , 2)
                            );
    
    auto t = std::get<0>(test_data);        
    auto expected_shape = std::get<1>(test_data);
    auto expected_size = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);        
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
    
}

TEST_CASE("test_expression_is_trivial","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using tensor_type = tensor<value_type, default_config>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<tensor_type, bool>;
    //tensor,expected_is_trivial
    auto test_data = GENERATE(                                
                                test_type(tensor_type{1}+tensor_type{1}, true),
                                test_type(tensor_type{1,2,3,4,5}+tensor_type{1}, false),
                                test_type(tensor_type{1,2,3,4,5}+tensor_type{1,2,3,4,5}, true),
                                test_type(tensor_type{{1},{2},{3},{4},{5}}+tensor_type{{1,2,3,4,5}}, false),
                                test_type(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{3,4,5}+tensor_type{3,4,5}, true),
                                test_type((tensor_type{1,2,3}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5}), true),
                                test_type((tensor_type{1}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5}), false)
                            );
    
    auto t = std::get<0>(test_data);
    auto expected_is_trivial = std::get<1>(test_data);
    auto e = t.as_expression();
    REQUIRE(e.is_trivial() == expected_is_trivial);    
}