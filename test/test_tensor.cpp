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
}


