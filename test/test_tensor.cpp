#include "catch.hpp"
#include "config.hpp"
#include "tensor.hpp"
#include <tuple>


namespace test_tensor_helpers{

using gtensor::tensor;
using gtensor::tensor_base;

template<typename ValT, typename CfgT>
struct test_tensor : public tensor<ValT, CfgT, tensor_base<ValT,CfgT>>{

    test_tensor(const tensor& base):
        tensor{base}
    {}
    auto is_trivial()const{return impl()->engine().is_trivial();}
};


}   //end of namespace test_tensor


TEST_CASE("test_tensor_construct_from_list","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;    
    using config_type = gtensor::config::default_config;
    using tensor_type = tensor<value_type, config_type>;
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
    REQUIRE(t.as_htensor().as_htensor().shape() == expected_shape);
    REQUIRE(t.as_htensor().size() == expected_size);
    REQUIRE(t.as_htensor().dim() == expected_dim);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
}

TEST_CASE("test_tensor_construct_given_shape","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using config_type = gtensor::config::default_config;
    using tensor_type = tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<tensor_type, shape_type, index_type, index_type>;
    //tensor,expected_shape,expected size,expected dim
    auto test_data = GENERATE(                                
                                test_type(tensor_type(1,1), shape_type{1}, 1 , 1),
                                test_type(tensor_type(1,10), shape_type{10}, 10 , 1),
                                test_type(tensor_type(1,1,1), shape_type{1,1}, 1 , 2),
                                test_type(tensor_type(1,1,3), shape_type{1,3}, 3 , 2),
                                test_type(tensor_type(1,2,3), shape_type{2,3}, 6 , 2),
                                test_type(tensor_type(1,1,1,4), shape_type{1,1,4}, 4 , 3),
                                test_type(tensor_type(1,1,4,1), shape_type{1,4,1}, 4 , 3),
                                test_type(tensor_type(0,1,4,3), shape_type{1,4,3}, 12 , 3)
                            );
    
    auto t = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_size = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
    REQUIRE(t.as_htensor().shape() == expected_shape);
    REQUIRE(t.as_htensor().size() == expected_size);
    REQUIRE(t.as_htensor().dim() == expected_dim);
}

TEST_CASE("test_tensor_construct_using_operator","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using gtensor::tensor_base;
    using gtensor::storage_tensor;
    using config_type = gtensor::config::default_config;
    using tensor_type = tensor<value_type, config_type>;
    using htensor_type = tensor<value_type, config_type, tensor_base<value_type, config_type>>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<htensor_type, shape_type, index_type, index_type>;
    //0htensor,1expected_shape,2expected size,3expected dim
    auto test_data = GENERATE(                                
                                test_type((tensor_type{1}+tensor_type{1}).as_htensor(), shape_type{1}, 1 , 1),
                                test_type(static_cast<htensor_type>(tensor_type{1}+tensor_type{1,2,3}), shape_type{3}, 3 , 1),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3}+tensor_type{1,2,3}), shape_type{3}, 3 , 1),
                                test_type(static_cast<htensor_type>(tensor_type{{1,2,3}}+tensor_type{1,2,3}), shape_type{1,3}, 3 , 2),
                                test_type(static_cast<htensor_type>(tensor_type{{1,2,3}}+tensor_type{{1},{2},{3}}), shape_type{3,3}, 9 , 2),
                                test_type(static_cast<htensor_type>(tensor_type{{1,2,3},{4,5,6}}+tensor_type{{1},{2}}), shape_type{2,3}, 6 , 2),
                                test_type(static_cast<htensor_type>(tensor_type{{{1,2,3},{4,5,6}}}+tensor_type{1,2,3}), shape_type{1,2,3}, 6 , 3),
                                test_type(static_cast<htensor_type>(tensor_type{1}+tensor_type{1}+tensor_type{1}) ,shape_type{1}, 1 , 1),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{1}) ,shape_type{3}, 3 , 1),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3}+tensor_type{{1},{2},{3}}+tensor_type{1,2,3}) ,shape_type{3,3}, 9 , 2),
                                test_type(static_cast<htensor_type>((tensor_type{1,2,3}+(tensor_type{{1},{2},{3}})+(tensor_type{1,2,3})+tensor_type{1})) ,shape_type{3,3}, 9 , 2)
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
    using gtensor::tensor_base;
    using test_tensor_helpers::test_tensor;
    using config_type = gtensor::config::default_config;
    using tensor_type = tensor<value_type, config_type>;
    using htensor_type = tensor<value_type, config_type, tensor_base<value_type, config_type>>;
    using test_tensor_type = test_tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<htensor_type, bool>;
    //tensor,expected_is_trivial
    auto test_data = GENERATE(                                
                                test_type(static_cast<htensor_type>(tensor_type{1}+tensor_type{1}), true),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3,4,5}+tensor_type{1}), false),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3,4,5}+tensor_type{1,2,3,4,5}), true),
                                test_type(static_cast<htensor_type>(tensor_type{{1},{2},{3},{4},{5}}+tensor_type{{1,2,3,4,5}}), false),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{3,4,5}+tensor_type{3,4,5}), true),
                                test_type(static_cast<htensor_type>((tensor_type{1,2,3}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5})), true),
                                test_type(static_cast<htensor_type>((tensor_type{1}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5})), false)
                            );
    
    auto t = test_tensor_type{std::get<0>(test_data)};
    auto expected_is_trivial = std::get<1>(test_data);
    REQUIRE(t.is_trivial() == expected_is_trivial);    
}

TEST_CASE("test_view_making_interface","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using gtensor::tensor_base;
    using config_type = gtensor::config::default_config;
    using tensor_type = tensor<value_type, config_type>;    
    using htensor_type = tensor<value_type, config_type, tensor_base<value_type,config_type>>;    
    using slice_type = typename config_type::slice_type;
    using nop_type = typename config_type::nop_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::subscript_exception;
    nop_type nop;
    SECTION("test_subscripts_correctenes_check"){
        SECTION("view_slice"){
            REQUIRE_NOTHROW(tensor_type{1}({}));
            REQUIRE_THROWS_AS((tensor_type{1}({{},{}})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1}({{0,1}}));
            REQUIRE_THROWS_AS((tensor_type{1}({{nop,2}})),subscript_exception);            
            REQUIRE_NOTHROW(tensor_type{1,2,3,4,5}({}));
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}({{0,4,-1}})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}({{0,0}})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}({{1,-1}}));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}({{1,-1},{1,-1}})),subscript_exception);
        }
        SECTION("view_transpose"){
            REQUIRE_NOTHROW(tensor_type{1}.transpose());            
            REQUIRE_NOTHROW(tensor_type{1}.transpose(0));                        
            REQUIRE_THROWS_AS((tensor_type{1}.transpose(1)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1}.transpose(0,1)),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4}}.transpose());
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4}}.transpose(0,1));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4}}.transpose(1,0));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4}}.transpose(0)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4}}.transpose(1,1)),subscript_exception);
        }
        SECTION("view_subdim"){
            REQUIRE_NOTHROW(tensor_type{1}());
            REQUIRE_THROWS_AS((tensor_type{1}(0)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}(0)),subscript_exception);
            REQUIRE_NOTHROW((tensor_type{{{1,2},{3,4},{5,6}}}(0)));
            REQUIRE_NOTHROW((tensor_type{{{1,2},{3,4},{5,6}}}(0,0)));
            REQUIRE_NOTHROW((tensor_type{{{1,2},{3,4},{5,6}}}(0,2)));
            REQUIRE_THROWS_AS((tensor_type{{{1,2},{3,4},{5,6}}}(1)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{{1,2},{3,4},{5,6}}}(0,3)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{{1,2},{3,4},{5,6}}}(0,2,0)),subscript_exception);
        }
        SECTION("view_reshape"){
            REQUIRE_NOTHROW(tensor_type{1}.reshape());
            REQUIRE_NOTHROW(tensor_type{1}.reshape(1));
            REQUIRE_THROWS_AS((tensor_type{1}.reshape(0)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1}.reshape(2)),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape());
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(6));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(6,1));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(1,6,1));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(2,3));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(3,2));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}.reshape(10)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}.reshape(3,3)),subscript_exception);
        }
    }    
    SECTION("test_slices_filling_and_result_view"){
        using test_type = std::tuple<htensor_type, shape_type, index_type, index_type>;
        //0view,1expected_shape,2expected size,3expected dim
        auto test_data = GENERATE_COPY(
            test_type{static_cast<htensor_type>(tensor_type{1}({{}})),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1}({{nop,nop,-1}})),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1,2,3,4,5}({{}})),shape_type{5}, 5, 1},
            test_type{static_cast<htensor_type>(tensor_type{1,2,3,4,5}({{nop,nop,2}})),shape_type{3}, 3, 1},
            test_type{static_cast<htensor_type>(tensor_type{1,2,3,4,5}({{nop,nop,-2}})),shape_type{3}, 3, 1},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}({{},{},{0,-1}})),shape_type{1,3,1}, 3, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}({{},{1,-1},{0,-1}})),shape_type{1,1,1}, 1, 3},
            test_type{static_cast<htensor_type>(tensor_type{1}.transpose()),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1}.transpose(0)),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1,2,3,4,5}.transpose()),shape_type{5}, 5, 1},
            test_type{static_cast<htensor_type>(tensor_type{{1,2,3,4,5}}.transpose()),shape_type{5,1}, 5, 2},
            test_type{static_cast<htensor_type>(tensor_type{{1,2,3,4,5}}.transpose(1,0)),shape_type{5,1}, 5, 2},
            test_type{static_cast<htensor_type>(tensor_type{{1,2,3,4,5}}.transpose(1,0)),shape_type{5,1}, 5, 2},
            test_type{static_cast<htensor_type>(tensor_type{{1,2,3,4,5}}.transpose(0,1)),shape_type{1,5}, 5, 2},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.transpose()),shape_type{2,3,1}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(0,2,1)),shape_type{1,2,3}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}()),shape_type{1,3,2}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}(0)),shape_type{3,2}, 6, 2},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}(0,1)),shape_type{2}, 2, 1},
            test_type{static_cast<htensor_type>(tensor_type{1}.reshape()),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1}.reshape(1)),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.reshape()),shape_type{1,3,2}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.reshape(6)),shape_type{6}, 6, 1},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.reshape(2,1,3)),shape_type{2,1,3}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.reshape(6,1)),shape_type{6,1}, 6, 2}
        );
        auto view = std::get<0>(test_data);
        auto expected_shape = std::get<1>(test_data);
        auto expected_size = std::get<2>(test_data);
        auto expected_dim = std::get<3>(test_data);
        REQUIRE(view.shape() == expected_shape);
        REQUIRE(view.size() == expected_size);
        REQUIRE(view.dim() == expected_dim);
    }
}

// TEST_CASE("test_tensor_construct_using_derived_operands","[test_tensor]"){
//     using value_type = float;
//     using gtensor::tensor;
//     using config_type = gtensor::config::default_config;
//     using tensor_type = tensor<value_type, config_type>;
//     using shape_type = typename config_type::shape_type;
//     using index_type = typename config_type::index_type;    
//     using test_type = std::tuple<tensor_type, shape_type, index_type, index_type>;
//     //tensor,2expected_shape,3expected size,4expected dim
//     auto test_data = GENERATE(                                
//                                 test_type(tensor_type{{1,2,3}}.as_storage_tensor()+tensor_type{{1},{2},{3}}.as_storage_tensor(), shape_type{3,3}, 9 , 2),
//                                 test_type((tensor_type{1}+tensor_type{1}).as_expression()+tensor_type{1}.as_storage_tensor() ,shape_type{1}, 1 , 1),
//                                 test_type((tensor_type{1,2,3}+tensor_type{{1},{2},{3}}).as_expression()+(tensor_type{1,2,3}+tensor_type{1}).as_expression() ,shape_type{3,3}, 9 , 2)
//                             );
    
//     auto t = std::get<0>(test_data);        
//     auto expected_shape = std::get<1>(test_data);
//     auto expected_size = std::get<2>(test_data);
//     auto expected_dim = std::get<3>(test_data);        
//     REQUIRE(t.shape() == expected_shape);
//     REQUIRE(t.size() == expected_size);
//     REQUIRE(t.dim() == expected_dim);
    
// }

// TEST_CASE("test_expression_trivial_at","[test_tensor]"){
//     using value_type = float;
//     using gtensor::tensor;
//     using config_type = gtensor::config::default_config;
//     using tensor_type = tensor<value_type, config_type>;
//     using shape_type = typename config_type::shape_type;
//     using index_type = typename config_type::index_type;
//     using test_type = std::tuple<tensor_type, index_type, value_type>;
//     //0tensor,1index,2expected_value
//     auto test_data = GENERATE(                                
//                                 test_type(tensor_type{1}+tensor_type{1}, 0, value_type{2}),
//                                 test_type(tensor_type{1}+tensor_type{1}+tensor_type{1}, 0, value_type{3}),
//                                 test_type(tensor_type{1,2,3}+tensor_type{1,2,3}, 0, value_type{2}),
//                                 test_type(tensor_type{1,2,3}+tensor_type{1,2,3}, 1, value_type{4}),
//                                 test_type(tensor_type{1,2,3}+tensor_type{1,2,3}, 2, value_type{6}),                                                                
//                                 test_type(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{3,4,5}+tensor_type{3,4,5}, 0, value_type{8}),
//                                 test_type(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{3,4,5}+tensor_type{3,4,5}, 1, value_type{12}),
//                                 test_type(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{3,4,5}+tensor_type{3,4,5}, 2, value_type{16}),
//                                 test_type((tensor_type{1,2,3}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5}),0, value_type{8}),
//                                 test_type((tensor_type{1,2,3}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5}),1, value_type{12}),
//                                 test_type((tensor_type{1,2,3}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5}),2, value_type{16})
//                             );
    
//     auto t = std::get<0>(test_data);
//     auto index = std::get<1>(test_data);
//     auto expected_value = std::get<2>(test_data);
//     auto e = t.as_expression();
//     REQUIRE(e.trivial_at(index) == expected_value);
// }
