#include <tuple>
#include "catch.hpp"
#include "tensor.hpp"

namespace test_walker_{

using gtensor::stensor_impl;
using gtensor::expression_impl;
using gtensor::config::default_config;


//make 3d stensor with data {{{1,2,3},{4,5,6}}}
template<typename ValT>
struct stensor_maker{
    using value_type = ValT;
    using tensor_type = gtensor::tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{1,2,3},{4,5,6}}};}
};

//make 3d trivial broadcast expression with data {{{1,2,3},{4,5,6}}}
template<typename ValT>
struct trivial_expression_maker{
    using value_type = ValT;
    using tensor_type = gtensor::tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{-1,-1,-1},{-1,-1,-1}}} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{1,1,1},{4,4,4}}};}
};

//make 3d complex expression with data {{{1,2,3},{4,5,6}}}
template<typename ValT>
struct not_trivial_expression_maker{
    using value_type = ValT;
    using tensor_type = gtensor::tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{2} * tensor_type{-1,-1,-1} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}} + tensor_type{5,5,5} - tensor_type{3} ;}
};


}   //end of namespace test_walker_



TEMPLATE_TEST_CASE("test_walker","test_walker", 
                    test_walker_::stensor_maker<float>, 
                    test_walker_::trivial_expression_maker<float>,
                    test_walker_::not_trivial_expression_maker<float>
                    ){
    using value_type = typename TestType::value_type;
    using gtensor::stensor_impl;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using stensor_type = stensor_impl<value_type, default_config>;    
    using walker_type = gtensor::walker<value_type,default_config>;
    using test_type = std::tuple<walker_type, value_type>;    

    auto walker_parent = TestType{}();
    REQUIRE(*walker_parent.create_walker() == value_type{1});
    REQUIRE(*walker_parent.create_walker().walk(0,1) == value_type{1});
    REQUIRE(*walker_parent.create_walker().walk(1,1) == value_type{4});
    REQUIRE(*walker_parent.create_walker().walk(2,1) == value_type{2});
    REQUIRE(*walker_parent.create_walker().walk(1,1).walk(2,2) == value_type{6});
    REQUIRE(*walker_parent.create_walker().walk(3,1).walk(2,2) == value_type{3});
    REQUIRE(*walker_parent.create_walker().walk(1,1).reset(1) == value_type{1});
    REQUIRE(*walker_parent.create_walker().walk(1,1).walk(2,2).reset(0) == value_type{6});
    REQUIRE(*walker_parent.create_walker().walk(1,1).walk(2,2).reset(3) == value_type{6});
    REQUIRE(*walker_parent.create_walker().walk(1,1).walk(2,2).reset(1) == value_type{3});
    REQUIRE(*walker_parent.create_walker().walk(1,1).walk(2,2).reset(2) == value_type{4});
    REQUIRE(*walker_parent.create_walker().walk(1,1).walk(2,2).reset() == value_type{1});
}