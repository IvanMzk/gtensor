#include "catch.hpp"
#include "benchmark_helpers.hpp"
#include "tensor.hpp"

TEST_CASE("test_deep_tree_compilation","[test_deep_tree_compilation]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type,config_type>;
    using benchmark_helpers::make_asymmetric_tree;
    using benchmark_helpers::make_symmetric_tree;
    
    auto e = make_asymmetric_tree<200>(tensor_type{1,2,3}, tensor_type{{1,1,1},{2,2,2}});
    //auto e1 = make_symmetric_tree<10>(tensor_type{1,2,3}, tensor_type{{1,1,1},{2,2,2}});
}

