#include "gtensor.hpp"
#include "test_config.hpp"
#include "benchmark_helpers.hpp"

TEST_CASE("benchmark_expression_template","[benchmark_expression_template]"){
    using value_type = float;
    using test_config_type = typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type;
    using tensor_type = gtensor::tensor<value_type,test_config_type>;
    using benchmark_helpers::asymmetric_tree_maker;
    using benchmark_helpers::symmetric_tree_maker;
    using benchmark_helpers::benchmark_with_making_iter;
    using benchmark_helpers::making_iter_iterate_deref;

    //benchmark_with_making_iter(asymmetric_tree_maker<50>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)),"",making_iter_iterate_deref);
}