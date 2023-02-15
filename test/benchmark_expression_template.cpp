#include "gtensor.hpp"
#include "test_config.hpp"
#include "benchmark_helpers.hpp"

namespace benchmark_expression_template_helpers{
}   //end of namespace benchmark_expression_template_helpers

TEST_CASE("benchmark_expression_template","[benchmark_expression_template]"){
    using value_type = float;
    using test_config_type = typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type;
    using tensor_type = gtensor::tensor<value_type,test_config_type>;
    using benchmark_helpers::asymmetric_tree_maker;
    using benchmark_helpers::symmetric_tree_maker;
    using benchmark_helpers::benchmark;
    using benchmark_helpers::making_iter_iterate_deref;
    using benchmark_helpers::making_riter_iterate_deref;

    auto benchmark_worker = making_iter_iterate_deref;
    //auto benchmark_worker = making_riter_iterate_deref;
    static constexpr std::size_t tree_depth = 5;
    static constexpr std::size_t deep_tree_depth = 50;


    // auto trivial_tensor_maker = []{return asymmetric_tree_maker<tree_depth>{}(tensor_type({100,100}, 0.0f),tensor_type({100,100}, 0.0f));};
    // auto broadcast_tensor_maker = []{return asymmetric_tree_maker<tree_depth>{}(tensor_type({10,100,10}, 0.0f),tensor_type({100,10}, 0.0f));};

    auto trivial_tensor_maker = []{return asymmetric_tree_maker<tree_depth>{}(tensor_type({1000,1000}, 0.0f),tensor_type({1000,1000}, 0.0f));};
    auto broadcast_tensor_maker = []{return asymmetric_tree_maker<tree_depth>{}(tensor_type({100,100,100}, 0.0f),tensor_type({100,100}, 0.0f));};
    auto deep_tree_tensor_maker = []{return asymmetric_tree_maker<deep_tree_depth>{}(tensor_type({10,10000}, 0.0f),tensor_type({10,10000}, 0.0f));};

    // auto trivial_tensor_maker = []{return asymmetric_tree_maker<deep_tree_depth>{}(tensor_type({10000,10}, 0.0f),tensor_type({10000,10}, 0.0f));};
    // auto broadcast_tensor_maker = []{return asymmetric_tree_maker<deep_tree_depth>{}(tensor_type({100,100,10}, 0.0f),tensor_type({100,10}, 0.0f));};


    benchmark("broadcast",benchmark_worker, broadcast_tensor_maker());
    benchmark("trivial_broadcast",benchmark_worker, trivial_tensor_maker());

    benchmark("slice_of_broadcast",benchmark_worker, broadcast_tensor_maker()({}));
    benchmark("transpose_of_broadcast",benchmark_worker, broadcast_tensor_maker().transpose());
    benchmark("reshape_of_broadcast",benchmark_worker, broadcast_tensor_maker().reshape());
    benchmark("subdim_of_broadcast",benchmark_worker, broadcast_tensor_maker()());

    benchmark("slice_of_trivial",benchmark_worker, trivial_tensor_maker()({}));
    benchmark("transpose_of_trivial",benchmark_worker, trivial_tensor_maker().transpose());
    benchmark("reshape_of_trivial",benchmark_worker, trivial_tensor_maker().reshape());
    benchmark("subdim_of_trivial",benchmark_worker, trivial_tensor_maker()());

    benchmark("benchmark_deep_tree",benchmark_worker, deep_tree_tensor_maker());

}