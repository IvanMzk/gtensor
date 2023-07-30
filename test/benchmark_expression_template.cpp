#include "benchmark_helpers.hpp"
#include "tensor.hpp"

namespace benchmark_expression_template_helpers{
}   //end of namespace benchmark_expression_template_helpers


TEST_CASE("test_benchmark_helpers_make_tree","[test_benchmark_helpers]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using benchmark_helpers::make_asymmetric_tree;

    REQUIRE(make_asymmetric_tree<1>(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}}) == tensor_type{{2,4,6},{8,10,12}});
    REQUIRE(make_asymmetric_tree<1>(tensor_type{{1,2,3},{4,5,6}},tensor_type{7,8,9}) == tensor_type{{8,10,12},{11,13,15}});
    REQUIRE(make_asymmetric_tree<2>(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}}) == tensor_type{{3,6,9},{12,15,18}});
    REQUIRE(make_asymmetric_tree<2>(tensor_type{{1,2,3},{4,5,6}},tensor_type{7,8,9}) == tensor_type{{15,18,21},{18,21,24}});
    REQUIRE(make_asymmetric_tree<3>(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}}) == tensor_type{{4,8,12},{16,20,24}});
    REQUIRE(make_asymmetric_tree<3>(tensor_type{{1,2,3},{4,5,6}},tensor_type{7,8,9}) == tensor_type{{22,26,30},{25,29,33}});
    REQUIRE(make_asymmetric_tree<4>(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}}) == tensor_type{{5,10,15},{20,25,30}});
    REQUIRE(make_asymmetric_tree<4>(tensor_type{{1,2,3},{4,5,6}},tensor_type{7,8,9}) == tensor_type{{29,34,39},{32,37,42}});

}

TEMPLATE_TEST_CASE("benchmark_expression_template","[benchmark_expression_template]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using benchmark_helpers::make_asymmetric_tree;
    using benchmark_helpers::benchmark;

    auto t1 = tensor_type({10,100,1000},2);
    auto t2 = tensor_type({100,1000},1);

    auto tree_50_1E6 = make_asymmetric_tree<50>(t1,t2);

    auto bench_iteration_deref = [](const auto& t, auto order){
        using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t)>>;
        using value_type = typename tensor_type::value_type;
        auto a = t.template traverse_order_adapter<decltype(order)>();
        value_type v{0};
        for (auto it=a.begin(),last=a.end(); it!=last; ++it){
            v += *it;
        }
        return v;
    };

    auto bench_copy = [](const auto& t, auto order){
        return t.copy(order);
    };

    benchmark("c_copy_depth50_10E6",bench_copy,tree_50_1E6,c_order{});
    benchmark("f_copy_depth50_10E6",bench_copy,tree_50_1E6,f_order{});
    // benchmark("c_iteration_deref_depth50_10E6",bench_iteration_deref,tree_50_1E6,c_order{});
    // benchmark("f_iteration_deref_depth50_10E6",bench_iteration_deref,tree_50_1E6,f_order{});
}