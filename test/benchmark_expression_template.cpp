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

TEST_CASE("benchmark_expression_template","[benchmark_expression_template]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using benchmark_helpers::make_asymmetric_tree;
    using benchmark_helpers::benchmark;

    auto t1 = tensor_type({10,100,1000},2);
    auto t2 = tensor_type({100,1000},1);

    auto tree_50_1E6 = make_asymmetric_tree<50>(t1,t2);

    auto bench_iteration_deref = [](const auto& t){
        using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t)>>;
        using value_type = typename tensor_type::value_type;
        value_type a;
        for (auto it=t.begin(),last=t.end(); it!=last; ++it){
            a = *it;
        }
        return a;
    };

    benchmark("bench_depth50_10E6",bench_iteration_deref,tree_50_1E6);
}