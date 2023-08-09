#include "benchmark_helpers.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"
#include "statistic.hpp"

namespace benchmark_expression_template_helpers{

}   //end of namespace benchmark_expression_template_helpers

TEMPLATE_TEST_CASE("benchmark_tensor_sum","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using benchmark_helpers::benchmark;

    auto bench_sum = [](const auto& t, auto...axes){
        auto tmp = t.sum({axes...});
        return tmp.size();
    };

    auto t = tensor_type({50,5,10,10,5,50},2);

    //like over flatten
    //benchmark("sum_all_10E6",bench_sum,t);

    //single axis
    benchmark("sum_axis_10E6",bench_sum,t,0);
    benchmark("sum_axis_10E6",bench_sum,t,1);
    benchmark("sum_axis_10E6",bench_sum,t,2);
    benchmark("sum_axis_10E6",bench_sum,t,3);
    benchmark("sum_axis_10E6",bench_sum,t,4);
    benchmark("sum_axis_10E6",bench_sum,t,5);

    //axes
    benchmark("sum_axes01_10E6",bench_sum,t,0,1);
    benchmark("sum_axes02_10E6",bench_sum,t,0,2);
    benchmark("sum_axes03_10E6",bench_sum,t,0,3);
    benchmark("sum_axes12_10E6",bench_sum,t,1,2);
    benchmark("sum_axes13_10E6",bench_sum,t,1,3);
    benchmark("sum_axes23_10E6",bench_sum,t,2,3);
    benchmark("sum_axes34_10E6",bench_sum,t,3,4);
    benchmark("sum_axes45_10E6",bench_sum,t,4,5);

    benchmark("sum_axes012_10E6",bench_sum,t,0,1,2);
    benchmark("sum_axes024_10E6",bench_sum,t,0,2,4);
    benchmark("sum_axes135_10E6",bench_sum,t,1,3,5);
    benchmark("sum_axes543_10E6",bench_sum,t,5,4,3);
}
