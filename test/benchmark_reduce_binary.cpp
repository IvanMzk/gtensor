#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"
#include "statistic.hpp"

namespace benchmark_expression_template_helpers{

using gtensor::basic_tensor;
using gtensor::reduce_binary;

template<typename Axes, typename...Ts, typename Initial=gtensor::detail::no_value>
auto sum(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false, const Initial& initial=Initial{}){
    using f_type = gtensor::math_reduce_operations::nan_propagate_operation<gtensor::math_reduce_operations::plus>;
    return reduce_binary(t,axes,f_type{},keep_dims,initial);
}

}   //end of namespace benchmark_expression_template_helpers


TEMPLATE_TEST_CASE("test_reduce_binary_sum","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    //using config_type = typename tensor_type::config_type;
    //using order = typename tensor_type::order;

    auto t = tensor_type{{{{{7,5,8,5},{0,5,5,1},{3,8,0,8}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}}}},{{{{4,8,0,7},{0,0,2,4},{1,5,8,5}}},{{{6,8,4,8},{4,1,3,2},{7,0,6,2}}}},{{{{7,3,6,4},{2,6,4,7},{0,3,3,1}}},{{{2,1,3,0},{4,7,4,4},{7,6,3,3}}}}};

    REQUIRE(benchmark_expression_template_helpers::sum(tensor_type{1,2,3,4,5},std::vector<int>{0}) == tensor_type(15));
    REQUIRE(benchmark_expression_template_helpers::sum(t,std::vector<int>{0}) == tensor_type{{{{18,16,14,16},{2,11,11,12},{4,16,11,14}}},{{{8,9,9,13},{9,10,10,6},{20,13,12,12}}}});
    REQUIRE(benchmark_expression_template_helpers::sum(t,std::vector<int>{0,1}) == tensor_type{{{26,25,23,29},{11,21,21,18},{24,29,23,26}}});
    REQUIRE(benchmark_expression_template_helpers::sum(t,std::vector<int>{0,2}) == tensor_type{{{18,16,14,16},{2,11,11,12},{4,16,11,14}},{{8,9,9,13},{9,10,10,6},{20,13,12,12}}});
    REQUIRE(benchmark_expression_template_helpers::sum(t,std::vector<int>{1,2,3}) == tensor_type{{17,27,21,26},{22,22,23,28},{22,26,23,19}});
    REQUIRE(benchmark_expression_template_helpers::sum(t,std::vector<int>{2,3}) == tensor_type{{{10,18,13,14},{7,9,8,12}},{{5,13,10,16},{17,9,13,12}},{{9,12,13,12},{13,14,10,7}}});
}

// TEMPLATE_TEST_CASE("benchmark_reduce_binary_sum","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::benchmark;

//     auto bench_sum = [](const auto& t, auto...axes){
//         auto tmp = benchmark_expression_template_helpers::sum(t,std::initializer_list<int>{axes...});
//         return tmp.size();
//     };

//     auto t = tensor_type({50,5,10,10,5,50},2);

//     //like over flatten
//     benchmark("sum_binary_flatten",bench_sum,t,0,1,2,3,4,5);

//     //single axis
//     benchmark("sum_binary_axis0",bench_sum,t,0);
//     benchmark("sum_binary_axis1",bench_sum,t,1);
//     benchmark("sum_binary_axis2",bench_sum,t,2);
//     benchmark("sum_binary_axis3",bench_sum,t,3);
//     benchmark("sum_binary_axis4",bench_sum,t,4);
//     benchmark("sum_binary_axis5",bench_sum,t,5);

//     //axes
//     benchmark("sum_binary_axes01",bench_sum,t,0,1);
//     benchmark("sum_binary_axes02",bench_sum,t,0,2);
//     benchmark("sum_binary_axes03",bench_sum,t,0,3);
//     benchmark("sum_binary_axes03",bench_sum,t,0,4);
//     benchmark("sum_binary_axes05",bench_sum,t,0,5);
//     benchmark("sum_binary_axes12",bench_sum,t,1,2);
//     benchmark("sum_binary_axes13",bench_sum,t,1,3);
//     benchmark("sum_binary_axes15",bench_sum,t,1,5);
//     benchmark("sum_binary_axes23",bench_sum,t,2,3);
//     benchmark("sum_binary_axes25",bench_sum,t,2,5);

//     benchmark("sum_binary_axes012",bench_sum,t,0,1,2);
//     benchmark("sum_binary_axes024",bench_sum,t,0,2,4);
//     benchmark("sum_binary_axes135",bench_sum,t,1,3,5);
//     benchmark("sum_binary_axes543",bench_sum,t,5,4,3);
// }


TEMPLATE_TEST_CASE("benchmark_reduce_bunary_sum_big","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using shape_type = typename tensor_type::shape_type;
    using order = typename tensor_type::order;
    using benchmark_helpers::benchmark;
    using benchmark_helpers::cpu_timer;
    using benchmark_helpers::order_to_str;
    using gtensor::detail::shape_to_str;
    using benchmark_helpers::axes_to_str;

    auto bench_sum = [](const auto& t_, const auto& axes){
        auto start = cpu_timer{};
        auto tmp = benchmark_expression_template_helpers::sum(t_,axes);
        auto stop = cpu_timer{};
        std::cout<<std::endl<<"sum axes "<<axes_to_str(axes)<<" "<<stop-start<<" ms";
    };

    std::vector<shape_type> shapes{
        shape_type{100000000,3,1,2},
        shape_type{10000000,3,1,20},
        shape_type{1000000,3,10,20},
        shape_type{100000,3,100,20},
        shape_type{10000,3,100,200},
        shape_type{1000,3,1000,200},
        shape_type{100,3,1000,2000},
        shape_type{50,6,1000,2000}
    };
    auto axeses = std::make_tuple(0,1,2,3,std::vector<int>{0,1},std::vector<int>{0,2},std::vector<int>{0,3},std::vector<int>{1,2},
        std::vector<int>{1,3},std::vector<int>{2,3},std::vector<int>{0,1,2},std::vector<int>{1,2,3},std::vector<int>{0,1,2,3}
    );
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t = tensor_type(*it,2);
        std::cout<<std::endl<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_sum(t,std::get<0>(axeses));
        bench_sum(t,std::get<1>(axeses));
        bench_sum(t,std::get<2>(axeses));
        bench_sum(t,std::get<3>(axeses));
        bench_sum(t,std::get<4>(axeses));
        bench_sum(t,std::get<5>(axeses));
        bench_sum(t,std::get<6>(axeses));
        bench_sum(t,std::get<7>(axeses));
        bench_sum(t,std::get<8>(axeses));
        bench_sum(t,std::get<9>(axeses));
        bench_sum(t,std::get<10>(axeses));
        bench_sum(t,std::get<11>(axeses));
        bench_sum(t,std::get<12>(axeses));
    }
}


