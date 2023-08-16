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

template<typename Axes, typename...Ts, typename Initial=gtensor::detail::no_value>
auto nansum(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false, const Initial& initial=Initial{}){
    using f_type = gtensor::math_reduce_operations::nan_ignoring_operation<gtensor::math_reduce_operations::plus>;
    return reduce_binary(t,axes,f_type{},keep_dims,initial);
}

template<typename Axes, typename...Ts>
auto mean(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    using value_type = typename basic_tensor<Ts...>::value_type;
    using res_type = gtensor::math::make_floating_point_t<value_type>;
    using f_type = gtensor::math_reduce_operations::nan_propagate_operation<std::plus<res_type>>;
    auto sum = reduce_binary(t,axes,f_type{},keep_dims);
    const auto axes_size = t.size() / sum.size();
    return sum/axes_size;
}

// template<typename Axes, typename...Ts>
// auto var(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
//     auto square = [](const auto& e){return e*e;};
//     auto mean_ = benchmark_expression_template_helpers::mean(t,axes,true);
//     auto tmp = gtensor::n_operator(square,t-std::move(mean_)).copy();
//     return benchmark_expression_template_helpers::mean(tmp,axes,keep_dims);
// }

template<typename Axes, typename...Ts>
auto var(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    using value_type = typename basic_tensor<Ts...>::value_type;
    using res_type = gtensor::math::make_floating_point_t<value_type>;
    using f_type = gtensor::math_reduce_operations::nan_propagate_operation<std::plus<res_type>>;
    auto square = [](const auto& e){return e*e;};
    auto sum = reduce_binary(t,axes,f_type{},keep_dims);
    auto sum_of_squared = reduce_binary(gtensor::n_operator(square,t).copy(),axes,f_type{},keep_dims);
    const auto axes_size = t.size() / sum.size();
    const auto axes_size_2 = axes_size*axes_size;
    auto res = ((axes_size*std::move(sum_of_squared) - gtensor::n_operator(square,std::move(sum)))/axes_size_2);
    return res.copy();
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

    REQUIRE(benchmark_expression_template_helpers::nansum(tensor_type{1,2,3,4,5},std::vector<int>{0}) == tensor_type(15));
    REQUIRE(benchmark_expression_template_helpers::nansum(t,std::vector<int>{0}) == tensor_type{{{{18,16,14,16},{2,11,11,12},{4,16,11,14}}},{{{8,9,9,13},{9,10,10,6},{20,13,12,12}}}});
    REQUIRE(benchmark_expression_template_helpers::nansum(t,std::vector<int>{0,1}) == tensor_type{{{26,25,23,29},{11,21,21,18},{24,29,23,26}}});
    REQUIRE(benchmark_expression_template_helpers::nansum(t,std::vector<int>{0,2}) == tensor_type{{{18,16,14,16},{2,11,11,12},{4,16,11,14}},{{8,9,9,13},{9,10,10,6},{20,13,12,12}}});
    REQUIRE(benchmark_expression_template_helpers::nansum(t,std::vector<int>{1,2,3}) == tensor_type{{17,27,21,26},{22,22,23,28},{22,26,23,19}});
    REQUIRE(benchmark_expression_template_helpers::nansum(t,std::vector<int>{2,3}) == tensor_type{{{10,18,13,14},{7,9,8,12}},{{5,13,10,16},{17,9,13,12}},{{9,12,13,12},{13,14,10,7}}});
}

TEMPLATE_TEST_CASE("test_reduce_binary_mean","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor_close;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    auto t = tensor_type{{{{{7,5,8,5},{0,5,5,1},{3,8,0,8}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}}}},{{{{4,8,0,7},{0,0,2,4},{1,5,8,5}}},{{{6,8,4,8},{4,1,3,2},{7,0,6,2}}}},{{{{7,3,6,4},{2,6,4,7},{0,3,3,1}}},{{{2,1,3,0},{4,7,4,4},{7,6,3,3}}}}};

    REQUIRE(tensor_close(benchmark_expression_template_helpers::mean(t,std::vector<int>{1}), tensor_type{{{{3.5,2.5,5.0,5.0},{0.5,3.5,4.0,0.5},{4.5,7.5,1.5,7.5}}},{{{5.0,8.0,2.0,7.5},{2.0,0.5,2.5,3.0},{4.0,2.5,7.0,3.5}}},{{{4.5,2.0,4.5,2.0},{3.0,6.5,4.0,5.5},{3.5,4.5,3.0,2.0}}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(benchmark_expression_template_helpers::mean(t,std::vector<int>{0,1}), tensor_type{{{4.333,4.166,3.833,4.833},{1.833,3.5,3.5,3.0},{4.0,4.833,3.833,4.333}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(benchmark_expression_template_helpers::mean(t,std::vector<int>{0,2}), tensor_type{{{6.0,5.333,4.667,5.333},{0.667,3.667,3.667,4.0},{1.333,5.333,3.667,4.667}},{{2.667,3.0,3.0,4.333},{3.0,3.333,3.333,2.0},{6.667,4.333,4.0,4.0}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(benchmark_expression_template_helpers::mean(t,std::vector<int>{1,2,3}), tensor_type{{2.833,4.5,3.5,4.333},{3.667,3.667,3.833,4.667},{3.667,4.333,3.833,3.167}}, 1E-2,1E-2));
    REQUIRE(tensor_close(benchmark_expression_template_helpers::mean(t,std::vector<int>{2,3}), tensor_type{{{3.333,6.0,4.333,4.667},{2.333,3.0,2.667,4.0}},{{1.667,4.333,3.333,5.333},{5.667,3.0,4.333,4.0}},{{3.0,4.0,4.333,4.0},{4.333,4.667,3.333,2.333}}},1E-2,1E-2));
}

TEMPLATE_TEST_CASE("test_reduce_binary_var","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor_close;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    auto t = tensor_type{{{{{7,5,8,5},{0,5,5,1},{3,8,0,8}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}}}},{{{{4,8,0,7},{0,0,2,4},{1,5,8,5}}},{{{6,8,4,8},{4,1,3,2},{7,0,6,2}}}},{{{{7,3,6,4},{2,6,4,7},{0,3,3,1}}},{{{2,1,3,0},{4,7,4,4},{7,6,3,3}}}}};

    REQUIRE(tensor_close(benchmark_expression_template_helpers::var(t,std::vector<int>{1}), tensor_type{{{{12.25,6.25,9.0,0.0},{0.25,2.25,1.0,0.25},{2.25,0.25,2.25,0.25}}},{{{1.0,0.0,4.0,0.25},{4.0,0.25,0.25,1.0},{9.0,6.25,1.0,2.25}}},{{{6.25,1.0,2.25,4.0},{1.0,0.25,0.0,2.25},{12.25,2.25,0.0,1.0}}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(benchmark_expression_template_helpers::var(t,std::vector<int>{0,1}), tensor_type{{{6.889,9.806,6.806,6.472},{2.806,6.917,0.917,5.333},{8.0,7.139,6.472,6.556}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(benchmark_expression_template_helpers::var(t,std::vector<int>{0,2}), tensor_type{{{2.0,4.222,11.556,1.556},{0.889,6.889,1.556,6.0},{1.556,4.222,10.889,8.222}},{{6.222,12.667,0.667,10.889},{2.0,6.889,0.222,2.667},{0.222,9.556,2.0,4.667}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(benchmark_expression_template_helpers::var(t,std::vector<int>{1,2,3}), tensor_type{{7.806,7.583,6.25,8.556},{6.222,12.222,6.806,5.222},{6.889,4.556,1.139,5.139}}, 1E-2,1E-2));
    REQUIRE(tensor_close(benchmark_expression_template_helpers::var(t,std::vector<int>{2,3}), tensor_type{{{8.222,2.0,10.889,8.222},{6.889,8.667,0.222,8.667}},{{2.889,10.889,11.556,1.556},{1.556,12.667,1.556,8.0}},{{8.667,2.0,1.556,6.0},{4.222,6.889,0.222,2.889}}},1E-2,1E-2));
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


// TEMPLATE_TEST_CASE("benchmark_reduce_bunary_sum_big","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using shape_type = typename tensor_type::shape_type;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::benchmark;
//     using benchmark_helpers::cpu_timer;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::axes_to_str;

//     auto bench_sum = [](const auto& t_, const auto& axes){
//         auto start = cpu_timer{};
//         auto t_copy = t_.copy(order{});
//         //auto tmp = benchmark_expression_template_helpers::sum(t_copy,axes);
//         auto stop = cpu_timer{};
//         std::cout<<std::endl<<"sum axes "<<axes_to_str(axes)<<" "<<stop-start<<" ms";
//         return t_copy.size();
//     };

//     std::vector<shape_type> shapes{
//         shape_type{100000000,3,1,2},
//         shape_type{10000000,3,1,20},
//         shape_type{1000000,3,10,20},
//         shape_type{100000,3,100,20},
//         shape_type{10000,3,100,200},
//         shape_type{1000,3,1000,200},
//         shape_type{100,3,1000,2000},
//         shape_type{50,6,1000,2000}
//     };
//     auto axeses = std::make_tuple(0,1,2,3,std::vector<int>{0,1},std::vector<int>{0,2},std::vector<int>{0,3},std::vector<int>{1,2},
//         std::vector<int>{1,3},std::vector<int>{2,3},std::vector<int>{0,1,2},std::vector<int>{1,2,3},std::vector<int>{0,1,2,3}
//     );
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t = 2*t_+3*t_*(2*t_+1)+4*(t_+2)*(t_-2)*(t_+3);
//         std::cout<<std::endl<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_sum(t,std::get<0>(axeses));
//         bench_sum(t,std::get<1>(axeses));
//         bench_sum(t,std::get<2>(axeses));
//         bench_sum(t,std::get<3>(axeses));
//         bench_sum(t,std::get<4>(axeses));
//         bench_sum(t,std::get<5>(axeses));
//         bench_sum(t,std::get<6>(axeses));
//         bench_sum(t,std::get<7>(axeses));
//         bench_sum(t,std::get<8>(axeses));
//         bench_sum(t,std::get<9>(axeses));
//         bench_sum(t,std::get<10>(axeses));
//         bench_sum(t,std::get<11>(axeses));
//         bench_sum(t,std::get<12>(axeses));
//     }
// }


TEMPLATE_TEST_CASE("benchmark_reduce_binary_var","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using order = typename tensor_type::order;
    using benchmark_helpers::benchmark;
    using benchmark_helpers::cpu_timer;
    using benchmark_helpers::order_to_str;
    using gtensor::detail::shape_to_str;
    using benchmark_helpers::axes_to_str;
    using benchmark_helpers::shapes;
    using helpers_for_testing::apply_by_element;


    auto bench_reduce_binary_var = [](const auto& t_, const auto& axes, auto mes){
        auto start = cpu_timer{};
        auto tmp = benchmark_expression_template_helpers::var(t_,axes);
        //auto tmp = gtensor::var(t_,axes);
        auto stop = cpu_timer{};
        std::cout<<std::endl<<mes<<" "<<axes_to_str(axes)<<" "<<stop-start<<" ms";
        return tmp.size();
    };

    auto axeses = std::make_tuple(0,1,2,3,std::vector<int>{0,1},std::vector<int>{0,2},std::vector<int>{0,3},std::vector<int>{1,2},
        std::vector<int>{1,3},std::vector<int>{2,3},std::vector<int>{0,1,2},std::vector<int>{1,2,3},std::vector<int>{0,1,2,3}
    );
    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t = tensor_type(*it,2);
        std::cout<<std::endl<<"tensor "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_reduce_binary_var(t,std::get<0>(axeses),"reduce_binary var");
        bench_reduce_binary_var(t,std::get<1>(axeses),"reduce_binary var");
        bench_reduce_binary_var(t,std::get<2>(axeses),"reduce_binary var");
        bench_reduce_binary_var(t,std::get<3>(axeses),"reduce_binary var");
        bench_reduce_binary_var(t,std::get<4>(axeses),"reduce_binary var");
        bench_reduce_binary_var(t,std::get<5>(axeses),"reduce_binary var");
        bench_reduce_binary_var(t,std::get<6>(axeses),"reduce_binary var");
        bench_reduce_binary_var(t,std::get<7>(axeses),"reduce_binary var");
        bench_reduce_binary_var(t,std::get<8>(axeses),"reduce_binary var");
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total,ms "<<stop-start;
}


