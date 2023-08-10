#include "benchmark_helpers.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"
#include "statistic.hpp"

namespace benchmark_expression_template_helpers{

using gtensor::basic_tensor;
using gtensor::tensor;

template<typename Axes, typename...Ts>
auto reduce_sum_(const basic_tensor<Ts...>& parent, const Axes& axes_, bool keep_dims=false){
    using parent_type = basic_tensor<Ts...>;
    using order = typename parent_type::order;
    using config_type = typename parent_type::config_type;
    using value_type = typename parent_type::value_type;
    using traverse_order = typename parent_type::traverse_order;
    const auto pdim = parent.dim();
    const auto& pshape = parent.shape();
    auto axes = gtensor::detail::make_axes<config_type>(pdim,axes_);
    gtensor::detail::check_reduce_args(pshape, axes);
    auto res = tensor<value_type,order,config_type>{gtensor::detail::make_reduce_shape(pshape, axes, keep_dims)};
    if (!res.empty()){
        auto axes_iterator_maker = gtensor::detail::make_axes_iterator_maker<config_type>(pshape,axes,traverse_order{});
        auto traverser = axes_iterator_maker.create_forward_traverser(parent.create_walker(),std::true_type{});
        auto a = res.traverse_order_adapter(order{});
        auto res_it = a.begin();
        do{
            *res_it = std::accumulate(
                axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
                axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
                value_type{0}
            );
            ++res_it;
        }while(traverser.template next<order>());
    }
    return res;
}



template<typename Axes, typename...Ts>
auto reduce_sum(const basic_tensor<Ts...>& parent, const Axes& axes_, bool keep_dims=false){
    return reduce_sum_(parent,axes_,keep_dims);
}

template<typename...Ts, typename U>
auto reduce_sum(const basic_tensor<Ts...>& parent, std::initializer_list<U> axes_, bool keep_dims=false){
    return reduce_sum_(parent,axes_,keep_dims);
}

}   //end of namespace benchmark_expression_template_helpers


// TEMPLATE_TEST_CASE("test_reduce_sum","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     //using config_type = typename tensor_type::config_type;
//     //using order = typename tensor_type::order;
//     using benchmark_expression_template_helpers::reduce_sum;

//     auto t = tensor_type{{{{{7,5,8,5},{0,5,5,1},{3,8,0,8}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}}}},{{{{4,8,0,7},{0,0,2,4},{1,5,8,5}}},{{{6,8,4,8},{4,1,3,2},{7,0,6,2}}}},{{{{7,3,6,4},{2,6,4,7},{0,3,3,1}}},{{{2,1,3,0},{4,7,4,4},{7,6,3,3}}}}};

//     REQUIRE(reduce_sum(tensor_type{1,2,3,4,5},std::vector<int>{0}) == tensor_type(15));
//     REQUIRE(reduce_sum(t,std::vector<int>{0}) == tensor_type{{{{18,16,14,16},{2,11,11,12},{4,16,11,14}}},{{{8,9,9,13},{9,10,10,6},{20,13,12,12}}}});
//     REQUIRE(reduce_sum(t,std::vector<int>{0,1}) == tensor_type{{{26,25,23,29},{11,21,21,18},{24,29,23,26}}});
//     REQUIRE(reduce_sum(t,std::vector<int>{0,2}) == tensor_type{{{18,16,14,16},{2,11,11,12},{4,16,11,14}},{{8,9,9,13},{9,10,10,6},{20,13,12,12}}});
//     REQUIRE(reduce_sum(t,std::vector<int>{1,2,3}) == tensor_type{{17,27,21,26},{22,22,23,28},{22,26,23,19}});
//     REQUIRE(reduce_sum(t,std::vector<int>{2,3}) == tensor_type{{{10,18,13,14},{7,9,8,12}},{{5,13,10,16},{17,9,13,12}},{{9,12,13,12},{13,14,10,7}}});
// }

// TEMPLATE_TEST_CASE("benchmark_axes_iterator","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using config_type = typename tensor_type::config_type;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::make_asymmetric_tree;
//     using benchmark_helpers::benchmark;

//     auto t = tensor_type({50,50,50,50},2);
//     auto axes_01 = gtensor::detail::make_axes<config_type>(t.dim(),std::vector<int>{0,1});
//     auto axes_23 = gtensor::detail::make_axes<config_type>(t.dim(),std::vector<int>{2,3});
//     auto axes_iterator_maker_01 = gtensor::detail::make_axes_iterator_maker<config_type>(t.shape(),axes_01,order{});
//     auto axes_iterator_maker_23 = gtensor::detail::make_axes_iterator_maker<config_type>(t.shape(),axes_23,order{});

//     auto bench_iteration_deref = [](const auto& t, const auto& it_maker){
//         value_type v{0};
//         auto it = it_maker.begin(t.create_walker());
//         auto last = it_maker.end(t.create_walker());
//         for (;it!=last; ++it){
//             v += *it;
//         }
//         return v;
//     };

//     benchmark("bench_axes_iterator_01",bench_iteration_deref,t,axes_iterator_maker_01);
//     benchmark("bench_axes_iterator_23",bench_iteration_deref,t,axes_iterator_maker_23);

// }

// TEMPLATE_TEST_CASE("benchmark_reduce_stdev","[benchmark_tensor]",
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
//         if constexpr (sizeof...(axes) == 1){
//             auto tmp = gtensor::std(t,axes...);
//             return tmp.size();
//         }else{
//             auto tmp = gtensor::std(t,{axes...});
//             return tmp.size();
//         }
//     };

//     //auto t = tensor_type({10,5,100,1000},2);
//     auto t = tensor_type({50,50,50,50},2);

//     //like over flatten
//     //benchmark("std_all_10E6",bench_sum,t);

//     //single axis
//     // benchmark("std_axis_10E6",bench_sum,t,0);
//     // benchmark("std_axis_10E6",bench_sum,t,1);
//     // benchmark("std_axis_10E6",bench_sum,t,2);
//     // benchmark("std_axis_10E6",bench_sum,t,3);

//     //axes
//     benchmark("std_axes2_10E6",bench_sum,t,0,1);
//     benchmark("std_axes2_10E6",bench_sum,t,0,2);
//     benchmark("std_axes2_10E6",bench_sum,t,0,3);
//     benchmark("std_axes2_10E6",bench_sum,t,1,2);
//     benchmark("std_axes2_10E6",bench_sum,t,1,3);
//     benchmark("std_axes2_10E6",bench_sum,t,2,3);


//     // benchmark("std_axes2_10E6",bench_sum,t,0,2);
//     // benchmark("std_axes3_10E6",bench_sum,t,0,1,2);
//     // benchmark("std_axes4_10E6",bench_sum,t,0,1,2,3);
// }


// TEMPLATE_TEST_CASE("benchmark_tensor_sum","[benchmark_tensor]",
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
//         auto tmp = t.sum({axes...});
//         return tmp.size();
//     };

//     //auto t = tensor_type({50,5,10,10,5,50},2);
//     //auto t = tensor_type({50,50,50,50},2);
//     auto t = tensor_type({500,3,1000,2000},2);
//     //auto t = tensor_type({100000000},2);

//     //like over flatten
//     //benchmark("sum_all_10E6",bench_sum,t);

//     //single axis
//     benchmark("sum_axis_10E6",bench_sum,t,0);
//     //benchmark("sum_axis_10E6",bench_sum,t,1);
//     //benchmark("sum_axis_10E6",bench_sum,t,2);
//     // benchmark("sum_axis_10E6",bench_sum,t,3);
//     // benchmark("sum_axis_10E6",bench_sum,t,4);
//     // benchmark("sum_axis_10E6",bench_sum,t,5);

//     //axes
//     // benchmark("sum_axes01_10E6",bench_sum,t,0,1);
//     // benchmark("sum_axes02_10E6",bench_sum,t,0,2);
//     // benchmark("sum_axes03_10E6",bench_sum,t,0,3);
//     // benchmark("sum_axes12_10E6",bench_sum,t,1,2);
//     // benchmark("sum_axes13_10E6",bench_sum,t,1,3);
//     // benchmark("sum_axes23_10E6",bench_sum,t,2,3);

//     // benchmark("sum_axes012_10E6",bench_sum,t,0,1,2);
//     // benchmark("sum_axes024_10E6",bench_sum,t,0,2,4);
//     // benchmark("sum_axes135_10E6",bench_sum,t,1,3,5);
//     // benchmark("sum_axes543_10E6",bench_sum,t,5,4,3);
// }


TEMPLATE_TEST_CASE("benchmark_tensor_big","[benchmark_tensor]",
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
    using benchmark_helpers::benchmark;
    using benchmark_helpers::cpu_timer;
    using benchmark_helpers::order_to_str;
    using gtensor::detail::shape_to_str;

    const auto axis=0;
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
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t = tensor_type(*it,2);
        std::cout<<std::endl<<order_to_str(typename tensor_type::order{})<<" "<<shape_to_str(t.shape())<<" axes "<<axis;
        //mean
        {
            auto start = cpu_timer{};
            auto t_mean = t.mean(axis);
            auto stop = cpu_timer{};
            std::cout<<std::endl<<"mean "<<stop-start<<" ms";
        }
        //std
        {
            auto start = cpu_timer{};
            auto t_std = t.stdev(axis);
            auto stop = cpu_timer{};
            std::cout<<std::endl<<"std "<<stop-start<<" ms";
        }
    }
}
