#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"
#include "statistic.hpp"

namespace benchmark_reduce_{

using gtensor::basic_tensor;
using gtensor::reduce_binary;
using gtensor::detail::shape_to_str;
using benchmark_helpers::order_to_str;
using benchmark_helpers::axes_to_str;
using gtensor::config::c_order;
using gtensor::config::f_order;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;

template<typename Axes, typename...Ts, typename Initial=gtensor::detail::no_value>
auto reduce_binary_sum(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false, const Initial& initial=Initial{}){
    using f_type = gtensor::math_reduce_operations::nan_propagate_operation<gtensor::math_reduce_operations::plus>;
    return reduce_binary(t,axes,f_type{},keep_dims,initial);
}



template<typename Tensor>
struct bench_reduce_helper{
    using tensor_type = Tensor;
    using layout = typename tensor_type::order;
    using shape_type = typename tensor_type::shape_type;

    template<typename Shapes, typename Axes, typename Builder, typename Reducer>
    auto operator()(std::string mes, std::size_t n_iters, Shapes shapes, Axes axes, Builder builder, Reducer reducer){
        std::cout<<std::endl<<"layout "<<order_to_str(layout{})<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto shapes_it=shapes.begin(), shapes_last=shapes.end(); shapes_it!=shapes_last; ++shapes_it){
            std::vector<double> shape_intervals{};
            const auto& shape = *shapes_it;
            shape_type t_shape{};
            bool is_t_shape{false};
            for (auto axes_it=axes.begin(), axes_last=axes.end(); axes_it!=axes_last; ++axes_it){
                std::vector<double> intervals{};
                auto ax = *axes_it;
                for (auto n=n_iters; n!=0; --n){
                    auto t_ = tensor_type(shape,2);
                    auto t=builder(t_);
                    if (!is_t_shape){
                        t_shape=t.shape();
                        is_t_shape=true;
                    }
                    //measure reduce time
                    double dt = 0;
                    dt = timing(reducer,t,ax);
                    intervals.push_back(dt);
                    shape_intervals.push_back(dt);
                    total_intervals.push_back(dt);
                }
                std::cout<<std::endl<<"input shape "<<shape_to_str(shape)<<" shape "<<shape_to_str(t_shape)<<" axes "<<axes_to_str(ax)<<" "<<statistic(intervals);
            }
            std::cout<<std::endl<<"TOTAL input shape "<<shape_to_str(shape)<<" shape "<<shape_to_str(t_shape)<<statistic(shape_intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};


template<typename Shapes, typename Axes, typename Builder, typename Reducer>
auto bench_reduce(std::string mes, std::size_t n_iters, Shapes shapes, Axes axes, Builder builder, Reducer reducer){
    using value_type = double;
    bench_reduce_helper<gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,axes,builder,reducer);
    bench_reduce_helper<gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,axes,builder,reducer);
}

}   //end of namespace benchmark_reduce_


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

    REQUIRE(benchmark_reduce_::reduce_binary_sum(tensor_type{1,2,3,4,5},std::vector<int>{0}) == tensor_type(15));
    REQUIRE(benchmark_reduce_::reduce_binary_sum(t,std::vector<int>{0}) == tensor_type{{{{18,16,14,16},{2,11,11,12},{4,16,11,14}}},{{{8,9,9,13},{9,10,10,6},{20,13,12,12}}}});
    REQUIRE(benchmark_reduce_::reduce_binary_sum(t,std::vector<int>{0,1}) == tensor_type{{{26,25,23,29},{11,21,21,18},{24,29,23,26}}});
    REQUIRE(benchmark_reduce_::reduce_binary_sum(t,std::vector<int>{0,2}) == tensor_type{{{18,16,14,16},{2,11,11,12},{4,16,11,14}},{{8,9,9,13},{9,10,10,6},{20,13,12,12}}});
    REQUIRE(benchmark_reduce_::reduce_binary_sum(t,std::vector<int>{1,2,3}) == tensor_type{{17,27,21,26},{22,22,23,28},{22,26,23,19}});
    REQUIRE(benchmark_reduce_::reduce_binary_sum(t,std::vector<int>{2,3}) == tensor_type{{{10,18,13,14},{7,9,8,12}},{{5,13,10,16},{17,9,13,12}},{{9,12,13,12},{13,14,10,7}}});
}

TEST_CASE("benchmark_reduce","[benchmark_tensor]")
{

    using benchmark_reduce_::bench_reduce;
    using benchmark_reduce_::reduce_binary_sum;

    auto reducer_reduce_binary_sum = [](const auto& t, const auto& axes){
        auto r = reduce_binary_sum(t,axes);
        return *r.begin();
    };

    auto reducer_reduce_sum = [](const auto& t, const auto& axes){
        auto r = t.sum(axes);
        return *r.begin();
    };

    const auto axes = benchmark_helpers::axes;

    // const auto n_iters = 100;
    // const auto shapes = benchmark_helpers::small_shapes;

    const auto n_iters = 1;
    const auto shapes = benchmark_helpers::shapes;

    bench_reduce("reduce sum on tensor",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_sum);
    //bench_reduce("reduce_binary sum on tensor",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_sum);
    //bench_reduce("reduce_binary sum on expression view t+t+t+t+t+t+t+t+t+t",n_iters,shapes,axes,[](auto&& t){return t+t+t+t+t+t+t+t+t+t;},reducer_reduce_binary_sum);

}

