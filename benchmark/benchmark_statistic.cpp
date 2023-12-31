/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"
#include "statistic.hpp"

namespace benchmark_statistic_{

using gtensor::tensor;
using gtensor::basic_tensor;
using gtensor::reduce_binary;
using gtensor::detail::shape_to_str;
using benchmark_helpers::order_to_str;
using benchmark_helpers::axes_to_str;
using gtensor::config::c_order;
using gtensor::config::f_order;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;

template<typename Tensor>
struct bench_statistic_helper{

    using tensor_type = Tensor;
    using shape_type = typename tensor_type::shape_type;
    using layout = typename tensor_type::order;

    template<typename Shapes, typename Axes, typename Builder, typename Command>
    auto operator()(std::string mes, std::size_t n_iters, Shapes shapes, Axes axes, Builder builder, Command command){
        std::cout<<std::endl<<"layout "<<order_to_str(layout{})<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto shapes_it=shapes.begin(), shapes_last=shapes.end(); shapes_it!=shapes_last; ++shapes_it){
            std::vector<double> shape_intervals{};
            const auto& shape = *shapes_it;
            shape_type t_shape{};
            bool is_t_shape{false};
            auto t_ = tensor_type(shape);
            auto t=builder(t_);
            for (auto axes_it=axes.begin(), axes_last=axes.end(); axes_it!=axes_last; ++axes_it){
                std::vector<double> intervals{};
                auto ax = *axes_it;
                for (auto n=n_iters; n!=0; --n){
                    if (!is_t_shape){
                        t_shape=t.shape();
                        is_t_shape=true;
                    }
                    //measure command time
                    double dt = 0;
                    dt = timing(command,t,ax);
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

template<typename Tensor>
struct bench_statistic_flatten_helper{

    using tensor_type = Tensor;
    using shape_type = typename tensor_type::shape_type;
    using layout = typename tensor_type::order;

    template<typename Shapes, typename Builder, typename Command>
    auto operator()(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
        std::cout<<std::endl<<"layout "<<order_to_str(layout{})<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto shapes_it=shapes.begin(), shapes_last=shapes.end(); shapes_it!=shapes_last; ++shapes_it){
            const auto& shape = *shapes_it;
            shape_type t_shape{};
            bool is_t_shape{false};
            auto t_ = tensor_type(shape);
            auto t=builder(t_);
            std::vector<double> intervals{};
            for (auto n=n_iters; n!=0; --n){
                if (!is_t_shape){
                    t_shape=t.shape();
                    is_t_shape=true;
                }
                //measure command time
                double dt = 0;
                dt = timing(command,t);
                intervals.push_back(dt);
                total_intervals.push_back(dt);
            }
            std::cout<<std::endl<<"input shape "<<shape_to_str(shape)<<" shape "<<shape_to_str(t_shape)<<statistic(intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

template<typename Shapes, typename Axes, typename Builder, typename Command>
auto bench_statistic(std::string mes, std::size_t n_iters, Shapes shapes, Axes axes, Builder builder, Command command){
    using value_type = double;
    bench_statistic_helper<gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,axes,builder,command);
    bench_statistic_helper<gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,axes,builder,command);
}

template<typename Shapes, typename Builder, typename Command>
auto bench_statistic_flatten(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
    using value_type = double;
    bench_statistic_flatten_helper<gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,builder,command);
    bench_statistic_flatten_helper<gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,builder,command);
}

}   //end of namespace benchmark_statistic_

TEST_CASE("benchmark_statistic","[benchmark_tensor]")
{
    using benchmark_statistic_::bench_statistic;
    using benchmark_statistic_::bench_statistic_flatten;
    using helpers_for_testing::generate_lehmer;

    auto builder = [](auto& t_){
        generate_lehmer(t_.begin(),t_.end(),[](const auto& e){return e%5;},123);
        return t_.clone_shallow();
    };

    const auto n_iters = 10;
    const std::vector<std::vector<int>> shapes{
        std::vector<int>{10000000,3,1,2},
        std::vector<int>{100000,3,10,20},
        std::vector<int>{1000,3,100,200},
        std::vector<int>{50,6,100,2000}
    };
    const auto axes = std::vector<std::vector<int>>{
        std::vector<int>{0},
        std::vector<int>{1},
        std::vector<int>{2},
        std::vector<int>{3},
        std::vector<int>{0,1},
        std::vector<int>{2,3},
        std::vector<int>{0,1,2}
    };

    //mean
    bench_statistic_flatten("mean flatten default",n_iters,shapes,builder,[](const auto& t){auto res = gtensor::mean(t); return *res.begin();});
    bench_statistic_flatten("mean flatten exec_pol<8>",n_iters,shapes,builder,[](const auto& t){auto res = gtensor::mean(multithreading::exec_pol<8>{}, t); return *res.begin();});
    bench_statistic("mean over axes default",n_iters,shapes,axes,builder,[](const auto& t, const auto& axes){auto res = gtensor::mean(t,axes); return *res.begin();});
    bench_statistic("mean over axes exec_pol<8>",n_iters,shapes,axes,builder,[](const auto& t, const auto& axes){auto res = gtensor::mean(multithreading::exec_pol<8>{},t,axes); return *res.begin();});

    //stdev
    bench_statistic_flatten("stdev flatten default",n_iters,shapes,builder,[](const auto& t){auto res = stdev(t); return *res.begin();});
    bench_statistic_flatten("stdev flatten exec_pol<8>",n_iters,shapes,builder,[](const auto& t){auto res = stdev(multithreading::exec_pol<8>{}, t); return *res.begin();});
    bench_statistic("stdev over axes default",n_iters,shapes,axes,builder,[](const auto& t, const auto& axes){auto res = stdev(t,axes); return *res.begin();});
    bench_statistic("stdev over axes exec_pol<8>",n_iters,shapes,axes,builder,[](const auto& t, const auto& axes){auto res = stdev(multithreading::exec_pol<8>{},t,axes); return *res.begin();});
}

