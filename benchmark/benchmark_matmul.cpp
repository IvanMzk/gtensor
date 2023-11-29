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
#include "tensor_math.hpp"


namespace benchmark_matmul_{

using gtensor::tensor;
using gtensor::basic_tensor;
using gtensor::detail::shape_to_str;
using benchmark_helpers::order_to_str;
using gtensor::config::c_order;
using gtensor::config::f_order;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;

template<typename Tensor1, typename Tensor2>
struct bench_matmul_helper{

    using shape_type = typename Tensor1::shape_type;
    using layout1 = typename Tensor1::order;
    using layout2 = typename Tensor2::order;

    template<typename Shapes, typename Builder, typename Command>
    auto operator()(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
        std::cout<<std::endl<<"layout1 "<<order_to_str(layout1{})<<" "<<"layout2 "<<order_to_str(layout2{})<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto shapes_it=shapes.begin(), shapes_last=shapes.end(); shapes_it!=shapes_last; ++shapes_it){
            const auto& shape1 = std::get<0>(*shapes_it);
            const auto& shape2 = std::get<1>(*shapes_it);
            auto t1 = builder(Tensor1(shape1));
            auto t2 = builder(Tensor2(shape2));
            std::vector<double> shape_intervals{};
            for (auto n=n_iters; n!=0; --n){
                //measure command time
                double dt = 0;
                dt = timing(command,t1,t2);
                shape_intervals.push_back(dt);
                total_intervals.push_back(dt);
            }
            std::cout<<std::endl<<"shape1 "<<shape_to_str(shape1)<<" shape2 "<<shape_to_str(shape2)<<" "<<statistic(shape_intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

template<typename T, typename Shapes, typename Builder, typename Command>
auto bench_matmul(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
    bench_matmul_helper<gtensor::tensor<T,c_order>,gtensor::tensor<T,c_order>>{}(mes,n_iters,shapes,builder,command);
    bench_matmul_helper<gtensor::tensor<T,f_order>,gtensor::tensor<T,f_order>>{}(mes,n_iters,shapes,builder,command);
    // bench_matmul_helper<gtensor::tensor<T,c_order>,gtensor::tensor<T,f_order>>{}(mes,n_iters,shapes,builder,command);
    // bench_matmul_helper<gtensor::tensor<T,f_order>,gtensor::tensor<T,c_order>>{}(mes,n_iters,shapes,builder,command);
}

}   //end of namespace benchmark_matmul_

TEST_CASE("benchmark_matmul","[benchmark_tensor]")
{
    using helpers_for_testing::generate_lehmer;
    using benchmark_matmul_::bench_matmul;

    auto builder = [](auto&& t_){
        generate_lehmer(t_.begin(),t_.end(),[](const auto& e){return e%5;},123);
        return t_.clone_shallow();
    };

    auto shapes = std::vector<std::pair<std::vector<int>,std::vector<int>>>{
        //std::make_pair(std::vector<int>{4000,4000},std::vector<int>{4000,4000})
        //std::make_pair(std::vector<int>{6000,6000},std::vector<int>{6000,6000})
        std::make_pair(std::vector<int>{10000,10000},std::vector<int>{10000,10000})
    };
    const auto n_iters = 1;

    auto command_matmul_par = [](const auto& t1, const auto& t2){
        auto r = matmul(multithreading::exec_pol<16>{}, t1,t2);
        return std::abs(*r.begin());
    };

    bench_matmul<double>("bench matmul_par double",n_iters,shapes,builder,command_matmul_par);
    // bench_matmul<float>("bench matmul_par float",n_iters,shapes,builder,command_matmul_par);
    // bench_matmul<std::complex<double>>("bench matmul_par std::complex<double>",n_iters,shapes,builder,command_matmul_par);
    // bench_matmul<std::complex<float>>("bench matmul_par std::complex<float>",n_iters,shapes,builder,command_matmul_par);
}