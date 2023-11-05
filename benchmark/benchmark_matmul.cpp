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

template<typename Shapes, typename Builder, typename Command>
auto bench_matmul(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
    using value_type = double;
    bench_matmul_helper<gtensor::tensor<value_type,c_order>,gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,builder,command);
    bench_matmul_helper<gtensor::tensor<value_type,f_order>,gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,builder,command);
    bench_matmul_helper<gtensor::tensor<value_type,c_order>,gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,builder,command);
    bench_matmul_helper<gtensor::tensor<value_type,f_order>,gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,builder,command);
}

}   //end of namespace benchmark_matmul_

TEST_CASE("benchmark_matmul","[benchmark_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::generate_lehmer;
    using benchmark_helpers::timing;
    using benchmark_matmul_::bench_matmul;

    auto builder = [](auto&& t_){
        generate_lehmer(t_.begin(),t_.end(),[](const auto& e){return e%5;},123);
        return t_.clone_shallow();
    };

    auto command = [](const auto& t1, const auto& t2){
        auto r = matmul(t1,t2);
        return *r.begin();
    };

    auto shapes = std::vector<std::pair<std::vector<int>,std::vector<int>>>{
        //1d x nd
        std::make_pair(std::vector<int>{10000},std::vector<int>{10000,10000}),
        //nd x 1d
        std::make_pair(std::vector<int>{10000,10000},std::vector<int>{10000}),
        //nd x nd
        std::make_pair(std::vector<int>{2000,1000},std::vector<int>{1000,3000})
    };
    const auto n_iters = 10;
    bench_matmul("bench matmul",n_iters,shapes,builder,command);
}