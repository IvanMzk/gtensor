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

namespace benchmark_copy_{

using gtensor::tensor;
using gtensor::basic_tensor;
using gtensor::detail::shape_to_str;
using benchmark_helpers::order_to_str;
using gtensor::config::c_order;
using gtensor::config::f_order;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;

template<typename Tensor>
struct bench_copy_helper{

    using tensor_type = Tensor;
    using shape_type = typename tensor_type::shape_type;
    using layout = typename tensor_type::order;

    template<typename Shapes, typename Builder, typename Command>
    auto operator()(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
        std::cout<<std::endl<<"layout "<<order_to_str(layout{})<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto shapes_it=shapes.begin(), shapes_last=shapes.end(); shapes_it!=shapes_last; ++shapes_it){
            std::vector<double> shape_intervals{};
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
            std::cout<<std::endl<<"input shape "<<shape_to_str(shape)<<" shape "<<shape_to_str(t_shape)<<" "<<statistic(intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

template<typename Shapes, typename Builder, typename Command>
auto bench_copy(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Command command){
    using value_type = double;
    bench_copy_helper<gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,builder,command);
    bench_copy_helper<gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,builder,command);
}

}   //end of namespace benchmark_statistic_

TEST_CASE("benchmark_copy","[benchmark_copy]")
{
    using benchmark_copy_::bench_copy;
    using helpers_for_testing::generate_lehmer;

    auto make_eval_seq = [](const auto& t){
        auto res = t.eval();
        return *res.begin();
    };
    auto make_eval_par_8 = [](const auto& t){
        auto res = t.eval(multithreading::exec_pol<8>{});
        return *res.begin();
    };


    const auto n_iters = 1;
    const std::vector<std::vector<int>> shapes{
        std::vector<int>{100000000,3,1,2},
        std::vector<int>{1000000,3,10,20},
        std::vector<int>{10000,3,100,200},
        std::vector<int>{50,6,1000,2000}
    };

    bench_copy("eval trivial expression (((((((((t+1)+2)+3)+4)+5)+6)+7)+8)+9)",n_iters,shapes,[](auto&& t){return (((((((((t+1)+2)+3)+4)+5)+6)+7)+8)+9);},make_eval_seq);
    bench_copy("eval trivial expression (((((((((t+1)+2)+3)+4)+5)+6)+7)+8)+9) exec_pol<8>",n_iters,shapes,[](auto&& t){return (((((((((t+1)+2)+3)+4)+5)+6)+7)+8)+9);},make_eval_par_8);

    bench_copy("eval trivial expression t+t+t+t+t+t+t+t+t+t",n_iters,shapes,[](auto&& t){return t+t+t+t+t+t+t+t+t+t;},make_eval_seq);
    bench_copy("eval trivial expression t+t+t+t+t+t+t+t+t+t exec_pol<8>",n_iters,shapes,[](auto&& t){return t+t+t+t+t+t+t+t+t+t;},make_eval_par_8);

    bench_copy("eval non trivial expression t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t",n_iters,shapes,[](auto&& t){return t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t;},make_eval_seq);
    bench_copy("eval non trivial expression t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t exec_pol<8>",n_iters,shapes,[](auto&& t){return t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t;},make_eval_par_8);

}

