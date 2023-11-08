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
    //bench_matmul_helper<gtensor::tensor<value_type,f_order>,gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,builder,command);
    //bench_matmul_helper<gtensor::tensor<value_type,c_order>,gtensor::tensor<value_type,f_order>>{}(mes,n_iters,shapes,builder,command);
    //bench_matmul_helper<gtensor::tensor<value_type,f_order>,gtensor::tensor<value_type,c_order>>{}(mes,n_iters,shapes,builder,command);
}

template<typename...Ts, typename...Us>
auto matmul_2d(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
    using tensor_type1 = basic_tensor<Ts...>;
    using tensor_type2 = basic_tensor<Us...>;
    using value_type1 = typename tensor_type1::value_type;
    using value_type2 = typename tensor_type2::value_type;
    using order1 = typename basic_tensor<Ts...>::order;
    using order2 = typename basic_tensor<Us...>::order;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using res_order = std::conditional_t<std::is_same_v<order1,order2>,order1,gtensor::config::c_order>;
    using config_type = typename tensor_type1::config_type;
    using res_type = gtensor::detail::tensor_copy_type_t<std::decay_t<decltype(std::declval<value_type1>()*std::declval<value_type2>())>,res_order,config_type>;
    using shape_type = typename res_type::shape_type;
    const auto& shape1 = t1.shape();
    const auto& shape2 = t2.shape();
    res_type res({shape1[0],shape2[1]},0);
    const auto& res_shape = res.shape();
    const auto res_dim = gtensor::detail::make_dim(res_shape);
    const auto k = shape1[1];

    auto matmul_same_order = [k](auto res_it, auto it1, auto it2, const auto& m, const auto& n){
        for (auto res_last=res_it+m*n; res_it!=res_last; res_it+=n){
            for (auto [last1_,it2_]=std::make_tuple(it1+k,it2); it1!=last1_; ++it1){
                const auto e1 = *it1;
                for(auto [last2_,res_it_]=std::make_tuple(it2_+n,res_it); it2_!=last2_; ++it2_,++res_it_){
                    *res_it_+=e1**it2_;
                }
            }
        }
    };

    if constexpr (std::is_same_v<order1,order2>){
        auto a1 = t1.traverse_order_adapter(order1{});
        auto a2 = t2.traverse_order_adapter(order2{});
        auto a_res = res.traverse_order_adapter(res_order{});
        if constexpr (std::is_same_v<order1,c_order>){
            matmul_same_order(a_res.begin(),a1.begin(),a2.begin(),res_shape[0],res_shape[1]);
        }else{
            matmul_same_order(a_res.begin(),a2.begin(),a1.begin(),res_shape[1],res_shape[0]);
        }
    }else{
        if constexpr (std::is_same_v<order1,c_order>){
        }else{
        }
    }
    return res;
}

template<typename...Ts, typename...Us>
auto matmul_2d_tiled(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
    using tensor_type1 = basic_tensor<Ts...>;
    using tensor_type2 = basic_tensor<Us...>;
    using value_type1 = typename tensor_type1::value_type;
    using value_type2 = typename tensor_type2::value_type;
    using order1 = typename basic_tensor<Ts...>::order;
    using order2 = typename basic_tensor<Us...>::order;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using res_order = std::conditional_t<std::is_same_v<order1,order2>,order1,gtensor::config::c_order>;
    using config_type = typename tensor_type1::config_type;
    using res_type = gtensor::detail::tensor_copy_type_t<std::decay_t<decltype(std::declval<value_type1>()*std::declval<value_type2>())>,res_order,config_type>;
    using shape_type = typename res_type::shape_type;
    using index_type = typename res_type::index_type;
    const auto& shape1 = t1.shape();
    const auto& shape2 = t2.shape();
    res_type res({shape1[0],shape2[1]},0);
    const auto& res_shape = res.shape();
    const auto res_dim = gtensor::detail::make_dim(res_shape);
    const auto k = shape1[1];
    // const index_type k_ = 8;
    // const index_type n_ = 8;
    const index_type k_ = 16;
    const index_type n_ = 128;

    auto matmul_same_order = [k,k_](auto res_it, auto it1, auto it2, const auto& m, const auto& n, const auto& n_){

        for (index_type kk=0; kk<k; kk+=k_){
            for (index_type jj=0; jj<n; jj+=n_){
                //for (index_type i=0; i!=m; ++i){
                for (auto [res_it_, it1_, res_last_] = std::make_tuple(res_it,it1,res_it+m*n); res_it_!=res_last_; res_it_+=n,it1_+=k){
                    //for (auto r=kk; r!=std::min(kk+k_,k); ++r){
                    for (auto [it1__, it1_last__, it2_] = std::make_tuple(it1_+kk,it1_+std::min(kk+k_,k),it2+kk*n); it1__!=it1_last__; ++it1__,it2_+=n){
                        //const auto e1 = it1[i*k+r];
                        const auto e1 = *it1__;
                        //for(auto j=jj; j!=std::min(jj+n_,n); ++j){
                        for(auto [res_it__,res_last__,it2__] = std::make_tuple(res_it_+jj,res_it_+std::min(jj+n_,n),it2_+jj); res_it__!=res_last__; ++res_it__,++it2__){
                            //res_it[i*n+j]+=e1*it2[r*n+j];
                            *res_it__+=e1**it2__;
                        }
                    }
                }
            }
        }

        // for (index_type kk=0; kk<k; kk+=k_){
        //     for (index_type jj=0; jj<n; jj+=n_){
        //         for (index_type i=0; i!=m; ++i){
        //             for (auto r=kk; r!=std::min(kk+k_,k); ++r){
        //                 const auto e1 = it1[i*k+r];
        //                 for(auto j=jj; j!=std::min(jj+n_,n); ++j){
        //                     res_it[i*n+j]+=e1*it2[r*n+j];
        //                 }
        //             }
        //         }
        //     }
        // }

    };

    if constexpr (std::is_same_v<order1,order2>){
        auto a1 = t1.traverse_order_adapter(order1{});
        auto a2 = t2.traverse_order_adapter(order2{});
        auto a_res = res.traverse_order_adapter(res_order{});
        if constexpr (std::is_same_v<order1,c_order>){
            matmul_same_order(a_res.begin(),a1.begin(),a2.begin(),res_shape[0],res_shape[1],n_);
        }else{
            matmul_same_order(a_res.begin(),a2.begin(),a1.begin(),res_shape[1],res_shape[0],n_);
        }
    }else{
        if constexpr (std::is_same_v<order1,c_order>){
        }else{
        }
    }
    return res;
}


}   //end of namespace benchmark_matmul_

TEST_CASE("benchmark_matmul","[benchmark_tensor]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using helpers_for_testing::generate_lehmer;
    using benchmark_helpers::timing;
    using benchmark_matmul_::bench_matmul;

    auto builder = [](auto&& t_){
        generate_lehmer(t_.begin(),t_.end(),[](const auto& e){return e%5;},123);
        return t_.clone_shallow();
    };

    auto command_matmul = [](const auto& t1, const auto& t2){
        auto r = matmul(t1,t2);
        return *r.begin();
    };

    auto shapes = std::vector<std::pair<std::vector<int>,std::vector<int>>>{
        // //1d x nd
        // std::make_pair(std::vector<int>{10000},std::vector<int>{10000,10000}),
        // //nd x 1d
        // std::make_pair(std::vector<int>{10000,10000},std::vector<int>{10000}),
        // //nd x nd
        std::make_pair(std::vector<int>{1000,1000},std::vector<int>{1000,1000})
        //std::make_pair(std::vector<int>{4000,4000},std::vector<int>{4000,4000})
        //std::make_pair(std::vector<int>{2000,1000},std::vector<int>{1000,3000}),
        //std::make_pair(std::vector<int>{10000,10000},std::vector<int>{10000,10000})
        //std::make_pair(std::vector<int>{3,2,300,1000},std::vector<int>{2,1000,900})
        //std::make_pair(std::vector<int>{100,100,200,100},std::vector<int>{100,100,300})
    };
    const auto n_iters = 10;
    //bench_matmul("bench matmul",n_iters,shapes,builder,command_matmul);


    using benchmark_matmul_::matmul_2d;
    using benchmark_matmul_::matmul_2d_tiled;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    auto aa = tensor<double>{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3,2,4,3,4,0,2,3,4,1,0,0,1,2,0,4,0,2,3,2,4},{4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2,2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0},{1,1,1,0,2,0,3,3,1,4,3,1,1,3,1,1,4,2,1,2,3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0,0,0,3,1,1,2,2,4,1,3,0,1,2,0,3,1,3,1,3,2},{4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4,1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2},{3,3,1,2,2,1,4,0,1,4,0,2,3,4,1,3,0,0,1,3,4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4,4,0,0,4,0,4,2,4,4,0,4,1,1,4,3,1,0,3,2,0},{0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4,1,0,1,3,2,1,1,2,0,3,1,3,2,0,2,2,1,3,3,1},{0,1,1,1,0,2,1,2,1,4,0,0,3,0,2,3,4,0,3,0,4,1,4,3,2,3,2,1,3,3,3,0,4,3,4,1,0,1,4,2},{3,4,1,1,1,2,1,3,3,1,3,3,2,2,1,2,1,0,4,3,0,4,3,1,0,3,2,4,3,3,0,1,2,1,3,2,3,2,3,3},{0,4,1,3,4,4,3,1,1,3,1,4,0,1,1,4,0,2,0,4,4,3,2,3,0,3,4,4,2,2,4,1,3,0,2,4,0,1,3,3},{0,2,4,1,2,0,2,1,0,4,1,0,3,4,3,2,0,1,3,1,3,3,2,4,1,0,0,1,4,4,4,3,3,2,0,4,3,4,3,4},{0,3,3,3,2,2,4,2,4,4,0,0,1,0,4,0,0,1,3,3,0,2,2,0,2,4,1,1,3,4,4,2,2,3,2,2,0,3,1,0},{0,0,2,3,3,0,3,0,1,4,2,3,3,0,2,4,3,1,2,1,4,1,0,3,1,4,2,3,3,4,0,1,4,2,4,4,1,4,2,0},{3,0,1,4,1,3,1,3,0,3,3,3,3,4,3,3,2,4,0,2,1,1,2,1,0,1,0,2,2,0,3,1,3,1,3,2,1,3,4,3},{4,4,1,0,2,3,1,3,2,1,1,0,2,0,2,3,0,1,2,2,4,1,3,3,4,4,2,4,0,3,2,4,4,0,2,2,3,2,1,0},{1,1,4,1,2,4,0,4,0,4,4,1,0,0,3,4,4,1,2,2,4,0,2,3,2,0,2,2,4,4,3,0,4,0,4,2,3,0,0,0},{2,1,2,4,2,4,1,4,3,0,0,0,2,1,4,1,1,0,4,1,1,4,4,3,3,0,2,2,1,4,3,3,4,3,2,0,2,3,1,4},{1,2,1,4,1,1,3,4,2,4,4,3,3,0,0,1,3,3,1,4,1,2,2,3,3,3,1,0,1,1,3,1,4,1,0,1,1,2,0,0},{2,4,3,1,3,1,1,1,2,3,0,1,4,4,2,4,1,2,0,0,0,3,1,2,3,3,4,3,1,1,1,1,0,1,0,1,0,2,4,3},{0,2,0,0,1,1,3,2,0,0,3,4,4,3,4,4,2,1,2,4,4,1,3,0,2,0,1,0,1,2,4,1,3,1,4,4,0,1,1,0},{1,0,3,1,3,4,3,4,1,0,0,1,4,4,1,4,4,4,0,3,2,4,4,0,2,1,4,3,0,2,0,3,2,0,0,2,0,1,0,3},{1,1,2,0,0,4,0,2,3,0,2,1,1,4,0,0,1,4,3,1,1,4,4,2,1,0,1,4,1,3,3,3,0,2,1,3,1,4,4,2},{4,0,2,4,4,3,1,1,2,2,1,2,0,2,4,1,4,1,2,0,3,3,1,2,3,3,2,0,2,0,0,4,2,1,4,0,4,2,4,2},{3,3,3,4,2,0,1,1,4,3,0,4,1,4,1,4,2,3,0,1,1,3,0,3,1,1,1,4,4,3,0,0,4,3,2,2,4,4,2,3},{1,4,2,0,3,3,3,2,1,2,1,1,0,0,0,3,3,4,1,0,1,1,3,2,3,1,1,2,2,3,3,1,3,0,2,0,3,0,3,4},{0,0,4,2,2,2,3,2,1,0,0,3,4,4,2,1,2,4,1,2,2,2,1,4,0,2,0,1,2,2,1,3,4,3,3,0,3,0,4,3},{0,2,3,3,3,2,1,0,3,1,0,4,0,4,2,2,2,4,1,1,4,1,1,4,4,0,3,2,3,3,4,0,2,1,4,0,1,1,2,3},{3,1,3,2,4,1,4,1,3,0,3,1,2,0,1,1,1,2,4,3,0,2,0,4,4,0,0,1,0,2,0,0,4,4,0,0,3,4,1,4},{0,4,1,2,2,0,3,1,3,4,1,2,1,0,4,4,0,4,0,0,0,3,0,4,4,3,0,3,3,0,2,0,1,3,4,2,3,3,4,3}};
    auto bb = tensor<double>{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3},{2,4,3,4,0,2,3,4,1,0,0,1,2,0,4,0,2,3,2,4},{4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2},{2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0},{1,1,1,0,2,0,3,3,1,4,3,1,1,3,1,1,4,2,1,2},{3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0},{0,0,3,1,1,2,2,4,1,3,0,1,2,0,3,1,3,1,3,2},{4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4},{1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2},{3,3,1,2,2,1,4,0,1,4,0,2,3,4,1,3,0,0,1,3},{4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4},{4,0,0,4,0,4,2,4,4,0,4,1,1,4,3,1,0,3,2,0},{0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4},{1,0,1,3,2,1,1,2,0,3,1,3,2,0,2,2,1,3,3,1},{0,1,1,1,0,2,1,2,1,4,0,0,3,0,2,3,4,0,3,0},{4,1,4,3,2,3,2,1,3,3,3,0,4,3,4,1,0,1,4,2},{3,4,1,1,1,2,1,3,3,1,3,3,2,2,1,2,1,0,4,3},{0,4,3,1,0,3,2,4,3,3,0,1,2,1,3,2,3,2,3,3},{0,4,1,3,4,4,3,1,1,3,1,4,0,1,1,4,0,2,0,4},{4,3,2,3,0,3,4,4,2,2,4,1,3,0,2,4,0,1,3,3},{0,2,4,1,2,0,2,1,0,4,1,0,3,4,3,2,0,1,3,1},{3,3,2,4,1,0,0,1,4,4,4,3,3,2,0,4,3,4,3,4},{0,3,3,3,2,2,4,2,4,4,0,0,1,0,4,0,0,1,3,3},{0,2,2,0,2,4,1,1,3,4,4,2,2,3,2,2,0,3,1,0},{0,0,2,3,3,0,3,0,1,4,2,3,3,0,2,4,3,1,2,1},{4,1,0,3,1,4,2,3,3,4,0,1,4,2,4,4,1,4,2,0},{3,0,1,4,1,3,1,3,0,3,3,3,3,4,3,3,2,4,0,2},{1,1,2,1,0,1,0,2,2,0,3,1,3,1,3,2,1,3,4,3},{4,4,1,0,2,3,1,3,2,1,1,0,2,0,2,3,0,1,2,2},{4,1,3,3,4,4,2,4,0,3,2,4,4,0,2,2,3,2,1,0},{1,1,4,1,2,4,0,4,0,4,4,1,0,0,3,4,4,1,2,2},{4,0,2,3,2,0,2,2,4,4,3,0,4,0,4,2,3,0,0,0},{2,1,2,4,2,4,1,4,3,0,0,0,2,1,4,1,1,0,4,1},{1,4,4,3,3,0,2,2,1,4,3,3,4,3,2,0,2,3,1,4},{1,2,1,4,1,1,3,4,2,4,4,3,3,0,0,1,3,3,1,4},{1,2,2,3,3,3,1,0,1,1,3,1,4,1,0,1,1,2,0,0},{2,4,3,1,3,1,1,1,2,3,0,1,4,4,2,4,1,2,0,0},{0,3,1,2,3,3,4,3,1,1,1,1,0,1,0,1,0,2,4,3}};
    auto rr = tensor<double>{{147,148,178,179,134,188,161,185,168,197,140,142,207,132,180,184,158,158,192,173},{176,174,204,203,165,208,172,194,185,217,166,144,229,159,208,179,170,173,178,176},{125,157,158,178,139,165,159,184,148,205,140,129,180,133,163,148,139,124,144,140},{118,121,140,144,102,140,126,133,117,161,89,99,165,108,150,145,121,119,137,125},{181,150,154,176,141,181,164,179,186,200,150,144,208,133,157,188,159,168,156,171},{137,168,157,212,124,174,170,204,170,177,163,155,167,107,170,147,151,166,158,191},{179,135,165,175,144,203,141,180,207,218,146,132,225,153,192,190,156,167,156,157},{131,140,147,134,116,142,106,149,149,157,101,114,172,111,142,144,137,122,147,145},{121,142,164,151,124,164,129,162,148,179,132,122,171,118,176,174,135,130,161,148},{164,162,170,186,128,182,166,197,160,195,136,135,218,138,193,184,143,161,176,182},{159,197,191,201,149,197,164,197,154,214,160,162,194,138,203,210,157,174,176,187},{174,186,171,199,146,179,160,209,170,192,190,162,205,150,162,173,152,187,166,189},{143,149,186,142,111,176,128,179,168,166,149,130,189,120,176,139,139,135,161,153},{144,158,167,192,149,181,136,169,162,212,168,161,205,136,177,184,170,165,151,158},{160,140,168,175,134,190,150,172,149,169,124,124,200,144,200,175,138,149,167,149},{133,156,185,173,145,184,160,189,157,213,135,146,198,109,194,174,142,167,172,189},{131,152,165,178,121,165,143,185,142,205,131,153,177,123,186,192,161,146,188,192},{162,143,196,162,135,187,158,197,168,190,148,132,215,114,196,192,147,154,214,187},{134,156,174,154,113,169,145,156,150,190,131,116,176,105,171,156,152,125,152,159},{146,128,146,164,124,149,160,151,161,168,122,115,185,128,166,147,126,162,148,140},{126,150,155,151,121,151,131,176,132,164,100,117,162,115,180,144,130,111,148,166},{137,119,172,156,129,171,162,175,147,201,131,115,185,128,184,169,144,146,183,148},{184,148,164,162,127,163,143,152,157,173,132,101,212,142,167,170,96,135,159,148},{141,144,169,182,146,187,159,179,156,201,143,154,201,121,178,187,152,144,170,159},{188,151,174,231,144,200,161,197,180,211,172,145,231,130,200,192,168,195,177,177},{123,147,155,159,115,150,136,173,131,182,125,105,154,105,160,149,125,127,163,152},{177,141,167,171,136,178,130,186,176,193,154,123,192,144,173,183,159,155,164,154},{169,157,161,193,138,182,142,169,173,185,134,124,182,128,186,187,129,151,185,173},{137,149,153,152,124,140,141,158,169,194,140,108,176,103,135,147,144,128,156,167},{159,160,175,203,129,172,155,184,179,196,147,134,203,122,179,144,134,154,161,167}};

    //std::cout<<std::endl<<matmul_2d_tiled(aa,bb);

    //REQUIRE(matmul_2d(tensor<double,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    //REQUIRE(matmul_2d(tensor<double,f_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,f_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    // REQUIRE(matmul_2d(aa,bb)==rr);
    // REQUIRE(matmul_2d(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    // REQUIRE(matmul_2d_tiled(aa,bb)==rr);
    // REQUIRE(matmul_2d_tiled(aa.copy(f_order{}),bb.copy(f_order{}))==rr);
    // REQUIRE(matmul_2d_tiled(tensor<double,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    //REQUIRE(matmul_2d_tiled(tensor<double,f_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,f_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});

    //REQUIRE(matmul(tensor<double,c_order>{{1,2,4,2},{3,4,2,0},{5,3,1,1}},tensor<double,c_order>{{2,1,2},{0,3,1},{1,1,4},{4,3,3}})==tensor<double>{{14,17,26},{8,17,18},{15,18,20}});
    REQUIRE(matmul(aa,bb)==rr);
    REQUIRE(matmul(aa.copy(f_order{}),bb.copy(f_order{}))==rr);

    auto command_matmul_2d = [](const auto& t1, const auto& t2){
        auto r = matmul_2d(t1,t2);
        return *r.begin();
    };
    auto command_matmul_2d_tiled = [](const auto& t1, const auto& t2){
        auto r = matmul_2d_tiled(t1,t2);
        return *r.begin();
    };
    //bench_matmul("bench matmul_2d_tiled",n_iters,shapes,builder,command_matmul_2d_tiled);
    //bench_matmul("bench matmul_2d",n_iters,shapes,builder,command_matmul_2d);
    bench_matmul("bench matmul",n_iters,shapes,builder,command_matmul);

}