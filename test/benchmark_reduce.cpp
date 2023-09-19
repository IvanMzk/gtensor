#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"
//#include "statistic.hpp"

namespace benchmark_reduce_{

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

template<typename Axes, typename...Ts, typename Initial=gtensor::detail::no_value>
auto reduce_binary_sum(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false, const Initial& initial=Initial{}){
    using f_type = gtensor::math_reduce_operations::nan_propagate_operation<gtensor::math_reduce_operations::plus<void>>;
    return reduce_binary(t,axes,f_type{},keep_dims,initial);
}

template<typename Axes, typename...Ts>
auto ptp(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    return (t.max(axes,keep_dims) - t.min(axes,keep_dims)).copy();
}


struct ptp_binary_operation{

    template<typename E>
    auto operator()(const E& e1, const E& e2){
        return std::make_pair(std::min(e1,e2),std::max(e1,e2));
    }

    template<typename E>
    auto operator()(const std::pair<E,E>& r, const E& e){
        return std::make_pair(std::min(r.first,e),std::max(r.second,e));
    }
};


template<typename Axes, typename...Ts>
auto ptp_pair(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    using tensor_type = basic_tensor<Ts...>;
    using order = typename tensor_type::order;
    using value_type = typename tensor_type::value_type;
    using config_type = typename tensor_type::config_type;
    if (t.empty()){
        throw gtensor::value_error("no initial");
    }else{
        auto tmp = reduce_binary(t,axes,ptp_binary_operation{},keep_dims,gtensor::detail::no_value{});
        tensor<value_type,order,config_type> res(tmp.shape());
        std::transform(tmp.begin(),tmp.end(),res.begin(),[](const auto& min_max_pair){return min_max_pair.second-min_max_pair.first;});
        return res;
    }
}

template<typename Axes, typename...Ts>
auto mean(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    using value_type = typename basic_tensor<Ts...>::value_type;
    using res_type = gtensor::math::make_floating_point_t<value_type>;
    using f_type = gtensor::math_reduce_operations::nan_propagate_operation<std::plus<res_type>>;
    auto tmp = reduce_binary(t,axes,f_type{},keep_dims);
    const auto axes_size = t.size() / tmp.size();
    return (tmp/axes_size).copy();
}

template<typename Axes, typename...Ts>
auto mean_inplace(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    using value_type = typename basic_tensor<Ts...>::value_type;
    using res_type = gtensor::math::make_floating_point_t<value_type>;
    using f_type = gtensor::math_reduce_operations::nan_propagate_operation<std::plus<res_type>>;
    auto tmp = reduce_binary(t,axes,f_type{},keep_dims);
    const auto axes_size = t.size() / tmp.size();
    return tmp/=axes_size;
}

template<typename Axes, typename...Ts>
auto var_temp_pair(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    using value_type = typename basic_tensor<Ts...>::value_type;
    using order = typename basic_tensor<Ts...>::order;
    using config_type = typename basic_tensor<Ts...>::config_type;
    using res_type = gtensor::math::make_floating_point_t<value_type>;
    using res_config_type = gtensor::config::extend_config_t<config_type,res_type>;
    auto initial = std::make_pair(res_type{0},res_type{0}); //e,e^2
    auto f = [](const auto& l, const auto& r){
        using l_type = std::remove_cv_t<std::remove_reference_t<decltype(l)>>;
        return l_type{l.first+r, l.second+r*r};
    };
    auto tmp = gtensor::reduce_binary(t,axes,f,keep_dims,initial);
    const auto axes_size = t.size() / tmp.size();
    const auto axes_size_2 = axes_size*axes_size;
    gtensor::tensor<res_type,order,res_config_type> res(tmp.shape());
    std::transform(tmp.begin(),tmp.end(),res.begin(),
        [axes_size,axes_size_2](const auto& e){
            return (axes_size*e.second - e.first*e.first)/(axes_size_2);
        }
    );
    return res;
}

template<typename Axes, typename...Ts>
auto var_temp_mean(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    auto square = [](const auto& e){return e*e;};
    auto mean_ = mean_inplace(t,axes,true);
    auto tmp = gtensor::n_operator(square,t-std::move(mean_));
    return mean_inplace(tmp,axes,keep_dims);
}

template<typename Axes, typename...Ts>
auto var_temp_mean_copy(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
    auto square = [](const auto& e){return e*e;};
    auto mean_ = mean_inplace(t,axes,true);
    auto tmp = gtensor::n_operator(square,t-std::move(mean_)).copy();
    return mean_inplace(tmp,axes,keep_dims);
}

// template<typename Axes, typename...Ts>
// auto var(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims=false){
//     using value_type = typename basic_tensor<Ts...>::value_type;
//     using res_type = gtensor::math::make_floating_point_t<value_type>;
//     using f_type = gtensor::math_reduce_operations::nan_propagate_operation<std::plus<res_type>>;
//     auto square = [](const auto& e){return e*e;};
//     auto sum = reduce_binary(t,axes,f_type{},keep_dims);
//     auto sum_of_squared = reduce_binary(gtensor::n_operator(square,t).copy(),axes,f_type{},keep_dims);
//     const auto axes_size = t.size() / sum.size();
//     const auto axes_size_2 = axes_size*axes_size;
//     auto res = ((axes_size*std::move(sum_of_squared) - gtensor::n_operator(square,std::move(sum)))/axes_size_2);
//     return res.copy();
// }

struct var_range_tmp_mean
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        if (first == last){
            return gtensor::statistic_reduce_operations::reduce_empty<res_type>();
        }
        const auto n = static_cast<res_type>(last-first);
        auto mean_ = gtensor::statistic_reduce_operations::mean{}(first,last);
        auto init = static_cast<const res_type&>(*first) - mean_;
        init *= init;
        const auto res = std::accumulate(++first,last,init,
            [mean_](const auto& r, const auto& e){
                const auto dif = e-mean_;
                return r+dif*dif;
            }
        );
        return res/n;
    }
};

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


TEMPLATE_TEST_CASE("test_reduce_binary_ptp_pair","[benchmark_tensor]",
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

    REQUIRE(benchmark_reduce_::ptp(t,std::vector<int>{0}) == tensor_type{{{{3,5,8,3},{2,6,3,6},{3,5,8,7}}},{{{6,8,2,8},{3,6,1,4},{1,7,3,5}}}});
    REQUIRE(benchmark_reduce_::ptp(t,std::vector<int>{0,1}) == tensor_type{{{7,8,8,8},{4,7,3,7},{7,8,8,7}}});
    REQUIRE(benchmark_reduce_::ptp(t,std::vector<int>{1,2,3}) == tensor_type{{7,8,8,8},{7,8,8,6},{7,6,3,7}});
}

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

TEMPLATE_TEST_CASE("test_var_range_tmp_mean","[benchmark_tensor]",
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

    auto var = [](const auto& t, const auto& axes){
        return gtensor::reduce_range(t,axes,benchmark_reduce_::var_range_tmp_mean{},false,true);
    };

    auto t = tensor_type{{{{{7,5,8,5},{0,5,5,1},{3,8,0,8}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}}}},{{{{4,8,0,7},{0,0,2,4},{1,5,8,5}}},{{{6,8,4,8},{4,1,3,2},{7,0,6,2}}}},{{{{7,3,6,4},{2,6,4,7},{0,3,3,1}}},{{{2,1,3,0},{4,7,4,4},{7,6,3,3}}}}};

    REQUIRE(tensor_close(var(t,std::vector<int>{1}), tensor_type{{{{12.25,6.25,9.0,0.0},{0.25,2.25,1.0,0.25},{2.25,0.25,2.25,0.25}}},{{{1.0,0.0,4.0,0.25},{4.0,0.25,0.25,1.0},{9.0,6.25,1.0,2.25}}},{{{6.25,1.0,2.25,4.0},{1.0,0.25,0.0,2.25},{12.25,2.25,0.0,1.0}}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(var(t,std::vector<int>{0,1}), tensor_type{{{6.889,9.806,6.806,6.472},{2.806,6.917,0.917,5.333},{8.0,7.139,6.472,6.556}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(var(t,std::vector<int>{0,2}), tensor_type{{{2.0,4.222,11.556,1.556},{0.889,6.889,1.556,6.0},{1.556,4.222,10.889,8.222}},{{6.222,12.667,0.667,10.889},{2.0,6.889,0.222,2.667},{0.222,9.556,2.0,4.667}}}, 1E-2,1E-2));
    REQUIRE(tensor_close(var(t,std::vector<int>{1,2,3}), tensor_type{{7.806,7.583,6.25,8.556},{6.222,12.222,6.806,5.222},{6.889,4.556,1.139,5.139}}, 1E-2,1E-2));
    REQUIRE(tensor_close(var(t,std::vector<int>{2,3}), tensor_type{{{8.222,2.0,10.889,8.222},{6.889,8.667,0.222,8.667}},{{2.889,10.889,11.556,1.556},{1.556,12.667,1.556,8.0}},{{8.667,2.0,1.556,6.0},{4.222,6.889,0.222,2.889}}},1E-2,1E-2));
}


TEST_CASE("benchmark_reduce","[benchmark_tensor]")
{

    using benchmark_reduce_::bench_reduce;
    using benchmark_reduce_::reduce_binary_sum;

    auto reducer_reduce_binary_sum = [](const auto& t, const auto& axes){
        auto r = reduce_binary_sum(t,axes);
        return *r.begin();
    };

    auto reducer_reduce_range_sum = [](const auto& t, const auto& axes){
        auto r = gtensor::reduce(t,axes,gtensor::math_reduce_operations::sum{},false,true);
        return *r.begin();
    };

    auto reducer_reduce_binary_ptp = [](const auto& t, const auto& axes){
        auto r = benchmark_reduce_::ptp(t,axes);
        return *r.begin();
    };

    auto reducer_reduce_binary_ptp_pair = [](const auto& t, const auto& axes){
        auto r = benchmark_reduce_::ptp_pair(t,axes);
        return *r.begin();
    };

    // auto reducer_reduce_range_ptp = [](const auto& t, const auto& axes){
    //     auto r = gtensor::ptp(t,axes);
    //     return *r.begin();
    // };

    // auto reducer_reduce_range_mean = [](const auto& t, const auto& axes){
    //     auto r = gtensor::mean(t,axes);
    //     return *r.begin();
    // };

    auto reducer_reduce_binary_mean = [](const auto& t, const auto& axes){
        auto r = benchmark_reduce_::mean(t,axes);
        return *r.begin();
    };

    auto reducer_reduce_binary_mean_div_inplace = [](const auto& t, const auto& axes){
        auto r = benchmark_reduce_::mean(t,axes);
        return *r.begin();
    };


    auto reducer_reduce_binary_var_temp_pair = [](const auto& t, const auto& axes){
        auto r = benchmark_reduce_::var_temp_pair(t,axes);
        return *r.begin();
    };
    auto reducer_reduce_binary_var_temp_mean = [](const auto& t, const auto& axes){
        auto r = benchmark_reduce_::var_temp_mean(t,axes);
        return *r.begin();
    };
    auto reducer_reduce_binary_var_temp_mean_copy = [](const auto& t, const auto& axes){
        auto r = benchmark_reduce_::var_temp_mean_copy(t,axes);
        return *r.begin();
    };

    // auto reducer_reduce_range_var = [](const auto& t, const auto& axes){
    //     auto r = gtensor::var(t,axes);
    //     return *r.begin();
    // };

    auto reducer_reduce_range_var_tmp_mean = [](const auto& t, const auto& axes){
        auto r = gtensor::reduce(t,axes,benchmark_reduce_::var_range_tmp_mean{},false,true);
        return *r.begin();
    };


    const auto axes = benchmark_helpers::axes;
    //const auto axes = benchmark_helpers::axes_scalar_2d;
    //const auto axes = benchmark_helpers::axes_scalar;
    //const auto axes = benchmark_helpers::axes_container;

    // const auto n_iters = 100;
    // const auto shapes = benchmark_helpers::small_shapes;

    const auto n_iters = 1;
    const auto shapes = benchmark_helpers::shapes;
    //const auto shapes = benchmark_helpers::shapes_2d;


    //bench_reduce("reduce range sum on tensor",n_iters,benchmark_helpers::shapes_2d,std::vector<int>{0},[](auto&& t){return t;},reducer_reduce_range_sum);



    //bench_reduce("reduce range var tmp mean",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_range_var_tmp_mean);
    //bench_reduce("reduce range var",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_range_var);
    //bench_reduce("reduce binary var on tensor temp pair",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_var_temp_pair);
    //bench_reduce("reduce binary var on tensor temp mean copy",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_var_temp_mean_copy);
    //bench_reduce("reduce binary var on tensor temp mean",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_var_temp_mean);


    //bench_reduce("reduce range mean on tensor",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_range_mean);
    //bench_reduce("reduce binary mean on tensor div inplace",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_mean_div_inplace);
    //bench_reduce("reduce binary mean on tensor",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_mean);

    //bench_reduce("reduce range ptp on tensor",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_range_ptp);
    //bench_reduce("reduce binary ptp min-max pair on tensor, single reduction for min max",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_ptp_pair);
    //bench_reduce("reduce ptp on tensor",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_ptp);

    //bench_reduce("reduce range sum on tensor",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_range_sum);
    //bench_reduce("reduce range sum on trivial expression view t+t+t+t+t+t+t+t+t+t",n_iters,shapes,axes,[](auto&& t){return t+t+t+t+t+t+t+t+t+t;},reducer_reduce_range_sum);
    //bench_reduce("reduce range sum on not trivial expression view t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t",n_iters,shapes,axes,[](auto&& t){return t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t;},reducer_reduce_range_sum);

    //bench_reduce("reduce_binary sum on tensor",n_iters,shapes,axes,[](auto&& t){return t;},reducer_reduce_binary_sum);
    //bench_reduce("reduce_binary sum on trivial expression view t+t+t+t+t+t+t+t+t+t",n_iters,shapes,axes,[](auto&& t){return t+t+t+t+t+t+t+t+t+t;},reducer_reduce_binary_sum);
    //bench_reduce("reduce binary sum on not trivial expression view t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t",n_iters,shapes,axes,[](auto&& t){return t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t;},reducer_reduce_binary_sum);

}

