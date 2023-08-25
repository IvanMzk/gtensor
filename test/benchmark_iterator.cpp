#include "benchmark_helpers.hpp"
#include "tensor.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("benchmark_iterator_small_shapes","[benchmark_tensor]",
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using layout = std::tuple_element_t<0,TestType>;
    using traverse_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using gtensor::tensor;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<gtensor::basic_storage>,value_type>;
    using tensor_type = gtensor::tensor<value_type,layout,config_type>;
    using slice_type = typename tensor_type::slice_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using config_type = typename tensor_type::config_type;
    using order = typename tensor_type::order;
    using benchmark_helpers::order_to_str;
    using gtensor::detail::shape_to_str;
    using benchmark_helpers::timing;
    using benchmark_helpers::statistic;
    using benchmark_helpers::opposite_order_t;

    //traverse range with deref
    auto traverse_forward = [](auto&& first, auto&& last){
        value_type r{0};
        for(;first!=last; ++first){
            r+=*first;
        }
        return r;
    };

    auto traverse_backward = [](auto&& first, auto&& last){
        value_type r{0};
        while(last!=first){
            --last;
            r+=*last;
        }
        return r;
    };

    //traverse range no deref
    auto traverse_no_deref = [](auto&& first, auto&& last){
        for(;first!=last; ++first){}
        return first==last;
    };

    auto bench_iterator = [](auto mes, auto n_iters, auto shapes, auto builder_f, auto traverse_f){
        std::cout<<std::endl<<"layout "<<order_to_str(layout{})<<" traverse "<<order_to_str(traverse_order{})<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
            const auto& shape = *it;
            std::vector<double> intervals{};
            for (auto n=n_iters; n!=0; --n){
                auto t_ = tensor_type(shape,2);
                auto t=builder_f(t_);
                auto a = t.traverse_order_adapter(traverse_order{});
                //measure traverse time
                auto dt = timing(traverse_f,a.begin(),a.end());
                //save measurements
                auto interval = dt.interval();
                intervals.push_back(interval);
                total_intervals.push_back(interval);
            }
            std::cout<<std::endl<<"shape "<<shape_to_str(shape)<<" "<<statistic(intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    };

    const auto n_iters = 100;
    const auto shapes = benchmark_helpers::small_shapes;
    //tensor
    //bench_iterator("tensor traverse",n_iters,shapes,[](auto& t){return t;},traverse_forward);
    //bench_iterator("tensor traverse backward",n_iters,shapes,[](auto& t){return t;},traverse_backward);
    //expression view
    //bench_iterator("expression t+t+t+t+t+t+t+t+t+t traverse",n_iters,shapes,[](auto& t){return t+t+t+t+t+t+t+t+t+t;},traverse_forward);
    //bench_iterator("expression t+t+t+t+t+t+t+t+t+t traverse backward",n_iters,shapes,[](auto& t){return t+t+t+t+t+t+t+t+t+t;},traverse_backward);
    //transpose view
    //bench_iterator("transpose view traverse forward",n_iters,shapes,[](auto& t){return t.transpose();},traverse_forward);
    //bench_iterator("transpose view traverse backward",n_iters,shapes,[](auto& t){return t.transpose();},traverse_backward);
    //slice view
    //bench_iterator("slice view t[0:-1,:,:,::-1] traverse forward",n_iters,shapes,[](auto& t){return t({{0,-1,1},{},{},{{},{},-1}});},traverse_forward);
    //bench_iterator("slice view t[0:-1,:,:,::-1] traverse backward",n_iters,shapes,[](auto& t){return t({{0,-1,1},{},{},{{},{},-1}});},traverse_backward);
    //bench_iterator("slice view t[:,1,:,:] traverse forward",n_iters,shapes,[](auto& t){return t(slice_type{},1,slice_type{},slice_type{});},traverse_forward);
    //bench_iterator("slice view t[:,1,:,:] traverse backward",n_iters,shapes,[](auto& t){return t(slice_type{},1,slice_type{},slice_type{});},traverse_backward);
    //bench_iterator("slice view t[0:-1,1,:,::-1] traverse forward",n_iters,shapes,[](auto& t){return t(slice_type{0,-1,1},1,slice_type{},slice_type{{},{},-1});},traverse_forward);
    //bench_iterator("slice view t[0:-1,1,:,::-1] traverse backward",n_iters,shapes,[](auto& t){return t(slice_type{0,-1,1},1,slice_type{},slice_type{{},{},-1});},traverse_backward);
    //reshape_view
    //bench_iterator("reshape view t.reshape((2000,3000)), same order, traverse forward",n_iters,shapes,[](auto& t){return t.reshape({2000,3000},layout{});},traverse_forward);
    //bench_iterator("reshape view t.reshape((2000,3000)), same order, traverse backward",n_iters,shapes,[](auto& t){return t.reshape({2000,3000},layout{});},traverse_backward);
    //bench_iterator("reshape view t.reshape((2000,3000)), opposite order, traverse forward",n_iters,shapes,[](auto& t){return t.reshape({2000,3000},opposite_order_t<layout>{});},traverse_forward);
    //bench_iterator("reshape view t.reshape((2000,3000)), opposite order, traverse backward",n_iters,shapes,[](auto& t){return t.reshape({2000,3000},opposite_order_t<layout>{});},traverse_backward);




}

