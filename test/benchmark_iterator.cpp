#include "benchmark_helpers.hpp"
#include "tensor.hpp"
#include "test_config.hpp"


namespace benchmark_iterator_{

using gtensor::detail::shape_to_str;
using benchmark_helpers::order_to_str;
using gtensor::config::c_order;
using gtensor::config::f_order;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;
using benchmark_helpers::opposite_order_t;

template<typename Tensor, typename Layout, typename TraverseOrder>
struct bench_iterator_helper{
    using tensor_type = Tensor;
    using layout = Layout;
    using traverse_order = TraverseOrder;
    using shape_type = typename tensor_type::shape_type;

    template<typename Shapes, typename Builder, typename Traverser>
    auto operator()(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder_f, Traverser traverse_f, bool reverse_iterator=false){
        std::cout<<std::endl<<"layout "<<order_to_str(layout{})<<" traverse "<<order_to_str(traverse_order{})<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
            const auto& shape = *it;
            std::vector<double> intervals{};
            for (auto n=n_iters; n!=0; --n){
                auto t_ = tensor_type(shape_type(shape.begin(),shape.end()),2);
                auto t=builder_f(t_);
                //measure traverse time
                double dt = 0;
                auto a = t.traverse_order_adapter(traverse_order{});
                if (reverse_iterator){
                    dt = timing(traverse_f,a.rbegin(),a.rend());
                }else{
                    dt = timing(traverse_f,a.begin(),a.end());
                }
                intervals.push_back(dt);
                total_intervals.push_back(dt);
            }
            std::cout<<std::endl<<"shape "<<shape_to_str(shape)<<" "<<statistic(intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

//traverse range with deref
auto traverse_forward = [](auto&& first, auto&& last){
    using iterator_type = std::remove_cv_t<std::remove_reference_t<decltype(first)>>;
    using value_type = typename std::iterator_traits<iterator_type>::value_type;
    value_type r{0};
    for(;first!=last; ++first){
        r+=*first;
    }
    return r;
};
auto traverse_backward = [](auto&& first, auto&& last){
    using iterator_type = std::remove_cv_t<std::remove_reference_t<decltype(first)>>;
    using value_type = typename std::iterator_traits<iterator_type>::value_type;
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

template<typename Shapes, typename Builder, typename Traverser>
auto bench_iterator(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder_f, Traverser traverse_f, bool reverse_iterator=false){
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<gtensor::basic_storage>,value_type>;
    bench_iterator_helper<gtensor::tensor<value_type,c_order,config_type>,c_order,c_order>{}(mes,n_iters,shapes,builder_f,traverse_f);
    bench_iterator_helper<gtensor::tensor<value_type,c_order,config_type>,c_order,f_order>{}(mes,n_iters,shapes,builder_f,traverse_f);
    bench_iterator_helper<gtensor::tensor<value_type,f_order,config_type>,f_order,c_order>{}(mes,n_iters,shapes,builder_f,traverse_f);
    bench_iterator_helper<gtensor::tensor<value_type,f_order,config_type>,f_order,f_order>{}(mes,n_iters,shapes,builder_f,traverse_f);
}

}


TEST_CASE("benchmark_iterator_small_shapes","[benchmark_tensor]")
{

    using tensor_type = gtensor::tensor<double>;
    using slice_type = typename tensor_type::slice_type;
    using benchmark_helpers::opposite_order_t;
    using benchmark_iterator_::bench_iterator;
    using benchmark_iterator_::traverse_forward;
    using benchmark_iterator_::traverse_backward;


    const auto n_iters = 100;
    const auto shapes = benchmark_helpers::small_shapes;
    //tensor
    bench_iterator("tensor traverse",n_iters,shapes,[](auto& t){return t;},traverse_forward,true);
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

