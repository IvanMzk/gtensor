#include "../benchmark_helpers.hpp"
#include "../test_config.hpp"
#include "tensor.hpp"


namespace benchmark_iterator_{

using gtensor::detail::shape_to_str;
using benchmark_helpers::order_to_str;
using gtensor::config::c_order;
using gtensor::config::f_order;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;
using benchmark_helpers::opposite_order_t;

template<typename Tensor, typename TraverseOrder>
struct bench_iterator_helper{
    using tensor_type = Tensor;
    using layout = typename tensor_type::order;
    using traverse_order = TraverseOrder;
    using shape_type = typename tensor_type::shape_type;

    template<typename Shapes, typename Builder, typename Traverser>
    auto operator()(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Traverser traverser, bool reverse_iterator=false, bool trivial_iterator=false){
        std::cout<<std::endl<<"layout "<<order_to_str(layout{})<<" traverse "<<order_to_str(traverse_order{})<<" reverse "<<reverse_iterator<<" trivial "<<trivial_iterator<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
            const auto& shape = *it;
            std::vector<double> intervals{};
            shape_type t_shape{};
            bool is_t_shape{false};
            for (auto n=n_iters; n!=0; --n){
                auto t_ = tensor_type(shape,2);
                auto t=builder(t_);
                if (!is_t_shape){
                    t_shape=t.shape();
                    is_t_shape=true;
                }
                //measure traverse time
                double dt = 0;
                auto a = t.traverse_order_adapter(traverse_order{});
                if (reverse_iterator){
                    if (trivial_iterator){
                        dt = timing(traverser,a.rbegin_trivial(),a.rend_trivial());
                    }else{
                        dt = timing(traverser,a.rbegin(),a.rend());
                    }
                }else{
                    if (trivial_iterator){
                        dt = timing(traverser,a.begin_trivial(),a.end_trivial());
                    }else{
                        dt = timing(traverser,a.begin(),a.end());
                    }
                }
                intervals.push_back(dt);
                total_intervals.push_back(dt);
            }
            std::cout<<std::endl<<"input shape "<<shape_to_str(shape)<<" shape "<<shape_to_str(t_shape)<<" "<<statistic(intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

//traverse range with deref
auto traverse_forward = [](auto first, auto last){
    using iterator_type = std::remove_cv_t<std::remove_reference_t<decltype(first)>>;
    using value_type = typename std::iterator_traits<iterator_type>::value_type;
    value_type r{0};
    for(;first!=last; ++first){
        r+=*first;
    }
    return r;
};
auto traverse_backward = [](auto first, auto last){
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
auto traverse_no_deref = [](auto first, auto last){
    for(;first!=last; ++first){}
    return first==last;
};

template<typename Shapes, typename Builder, typename Traverser>
auto bench_iterator(std::string mes, std::size_t n_iters, Shapes shapes, Builder builder, Traverser traverser, bool reverse_iterator=false, bool trivial_iterator=false){
    using value_type = double;
    bench_iterator_helper<gtensor::tensor<value_type,c_order>,c_order>{}(mes,n_iters,shapes,builder,traverser,reverse_iterator,trivial_iterator);
    // bench_iterator_helper<gtensor::tensor<value_type,c_order>,f_order>{}(mes,n_iters,shapes,builder,traverser,reverse_iterator,trivial_iterator);
    // bench_iterator_helper<gtensor::tensor<value_type,f_order>,c_order>{}(mes,n_iters,shapes,builder,traverser,reverse_iterator,trivial_iterator);
    // bench_iterator_helper<gtensor::tensor<value_type,f_order>,f_order>{}(mes,n_iters,shapes,builder,traverser,reverse_iterator,trivial_iterator);
}

}


TEST_CASE("benchmark_iterator","[benchmark_tensor]")
{
    using tensor_type = gtensor::tensor<double>;
    using slice_type = typename tensor_type::slice_type;
    using benchmark_helpers::opposite_order_t;
    using benchmark_iterator_::bench_iterator;
    using gtensor::config::c_order;
    using gtensor::config::f_order;


    const auto n_iters = 1;
    const auto shapes = benchmark_helpers::shapes;
    auto traverser = benchmark_iterator_::traverse_forward;
    const bool reverse_iterator = false;

    // //tensor
    // bench_iterator("tensor traverse forward",n_iters,shapes,[](auto&& t){return t;},traverser,reverse_iterator,false);
    // //expression view
    bench_iterator("trivial expression t+t+t+t+t+t+t+t+t+t traverse forward",n_iters,shapes,[](auto&& t){return t+t+t+t+t+t+t+t+t+t;},traverser,reverse_iterator,false);
    // bench_iterator("trivial expression t+t+t+t+t+t+t+t+t+t traverse forward",n_iters,shapes,[](auto&& t){return t+t+t+t+t+t+t+t+t+t;},traverser,reverse_iterator,true);
    // bench_iterator("non trivial expression t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t traverse forward",n_iters,shapes,[](auto&& t){return t+t(0)+t(1)+t(2)+t(3,0)+t(4,1)+t(1,2)+t+t+t;},traverser,reverse_iterator,false);
    // //transpose view
    // bench_iterator("transpose view traverse forward",n_iters,shapes,[](auto&& t){return t.transpose();},traverser,reverse_iterator,false);
    // //slice view
    // bench_iterator("slice view t[0:-1,:,:,::-1] traverse forward",n_iters,shapes,[](auto&& t){return t({{0,-1,1},{},{},{{},{},-1}});},traverser,reverse_iterator,false);
    // bench_iterator("slice view t[:,1,:,:] traverse forward",n_iters,shapes,[](auto&& t){return t(slice_type{},1,slice_type{},slice_type{});},traverser,reverse_iterator,false);
    // bench_iterator("slice view t[0:-1,1,:,::-1] traverse forward",n_iters,shapes,[](auto&& t){return t(slice_type{0,-1,1},1,slice_type{},slice_type{{},{},-1});},traverser,reverse_iterator,false);
    // //reshape_view
    // bench_iterator("reshape view t.reshape((-1,3000)), c_order, traverse forward",n_iters,shapes,[](auto&& t){return t.reshape({-1,3000},c_order{});},traverser,reverse_iterator,false);
    // bench_iterator("reshape view t.reshape((-1,3000)), f_order, traverse forward",n_iters,shapes,[](auto&& t){return t.reshape({-1,3000},f_order{});},traverser,reverse_iterator,false);
    // //mapping view
    // bench_iterator("mapping view t(t>0), traverse forward",n_iters,shapes,[](auto&& t){return t(t>0);},traverser,reverse_iterator,false);
    // //view of view
    // bench_iterator("transpose of trivial expression t+t+t+t+t+t+t+t+t+t traverse forward",n_iters,shapes,[](auto&& t){return (t+t+t+t+t+t+t+t+t+t).transpose();},traverser,reverse_iterator,false);
    // bench_iterator("transpose of slice view t[0:-1,:,:,::-1] traverse forward",n_iters,shapes,[](auto&& t){return t({{0,-1,1},{},{},{{},{},-1}}).transpose();},traverser,reverse_iterator,false);
}
