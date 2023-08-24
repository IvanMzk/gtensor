#include "benchmark_helpers.hpp"
#include "tensor.hpp"

namespace benchmark_iterator_{

template<typename Config, typename Walker, typename Order>
class forward_iterator
{
protected:
    using config_type = Config;
    using walker_type = Walker;
    using result_type = decltype(*std::declval<walker_type>());
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using strides_div_type = gtensor::detail::strides_div_t<config_type>;
public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = typename config_type::index_type;
    using value_type = typename gtensor::detail::iterator_internals_selector<result_type>::value_type;
    using pointer = typename gtensor::detail::iterator_internals_selector<result_type>::pointer;
    using reference = typename gtensor::detail::iterator_internals_selector<result_type>::reference;
    using const_reference = typename gtensor::detail::iterator_internals_selector<result_type>::const_reference;

    template<typename Walker_>
    forward_iterator(Walker_&& walker_, const shape_type& shape_, const index_type& pos):
        walker{std::forward<Walker_>(walker_)},
        shape{&shape_},
        index(shape_.size(),0),
        flat_index{pos},
        dim{gtensor::detail::make_dim(*shape)}
    {}
    forward_iterator& operator++(){
        gtensor::detail::next<Order>(walker,index,*shape);
        ++flat_index;
        return *this;
    }
    result_type operator*() const{return *walker;}
    bool operator==(const forward_iterator& rhs){return flat_index == rhs.flat_index;}
    bool operator!=(const forward_iterator& rhs){return !(*this == rhs);}
private:
    walker_type walker;
    const shape_type* shape;
    shape_type index;
    difference_type flat_index;
    dim_type dim;
};

template<typename Config, typename Walker, typename Order>
class bidirectional_iterator
{
protected:
    using config_type = Config;
    using walker_type = Walker;
    using result_type = decltype(*std::declval<walker_type>());
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using strides_div_type = gtensor::detail::strides_div_t<config_type>;
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = typename config_type::index_type;
    using value_type = typename gtensor::detail::iterator_internals_selector<result_type>::value_type;
    using pointer = typename gtensor::detail::iterator_internals_selector<result_type>::pointer;
    using reference = typename gtensor::detail::iterator_internals_selector<result_type>::reference;
    using const_reference = typename gtensor::detail::iterator_internals_selector<result_type>::const_reference;

    template<typename Walker_>
    bidirectional_iterator(Walker_&& walker_, const shape_type& shape_, const index_type& pos):
        walker{std::forward<Walker_>(walker_)},
        shape{&shape_},
        index(shape_.size(),0),
        flat_index{pos},
        dim{gtensor::detail::make_dim(*shape)}
    {}
    bidirectional_iterator& operator++(){
        if constexpr (std::is_same_v<gtensor::config::c_order,Order>){
            gtensor::detail::next_13::next_c(walker,index.begin(),shape->begin(),dim);
        }else{
            gtensor::detail::next_13::next_f(walker,index.begin(),shape->begin(),dim);
        }
        ++flat_index;
        return *this;
    }
    bidirectional_iterator& operator--(){
        if constexpr (std::is_same_v<gtensor::config::c_order,Order>){
            gtensor::detail::next_13::prev_c(walker,index.begin(),shape->begin(),dim);
        }else{
            gtensor::detail::next_13::prev_f(walker,index.begin(),shape->begin(),dim);
        }
        --flat_index;
        return *this;
    }
    result_type operator*()const{return *walker;}
    bool operator==(const bidirectional_iterator& rhs)const{return flat_index == rhs.flat_index;}
    bool operator!=(const bidirectional_iterator& rhs)const{return flat_index != rhs.flat_index;}
private:
    walker_type walker;
    const shape_type* shape;
    shape_type index;
    difference_type flat_index;
    dim_type dim;
};


auto bench_iterator = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;

    auto a = t_.traverse_order_adapter(order);
    auto it = a.begin();
    auto last = a.end();

    auto f = [it,last]()mutable{
        value_type r{0};
        for(;it!=last; ++it){
            r+=*it;
        }
        return r;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
    return dt;
};

auto bench_iterator_no_deref = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;

    auto a = t_.traverse_order_adapter(order);
    auto it = a.begin();
    auto last = a.end();

    auto f = [it,last]()mutable{
        for(;it!=last; ++it){}
        return last-it;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

}   //end of namespace benchmark_iterator_

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
    using tensor_type = gtensor::tensor<value_type,layout>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using config_type = typename tensor_type::config_type;
    using order = typename tensor_type::order;
    using benchmark_helpers::order_to_str;
    using gtensor::detail::shape_to_str;
    using benchmark_iterator_::bench_iterator;
    using benchmark_iterator_::bench_iterator_no_deref;
    using benchmark_helpers::cpu_timer;
    using benchmark_helpers::timing;
    using benchmark_helpers::fake_use;
    using benchmark_helpers::statistic;


    auto make_forward = [](const auto& t, auto pos){
        using walker_type = decltype(t.create_walker());
        using iterator_type = benchmark_iterator_::forward_iterator<config_type, walker_type, traverse_order>;
        return iterator_type{t.create_walker(),t.shape(),pos};
    };

    auto make_bidirectional = [](const auto& t, auto pos){
        using walker_type = decltype(t.create_walker());
        using iterator_type = benchmark_iterator_::bidirectional_iterator<config_type, walker_type, traverse_order>;
        return iterator_type{t.create_walker(),t.shape(),pos};
    };

    auto forward_begin = [make_forward](const auto& t){return make_forward(t,0);};
    auto forward_end = [make_forward](const auto& t){return make_forward(t,t.size());};

    auto bidirectional_begin = [make_bidirectional](const auto& t){return make_bidirectional(t,0);};
    auto bidirectional_end = [make_bidirectional](const auto& t){return make_bidirectional(t,t.size());};


    //traverse range with deref
    auto f = [](auto&& first, auto&& last){
        value_type r{0};
        for(;first!=last; ++first){
            r+=*first;
        }
        return r;
    };

    //traverse range with deref
    auto f_accum = [](auto&& first, auto&& last){
        return std::accumulate(first,last,value_type{0});
    };

    auto f_reverse = [](auto&& first, auto&& last){
        value_type r{0};
        while(last!=first){
            --last;
            r+=*last;
        }
        return r;
    };

    auto f_reverse_no_deref = [](auto&& first, auto&& last){
        while(last!=first){
            --last;
        }
        return first==last;
    };

    //traverse range no deref
    auto f_traverse = [](auto&& first, auto&& last){
        for(;first!=last; ++first){}
        return first==last;
    };

    //deref no traverse
    auto f_deref = [](auto&& first, auto&& last){
        value_type r{0};
        auto n = last-first;
        while(n!=0){
            --n;
            r+=*first;
        }
        return r;
    };

    const auto& shapes = benchmark_helpers::small_shapes_1;
    constexpr std::size_t bench_iterations = 1000;
    std::vector<double> intervals{};
    for (auto n=bench_iterations; n!=0; --n){
        //make tensors and iterators
        double dt_iteration{0};
        for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
            auto t_ = tensor_type(*it,2);
            auto t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;   //10
            auto a = t.traverse_order_adapter(traverse_order{});
            //measure traverse time
            //auto t_last = bidirectional_end(t);
            //--t_last;
            //auto dt = timing(f_reverse,bidirectional_begin(t),t_last);
            //auto dt = timing(f_reverse,bidirectional_begin(t),bidirectional_end(t));
            //auto dt = timing(f,bidirectional_begin(t),bidirectional_end(t));
            //auto dt = timing(f,forward_begin(t),forward_end(t));
            //auto dt = timing(f_traverse,forward_begin(t),forward_end(t));

            auto dt = timing(f,a.begin(),a.end());
            //auto dt = timing(f_reverse,a.begin(),a.end());

            //auto dt = timing(f_accum,a.begin(),a.end());
            //auto dt = timing(f_reverse_no_deref,a.begin(),a.end());
            //auto dt = timing(f_traverse,a.begin(),a.end());
            //auto dt = timing(f_deref,a.begin(),a.end());
            dt_iteration+=dt.interval();
        }
        intervals.push_back(dt_iteration);
    }
    std::cout<<std::endl<<"layout "<<order_to_str(layout{})<<" traverse "<<order_to_str(traverse_order{})<<" "<<statistic(intervals);
}

// TEMPLATE_TEST_CASE("benchmark_iterator_experiment","[benchmark_tensor]",
//     (std::tuple<gtensor::config::c_order,gtensor::config::c_order>)
//     // (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
//     // (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
//     // (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
// )
// {
//     using layout = std::tuple_element_t<0,TestType>;
//     using traverse_order = std::tuple_element_t<1,TestType>;
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = gtensor::tensor<value_type,layout>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::shapes;
//     using benchmark_iterator_::bench_iterator;
//     using benchmark_iterator_::bench_iterator_no_deref;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;   //10
//         std::cout<<std::endl<<"expression t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;"<<order_to_str(layout{})<<" "<<shape_to_str(t.shape());
//         bench_iterator(t,traverse_order{},"iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

// TEMPLATE_TEST_CASE("benchmark_iterator_tensor","[benchmark_tensor]",
//     (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
// )
// {
//     using layout = std::tuple_element_t<0,TestType>;
//     using traverse_order = std::tuple_element_t<1,TestType>;
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = gtensor::tensor<value_type,layout>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::shapes;
//     using benchmark_iterator_::bench_iterator;
//     using benchmark_helpers::cpu_timer;

//     double total{0};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t = tensor_type(*it,2);
//         std::cout<<std::endl<<"tensor "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         auto dt = bench_iterator(t,traverse_order{},"iterator");
//         total+=dt.interval();
//     }
//     std::cout<<std::endl<<"total, ms "<<total;
// }

// TEMPLATE_TEST_CASE("benchmark_iterator_expression","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::shapes;
//     using benchmark_iterator_::bench_iterator;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=t_+t_;
//         std::cout<<std::endl<<"expression t+t "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_iterator(t,c_order{},"iterator");
//         bench_iterator(t,f_order{},"iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

// TEMPLATE_TEST_CASE("benchmark_iterator_deep_expression","[benchmark_tensor]",
//     (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
// )
// {
//     using layout = std::tuple_element_t<0,TestType>;
//     using traverse_order = std::tuple_element_t<1,TestType>;
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = gtensor::tensor<value_type,layout>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::shapes;
//     using benchmark_iterator_::bench_iterator;
//     using benchmark_helpers::cpu_timer;

//     double total{0};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;   //10
//         std::cout<<std::endl<<"deep_expression  t_+t_+t_+t_+t_+t_+t_+t_+t_+t_ "<<order_to_str(layout{})<<" "<<shape_to_str(t.shape());
//         auto dt = bench_iterator(t,traverse_order{},"iterator");
//         total+=dt.interval();
//     }
//     std::cout<<std::endl<<"total, ms "<<total;
// }

// TEMPLATE_TEST_CASE("benchmark_iterator_expression_with_scalar","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::shapes;
//     using benchmark_iterator_::bench_iterator;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=t_+1;
//         std::cout<<std::endl<<"expression with scalar t+1 "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_iterator(t,c_order{},"iterator");
//         bench_iterator(t,f_order{},"iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

// TEMPLATE_TEST_CASE("benchmark_iterator_deep_expression_with_scalar","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::shapes;
//     using benchmark_iterator_::bench_iterator;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=((((((((((t_+1)+1)+1)+1)+1)+1)+1)+1)+1)+1);  //10
//         std::cout<<std::endl<<"expression with scalar ((((((((((t_+1)+1)+1)+1)+1)+1)+1)+1)+1)+1) "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_iterator(t,c_order{},"iterator");
//         bench_iterator(t,f_order{},"iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

