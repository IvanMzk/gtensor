#include "benchmark_helpers.hpp"
#include "tensor.hpp"

namespace benchmark_traverser_{

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
        //gtensor::detail::next<Order>(walker,*shape,dim,index);
        gtensor::detail::next<Order>(walker,*shape,index);
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

// template<typename Config, typename Traverser>
// class forward_iterator
// {
// protected:
//     using config_type = Config;
//     using traverser_type = Traverser;
//     using result_type = decltype(*std::declval<traverser_type>());
//     using shape_type = typename config_type::shape_type;
//     using index_type = typename config_type::index_type;
//     using strides_div_type = gtensor::detail::strides_div_t<config_type>;
// public:
//     using iterator_category = std::forward_iterator_tag;
//     using difference_type = typename config_type::index_type;
//     using value_type = typename gtensor::detail::iterator_internals_selector<result_type>::value_type;
//     using pointer = typename gtensor::detail::iterator_internals_selector<result_type>::pointer;
//     using reference = typename gtensor::detail::iterator_internals_selector<result_type>::reference;
//     using const_reference = typename gtensor::detail::iterator_internals_selector<result_type>::const_reference;

//     template<typename Traverser_>
//     forward_iterator(Traverser_&& traverser_, const index_type& pos):
//         traverser{std::forward<Traverser_>(traverser_)},
//         flat_index{pos}
//     {}
//     forward_iterator& operator++(){
//         traverser.next();
//         ++flat_index;
//         return *this;
//     }
//     result_type operator*() const{return *traverser;}
//     bool operator==(const forward_iterator& rhs){return flat_index == rhs.flat_index;}
//     bool operator!=(const forward_iterator& rhs){return !(*this == rhs);}
// private:
//     traverser_type traverser;
//     difference_type flat_index;
// };


auto bench_forward_traverser_no_deref = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using config_type = typename tensor_type::config_type;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;
    using traverse_order = decltype(order);


    using walker_type = decltype(t_.create_walker());
    using traverser_type = gtensor::walker_forward_traverser<config_type,walker_type>;

    traverser_type traverser{t_.shape(), t_.create_walker()};

    auto f = [&traverser](){
        value_type r{0};
        do{
            if (traverser.index().size()==0){
                ++r;
            }
        }while(traverser.template next<traverse_order>());
        return r;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

auto bench_forward_traverser_no_deref_counter_loop = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using config_type = typename tensor_type::config_type;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;
    using traverse_order = decltype(order);


    using walker_type = decltype(t_.create_walker());
    using traverser_type = gtensor::walker_forward_traverser<config_type,walker_type>;

    traverser_type traverser{t_.shape(), t_.create_walker()};

    auto f = [&traverser,&t_](){
        value_type r{0};
        auto n = t_.size();

        while(n!=0){
            if (traverser.index().size()==0){
                ++r;
            }
            --n;
            traverser.template next<traverse_order>();
        }
        return r;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

auto bench_random_traverser_no_deref_counter_loop = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using config_type = typename tensor_type::config_type;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;
    using traverse_order = decltype(order);


    using walker_type = decltype(t_.create_walker());
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order>;
    traverser_type traverser{t_.shape(), t_.descriptor().strides_div(traverse_order{}), t_.create_walker()};

    auto f = [&traverser,&t_](){
        value_type r{0};
        auto n = t_.size();
        while (n!=0){
            --n;
            traverser.next();
        }
        return *traverser;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

auto bench_forward_iterator_no_deref = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using config_type = typename tensor_type::config_type;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;
    using traverse_order = decltype(order);


    using walker_type = decltype(t_.create_walker());
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order>;
    // using iterator_type = forward_iterator<config_type, traverser_type>;
    // iterator_type it{traverser_type{t_.shape(), t_.descriptor().strides_div(traverse_order{}), t_.create_walker()},0};
    // iterator_type last{traverser_type{t_.shape(), t_.descriptor().strides_div(traverse_order{}), t_.create_walker()},t_.size()};
    using iterator_type = forward_iterator<config_type, walker_type, traverse_order>;
    iterator_type it{t_.create_walker(),t_.shape(),0};
    iterator_type last{t_.create_walker(),t_.shape(),t_.size()};

    auto f = [it,last]()mutable{
        for (;it!=last; ++it){}
        return last==it;
    };
    // auto f = [it,n]()mutable{
    //     while (n!=0){
    //         --n;
    //         ++it;
    //     }
    //     return it==it;
    // };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
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
        while (it!=last){
            ++it;
        }
        return last-it;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

}



TEMPLATE_TEST_CASE("benchmark_traverser_deep_expression","[benchmark_tensor]",
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>)
    // (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    // (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    // (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using layout = std::tuple_element_t<0,TestType>;
    using traverse_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,layout>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using benchmark_helpers::order_to_str;
    using gtensor::detail::shape_to_str;
    using benchmark_helpers::shapes;
    using benchmark_traverser_::bench_forward_traverser_no_deref;
    using benchmark_traverser_::bench_forward_traverser_no_deref_counter_loop;
    using benchmark_traverser_::bench_random_traverser_no_deref_counter_loop;
    using benchmark_traverser_::bench_iterator_no_deref;
    using benchmark_traverser_::bench_forward_iterator_no_deref;
    using benchmark_helpers::cpu_timer;

    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t_ = tensor_type(*it,2);
        auto t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;   //10
        std::cout<<std::endl<<"deep_expression  t_+t_+t_+t_+t_+t_+t_+t_+t_+t_ "<<order_to_str(layout{})<<" "<<shape_to_str(t.shape());
        bench_forward_iterator_no_deref(t,traverse_order{},"bench_forward_iterator_no_deref");

        //bench_random_traverser_no_deref_counter_loop(t,traverse_order{},"bench_random_traverser_no_deref_counter_loop");
        //bench_iterator_no_deref(t,traverse_order{},"bench_iterator_no_deref");

        //bench_forward_traverser_no_deref_counter_loop(t,traverse_order{},"bench_forward_traverser_no_deref_counter_loop");

    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total, ms "<<stop-start;
}