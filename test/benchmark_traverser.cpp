#include "benchmark_helpers.hpp"
#include "tensor.hpp"

namespace benchmark_traverser_{

// auto bench_traverser = [](const auto& t_, auto order, auto mes){
//     using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
//     using config_type = typename tensor_type::config_type;
//     using value_type = typename tensor_type::value_type;
//     using benchmark_helpers::order_to_str;
//     using benchmark_helpers::timing;

//     //auto a = t_.traverse_order_adapter(order);

//     using traverse_order = decltype(order);
//     using walker_type = decltype(t_.create_walker());
//     using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order>;
//     traverser_type traverser{t_.shape(), t_.descriptor().strides_div(traverse_order{}), t_.create_walker()};

//     auto f = [&traverser,&t_](){
//         value_type r{0};
//         auto n = t_.size();
//         while(n!=0){
//             r+=*traverser;
//             (void)traverser.next();
//             --n;
//         }
//         std::cout<<std::endl<<r;
//         return r;
//     };
//     auto dt = timing(f);
//     std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
// };

// auto bench_traverser = [](const auto& t_, auto order, auto mes){
//     using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
//     using config_type = typename tensor_type::config_type;
//     using value_type = typename tensor_type::value_type;
//     using benchmark_helpers::order_to_str;
//     using benchmark_helpers::timing;

//     //auto a = t_.traverse_order_adapter(order);

//     using walker_type = decltype(t_.create_walker());
//     using traverser_type = gtensor::walker_forward_traverser<config_type,walker_type>;
//     traverser_type traverser{t_.shape(), t_.create_walker()};

//     auto f = [&traverser,&t_](){
//         value_type r{0};
//         auto n = t_.size();
//         while(n!=0){
//             r+=*traverser;
//             (void)traverser.template next<decltype(order)>();
//             --n;
//         }
//         std::cout<<std::endl<<r;
//         return r;
//     };
//     auto dt = timing(f);
//     std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
// };

auto bench_traverser = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using config_type = typename tensor_type::config_type;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;
    using traverse_order = decltype(order);


    using walker_type = decltype(t_.create_walker());
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order>;
    traverser_type traverser{t_.shape(), t_.descriptor().strides_div(traverse_order{}), t_.create_walker()};

    auto f = [&traverser](){
        value_type r{0};
        do{
            r+=*traverser;
        }while(traverser.next());
        std::cout<<std::endl<<r;
        return r;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

auto bench_iterator = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;

    auto a = t_.traverse_order_adapter(order);

    auto f = [a](){
        value_type r{0};
        for(auto it=a.begin(),last=a.end(); it!=last; ++it){
            r+=*it;
        }
        std::cout<<std::endl<<r;
        return r;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

}



TEMPLATE_TEST_CASE("benchmark_traverser_deep_expression","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using order = typename tensor_type::order;
    using benchmark_helpers::order_to_str;
    using gtensor::detail::shape_to_str;
    using benchmark_helpers::shapes;
    using benchmark_traverser_::bench_traverser;
    using benchmark_traverser_::bench_iterator;
    using benchmark_helpers::cpu_timer;

    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t_ = tensor_type(*it,2);
        //auto t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;   //10
        auto t = t_+t_;
        std::cout<<std::endl<<"deep_expression  t_+t_+t_+t_+t_+t_+t_+t_+t_+t_ "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_iterator(t,c_order{},"iterator");
        bench_traverser(t,c_order{},"traverser");
        //bench_traverser(t,f_order{},"iterator");
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total, ms "<<stop-start;
}