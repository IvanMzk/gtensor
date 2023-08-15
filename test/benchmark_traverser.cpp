#include "benchmark_helpers.hpp"
#include "tensor.hpp"

namespace benchmark_traverser_{

const std::vector<std::vector<int>> shapes{
    std::vector<int>{100000000,3,1,2},
    std::vector<int>{10000000,3,1,20},
    std::vector<int>{1000000,3,10,20},
    std::vector<int>{100000,3,100,20},
    std::vector<int>{10000,3,100,200},
    std::vector<int>{1000,3,1000,200},
    std::vector<int>{100,3,1000,2000},
    std::vector<int>{50,6,1000,2000}
};

auto bench_iterator = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::cpu_timer;
    using benchmark_helpers::order_to_str;

    auto a = t_.traverse_order_adapter(order);
    value_type v{0};
    auto start = cpu_timer{};
    for(auto it=a.begin(),last=a.end(); it!=last; ++it){
        v+=*it;
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<stop-start<<" ms";
    return v;
};

}

// TEMPLATE_TEST_CASE("benchmark_traverser_tensor_big","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using config_type = typename tensor_type::config_type;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::benchmark;
//     using benchmark_helpers::cpu_timer;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::axes_to_str;

//     auto make_traverser = [](const auto& t_){
//         using walker_type = decltype(t_.create_walker());
//         //return gtensor::walker_forward_range_traverser<config_type,walker_type>{t_.shape(),t_.create_walker(),0,t_.dim()};
//         return gtensor::walker_forward_traverser<config_type,walker_type>{t_.shape(),t_.create_walker()};
//     };

//     auto bench_traverser = [make_traverser](const auto& t_, auto order){
//         auto tr = make_traverser(t_);
//         using value_type = std::remove_cv_t<std::remove_reference_t<decltype(*tr)>>;
//         using order_type = decltype(order);
//         value_type v{0};
//         auto start = cpu_timer{};
//         do{
//             v+=*tr;
//         }while(tr.template next<order_type>());
//         auto stop = cpu_timer{};
//         std::cout<<std::endl<<"traverser "<<order_to_str(order)<<" "<<stop-start<<" ms";
//         return v;
//     };

//     using benchmark_traverser_::shapes;
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t = tensor_type(*it,2);
//         std::cout<<std::endl<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_traverser(t,c_order{});
//         bench_traverser(t,f_order{});
//     }
// }

// TEMPLATE_TEST_CASE("benchmark_iterator_tensor_big","[benchmark_tensor]",
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
//     using benchmark_traverser_::shapes;
//     using benchmark_traverser_::bench_iterator;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t = tensor_type(*it,2);
//         std::cout<<std::endl<<"tensor "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_iterator(t,c_order{},"iterator");
//         bench_iterator(t,f_order{},"iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

TEMPLATE_TEST_CASE("benchmark_iterator_expression_big","[benchmark_tensor]",
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
    using benchmark_traverser_::shapes;
    using benchmark_traverser_::bench_iterator;
    using benchmark_helpers::cpu_timer;

    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t_ = tensor_type(*it,2);
        auto t=t_+t_;
        std::cout<<std::endl<<"expression t+t "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_iterator(t,c_order{},"iterator");
        bench_iterator(t,f_order{},"iterator");
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total, ms "<<stop-start;
}




// TEMPLATE_TEST_CASE("benchmark_traverser_expression_big","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using config_type = typename tensor_type::config_type;
//     using shape_type = typename tensor_type::shape_type;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::benchmark;
//     using benchmark_helpers::cpu_timer;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::axes_to_str;

//     auto make_traverser = [](const auto& t_){
//         using walker_type = decltype(t_.create_walker());
//         //return gtensor::walker_forward_range_traverser<config_type,walker_type>{t_.shape(),t_.create_walker(),0,t_.dim()};
//         return gtensor::walker_forward_traverser<config_type,walker_type>{t_.shape(),t_.create_walker()};
//     };

//     auto bench_traverser = [make_traverser](const auto& t_, auto order){
//         auto tr = make_traverser(t_);
//         using value_type = std::remove_cv_t<std::remove_reference_t<decltype(*tr)>>;
//         using order_type = decltype(order);
//         value_type v{0};
//         auto start = cpu_timer{};
//         do{
//             v+=*tr;
//         }while(tr.template next<order_type>());
//         auto stop = cpu_timer{};
//         std::cout<<std::endl<<"traverser "<<order_to_str(order)<<" "<<stop-start<<" ms";
//         return v;
//     };

//     std::vector<shape_type> shapes{
//         shape_type{100000000,3,1,2},
//         shape_type{10000000,3,1,20},
//         shape_type{1000000,3,10,20},
//         shape_type{100000,3,100,20},
//         shape_type{10000,3,100,200},
//         shape_type{1000,3,1000,200},
//         shape_type{100,3,1000,2000},
//         shape_type{50,6,1000,2000}
//     };
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         //auto t = 2*t_+3*t_*(2*t_+1)+4*(t_+2)*(t_-2)*(t_+3);
//         auto t = t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;
//         std::cout<<std::endl<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_traverser(t,c_order{});
//         bench_traverser(t,f_order{});
//     }
// }

// TEMPLATE_TEST_CASE("benchmark_iterator_expression_big","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using shape_type = typename tensor_type::shape_type;
//     using order = typename tensor_type::order;
//     using benchmark_helpers::benchmark;
//     using benchmark_helpers::cpu_timer;
//     using benchmark_helpers::order_to_str;
//     using gtensor::detail::shape_to_str;
//     using benchmark_helpers::axes_to_str;

//     auto bench_iterator = [](const auto& t_, auto order){
//         using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
//         using value_type = typename tensor_type::value_type;
//         auto a = t_.traverse_order_adapter(order);
//         value_type v{0};
//         auto start = cpu_timer{};
//         for(auto it=a.begin(),last=a.end(); it!=last; ++it){
//             v+=*it;
//         }
//         auto stop = cpu_timer{};
//         std::cout<<std::endl<<"iterator "<<order_to_str(order)<<" "<<stop-start<<" ms";
//         return v;
//     };

//     std::vector<shape_type> shapes{
//         shape_type{100000000,3,1,2},
//         shape_type{10000000,3,1,20},
//         shape_type{1000000,3,10,20},
//         shape_type{100000,3,100,20},
//         shape_type{10000,3,100,200},
//         shape_type{1000,3,1000,200},
//         shape_type{100,3,1000,2000},
//         shape_type{50,6,1000,2000}
//     };
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         //auto t = 2*t_+3*t_*(2*t_+1)+4*(t_+2)*(t_-2)*(t_+3);
//         auto t = t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;
//         std::cout<<std::endl<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_iterator(t,c_order{});
//         bench_iterator(t,f_order{});
//     }
// }

