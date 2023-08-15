#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"


namespace benchmark_copy_{

auto bench_copy = [](const auto& t_, auto order, auto mes){
    using benchmark_helpers::cpu_timer;
    using benchmark_helpers::order_to_str;
    auto start = cpu_timer{};
    auto res = t_.copy(order);
    auto stop = cpu_timer{};
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<stop-start<<" ms";
    return res.size();
};

}   //end of namespace benchmark_copy_

// TEMPLATE_TEST_CASE("benchmark_copy_tensor","[benchmark_tensor]",
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
//     using benchmark_copy_::bench_copy;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t = tensor_type(*it,2);
//         std::cout<<std::endl<<"tensor "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_copy(t,c_order{},"copy");
//         bench_copy(t,f_order{},"copy");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

// TEMPLATE_TEST_CASE("benchmark_copy_expression","[benchmark_tensor]",
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
//     using benchmark_copy_::bench_copy;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t = t_+t_;
//         std::cout<<std::endl<<"expression t+t "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_copy(t,c_order{},"copy");
//         bench_copy(t,f_order{},"copy");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

TEMPLATE_TEST_CASE("benchmark_copy_deep_expression","[benchmark_tensor]",
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
    using benchmark_copy_::bench_copy;
    using benchmark_helpers::cpu_timer;

    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t_ = tensor_type(*it,2);
        auto t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;   //10
        std::cout<<std::endl<<"deep expression t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_ "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_copy(t,c_order{},"copy");
        bench_copy(t,f_order{},"copy");
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total, ms "<<stop-start;
}