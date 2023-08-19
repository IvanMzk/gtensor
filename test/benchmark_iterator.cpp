#include "benchmark_helpers.hpp"
#include "tensor.hpp"

namespace benchmark_iterator_{

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
        return r;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

}   //end of namespace benchmark_iterator_

TEMPLATE_TEST_CASE("benchmark_iterator_tensor","[benchmark_tensor]",
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
    using benchmark_iterator_::bench_iterator;
    using benchmark_helpers::cpu_timer;

    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t = tensor_type(*it,2);
        std::cout<<std::endl<<"tensor "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_iterator(t,c_order{},"iterator");
        bench_iterator(t,f_order{},"iterator");
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total, ms "<<stop-start;
}

TEMPLATE_TEST_CASE("benchmark_iterator_expression","[benchmark_tensor]",
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
    using benchmark_iterator_::bench_iterator;
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

TEMPLATE_TEST_CASE("benchmark_iterator_deep_expression","[benchmark_tensor]",
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
    using benchmark_iterator_::bench_iterator;
    using benchmark_helpers::cpu_timer;

    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t_ = tensor_type(*it,2);
        auto t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;   //10
        std::cout<<std::endl<<"deep_expression  t_+t_+t_+t_+t_+t_+t_+t_+t_+t_ "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_iterator(t,c_order{},"iterator");
        bench_iterator(t,f_order{},"iterator");
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total, ms "<<stop-start;
}

TEMPLATE_TEST_CASE("benchmark_iterator_expression_with_scalar","[benchmark_tensor]",
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
    using benchmark_iterator_::bench_iterator;
    using benchmark_helpers::cpu_timer;

    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t_ = tensor_type(*it,2);
        auto t=t_+1;
        std::cout<<std::endl<<"expression with scalar t+1 "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_iterator(t,c_order{},"iterator");
        bench_iterator(t,f_order{},"iterator");
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total, ms "<<stop-start;
}

TEMPLATE_TEST_CASE("benchmark_iterator_deep_expression_with_scalar","[benchmark_tensor]",
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
    using benchmark_iterator_::bench_iterator;
    using benchmark_helpers::cpu_timer;

    auto start = cpu_timer{};
    for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
        auto t_ = tensor_type(*it,2);
        auto t=((((((((((t_+1)+1)+1)+1)+1)+1)+1)+1)+1)+1);  //10
        std::cout<<std::endl<<"expression with scalar ((((((((((t_+1)+1)+1)+1)+1)+1)+1)+1)+1)+1) "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
        bench_iterator(t,c_order{},"iterator");
        bench_iterator(t,f_order{},"iterator");
    }
    auto stop = cpu_timer{};
    std::cout<<std::endl<<"total, ms "<<stop-start;
}

