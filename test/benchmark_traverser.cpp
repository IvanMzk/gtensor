#include "benchmark_helpers.hpp"
#include "tensor.hpp"


// TEMPLATE_TEST_CASE("benchmark_iterator","[benchmark_traverser]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::benchmark;

//     auto t = tensor_type({100,100,1000},2);

//     auto bench_iterator = [](const auto& t_, auto order){
//         using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
//         using value_type = typename tensor_type::value_type;
//         using order_type = decltype(order);
//         auto a = t_.traverse_order_adapter(order_type{});
//         value_type v{0};
//         for(auto it=a.begin(),last=a.end(); it!=last; ++it){
//             v+=*it;
//         }
//         return v;
//     };

//     benchmark("iterator_c_next_10E7",bench_iterator,t,c_order{});
//     benchmark("iterator_f_next_10E7",bench_iterator,t,f_order{});
// }

TEMPLATE_TEST_CASE("benchmark_traverser","[benchmark_traverser]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using benchmark_helpers::benchmark;

    auto t = tensor_type({100,100,1000},2);

    auto make_traverser = [](const auto& t_){
        using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
        using walker_type = decltype(t_.create_walker());
        return gtensor::walker_forward_traverser<typename tensor_type::config_type,walker_type>{t_.shape(),t_.create_walker()};
    };

    auto bench_traverser = [make_traverser](const auto& t_, auto order){
        auto tr = make_traverser(t_);
        using value_type = std::remove_cv_t<std::remove_reference_t<decltype(*tr)>>;
        using order_type = decltype(order);
        value_type v{0};
        do{
            v+=*tr;
        }while(tr.template next<order_type>());
        return v;
    };

    benchmark("traverser_c_next_10E7",bench_traverser,t,c_order{});
    benchmark("traverser_f_next_10E7",bench_traverser,t,f_order{});
}

// TEMPLATE_TEST_CASE("benchmark_traverser_","[benchmark_traverser]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::benchmark;

//     auto t = tensor_type({100,100,1000},2);

//     auto make_traverser = [](const auto& t_){
//         using walker_type = decltype(t_.create_walker());
//         return gtensor::walker_forward_traverser_<walker_type>{t_.shape(),t_.create_walker()};
//     };

//     auto bench_traverser = [make_traverser](const auto& t_, auto order){
//         auto tr = make_traverser(t_);
//         using value_type = std::remove_cv_t<std::remove_reference_t<decltype(*tr)>>;
//         using order_type = decltype(order);
//         value_type v{0};
//         do{
//             v+=*tr;
//         }while(tr.template next<order_type>());
//         return v;
//     };

//     benchmark("traverser__c_next_10E7",bench_traverser,t,c_order{});
//     benchmark("traverser__f_next_10E7",bench_traverser,t,f_order{});
// }

TEMPLATE_TEST_CASE("benchmark_range_traverser","[benchmark_traverser]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using benchmark_helpers::benchmark;

    auto t = tensor_type({100,100,1000},2);

    auto make_traverser = [](const auto& t_){
        using walker_type = decltype(t_.create_walker());
        return gtensor::walker_forward_range_traverser<walker_type>{t_.shape(),t_.create_walker(),0,t_.dim()};
    };

    auto bench_traverser = [make_traverser](const auto& t_, auto order){
        auto tr = make_traverser(t_);
        using value_type = std::remove_cv_t<std::remove_reference_t<decltype(*tr)>>;
        using order_type = decltype(order);
        value_type v{0};
        do{
            v+=*tr;
        }while(tr.template next<order_type>());
        return v;
    };

    benchmark("range_traverser_c_next_10E7",bench_traverser,t,c_order{});
    benchmark("range_traverser_f_next_10E7",bench_traverser,t,f_order{});
}

