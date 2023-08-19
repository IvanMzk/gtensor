#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"

namespace benchmark_trivial_iterator_{

auto bench_trivial_iterator = [](const auto& t_, auto order, auto mes){
    using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t_)>>;
    using value_type = typename tensor_type::value_type;
    using benchmark_helpers::cpu_timer;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;

    REQUIRE(t_.is_trivial());
    auto a = t_.traverse_order_adapter(order);
    auto f = [a](){
        value_type r{0};
        for(auto it=a.begin_trivial(),last=a.end_trivial(); it!=last; ++it){
            r+=*it;
        }
        return r;
    };
    auto dt = timing(f);
    std::cout<<std::endl<<mes<<" "<<order_to_str(order)<<" "<<dt.interval()<<" ms";
};

}   //end of namespace benchmark_trivial_iterator_

TEMPLATE_TEST_CASE("test_trivial_iterator_tensor","[benchmark_tensor]",
    gtensor::config::c_order
    //gtensor::config::f_order
)
{
    using value_type = double;
    using layout = TestType;
    using tensor_type = gtensor::tensor<value_type,layout>;
    using shape_type = typename tensor_type::shape_type;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using benchmark_helpers::order_to_str;
    using benchmark_helpers::timing;
    using gtensor::detail::shape_to_str;
    using helpers_for_testing::generate_lehmer;

    auto t = tensor_type(shape_type{100,3,2000,1000});
    generate_lehmer(t.begin(),t.end(),123);
    auto t1 = tensor_type(shape_type{3,2000,1000});
    generate_lehmer(t1.begin(),t1.end(),123);

    //auto e = t+t+t+t+t+t+t+t+t+t;
    //auto e = (t+1)*(t-1) - (t+2)*(t-2) + (t+3)*(t-3);
    //auto e = (t+t)/(t+t) + (t+t)/(t+t) + (t+t)/(t+t);
    //auto e = (t+t)/(t-1) + (t+t)/(t-1) + (t+t)/(t-1);
    //auto e = (((((((((t+1)+2)+3)+4)+5)+6)+7)+8)+9+10);
    //auto e = (t+1)/(t-1) + (t+2)/(t-2) + (t+3)/(t-3);
    //auto e = (t+1)/(t-1);
    auto e = (t-t1)/(t+t1);

    auto bench_trivial = [](const auto& t, auto order){
        auto f = [&t,&order](){
            auto a = t.traverse_order_adapter(order);
            value_type r = 0;
            for (auto it=a.begin_trivial(),last=a.end_trivial(); it!=last; ++it){
                r+=*it;
            }
            //REQUIRE(tensor_close(tensor_type(r),tensor_type(1.288511044226555e+18),1,1));
            //REQUIRE(tensor_close(tensor_type(r),tensor_type(600000005.1506355),1,1));
            return r;
        };
        auto dt = timing(f);
        std::cout<<std::endl<<"trivial iterator "<<order_to_str(order)<<" test time,ms "<<dt.interval();
    };
    auto bench = [](const auto& t, auto order){
        auto f = [&t,&order](){
            auto a = t.traverse_order_adapter(order);
            value_type r = 0;
            for (auto it=a.begin(),last=a.end(); it!=last; ++it){
                r+=*it;
            }
            //REQUIRE(tensor_close(tensor_type(r),tensor_type(1.288511044226555e+18),1,1));
            //REQUIRE(tensor_close(tensor_type(r),tensor_type(600000005.1506355),1,1));
            std::cout<<std::endl<<r;
            return r;
        };
        auto dt = timing(f);
        std::cout<<std::endl<<"iterator "<<order_to_str(order)<<" test time,ms "<<dt.interval();
    };
    //REQUIRE(e.is_trivial());
    std::cout<<std::endl<<"layout "<<order_to_str(layout{});
    bench(e,c_order{});
    //bench_trivial(e,c_order{});
    //bench_trivial(e,f_order{});

}




// TEMPLATE_TEST_CASE("benchmark_trivial_iterator_tensor","[benchmark_tensor]",
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
//     using benchmark_trivial_iterator_::bench_trivial_iterator;
//     using benchmark_helpers::cpu_timer;
//     using benchmark_helpers::fake_use;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t = tensor_type(*it,2);
//         std::cout<<std::endl<<"tensor "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_trivial_iterator(t,c_order{},"trivial iterator");
//         bench_trivial_iterator(t,f_order{},"trivial iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

// TEMPLATE_TEST_CASE("benchmark_trivial_iterator_expression","[benchmark_tensor]",
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
//     using benchmark_trivial_iterator_::bench_trivial_iterator;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=t_+t_;
//         std::cout<<std::endl<<"expression t+t "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_trivial_iterator(t,c_order{},"trivial iterator");
//         bench_trivial_iterator(t,f_order{},"trivial iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

// TEMPLATE_TEST_CASE("benchmark_trivial_iterator_deep_expression","[benchmark_tensor]",
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
//     using benchmark_trivial_iterator_::bench_trivial_iterator;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=t_+t_+t_+t_+t_+t_+t_+t_+t_+t_;   //10
//         std::cout<<std::endl<<"deep_expression  t_+t_+t_+t_+t_+t_+t_+t_+t_+t_ "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_trivial_iterator(t,c_order{},"trivial iterator");
//         bench_trivial_iterator(t,f_order{},"trivial iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

// TEMPLATE_TEST_CASE("benchmark_trivial_iterator_expression_with_scalar","[benchmark_tensor]",
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
//     using benchmark_trivial_iterator_::bench_trivial_iterator;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=t_+1;
//         std::cout<<std::endl<<"expression with scalar t+1 "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_trivial_iterator(t,c_order{},"trivial iterator");
//         bench_trivial_iterator(t,f_order{},"trivial iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

// TEMPLATE_TEST_CASE("benchmark_trivial_iterator_deep_expression_with_scalar","[benchmark_tensor]",
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
//     using benchmark_trivial_iterator_::bench_trivial_iterator;
//     using benchmark_helpers::cpu_timer;

//     auto start = cpu_timer{};
//     for (auto it=shapes.begin(), last=shapes.end(); it!=last; ++it){
//         auto t_ = tensor_type(*it,2);
//         auto t=((((((((((t_+1)+1)+1)+1)+1)+1)+1)+1)+1)+1);  //10
//         std::cout<<std::endl<<"expression with scalar ((((((((((t_+1)+1)+1)+1)+1)+1)+1)+1)+1)+1) "<<order_to_str(order{})<<" "<<shape_to_str(t.shape());
//         bench_trivial_iterator(t,c_order{},"trivial iterator");
//         bench_trivial_iterator(t,f_order{},"trivial iterator");
//     }
//     auto stop = cpu_timer{};
//     std::cout<<std::endl<<"total, ms "<<stop-start;
// }

