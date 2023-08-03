#include "benchmark_helpers.hpp"
#include "tensor.hpp"

namespace benchmark_expression_template_helpers{
}   //end of namespace benchmark_expression_template_helpers


TEST_CASE("test_benchmark_helpers_make_tree","[test_benchmark_helpers]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using benchmark_helpers::make_asymmetric_tree;

    REQUIRE(make_asymmetric_tree<1>(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}}) == tensor_type{{2,4,6},{8,10,12}});
    REQUIRE(make_asymmetric_tree<1>(tensor_type{{1,2,3},{4,5,6}},tensor_type{7,8,9}) == tensor_type{{8,10,12},{11,13,15}});
    REQUIRE(make_asymmetric_tree<2>(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}}) == tensor_type{{3,6,9},{12,15,18}});
    REQUIRE(make_asymmetric_tree<2>(tensor_type{{1,2,3},{4,5,6}},tensor_type{7,8,9}) == tensor_type{{15,18,21},{18,21,24}});
    REQUIRE(make_asymmetric_tree<3>(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}}) == tensor_type{{4,8,12},{16,20,24}});
    REQUIRE(make_asymmetric_tree<3>(tensor_type{{1,2,3},{4,5,6}},tensor_type{7,8,9}) == tensor_type{{22,26,30},{25,29,33}});
    REQUIRE(make_asymmetric_tree<4>(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}}) == tensor_type{{5,10,15},{20,25,30}});
    REQUIRE(make_asymmetric_tree<4>(tensor_type{{1,2,3},{4,5,6}},tensor_type{7,8,9}) == tensor_type{{29,34,39},{32,37,42}});

}

// TEMPLATE_TEST_CASE("benchmark_expression_view_iterator","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::make_asymmetric_tree;
//     using benchmark_helpers::benchmark;

//     auto t1 = tensor_type({10,100,1000},2);
//     auto t2 = tensor_type({100,1000},1);

//     auto tree_50_1E6 = make_asymmetric_tree<50>(t1,t2);

//     auto bench_iteration_deref = [](const auto& t, auto order){
//         using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t)>>;
//         using value_type = typename tensor_type::value_type;
//         auto a = t.template traverse_order_adapter<decltype(order)>();
//         value_type v{0};
//         for (auto it=a.begin(),last=a.end(); it!=last; ++it){
//             v += *it;
//         }
//         return v;
//     };

//     benchmark("c_iteration_deref_depth50_10E6",bench_iteration_deref,tree_50_1E6,c_order{});
//     benchmark("f_iteration_deref_depth50_10E6",bench_iteration_deref,tree_50_1E6,f_order{});
// }

// TEMPLATE_TEST_CASE("benchmark_expression_view_reverse_iterator","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::make_asymmetric_tree;
//     using benchmark_helpers::benchmark;

//     auto t1 = tensor_type({10,100,1000},2);
//     auto t2 = tensor_type({100,1000},1);

//     auto tree_50_1E6 = make_asymmetric_tree<50>(t1,t2);

//     auto bench_iteration_deref = [](const auto& t, auto order){
//         using tensor_type = std::remove_cv_t<std::remove_reference_t<decltype(t)>>;
//         using value_type = typename tensor_type::value_type;
//         auto a = t.template traverse_order_adapter<decltype(order)>();
//         value_type v{0};
//         for (auto it=a.rbegin(),last=a.rend(); it!=last; ++it){
//             v += *it;
//         }
//         return v;
//     };

//     benchmark("c_reverse_iteration_deref_depth50_10E6",bench_iteration_deref,tree_50_1E6,c_order{});
//     benchmark("f_reverse_iteration_deref_depth50_10E6",bench_iteration_deref,tree_50_1E6,f_order{});
// }

// TEMPLATE_TEST_CASE("benchmark_expression_view_copy","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::make_asymmetric_tree;
//     using benchmark_helpers::benchmark;

//     auto bench_copy = [](const auto& t, auto order){
//         return t.copy(order);
//     };

//     auto t1 = tensor_type({10,100,1000},2);
//     auto t2 = tensor_type({100,1000},1);
//     auto tree_50_1E6 = make_asymmetric_tree<50>(t1,t2);
//     benchmark("c_copy_depth50_10E6",bench_copy,tree_50_1E6,c_order{});
//     benchmark("f_copy_depth50_10E6",bench_copy,tree_50_1E6,f_order{});

//     // auto t1 = tensor_type({5,1,100},2);
//     // auto t2 = tensor_type({100},1);
//     // auto tree_50_1E2 = make_asymmetric_tree<50>(t1,t2);
//     // benchmark("c_copy_depth50_10E2",bench_copy,tree_50_1E2,c_order{});
//     // benchmark("f_copy_depth50_10E2",bench_copy,tree_50_1E2,f_order{});
// }

// TEMPLATE_TEST_CASE("benchmark_tensor_copy","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::make_asymmetric_tree;
//     using benchmark_helpers::benchmark;

//     auto bench_copy = [](const auto& t, auto order){
//         return t.copy(order);
//     };

//     auto t = tensor_type({10,100,1000},2);
//     benchmark("c_copy_10E6",bench_copy,t,c_order{});
//     benchmark("f_copy_10E6",bench_copy,t,f_order{});
// }

// TEMPLATE_TEST_CASE("benchmark_tensor_transpose_view_copy","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::make_asymmetric_tree;
//     using benchmark_helpers::benchmark;

//     auto bench_copy = [](const auto& t, auto order){
//         return t.copy(order);
//     };

//     //auto v = tensor_type({10,100,1000},2).transpose();
//     auto v = tensor_type({10,100,1000},2).transpose().transpose(2,0,1).transpose(0,2,1);
//     //auto v = tensor_type({10,100,1000},2).transpose().transpose(1,0,2);
//     //auto v = tensor_type({10,100,1000},2)({{0,-1,1},{0,-1,2},{0,-1,1}});
//     //auto v = tensor_type({10,100,1000},2)({{0,-1,1},{0,-1,1},{0,-1,1}})({{0,-1,1},{0,-1,1},{0,-1,1}});
//     //std::cout<<std::endl<<gtensor::detail::shape_to_str(v.shape());

//     benchmark("c_copy_transpose_view_10E6",bench_copy,v,c_order{});
//     benchmark("f_copy_transpose_view_10E6",bench_copy,v,f_order{});
// }

// TEMPLATE_TEST_CASE("benchmark_tensor_slice_view_copy","[benchmark_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using tensor_type = gtensor::tensor<value_type,TestType>;
//     using benchmark_helpers::make_asymmetric_tree;
//     using benchmark_helpers::benchmark;

//     auto bench_copy = [](const auto& t, auto order){
//         return t.copy(order);
//     };

//     auto v1 = tensor_type({10,100,1000},2)({{0,-1},{},{0,-1}});
//     //auto v2 = tensor_type({10,100,1000},2)({{},{},{0,-1}})({{},{},{0,-1}});
//     // auto v3 = tensor_type({10,100,1000},2)({{},{},{}})({{},{},{}})({{},{},{}});

//     benchmark("c_copy_slice1_view_10E6",bench_copy,v1,c_order{});
//     benchmark("f_copy_slice1_view_10E6",bench_copy,v1,f_order{});
//     //benchmark("c_copy_slice2_view_10E6",bench_copy,v2,c_order{});
//     //benchmark("f_copy_slice2_view_10E6",bench_copy,v2,f_order{});
//     // benchmark("c_copy_slice3_view_10E6",bench_copy,v3,c_order{});
//     // benchmark("f_copy_slice3_view_10E6",bench_copy,v3,f_order{});
// }


TEMPLATE_TEST_CASE("benchmark_tensor_tree_slice_view_copy","[benchmark_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using benchmark_helpers::make_asymmetric_tree;
    using benchmark_helpers::benchmark;

    auto bench_copy = [](const auto& t, auto order){
        return t.copy(order);
    };

    auto t1 = tensor_type({10,100,1000},2);
    auto t2 = tensor_type({100,1000},1);
    auto tree_50_1E6 = make_asymmetric_tree<50>(t1,t2);
    auto v1 = tree_50_1E6({{0,-1},{},{0,-1}});


    benchmark("c_copy_tree_slice1_view_10E6",bench_copy,v1,c_order{});
    benchmark("f_copy_tree_slice1_view_10E6",bench_copy,v1,f_order{});
    //benchmark("c_copy_slice2_view_10E6",bench_copy,v2,c_order{});
    //benchmark("f_copy_slice2_view_10E6",bench_copy,v2,f_order{});
    // benchmark("c_copy_slice3_view_10E6",bench_copy,v3,c_order{});
    // benchmark("f_copy_slice3_view_10E6",bench_copy,v3,f_order{});
}

