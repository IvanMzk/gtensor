#include "gtensor.hpp"
#include "test_config.hpp"
#include "benchmark_helpers.hpp"
#include "expression_template_test_helpers.hpp"

namespace benchmark_expression_template_helpers{

using gtensor::detail::begin_multiindex;
using gtensor::detail::end_multiindex;
using gtensor::detail::begin_flatindex;
using gtensor::detail::end_flatindex;
using gtensor::detail::begin_flatindex_indexer;
using gtensor::detail::end_flatindex_indexer;

template<typename T>
struct test_tensor_broadcast : public T{
    test_tensor_broadcast(const T& base):
        tensor{base}
    {}
    auto& engine()const{return impl()->engine();}
    bool is_trivial()const{return engine().is_trivial();}
    auto begin()const{return begin_multiindex(engine());}
    auto end()const{return end_multiindex(engine());}
};
template<typename T>
struct test_tensor_trivial : public T{
    test_tensor_trivial(const T& base):
        tensor{base}
    {}
    auto& engine()const{return impl()->engine();}
    bool is_trivial()const{return engine().is_trivial();}
    auto begin()const{return begin_flatindex(engine());}
    auto end()const{return end_flatindex(engine());}
};

template<typename T>
struct test_tensor_indexer : public T{
    test_tensor_indexer(const T& base):
        tensor{base}
    {}
    auto& engine()const{return impl()->engine();}
    bool is_trivial()const{return engine().is_trivial();}
    auto begin()const{return begin_flatindex_indexer(engine());}
    auto end()const{return end_flatindex_indexer(engine());}
};

template<typename T>
auto make_test_tensor_broadcast(T&& t){return test_tensor_broadcast<std::decay_t<T>>{t};}
template<typename T>
auto make_test_tensor_trivial(T&& t){return test_tensor_trivial<std::decay_t<T>>{t};}
template<typename T>
auto make_test_tensor_indexer(T&& t){return test_tensor_indexer<std::decay_t<T>>{t};}

}   //end of namespace benchmark_expression_template_helpers

// TEST_CASE("benchmark_expression_template_trivial_tree","[benchmark_expression_template]"){
//     using value_type = float;
//     using test_config_type = typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type;
//     using tensor_type = gtensor::tensor<value_type,test_config_type>;
//     using benchmark_expression_template_helpers::test_tensor_broadcast;
//     using benchmark_expression_template_helpers::test_tensor_trivial;
//     using benchmark_expression_template_helpers::make_test_tensor_broadcast;
//     using benchmark_expression_template_helpers::make_test_tensor_trivial;
//     using benchmark_expression_template_helpers::make_test_tensor_indexer;
//     using benchmark_helpers::asymmetric_tree_maker;
//     using benchmark_helpers::symmetric_tree_maker;
//     using benchmark_helpers::benchmark_with_making_iter;
//     using benchmark_helpers::making_iter_iterate_deref;
//     using benchmark_helpers::making_iter_reverse_iterate_deref;

//     //auto benchmark_worker = making_iter_iterate_deref;
//     auto benchmark_worker = making_iter_reverse_iterate_deref;

//     benchmark_with_making_iter(
//         make_test_tensor_broadcast(asymmetric_tree_maker<50>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f))),
//         "asymmetric_tree_trivial_depth50_multiiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_trivial(asymmetric_tree_maker<50>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f))),
//         "asymmetric_tree_trivial_depth50_flatiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_indexer(asymmetric_tree_maker<50>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f))),
//         "asymmetric_tree_trivial_depth50_flatiter_indexer",
//         benchmark_worker
//     );


//     benchmark_with_making_iter(
//         make_test_tensor_trivial(
//             asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f)({{}}) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f)({{}})
//         ),
//         "asymmetric_tree_trivial_view_slice_operand_depth50_flatiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_broadcast(
//             asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f)({{}}) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f)({{}})
//         ),
//         "asymmetric_tree_trivial_view_slice_operand_depth50_multiiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_trivial(
//             asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f).transpose(0,1) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f).transpose(0,1)
//         ),
//         "asymmetric_tree_trivial_view_transpose_operand_depth50_flatiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_broadcast(
//             asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f).transpose(0,1) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f).transpose(0,1)
//         ),
//         "asymmetric_tree_trivial_view_transpose_operand_depth50_multiiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_trivial(
//             asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f)() +
//             tensor_type(std::vector<int>{10,10000}, 0.0f)()
//         ),
//         "asymmetric_tree_trivial_view_subdim_operand_depth50_flatiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_broadcast(
//             asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f)() +
//             tensor_type(std::vector<int>{10,10000}, 0.0f)()
//         ),
//         "asymmetric_tree_trivial_view_subdim_operand_depth50_multiiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_trivial(
//             asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f).reshape(10,10000) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f).reshape(10,10000)
//         ),
//         "asymmetric_tree_trivial_view_reshape_operand_depth50_flatiter",
//         benchmark_worker
//     );
//     benchmark_with_making_iter(
//         make_test_tensor_broadcast(
//             asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f).reshape(10,10000) +
//             tensor_type(std::vector<int>{10,10000}, 0.0f).reshape(10,10000)
//         ),
//         "asymmetric_tree_trivial_view_reshape_operand_depth50_multiiter",
//         benchmark_worker
//     );
// }

TEST_CASE("benchmark_expression_template_view","[benchmark_expression_template]"){
    using value_type = float;
    using test_config_type = typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type;
    using tensor_type = gtensor::tensor<value_type,test_config_type>;
    using benchmark_expression_template_helpers::test_tensor_broadcast;
    using benchmark_expression_template_helpers::test_tensor_trivial;
    using benchmark_expression_template_helpers::make_test_tensor_broadcast;
    using benchmark_expression_template_helpers::make_test_tensor_trivial;
    using benchmark_expression_template_helpers::make_test_tensor_indexer;
    using benchmark_helpers::asymmetric_tree_maker;
    using benchmark_helpers::symmetric_tree_maker;
    using benchmark_helpers::benchmark_with_making_iter;
    using benchmark_helpers::making_iter_iterate_deref;
    using benchmark_helpers::making_iter_reverse_iterate_deref;

    auto benchmark_worker = making_iter_iterate_deref;

    // auto trivial = []{return asymmetric_tree_maker<1>{}(tensor_type({100,100}, 0.0f),tensor_type({100,100}, 0.0f));};
    // auto trivial_converting_view_operand = []{return asymmetric_tree_maker<1>{}(tensor_type({100,100}, 0.0f).transpose(1,0),tensor_type({100,100}, 0.0f));};
    // auto not_trivial = []{return asymmetric_tree_maker<1>{}(tensor_type({10,100,10}, 0.0f),tensor_type({100,10}, 0.0f));};
    auto trivial = []{return asymmetric_tree_maker<1>{}(tensor_type({1000,1000}, 0.0f),tensor_type({1000,1000}, 0.0f));};
    auto trivial_converting_view_operand = []{return asymmetric_tree_maker<1>{}(tensor_type({1000,1000}, 0.0f).transpose(1,0),tensor_type({1000,1000}, 0.0f));};
    auto not_trivial = []{return asymmetric_tree_maker<1>{}(tensor_type({100,100,100}, 0.0f),tensor_type({100,100}, 0.0f));};

    //benchmark efficiency view of expression
    //must be run with and without dispatch based on is_trivial of expression
    //view requires making its parent indexer
    //in general expression indexer is not efficient, every flat subscription must be transformed to multiindex, such transformation uses division
    //for trivial expression trivial indexer, that use only flat index, can be used, but polymorphic wrapper needed to have single return type in interface
    //benchmark result:
    //wrapper has small overhead for non trivial expresion, but using trivial indexer with wrapper 2-3 times faster

    //view of trivial expression
    benchmark_with_making_iter(trivial().transpose(),"transpose_view_of_trivial_expression",benchmark_worker);
    benchmark_with_making_iter(trivial()({{}}),"slice_view_of_trivial_expression",benchmark_worker);
    benchmark_with_making_iter(trivial().reshape(),"reshape_view_of_trivial_expression",benchmark_worker);
    benchmark_with_making_iter(trivial()(),"subdim_view_of_trivial_expression",benchmark_worker);

    //view of not trivial expression
    benchmark_with_making_iter(not_trivial().transpose(),"transpose_view_of_not_trivial_expression",benchmark_worker);
    benchmark_with_making_iter(not_trivial()({{}}),"slice_view_of_not_trivial_expression",benchmark_worker);
    benchmark_with_making_iter(not_trivial().reshape(),"reshape_view_of_not_trivial_expression",benchmark_worker);
    benchmark_with_making_iter(not_trivial()(),"subdim_view_of_not_trivial_expression",benchmark_worker);

    //view of not trivial expression with converting view operand
    benchmark_with_making_iter(trivial_converting_view_operand().transpose(),"transpose_view_of_trivial_converting_view_operand_expression",benchmark_worker);
    benchmark_with_making_iter(trivial_converting_view_operand()({{}}),"slice_view_of_trivial_converting_view_operand_expression",benchmark_worker);
    benchmark_with_making_iter(trivial_converting_view_operand().reshape(),"reshape_view_of_trivial_converting_view_operand_expression",benchmark_worker);
    benchmark_with_making_iter(trivial_converting_view_operand()(),"subdim_view_of_trivial_converting_view_operand_expression",benchmark_worker);
}