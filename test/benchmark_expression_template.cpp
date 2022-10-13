#include "gtensor.hpp"
#include "test_config.hpp"
#include "benchmark_helpers.hpp"
#include "expression_template_test_helpers.hpp"

namespace benchmark_expression_template_helpers{

using gtensor::detail::begin_multiindex;
using gtensor::detail::end_multiindex;
using gtensor::detail::begin_flatindex;
using gtensor::detail::end_flatindex;

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
auto make_test_tensor_broadcast(T&& t){return test_tensor_broadcast<std::decay_t<T>>{t};}
template<typename T>
auto make_test_tensor_trivial(T&& t){return test_tensor_trivial<std::decay_t<T>>{t};}

}   //end of namespace benchmark_expression_template_helpers

TEST_CASE("benchmark_expression_template_trivial_tree","[benchmark_expression_template]"){
    using value_type = float;
    using test_config_type = typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type;
    using tensor_type = gtensor::tensor<value_type,test_config_type>;
    using benchmark_expression_template_helpers::test_tensor_broadcast;
    using benchmark_expression_template_helpers::test_tensor_trivial;
    using benchmark_expression_template_helpers::make_test_tensor_broadcast;
    using benchmark_expression_template_helpers::make_test_tensor_trivial;
    using benchmark_helpers::asymmetric_tree_maker;
    using benchmark_helpers::symmetric_tree_maker;
    using benchmark_helpers::benchmark_with_making_iter;
    using benchmark_helpers::making_iter_iterate_deref;
    using benchmark_helpers::making_iter_reverse_iterate_deref;

    //auto benchmark_worker = making_iter_iterate_deref;
    auto benchmark_worker = making_iter_reverse_iterate_deref;

    benchmark_with_making_iter(
        make_test_tensor_broadcast(asymmetric_tree_maker<50>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f))),
        "asymmetric_tree_trivial_depth50_multiiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_trivial(asymmetric_tree_maker<50>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f))),
        "asymmetric_tree_trivial_depth50_flatiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_trivial(
            asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
            tensor_type(std::vector<int>{10,10000}, 0.0f)({{}}) +
            tensor_type(std::vector<int>{10,10000}, 0.0f)({{}})
        ),
        "asymmetric_tree_trivial_view_slice_operand_depth50_flatiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_broadcast(
            asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
            tensor_type(std::vector<int>{10,10000}, 0.0f)({{}}) +
            tensor_type(std::vector<int>{10,10000}, 0.0f)({{}})
        ),
        "asymmetric_tree_trivial_view_slice_operand_depth50_multiiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_trivial(
            asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
            tensor_type(std::vector<int>{10,10000}, 0.0f).transpose(0,1) +
            tensor_type(std::vector<int>{10,10000}, 0.0f).transpose(0,1)
        ),
        "asymmetric_tree_trivial_view_transpose_operand_depth50_flatiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_broadcast(
            asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
            tensor_type(std::vector<int>{10,10000}, 0.0f).transpose(0,1) +
            tensor_type(std::vector<int>{10,10000}, 0.0f).transpose(0,1)
        ),
        "asymmetric_tree_trivial_view_transpose_operand_depth50_multiiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_trivial(
            asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
            tensor_type(std::vector<int>{10,10000}, 0.0f)() +
            tensor_type(std::vector<int>{10,10000}, 0.0f)()
        ),
        "asymmetric_tree_trivial_view_subdim_operand_depth50_flatiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_broadcast(
            asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
            tensor_type(std::vector<int>{10,10000}, 0.0f)() +
            tensor_type(std::vector<int>{10,10000}, 0.0f)()
        ),
        "asymmetric_tree_trivial_view_subdim_operand_depth50_multiiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_trivial(
            asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
            tensor_type(std::vector<int>{10,10000}, 0.0f).reshape(10,10000) +
            tensor_type(std::vector<int>{10,10000}, 0.0f).reshape(10,10000)
        ),
        "asymmetric_tree_trivial_view_reshape_operand_depth50_flatiter",
        benchmark_worker
    );
    benchmark_with_making_iter(
        make_test_tensor_broadcast(
            asymmetric_tree_maker<48>{}(tensor_type(std::vector<int>{10,10000}, 0.0f),tensor_type(std::vector<int>{10,10000}, 0.0f)) +
            tensor_type(std::vector<int>{10,10000}, 0.0f).reshape(10,10000) +
            tensor_type(std::vector<int>{10,10000}, 0.0f).reshape(10,10000)
        ),
        "asymmetric_tree_trivial_view_reshape_operand_depth50_multiiter",
        benchmark_worker
    );
}