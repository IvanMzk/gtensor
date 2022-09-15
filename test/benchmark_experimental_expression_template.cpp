//#pragma inline_depth(255)
#include <string>
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "benchmark_helpers.hpp"
#include "experimental_expression_template.hpp"


namespace benchmark_experimental_expression_template{

using value_type = float;
using config_type = gtensor::config::default_config;
using shape_type = typename config_type::shape_type;
using benchmark_helpers::asymmetric_tree_maker;
using benchmark_helpers::symmetric_tree_maker;
using tensor_no_dispatch_type = expression_template_without_dispatching::test_tensor<value_type,config_type>;
using tensor_walker_dispatch_type = expression_template_dispatch_in_walker::test_tensor<value_type,config_type>;
using tensor_variant_dispatch_type = expression_template_variant_dispatch::test_tensor<value_type,config_type>;
using tensor_polywalker_type = expression_template_polywalker::test_tensor<value_type,config_type>;

using gtensor::detail::shape_to_str;

auto iterate_without_deref = [](const auto& t){
    auto t_it = t.begin();
    auto t_end = t.end();
    std::size_t c{};
    while (t_it!=t_end){
        ++c;
        ++t_it;
    }
    return c;
};
auto making_iter_iterate_deref = [](const auto& t){
    auto t_it = t.begin();
    auto t_end = t.end();
    std::size_t c{};
    while (t_it!=t_end){
        if (*t_it > 2){
            ++c;
        }
        ++t_it;
    }
    return c;
};
auto iterate_deref = [](auto& it_begin, auto& it_end){
    std::size_t c{};
    while (it_begin!=it_end){
        if (*it_begin > 2){
            ++c;
        }
        ++it_begin;
    }
    return c;
};

template<bool>
struct making_iter_tag{};

template<typename T>
auto add_label(std::pair<T,std::string>& arg, std::string label){
    arg.second = label;
    return arg;
}

template<typename Arg>
auto make_iterators(std::size_t n, const Arg& t){
    return std::vector<std::pair<decltype(t.begin()), decltype(t.end())>>(n, std::make_pair(t.begin(), t.end()));
};

template<typename F, typename Arg>
auto benchmark_with_making_iter(const F& f, const Arg& arg, std::string label){
    BENCHMARK_ADVANCED(label.c_str())(Catch::Benchmark::Chronometer meter) {
        meter.measure([&] { return f(arg); });
    };
    return 0;
}
template<typename F, typename Arg>
auto benchmark_without_making_iter(const F& f, const Arg& arg, std::string label){
    BENCHMARK_ADVANCED(label.c_str())(Catch::Benchmark::Chronometer meter) {
        auto v = make_iterators(meter.runs(),arg);
        meter.measure([&f,&v,&arg](int i) { return f(v[i].first, v[i].second); });
    };
    return 0;
}

template<typename...Args>
auto run_benchmarks(making_iter_tag<true>, const Args&...args){
    return (benchmark_with_making_iter(making_iter_iterate_deref,args.first, args.second)+...);
}
template<typename...Args>
auto run_benchmarks(making_iter_tag<false>, const Args&...args){
    return (benchmark_without_making_iter(iterate_deref,args.first, args.second)+...);
}


template<typename IterTag, typename TreeMakerT>
struct benchmark_binary_tree{
    static constexpr std::size_t tree_depth = TreeMakerT::depth;
    auto operator()(const shape_type& shape1, const shape_type& shape2){
        auto make_tree = [](const auto& t1, const auto& t2){return TreeMakerT{}(t1,t2);};
        auto e_no_dispatch = std::make_pair(make_tree(tensor_no_dispatch_type(shape1,0.0f),tensor_no_dispatch_type(shape2,0.0f)), std::string{});
        auto e_walker_dispatch = std::make_pair(make_tree(tensor_walker_dispatch_type(shape1,0.0f), tensor_walker_dispatch_type(shape2,0.0f)), std::string{});
        auto e_variant_dispatch = std::make_pair(make_tree(tensor_variant_dispatch_type(shape1,0.0f), tensor_variant_dispatch_type(shape2,0.0f)), std::string{});
        auto e_polywalker = std::make_pair(make_tree(tensor_polywalker_type(shape1,0.0f), tensor_polywalker_type(shape2,0.0f)), std::string{});
        auto label_suffix = std::stringstream{};
        auto trivial_to_str = [](const auto& e){return e.engine().is_trivial() ? std::string("_trivial_") : std::string("_not_trivial_"); };
        label_suffix<<shape_to_str(shape1)<<"_"<<shape_to_str(shape2)<<"_"<<TreeMakerT::name<<"_depth"<<tree_depth;
        run_benchmarks(
            IterTag{},
            add_label(e_no_dispatch,"no_dispatch"+trivial_to_str(e_no_dispatch.first)+label_suffix.str()),
            add_label(e_walker_dispatch,"walker_dispatch"+trivial_to_str(e_walker_dispatch.first)+label_suffix.str()),
            add_label(e_polywalker,"polywalker"+trivial_to_str(e_polywalker.first)+label_suffix.str()),
            add_label(e_variant_dispatch,"variant_dispatch_"+trivial_to_str(e_variant_dispatch.first)+label_suffix.str())
        );
    }
};

}   //end of namespace benchmark_experimental_expression_template


TEMPLATE_TEST_CASE("benchmark_experimental_expression_template","[benchmark_experimental_expression_template]", benchmark_experimental_expression_template::making_iter_tag<true>)
{
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using shape_type = typename config_type::shape_type;
    using benchmark_helpers::asymmetric_tree_maker;
    using benchmark_helpers::symmetric_tree_maker;
    using tensor_no_dispatch_type = expression_template_without_dispatching::test_tensor<value_type,config_type>;
    using tensor_walker_dispatch_type = expression_template_dispatch_in_walker::test_tensor<value_type,config_type>;
    using tensor_variant_dispatch_type = expression_template_variant_dispatch::test_tensor<value_type,config_type>;
    using benchmark_experimental_expression_template::run_benchmarks;
    using benchmark_experimental_expression_template::add_label;
    using benchmark_experimental_expression_template::making_iter_tag;
    using benchmark_experimental_expression_template::benchmark_binary_tree;


    // shape_type shape1{1,1,3,1,5,1,7,1,9,1};
    // shape_type shape2{1,2,1,4,1,6,1,8,1,10};

    // shape_type shape1{1,2,1,4,1,6,1,8,1,10};
    // shape_type shape2{1,2,3,4,5,6,7,8,9,10};

    shape_type shape1{1, 10000};
    shape_type shape2{10,10000};

    // shape_type shape1{1,3000};
    // shape_type shape2{3000,1};

    // shape_type shape1{1,10000};
    // shape_type shape2{10000,1};

    benchmark_binary_tree<TestType,asymmetric_tree_maker<50>>{}(shape_type{1, 10000}, shape_type{10,10000});
    benchmark_binary_tree<TestType,asymmetric_tree_maker<50>>{}(shape_type{10, 10000}, shape_type{10,10000});
    // benchmark_binary_tree<TestType,symmetric_tree_maker<10>>{}(shape_type{1, 10000}, shape_type{10,10000});
    // benchmark_binary_tree<TestType,symmetric_tree_maker<10>>{}(shape_type{10, 10000}, shape_type{10,10000});

    // static constexpr std::size_t tree_depth = 100;
    // auto make_tree = [](const auto& t1, const auto& t2){return make_asymmetric_tree<tree_depth>(t1,t2);};
    // auto e_no_dispatch = make_tree(tensor_no_dispatch_type(shape1,0.0f),tensor_no_dispatch_type(shape2,0.0f));
    // auto w = e_no_dispatch.engine().create_walker();
}