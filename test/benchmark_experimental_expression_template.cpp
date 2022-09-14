
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "benchmark_helpers.hpp"
#include "experimental_expression_template.hpp"

TEST_CASE("benchmark_expression_template","[benchmark_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using shape_type = typename config_type::shape_type;
    using benchmark_helpers::make_symmetric_tree;
    using benchmark_helpers::make_asymmetric_tree;
    using tensor_no_dispatch_type = expression_template_without_dispatching::test_tensor<value_type,config_type>;
    using tensor_walker_dispatch_type = expression_template_dispatch_in_walker::test_tensor<value_type,config_type>;
    using tensor_variant_dispatch_type = expression_template_variant_dispatch::test_tensor<value_type,config_type>;

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

    auto iterate_with_deref = [](const auto& t){
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

    auto just_iterate_with_deref = [](auto& it_begin, auto& it_end){
        std::size_t c{};
        while (it_begin!=it_end){
            if (*it_begin > 2){
                ++c;
            }
            ++it_begin;
        }
        return c;
    };

    auto make_iterators = [](std::size_t n, const auto& t){
        return std::vector<std::pair<decltype(t.begin()), decltype(t.end())>>(n, std::make_pair(t.begin(), t.end()));
    };

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

    enum class benchmark_kinds {iteration_and_dereference, iterator_construction_iteration_and_dereference};
    auto benchmark_kind = benchmark_kinds::iterator_construction_iteration_and_dereference;
    //auto benchmark_kind = benchmark_kinds::iteration_and_dereference;

    SECTION("benchmark_shape(1,10000)_shape(10,10000)_asymmetric_tree_depth_50"){

        shape_type shape1{1, 10000};
        shape_type shape2{10,10000};

        static constexpr std::size_t tree_depth = 50;
        auto make_tree = [](const auto& t1, const auto& t2){return make_asymmetric_tree<tree_depth>(t1,t2);};

        auto e_no_dispatch = make_tree(tensor_no_dispatch_type(shape1,0.0f),tensor_no_dispatch_type(shape2,0.0f));
        auto e_walker_dispatch = make_tree(tensor_walker_dispatch_type(shape1,0.0f), tensor_walker_dispatch_type(shape2,0.0f));
        auto e_variant_dispatch = make_tree(tensor_variant_dispatch_type(shape1,0.0f), tensor_variant_dispatch_type(shape2,0.0f));

        if (benchmark_kind == benchmark_kinds::iteration_and_dereference)
        {
            BENCHMARK_ADVANCED("no_dispatch_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
                auto v = make_iterators(meter.runs(),e_no_dispatch);
                meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
            };
            BENCHMARK_ADVANCED("walker_dispatch_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
                auto v = make_iterators(meter.runs(),e_walker_dispatch);
                meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
            };
            BENCHMARK_ADVANCED("variant_dispatch_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
                auto v = make_iterators(meter.runs(),e_variant_dispatch);
                meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
            };
        }

        if (benchmark_kind == benchmark_kinds::iterator_construction_iteration_and_dereference)
        {
            BENCHMARK_ADVANCED("no_dispatch_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
                meter.measure([&iterate_with_deref, &e_no_dispatch] { return iterate_with_deref(e_no_dispatch); });
            };
            BENCHMARK_ADVANCED("walker_dispatch_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
                meter.measure([&iterate_with_deref, &e_walker_dispatch] { return iterate_with_deref(e_walker_dispatch); });
            };
            BENCHMARK_ADVANCED("variant_dispatch_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
                meter.measure([&iterate_with_deref, &e_variant_dispatch] { return iterate_with_deref(e_variant_dispatch); });
            };
        }
    }   //end of SECTION("benchmark_shape(1,10000)_shape(10,10000)_assymetric_tree_depth_50")


}