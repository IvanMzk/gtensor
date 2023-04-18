#include "catch.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_basic_indexer_reference_specialization","test_data_accessor",
    std::vector<int>,
    gtensor::storage_vector<int>
)
{
    using storage_type = TestType;
    using gtensor::basic_indexer;
    using helpers_for_testing::apply_by_element;

    //0parent,1index,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(storage_type{1,2,3,0}, 0, 1),
        std::make_tuple(storage_type{1,2,3,0}, 1, 2),
        std::make_tuple(storage_type{1,2,3,0}, 2, 3),
        std::make_tuple(storage_type{1,2,3,0}, 3, 0)
    );
    auto test = [](const auto& t, auto& parent){
        auto index = std::get<1>(t);
        auto expected = std::get<2>(t);
        using parent_type = decltype(parent);
        auto indexer = basic_indexer<parent_type>{parent};
        auto result = indexer[index];
        REQUIRE(std::is_same_v<decltype(parent[index]),decltype(indexer[index])>);
        REQUIRE(result == expected);
    };
    SECTION("test_basic_indexer_reference_to_non_const_specialization")
    {
        auto test_ = [test](const auto& t){
            auto parent = std::get<0>(t);
            using parent_type = decltype(parent);
            static_assert(!std::is_const_v<parent_type>);
            test(t,parent);
        };
        apply_by_element(test_, test_data);
    }
    SECTION("test_basic_indexer_reference_to_const_specialization")
    {
        auto test_ = [test](const auto& t){
            const auto parent = std::get<0>(t);
            using parent_type = decltype(parent);
            static_assert(std::is_const_v<parent_type>);
            test(t,parent);
        };
        apply_by_element(test_, test_data);
    }
}

TEMPLATE_TEST_CASE("test_basic_indexer_converter_specialization","test_data_accessor",
    std::vector<int>,
    gtensor::storage_vector<int>
)
{
    using stoarge_type = TestType;
    using gtensor::basic_indexer;
    using helpers_for_testing::apply_by_element;

    //0parent,1convertor,2index,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(stoarge_type{1,2,3,0}, [](auto i){return i;}, 0, 1),
        std::make_tuple(stoarge_type{1,2,3,0}, [](auto i){return ++i;}, 1, 3),
        std::make_tuple(stoarge_type{1,2,3,0}, [](auto i){return --i;}, 2, 2),
        std::make_tuple(stoarge_type{1,2,3,0}, [](auto i){return i;}, 3, 0)
    );
    auto test = [](const auto& t, auto& parent){
        auto converter = std::get<1>(t);
        auto index = std::get<2>(t);
        auto expected = std::get<3>(t);
        using converter_type = decltype(converter);
        using parent_type = decltype(parent);
        static_assert(std::is_lvalue_reference_v<parent_type>);
        auto indexer = basic_indexer<basic_indexer<parent_type>,converter_type>{basic_indexer<parent_type>{parent}, converter};
        auto result = indexer[index];
        REQUIRE(std::is_same_v<decltype(parent[index]),decltype(indexer[index])>);
        REQUIRE(result == expected);
    };
    SECTION("test_basic_indexer_converter_specialization_non_const_parent")
    {
        auto test_ = [test](const auto& t){
            auto parent = std::get<0>(t);
            using parent_type = decltype(parent);
            static_assert(!std::is_const_v<parent_type>);
            test(t, parent);
        };
        apply_by_element(test_, test_data);
    }
    SECTION("test_basic_indexer_converter_specialization_const_parent")
    {
        auto test_ = [test](const auto& t){
            const auto parent = std::get<0>(t);
            using parent_type = decltype(parent);
            static_assert(std::is_const_v<parent_type>);
            test(t, parent);
        };
        apply_by_element(test_, test_data);
    }
}

TEMPLATE_TEST_CASE("test_walker","test_data_accessor",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<gtensor::storage_vector>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using gtensor::basic_indexer;
    using gtensor::walker;
    using shape_type = typename config_type::shape_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = basic_indexer<storage_type&>;
    using walker_type = walker<config_type, indexer_type>;
    using helpers_for_testing::apply_by_element;
    //0storage,1adapted_strides,2reset_strides,3offset,4max_dim,5mover,6expected
    auto test_data = std::make_tuple(
        //1-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.step(0);}, value_type{2}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.step(0); w.step(0);}, value_type{3}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.walk(0,0);}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.walk(0,1);}, value_type{2}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.walk(0,2);}, value_type{3}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.walk(0,5);}, value_type{6}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.step(0), w.walk(0,2);}, value_type{4}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.walk(0,2); w.step(0);}, value_type{4}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.step(0); w.step_back(0);}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.walk(0,2); w.step_back(0);}, value_type{2}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.walk(0,5); w.reset_back(0);}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.reset(0);}, value_type{6}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.reset_back();}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto& w){w.walk(0,2); w.step(0); w.reset_back();}, value_type{1}),
        //1-d,offset,max_dim
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.step(0);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.step(1);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.step(2);}, value_type{2}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.walk(0,2);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.walk(1,2);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,0);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,2);}, value_type{3}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.step(2); w.step_back(0);}, value_type{2}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.step(2); w.step_back(1);}, value_type{2}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.step(2); w.step_back(2);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,5); w.reset_back(0);}, value_type{6}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,5); w.reset_back(1);}, value_type{6}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,5); w.reset_back(2);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.reset(0);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.reset(1);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.reset(2);}, value_type{6}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.reset_back();}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,5); w.reset_back();}, value_type{1}),
        //2-d,offset,max_dim
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.step(0);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.step(1);}, value_type{4}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.step(2);}, value_type{2}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.step(2); w.step(1);}, value_type{5}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.step(1); w.step(2);}, value_type{5}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(0,2);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(1,1);}, value_type{4}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(1,0);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,2);}, value_type{3}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.step(1); w.step_back(0);}, value_type{4}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.step(1); w.step_back(1);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.step(1); w.step_back(2);}, value_type{3}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(1,1); w.reset_back(0);}, value_type{4}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(1,1); w.reset_back(1);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,2); w.reset_back(0);}, value_type{3}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(2,2); w.reset_back(2);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.reset(0);}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.reset(1);}, value_type{4}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.reset(2);}, value_type{3}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.reset(1); w.reset(2);}, value_type{6}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.reset_back();}, value_type{1}),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{3,1}, shape_type{3,2}, index_type{3}, dim_type{3}, [](auto& w){w.walk(1,1); w.reset_back();}, value_type{1}),
        //3-d,unit dimension size
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.step(0);}, value_type{4}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.step(1);}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.step(2);}, value_type{2}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.step(2); w.step(1); w.step(0);}, value_type{5}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.walk(0,1);}, value_type{4}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.walk(1,1);}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.walk(1,2);}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.walk(2,2);}, value_type{3}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.step_back(1);}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.step(2); w.step_back(1);}, value_type{2}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.reset_back(1);}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.walk(2,2); w.reset_back(1);}, value_type{3}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{3,0,1}, shape_type{3,0,2}, index_type{0}, dim_type{3}, [](auto& w){w.reset(1);}, value_type{1})
    );
    auto test = [](const auto& t){
        auto storage = std::get<0>(t);
        auto adapted_strides = std::get<1>(t);
        auto reset_strides = std::get<2>(t);
        auto offset = std::get<3>(t);
        auto max_dim = std::get<4>(t);
        auto mover = std::get<5>(t);
        auto expected = std::get<6>(t);
        auto indexer = indexer_type{storage};
        auto walker =  walker_type{adapted_strides,reset_strides,offset,indexer,max_dim};
        mover(walker);
        auto result = *walker;
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_walker_result_type","test_data_accessor",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<gtensor::storage_vector>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using index_type = typename config_type::index_type;
    using gtensor::basic_indexer;
    using gtensor::walker;
    using helpers_for_testing::apply_by_element;
    SECTION("test_walker_non_const_storage")
    {
        using storage_type = typename config_type::template storage<value_type>;
        using indexer_type = basic_indexer<storage_type&>;
        using walker_type = walker<config_type,indexer_type>;
        REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(std::declval<walker_type>().operator*())>);
    }
    SECTION("test_walker_const_storage")
    {
        using storage_type = const typename config_type::template storage<value_type>;
        using indexer_type = basic_indexer<storage_type&>;
        using walker_type = walker<config_type,indexer_type>;
        REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(std::declval<walker_type>().operator*())>);
    }
}

TEMPLATE_TEST_CASE("test_walker_traverser_next","test_data_accessor",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<gtensor::storage_vector>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using gtensor::basic_indexer;
    using gtensor::walker;
    using gtensor::walker_forward_traverser;
    using gtensor::walker_bidirectional_traverser;
    using shape_type = typename config_type::shape_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = basic_indexer<storage_type&>;
    using walker_type = walker<config_type, indexer_type>;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_adapted_strides;
    using gtensor::detail::make_reset_strides;
    using helpers_for_testing::apply_by_element;

    auto do_next = [](auto& tr, auto n){
        bool result{true};
        while(n!=0){
            result = tr.next();
            --n;
        }
        return result;
    };

    //0storage,1shape,2mover,3expected_index,4expected_element,5expected_is_next
    auto test_data = std::make_tuple(
        //1-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next](auto& tr){return do_next(tr,0);}, shape_type{0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next](auto& tr){return do_next(tr,1);}, shape_type{1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next](auto& tr){return do_next(tr,2);}, shape_type{2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next](auto& tr){return do_next(tr,3);}, shape_type{3} ,value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next](auto& tr){return do_next(tr,4);}, shape_type{4} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next](auto& tr){return do_next(tr,5);}, shape_type{5} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next](auto& tr){return do_next(tr,6);}, shape_type{0} ,value_type{1}, false),
        //3-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next](auto& tr){return do_next(tr,0);}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next](auto& tr){return do_next(tr,1);}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next](auto& tr){return do_next(tr,2);}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next](auto& tr){return do_next(tr,3);}, shape_type{1,0,0} ,value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next](auto& tr){return do_next(tr,4);}, shape_type{1,0,1} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next](auto& tr){return do_next(tr,5);}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next](auto& tr){return do_next(tr,6);}, shape_type{0,0,0} ,value_type{1}, false)
    );
    auto test = [](const auto& t, auto& traverser){
        auto mover = std::get<2>(t);
        auto expected_index = std::get<3>(t);
        auto expected_element = std::get<4>(t);
        auto expected_is_next = std::get<5>(t);

        auto result_is_next = mover(traverser);
        auto result_index = traverser.index();
        auto result_element = *traverser.walker();
        REQUIRE(result_is_next == expected_is_next);
        REQUIRE(result_index == expected_index);
        REQUIRE(result_element == expected_element);
    };
    SECTION("test_walker_forward_traverser_next")
    {
        auto test_ = [test](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            using traverser_type = walker_forward_traverser<config_type, walker_type>;
            auto indexer = indexer_type{storage};
            auto strides = make_strides(shape);
            auto adapted_strides = make_adapted_strides(shape,strides);
            auto reset_strides = make_reset_strides(shape,strides);
            index_type offset{0};
            dim_type max_dim = shape.size();
            auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer, max_dim};
            auto traverser = traverser_type{shape, walker};
            test(t,traverser);
        };
        apply_by_element(test_, test_data);
    }
    SECTION("test_walker_bidirectional_traverser_next")
    {
        auto test_ = [test](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            using traverser_type = walker_bidirectional_traverser<config_type, walker_type>;
            auto indexer = indexer_type{storage};
            auto strides = make_strides(shape);
            auto adapted_strides = make_adapted_strides(shape,strides);
            auto reset_strides = make_reset_strides(shape,strides);
            index_type offset{0};
            dim_type max_dim = shape.size();
            auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer, max_dim};
            auto traverser = traverser_type{shape, walker};
            test(t,traverser);
        };
        apply_by_element(test_, test_data);
    }
}

TEMPLATE_TEST_CASE("test_walker_traverser_prev","test_data_accessor",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<gtensor::storage_vector>::config_type
){
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using gtensor::basic_indexer;
    using gtensor::walker;
    using gtensor::walker_bidirectional_traverser;
    using gtensor::walker_random_access_traverser;
    using shape_type = typename config_type::shape_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = basic_indexer<storage_type&>;
    using walker_type = walker<config_type, indexer_type>;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_adapted_strides;
    using gtensor::detail::make_reset_strides;
    using gtensor::detail::make_dividers;
    using helpers_for_testing::apply_by_element;

    auto do_next = [](auto& tr, auto n){
        bool result{true};
        while(n!=0){
            result = tr.next();
            --n;
        }
        return result;
    };
    auto do_prev = [](auto& tr, auto n){
        bool result{true};
        while(n!=0){
            result = tr.prev();
            --n;
        }
        return result;
    };

    //0storage,1shape,2mover,3expected_index,4expected_element,5expected_is_next
    auto test_data = std::make_tuple(
        //1-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_prev](auto& tr){return do_prev(tr,1);}, shape_type{5} ,value_type{6}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,1);}, shape_type{0}, value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,2);}, shape_type{1}, value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,3);}, shape_type{2}, value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,4);}, shape_type{3}, value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,5);}, shape_type{4}, value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,6);}, shape_type{5}, value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,7);}, shape_type{0}, value_type{1}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6), do_prev(tr,1);},shape_type{5},value_type{6},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6), do_prev(tr,2);},shape_type{4},value_type{5},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6), do_prev(tr,3);},shape_type{3},value_type{4},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6), do_prev(tr,4);},shape_type{2},value_type{3},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6), do_prev(tr,5);},shape_type{1},value_type{2},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6), do_prev(tr,6);},shape_type{0},value_type{1},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6), do_prev(tr,7);},shape_type{5},value_type{6},false),
        //3-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_prev](auto& tr){return do_prev(tr,1);}, shape_type{1,0,2} ,value_type{6}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,1);}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,2);}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,3);}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,4);}, shape_type{1,0,0} ,value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,5);}, shape_type{1,0,1} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,6);}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),do_next(tr,7);}, shape_type{0,0,0} ,value_type{1}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),do_prev(tr,1);}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),do_prev(tr,2);}, shape_type{1,0,1} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),do_prev(tr,3);}, shape_type{1,0,0} ,value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),do_prev(tr,4);}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),do_prev(tr,5);}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),do_prev(tr,6);}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),do_prev(tr,7);}, shape_type{1,0,2} ,value_type{6}, false)
    );
    auto test = [](const auto& t, auto& traverser){
        auto mover = std::get<2>(t);
        auto expected_index = std::get<3>(t);
        auto expected_element = std::get<4>(t);
        auto expected_is_next = std::get<5>(t);

        auto result_is_next = mover(traverser);
        auto result_index = traverser.index();
        auto result_element = *traverser.walker();
        REQUIRE(result_is_next == expected_is_next);
        REQUIRE(result_index == expected_index);
        REQUIRE(result_element == expected_element);
    };
    SECTION("test_walker_bidirectional_traverser_prev")
    {
        auto test_ = [test](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            using traverser_type = walker_bidirectional_traverser<config_type, walker_type>;
            auto indexer = indexer_type{storage};
            auto strides = make_strides(shape);
            auto adapted_strides = make_adapted_strides(shape,strides);
            auto reset_strides = make_reset_strides(shape,strides);
            index_type offset{0};
            dim_type max_dim = shape.size();
            auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer, max_dim};
            auto traverser = traverser_type{shape, walker};
            test(t,traverser);
        };
        apply_by_element(test_, test_data);
    }
    SECTION("test_walker_random_access_traverser_prev")
    {
        auto test_ = [test](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            using traverser_type = walker_random_access_traverser<config_type, walker_type>;
            auto indexer = indexer_type{storage};
            auto strides = make_strides(shape);
            auto adapted_strides = make_adapted_strides(shape,strides);
            auto reset_strides = make_reset_strides(shape,strides);
            index_type offset{0};
            dim_type max_dim = shape.size();
            auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer, max_dim};
            auto strides_div = make_dividers<config_type>(strides);
            auto traverser = traverser_type{shape, strides_div, walker};
            test(t,traverser);
        };
        apply_by_element(test_, test_data);
    }
}

TEMPLATE_TEST_CASE("test_walker_traverser_move","test_data_accessor",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<gtensor::storage_vector>::config_type
){
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using gtensor::basic_indexer;
    using gtensor::walker;
    using gtensor::walker_random_access_traverser;
    using shape_type = typename config_type::shape_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = basic_indexer<storage_type&>;
    using walker_type = walker<config_type, indexer_type>;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_adapted_strides;
    using gtensor::detail::make_reset_strides;
    using gtensor::detail::make_dividers;
    using helpers_for_testing::apply_by_element;

    auto do_next = [](auto& tr, auto n){
        bool result{true};
        while(n!=0){
            result = tr.next();
            --n;
        }
        return result;
    };
    auto do_prev = [](auto& tr, auto n){
        bool result{true};
        while(n!=0){
            result = tr.prev();
            --n;
        }
        return result;
    };

    //0storage,1shape,2mover,3expected_index,4expected_element,5expected_is_next
    auto test_data = std::make_tuple(
        //1-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(0); return true;}, shape_type{0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(1); return true;}, shape_type{1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(2); return true;}, shape_type{2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(5); return true;}, shape_type{5} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(0); tr.move(0); return true;}, shape_type{0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(1); tr.move(1); return true;}, shape_type{1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(1); tr.move(0); return true;}, shape_type{0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(0); tr.move(1); return true;}, shape_type{1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(5); tr.move(0); return true;}, shape_type{0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){tr.move(1); tr.move(5); return true;}, shape_type{5} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),tr.move(5),do_next(tr,1);}, shape_type{0} ,value_type{1}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_prev](auto& tr){return do_prev(tr,1),tr.move(5),do_prev(tr,1);}, shape_type{4} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_prev(tr,1),tr.move(4),do_next(tr,1);}, shape_type{5} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6),tr.move(0),do_prev(tr,1);}, shape_type{5} ,value_type{6}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next](auto& tr){return do_next(tr,6),tr.move(1),do_next(tr,1);}, shape_type{2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [do_next,do_prev](auto& tr){return do_next(tr,6),tr.move(1),do_prev(tr,1);}, shape_type{0} ,value_type{1}, true),
        //3-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(0); return true;}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(1); return true;}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(2); return true;}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(5); return true;}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(0); tr.move(0); return true;}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(1); tr.move(1); return true;}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(1); tr.move(0); return true;}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(0); tr.move(1); return true;}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(5); tr.move(0); return true;}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){tr.move(1); tr.move(5); return true;}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),tr.move(5),do_next(tr,1);}, shape_type{0,0,0} ,value_type{1}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_prev](auto& tr){return do_prev(tr,1),tr.move(5),do_prev(tr,1);}, shape_type{1,0,1} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_prev(tr,1),tr.move(4),do_next(tr,1);}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),tr.move(0),do_prev(tr,1);}, shape_type{1,0,2} ,value_type{6}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next](auto& tr){return do_next(tr,6),tr.move(1),do_next(tr,1);}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [do_next,do_prev](auto& tr){return do_next(tr,6),tr.move(1),do_prev(tr,1);}, shape_type{0,0,0} ,value_type{1}, true)
    );
    auto test = [](const auto& t, auto& traverser){
        auto mover = std::get<2>(t);
        auto expected_index = std::get<3>(t);
        auto expected_element = std::get<4>(t);
        auto expected_is_next = std::get<5>(t);
        auto result_is_next = mover(traverser);
        auto result_index = traverser.index();
        auto result_element = *traverser.walker();
        REQUIRE(result_is_next == expected_is_next);
        REQUIRE(result_index == expected_index);
        REQUIRE(result_element == expected_element);
    };
    SECTION("test_walker_random_access_traverser_move")
    {
        auto test_ = [test](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            using traverser_type = walker_random_access_traverser<config_type, walker_type>;
            auto indexer = indexer_type{storage};
            auto strides = make_strides(shape);
            auto adapted_strides = make_adapted_strides(shape,strides);
            auto reset_strides = make_reset_strides(shape,strides);
            index_type offset{0};
            dim_type max_dim = shape.size();
            auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer, max_dim};
            auto strides_div = make_dividers<config_type>(strides);
            auto traverser = traverser_type{shape, strides_div, walker};
            test(t,traverser);
        };
        apply_by_element(test_, test_data);
    }
}

