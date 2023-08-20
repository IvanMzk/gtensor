#include "catch.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEST_CASE("test_walker","test_data_accessor")
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using shape_type = typename config_type::shape_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using helpers_for_testing::apply_by_element;

    auto test_data = std::make_tuple(
        // //1-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{0}, dim_type{1}, [](auto&){}, value_type{1}),
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
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6}, shape_type{1}, shape_type{5}, index_type{3}, dim_type{3}, [](auto&){}, value_type{1}),
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
    SECTION("indexer_walker")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto adapted_strides = std::get<1>(t);
            auto reset_strides = std::get<2>(t);
            auto offset = std::get<3>(t);
            auto max_dim = std::get<4>(t);
            auto mover = std::get<5>(t);
            auto expected = std::get<6>(t);
            using indexer_type = gtensor::basic_indexer<storage_type&>;
            using walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, indexer_type>>;
            auto indexer = indexer_type{storage};
            auto walker =  walker_type{max_dim,adapted_strides,reset_strides,offset,indexer};
            mover(walker);
            auto result = *walker;
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("iterator_walker")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto adapted_strides = std::get<1>(t);
            auto reset_strides = std::get<2>(t);
            auto offset = std::get<3>(t);
            auto max_dim = std::get<4>(t);
            auto mover = std::get<5>(t);
            auto expected = std::get<6>(t);
            using iterator_type = decltype(storage.begin());
            using walker_type = gtensor::axes_correction_walker<gtensor::iterator_walker<config_type, iterator_type>>;
            auto walker =  walker_type{max_dim,adapted_strides,reset_strides,storage.begin()+offset};
            mover(walker);
            auto result = *walker;
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_walker_result_type","test_data_accessor")
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using index_type = typename config_type::index_type;
    using gtensor::basic_indexer;
    using gtensor::indexer_walker;
    using helpers_for_testing::apply_by_element;
    SECTION("test_walker_non_const_storage")
    {
        using storage_type = typename config_type::template storage<value_type>;
        using indexer_type = basic_indexer<storage_type&>;
        using walker_type = indexer_walker<config_type,indexer_type>;
        REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(std::declval<walker_type>().operator*())>);
    }
    SECTION("test_walker_const_storage")
    {
        using storage_type = const typename config_type::template storage<value_type>;
        using indexer_type = basic_indexer<storage_type&>;
        using walker_type = indexer_walker<config_type,indexer_type>;
        REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(std::declval<walker_type>().operator*())>);
    }
}
