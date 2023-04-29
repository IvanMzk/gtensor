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

TEMPLATE_TEST_CASE("test_walker_indexer","test_data_accessor",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<gtensor::storage_vector>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using gtensor::basic_indexer;
    using gtensor::walker_indexer;
    using gtensor::walker;
    using shape_type = typename config_type::shape_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = basic_indexer<storage_type&>;
    using walker_type = walker<config_type, indexer_type>;
    using walker_indexer_type = walker_indexer<walker_type>;
    using helpers_for_testing::apply_by_element;

    //0elements,1shape,2offset,3index_1,4expected_1,5index_2,6expected_2
    auto test_data = std::make_tuple(
        std::make_tuple(storage_type{2}, shape_type{1}, index_type{0}, 0 , 2, 0, 2),
        std::make_tuple(storage_type{1,2,3,4,5}, shape_type{5}, index_type{0}, 0,1, 4,5),
        std::make_tuple(storage_type{1,2,3,4,5}, shape_type{5}, index_type{0}, 1,2, 2,3),
        std::make_tuple(storage_type{1,2,3,4,5}, shape_type{5}, index_type{0}, 2,3, 1,2),
        std::make_tuple(storage_type{1,2,3,4,5}, shape_type{5}, index_type{0}, 3,4, 3,4),
        std::make_tuple(storage_type{1,2,3,4,5}, shape_type{5}, index_type{0}, 4,5, 0,1),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6,7,8,9,10,11,12}, shape_type{2,2,2}, index_type{3}, 0,1, 11,12),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6,7,8,9,10,11,12}, shape_type{2,2,2}, index_type{3}, 1,2, 2,3),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6,7,8,9,10,11,12}, shape_type{2,2,2}, index_type{3}, 2,3, 1,2),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6,7,8,9,10,11,12}, shape_type{2,2,2}, index_type{3}, 3,4, 0,1),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6,7,8,9,10,11,12}, shape_type{2,2,2}, index_type{3}, 4,5, 6,7),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6,7,8,9,10,11,12}, shape_type{2,2,2}, index_type{3}, 8,9, 5,6),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6,7,8,9,10,11,12}, shape_type{2,2,2}, index_type{3}, 10,11, 7,8),
        std::make_tuple(storage_type{0,0,0,1,2,3,4,5,6,7,8,9,10,11,12}, shape_type{2,2,2}, index_type{3}, 11,12, 11,12)
    );
    auto test = [](const auto& t){
        auto elements = std::get<0>(t);
        auto shape = std::get<1>(t);
        auto offset = std::get<2>(t);
        auto index_1 = std::get<3>(t);
        auto expected_1 = std::get<4>(t);
        auto index_2 = std::get<5>(t);
        auto expected_2 = std::get<6>(t);
        auto strides = gtensor::detail::make_strides(shape);
        auto adapted_strides = gtensor::detail::make_adapted_strides(shape, strides);
        auto reset_strides = gtensor::detail::make_reset_strides(shape, strides);
        auto strides_div = gtensor::detail::make_dividers<config_type>(strides);
        dim_type dim = shape.size();
        indexer_type indexer{elements};
        walker_type walker{adapted_strides, reset_strides, offset, indexer, dim};
        walker_indexer_type walker_indexer{strides_div, walker};
        auto result_1 = walker_indexer[index_1];
        auto result_2 = walker_indexer[index_2];
        REQUIRE(result_1 == expected_1);
        REQUIRE(result_2 == expected_2);
    };
    apply_by_element(test,test_data);
}
