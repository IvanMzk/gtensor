#include "catch.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_basic_indexer_reference_specialization","test_data_accessor"){
    using gtensor::basic_indexer;
    using helpers_for_testing::apply_by_element;

    //0parent,1index,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::vector<int>{1,2,3,0}, 0, 1),
        std::make_tuple(std::vector<int>{1,2,3,0}, 1, 2),
        std::make_tuple(std::vector<int>{1,2,3,0}, 2, 3),
        std::make_tuple(std::vector<int>{1,2,3,0}, 3, 0)
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

TEST_CASE("test_basic_indexer_converter_specialization","test_data_accessor"){
    using gtensor::basic_indexer;
    using helpers_for_testing::apply_by_element;

    //0parent,1convertor,2index,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::vector<int>{1,2,3,0}, [](auto i){return i;}, 0, 1),
        std::make_tuple(std::vector<int>{1,2,3,0}, [](auto i){return ++i;}, 1, 3),
        std::make_tuple(std::vector<int>{1,2,3,0}, [](auto i){return --i;}, 2, 2),
        std::make_tuple(std::vector<int>{1,2,3,0}, [](auto i){return i;}, 3, 0)
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

