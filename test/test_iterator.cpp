#include "catch.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_iterator_compare","[test_iterator]",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<gtensor::storage_vector>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::storage_type;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::indexer_iterator;
    using gtensor::reverse_indexer_iterator;
    using helpers_for_testing::apply_by_element;

    //0storage,1size,2comparator,3expected
    auto test_data = std::make_tuple(
        //empty range
        //self
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto){return first==first;}, true),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto){return first!=first;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto, auto last){return last==last;}, true),
        std::make_tuple(storage_type{}, index_type{0}, [](auto, auto last){return last!=last;}, false),
        //first,last
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first==last;}, true),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first!=last;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first>last;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first>=last;}, true),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first<last;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first<=last;}, true),
        //not empty range
        //self
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first==first;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first!=first;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last==last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last!=last;}, false),
        //first,last
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return first==last;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return first!=last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return first>last;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return first>=last;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return first<last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return first<=last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return last>first;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return last>=first;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return last<first;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return last<=first;}, false)
    );
    SECTION("test_indexer_iterator_compare")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto comparator = std::get<2>(t);
            auto expected = std::get<3>(t);
            indexer_type indexer{storage};
            using iterator_type = indexer_iterator<config_type,indexer_type>;
            auto first = iterator_type{indexer, 0};
            auto last = iterator_type{indexer, size};
            auto result = comparator(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_indexer_iterator_compare")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto comparator = std::get<2>(t);
            auto expected = std::get<3>(t);
            indexer_type indexer{storage};
            using iterator_type = reverse_indexer_iterator<config_type,indexer_type>;
            auto first = iterator_type{indexer, size};
            auto last = iterator_type{indexer, 0};
            auto result = comparator(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

