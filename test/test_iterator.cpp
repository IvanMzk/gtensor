#include "catch.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_iterator_equality","[test_iterator]"){
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
    using storage_type = std::vector<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::indexer_iterator;
    using helpers_for_testing::apply_by_element;

    //0storage,1size,1first_processor,2second_processor,
    auto test_data = std::make_tuple(
        std::make_tuple(storage_type{}, [](auto it){return it;}, [](auto it){return it;}, true)
    );

}