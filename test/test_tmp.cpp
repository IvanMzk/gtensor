#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

TEMPLATE_TEST_CASE("test_tmp","[test_tmp]",
    test_config::config_storage_selector<gtensor::storage_vector>::config_type,
    test_config::config_storage_selector<std::vector>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using tensor_type = gtensor::tensor<value_type,config_type>;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::indexer_iterator;
    using gtensor::reverse_indexer_iterator;
    using gtensor::walker_iterator;
    using gtensor::reverse_walker_iterator;
    using helpers_for_testing::apply_by_element;

    auto plus = [](auto it, auto n){auto res =  it+=n; return res;};


    //0storage,1size,2difference_maker,3expected
    auto test_data = std::make_tuple(
        //operator+=
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto old_first = first; first+=0; return first - old_first;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto old_first = first; first+=1; return first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){first+=6; return first - last;}, index_type{0}),
        //operator-=
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){auto old_last = last; last-=0; return last - old_last;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){auto old_last = last; last-=1; return last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){last-=6; return last - first;}, index_type{0}),
        //operator+
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [plus](auto first, auto){auto tmp = first+0; return tmp-tmp;}, index_type{0})
    );

    SECTION("test_walker_iterator_difference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto difference_maker = std::get<2>(t);
            auto expected = std::get<3>(t);
            using walker_type = gtensor::walker<config_type, indexer_type>;
            using iterator_type = walker_iterator<config_type,walker_type>;
            using dim_type = typename config_type::dim_type;
            using shape_type = typename config_type::shape_type;
            auto shape = shape_type{size};
            auto strides = gtensor::detail::make_strides(shape);
            auto strides_div = gtensor::detail::make_strides_div<config_type>(shape);
            auto adapted_strides = gtensor::detail::make_adapted_strides(shape, strides);
            auto reset_strides = gtensor::detail::make_reset_strides(shape, strides);
            index_type offset{0};
            dim_type max_dim = shape.size();
            indexer_type indexer{storage};
            walker_type walker{adapted_strides,reset_strides,offset,indexer,max_dim};
            auto first = iterator_type{walker, shape, strides_div, 0};
            auto last = iterator_type{walker, shape, strides_div, size};
            auto result = difference_maker(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}