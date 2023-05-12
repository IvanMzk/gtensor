#include <vector>
#include "catch.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

namespace test_iterator_{

template<typename T>
class subscriptable_storage_by_value
{
    using inner_storage_type = std::vector<T>;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = typename inner_storage_type::size_type;
    using difference_type = typename inner_storage_type::difference_type;
    subscriptable_storage_by_value(std::initializer_list<value_type> init_list):
        impl_(init_list)
    {}
    value_type operator[](size_type i){return impl_[i];}
};

template<typename T>
class subscriptable_storage_integral
{
    using inner_storage_type = std::vector<T>;
    using inner_size_type = typename inner_storage_type::size_type;
    using inner_difference_type = typename inner_storage_type::difference_type;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = integral_type::integral<inner_size_type>;
    using difference_type = integral_type::integral<inner_difference_type>;
    subscriptable_storage_integral(std::initializer_list<value_type> init_list):
        impl_(init_list)
    {}
    decltype(std::declval<inner_storage_type&>()[std::declval<inner_size_type&>()]) operator[](size_type i){return impl_[i.value()];}
};

}   //end of namespace test_iterator_

TEMPLATE_TEST_CASE("test_random_access_iterator_difference","[test_iterator]",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<test_iterator_::subscriptable_storage_by_value>::config_type,
    test_config::config_storage_selector<test_iterator_::subscriptable_storage_integral>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::indexer_iterator;
    using gtensor::reverse_indexer_iterator;
    using gtensor::walker_iterator;
    using gtensor::reverse_walker_iterator;
    using helpers_for_testing::apply_by_element;

    //0storage,1size,2difference_maker,3expected
    auto test_data = std::make_tuple(
        //empty range
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto){return first-first;}, index_type{0}),
        std::make_tuple(storage_type{}, index_type{0}, [](auto, auto last){return last-last;}, index_type{0}),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return last-first;}, index_type{0}),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first-last;}, index_type{0}),
        //not empty range
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first-first;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last-last;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return last-first;}, index_type{6}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return first-last;}, index_type{-6}),
        //operator effect on difference
        //operator=
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto first_copy = first; return first - first_copy;}, index_type{0}),
        //operator++
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto old_first = first; ++first; return first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto old_first = first; return ++first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto old_first = first; first++; return first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto old_first = first; return first++ - old_first;}, index_type{0}),
        //operator--
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){auto old_last = last; --last; return last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){auto old_last = last; return --last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){auto old_last = last; last--; return last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){auto old_last = last; return last-- - old_last;}, index_type{0}),
        //operator+=
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto old_first = first; first+=0; return first - old_first;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto old_first = first; first+=1; return first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){first+=6; return first - last;}, index_type{0}),
        //operator-=
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){auto old_last = last; last-=0; return last - old_last;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){auto old_last = last; last-=1; return last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){last-=6; return last - first;}, index_type{0}),
        //operator+
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto tmp = first; tmp+=0; return tmp - first;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first+1 - first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return first+6 - last;}, index_type{0}),
        //operator-
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last-0 - last;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last-1 - last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto last){return last-6 - first;}, index_type{0})
    );

    SECTION("test_indexer_iterator_difference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto difference_maker = std::get<2>(t);
            auto expected = std::get<3>(t);
            indexer_type indexer{storage};
            using iterator_type = indexer_iterator<config_type,indexer_type>;
            auto first = iterator_type{indexer, 0};
            auto last = iterator_type{indexer, size};
            auto result = difference_maker(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_indexer_iterator_difference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto difference_maker = std::get<2>(t);
            auto expected = std::get<3>(t);
            indexer_type indexer{storage};
            using iterator_type = reverse_indexer_iterator<config_type,indexer_type>;
            auto first = iterator_type{indexer, size};
            auto last = iterator_type{indexer, 0};
            auto result = difference_maker(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
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
    SECTION("test_reverse_walker_iterator_difference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto difference_maker = std::get<2>(t);
            auto expected = std::get<3>(t);
            using walker_type = gtensor::walker<config_type, indexer_type>;
            using iterator_type = reverse_walker_iterator<config_type,walker_type>;
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
            auto first = iterator_type{walker, shape, strides_div, size};
            auto last = iterator_type{walker, shape, strides_div, 0};
            auto result = difference_maker(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_random_access_iterator_compare","[test_iterator]",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<test_iterator_::subscriptable_storage_by_value>::config_type,
    test_config::config_storage_selector<test_iterator_::subscriptable_storage_integral>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::indexer_iterator;
    using gtensor::reverse_indexer_iterator;
    using gtensor::walker_iterator;
    using gtensor::reverse_walker_iterator;
    using helpers_for_testing::apply_by_element;

    //0storage,1size,2comparator,3expected
    auto test_data = std::make_tuple(
        //empty range
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto){return first==first;}, true),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto){return first!=first;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto, auto last){return last==last;}, true),
        std::make_tuple(storage_type{}, index_type{0}, [](auto, auto last){return last!=last;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first==last;}, true),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first!=last;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first>last;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first>=last;}, true),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first<last;}, false),
        std::make_tuple(storage_type{}, index_type{0}, [](auto first, auto last){return first<=last;}, true),
        //not empty range
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first==first;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first!=first;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last==last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last!=last;}, false),
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
    SECTION("test_walker_iterator_compare")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto comparator = std::get<2>(t);
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
            auto result = comparator(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_walker_iterator_compare")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto comparator = std::get<2>(t);
            auto expected = std::get<3>(t);
            using walker_type = gtensor::walker<config_type, indexer_type>;
            using iterator_type = reverse_walker_iterator<config_type,walker_type>;
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
            auto first = iterator_type{walker, shape, strides_div, size};
            auto last = iterator_type{walker, shape, strides_div, 0};
            auto result = comparator(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_random_access_iterator_dereference","[test_iterator]",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<test_iterator_::subscriptable_storage_by_value>::config_type,
    test_config::config_storage_selector<test_iterator_::subscriptable_storage_integral>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using walker_type = gtensor::walker<config_type, indexer_type>;
    using indexer_iterator_type = gtensor::indexer_iterator<config_type,indexer_type>;
    using reverse_indexer_iterator_type = gtensor::reverse_indexer_iterator<config_type,indexer_type>;
    using walker_iterator_type = gtensor::walker_iterator<config_type,walker_type>;
    using reverse_walker_iterator_type = gtensor::reverse_walker_iterator<config_type,walker_type>;
    using gtensor::walker_iterator;
    using gtensor::reverse_walker_iterator;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(*std::declval<indexer_iterator_type>())>);
    REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(*std::declval<reverse_indexer_iterator_type>())>);
    REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(*std::declval<walker_iterator_type>())>);
    REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(*std::declval<reverse_walker_iterator_type>())>);

    //0storage,1size,2dereference_maker,3forward_expected,4reverse_expected
    auto test_data = std::make_tuple(
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return *first;}, value_type{1}, value_type{6}),
        //operator effect on dereference
        //operator=
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){auto first_copy = first; return *first_copy;}, value_type{1}, value_type{6}),
        //operator++
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){++first; return *first;}, value_type{2}, value_type{5}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return *++first;}, value_type{2}, value_type{5}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){first++; return *first;}, value_type{2}, value_type{5}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return *first++;}, value_type{1}, value_type{6}),
        //operator--
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){--last; return *last;}, value_type{6}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return *--last;}, value_type{6}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){last--; return *last;}, value_type{6}, value_type{1}),
        //operator+=
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){first+=0; return *first;}, value_type{1}, value_type{6}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){first+=1; return *first;}, value_type{2}, value_type{5}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){first+=5; return *first;}, value_type{6}, value_type{1}),
        //operator-=
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){last-=1; return *last;}, value_type{6}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){last-=6; return *last;}, value_type{1}, value_type{6}),
        //operator+
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return *(first+0);}, value_type{1}, value_type{6}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return *(first+1);}, value_type{2}, value_type{5}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return *(first+5);}, value_type{6}, value_type{1}),
        //operator-
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return *(last-1);}, value_type{6}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return *(last-6);}, value_type{1}, value_type{6}),
        //operator[]
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first[0];}, value_type{1}, value_type{6}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first[1];}, value_type{2}, value_type{5}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto first, auto){return first[5];}, value_type{6}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last[-1];}, value_type{6}, value_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, index_type{6}, [](auto, auto last){return last[-6];}, value_type{1}, value_type{6})
    );
    SECTION("test_indexer_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto dereference_maker = std::get<2>(t);
            auto forward_expected = std::get<3>(t);
            indexer_type indexer{storage};
            using iterator_type = indexer_iterator_type;
            auto first = iterator_type{indexer, 0};
            auto last = iterator_type{indexer, size};
            auto result = dereference_maker(first, last);
            REQUIRE(result == forward_expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_indexer_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto dereference_maker = std::get<2>(t);
            auto reverse_expected = std::get<4>(t);
            indexer_type indexer{storage};
            using iterator_type = reverse_indexer_iterator_type;
            auto first = iterator_type{indexer, size};
            auto last = iterator_type{indexer, 0};
            auto result = dereference_maker(first, last);
            REQUIRE(result == reverse_expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_walker_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto dereference_maker = std::get<2>(t);
            auto expected = std::get<3>(t);
            using iterator_type = walker_iterator_type;
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
            auto result = dereference_maker(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_walker_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto size = std::get<1>(t);
            auto dereference_maker = std::get<2>(t);
            auto reverse_expected = std::get<4>(t);
            using iterator_type = reverse_walker_iterator_type;
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
            auto first = iterator_type{walker, shape, strides_div, size};
            auto last = iterator_type{walker, shape, strides_div, 0};
            auto result = dereference_maker(first, last);
            REQUIRE(result == reverse_expected);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_gtensor_iterator_std_reverse_adapter","[test_iterator]",
    test_config::config_storage_selector<std::vector>::config_type,
    test_config::config_storage_selector<test_iterator_::subscriptable_storage_by_value>::config_type,
    test_config::config_storage_selector<test_iterator_::subscriptable_storage_integral>::config_type
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using index_type = typename config_type::index_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::indexer_iterator;
    using gtensor::walker_iterator;
    using gtensor::reverse_iterator_generic;
    using helpers_for_testing::apply_by_element;

    storage_type elements{1,2,3,4,5,6};
    index_type size{6};
    indexer_type indexer{elements};

    SECTION("test_indexer_iterator_std_reverse_adapter")
    {
        using iterator_type = indexer_iterator<config_type,indexer_type>;
        using reverse_iterator_type = reverse_iterator_generic<iterator_type>;
        reverse_iterator_type rfirst{indexer, size};
        reverse_iterator_type rlast{indexer, 0};
        auto std_rfirst = std::make_reverse_iterator(iterator_type{indexer, size});
        auto std_rlast = std::make_reverse_iterator(iterator_type{indexer, 0});
        REQUIRE(std::equal(std_rfirst,std_rlast,rfirst,rlast));
    }
    SECTION("test_walker_iterator_std_reverse_adapter")
    {
        using walker_type = gtensor::walker<config_type, indexer_type>;
        using iterator_type = walker_iterator<config_type,walker_type>;
        using reverse_iterator_type = reverse_iterator_generic<iterator_type>;
        using dim_type = typename config_type::dim_type;
        using shape_type = typename config_type::shape_type;
        auto shape = shape_type{size};
        auto strides = gtensor::detail::make_strides(shape);
        auto strides_div = gtensor::detail::make_strides_div<config_type>(shape);
        auto adapted_strides = gtensor::detail::make_adapted_strides(shape, strides);
        auto reset_strides = gtensor::detail::make_reset_strides(shape, strides);
        index_type offset{0};
        dim_type max_dim = shape.size();
        indexer_type indexer{elements};
        walker_type walker{adapted_strides,reset_strides,offset,indexer,max_dim};
        auto first = iterator_type{walker, shape, strides_div, 0};
        auto last = iterator_type{walker, shape, strides_div, size};
        auto rfirst = reverse_iterator_type{last};
        auto rlast = reverse_iterator_type{first};
        auto std_rfirst = std::make_reverse_iterator(last);
        auto std_rlast = std::make_reverse_iterator(first);
        REQUIRE(std::equal(std_rfirst,std_rlast,rfirst,rlast));
    }
    SECTION("test_reverse_reverse_indexer_iterator")
    {
        using iterator_type = indexer_iterator<config_type,indexer_type>;
        using reverse_iterator_type = reverse_iterator_generic<iterator_type>;
        using reverse_reverse_iterator_type = reverse_iterator_generic<reverse_iterator_type>;
        iterator_type first{indexer, 0};
        iterator_type last{indexer, size};
        reverse_iterator_type rfirst{last};
        reverse_iterator_type rlast{first};
        reverse_reverse_iterator_type rrfirst{rlast};
        reverse_reverse_iterator_type rrlast{rfirst};
        REQUIRE(std::equal(first,last,rrfirst,rrlast));
    }
}

