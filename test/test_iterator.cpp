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

}   //end of namespace test_iterator_

TEST_CASE("test_random_access_iterator_difference","[test_iterator]")
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::indexer_iterator;
    using gtensor::reverse_indexer_iterator;
    using gtensor::walker_iterator;
    using gtensor::reverse_walker_iterator;
    using gtensor::broadcast_iterator;
    using gtensor::reverse_broadcast_iterator;
    using gtensor::detail::make_dim;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_adapted_strides;
    using gtensor::detail::make_reset_strides;
    using gtensor::detail::make_strides_div;
    using helpers_for_testing::apply_by_element;

    //0storage,1shape,2command,3expected
    auto test_data = std::make_tuple(
        //empty range
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto){return first-first;}, index_type{0}),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto, auto last){return last-last;}, index_type{0}),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto last){return last-first;}, index_type{0}),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto last){return first-last;}, index_type{0}),
        //not empty range
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){return first-first;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){return last-last;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return last-first;}, index_type{6}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return first-last;}, index_type{-6}),
        //operator effect on difference
        //operator=
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){auto first_copy = first; return first - first_copy;}, index_type{0}),
        //operator++
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){auto old_first = first; ++first; return first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){auto old_first = first; return ++first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){auto old_first = first; first++; return first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){auto old_first = first; return first++ - old_first;}, index_type{0}),
        //operator--
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){auto old_last = last; --last; return last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){auto old_last = last; return --last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){auto old_last = last; last--; return last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){auto old_last = last; return last-- - old_last;}, index_type{0}),
        //operator+=
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){auto old_first = first; first+=0; return first - old_first;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){auto old_first = first; first+=1; return first - old_first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){first+=6; return first - last;}, index_type{0}),
        //operator-=
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){auto old_last = last; last-=0; return last - old_last;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){auto old_last = last; last-=1; return last - old_last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){last-=6; return last - first;}, index_type{0}),
        //operator+
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){auto tmp = first; tmp+=0; return tmp - first;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){return first+1 - first;}, index_type{1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return first+6 - last;}, index_type{0}),
        //operator-
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){return last-0 - last;}, index_type{0}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){return last-1 - last;}, index_type{-1}),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return last-6 - first;}, index_type{0}),
        //iterators compare
        //empty range
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto){return first==first;}, true),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto){return first!=first;}, false),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto, auto last){return last==last;}, true),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto, auto last){return last!=last;}, false),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto last){return first==last;}, true),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto last){return first!=last;}, false),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto last){return first>last;}, false),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto last){return first>=last;}, true),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto last){return first<last;}, false),
        std::make_tuple(storage_type{}, shape_type{0}, [](auto first, auto last){return first<=last;}, true),
        //not empty range
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){return first==first;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto){return first!=first;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){return last==last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto, auto last){return last!=last;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return first==last;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return first!=last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return first>last;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return first>=last;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return first<last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return first<=last;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return last>first;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return last>=first;}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return last<first;}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto first, auto last){return last<=first;}, false)
    );

    SECTION("test_indexer_iterator_difference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto command = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto size = make_size(shape);
            using iterator_type = indexer_iterator<config_type,indexer_type>;
            auto first = iterator_type{indexer_type{storage},index_type{0}};
            auto last = iterator_type{indexer_type{storage},size};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_indexer_iterator_difference")
    {
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto command = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto size = make_size(shape);
            using iterator_type = reverse_indexer_iterator<config_type,indexer_type>;
            auto first = iterator_type{indexer_type{storage},size};
            auto last = iterator_type{indexer_type{storage},index_type{0}};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_walker_iterator_difference")
    {
        using elements_order = c_order;
        using traverse_order = c_order;
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto command = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order{});
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using walker_type = gtensor::indexer_walker<config_type, indexer_type,elements_order>;
            walker_type walker{adapted_strides,reset_strides,offset,indexer};
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order>;
            using iterator_type = walker_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order{});
            auto first = iterator_type{walker, shape, strides_div, index_type{0}};
            auto last = iterator_type{walker, shape, strides_div, size};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_walker_iterator_difference")
    {
        using elements_order = c_order;
        using traverse_order = c_order;
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto command = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order{});
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using walker_type = gtensor::indexer_walker<config_type, indexer_type,elements_order>;
            walker_type walker{adapted_strides,reset_strides,offset,indexer};
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order>;
            using iterator_type = reverse_walker_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order{});
            auto first = iterator_type{walker, shape, strides_div, size};
            auto last = iterator_type{walker, shape, strides_div, index_type{0}};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_broadcast_iterator_difference")
    {
        using elements_order = c_order;
        using traverse_order = c_order;
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto command = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order{});
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, indexer_type,elements_order>>;
            walker_type walker{0,adapted_strides,reset_strides,offset,indexer};
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order>;
            using iterator_type = broadcast_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order{});
            auto first = iterator_type{walker, shape, strides_div, index_type{0}};
            auto last = iterator_type{walker, shape, strides_div, size};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_broadcast_iterator_difference")
    {
        using elements_order = c_order;
        using traverse_order = c_order;
        auto test = [](const auto& t){
            auto storage = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto command = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order{});
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, indexer_type,elements_order>>;
            walker_type walker{0,adapted_strides,reset_strides,offset,indexer};
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order>;
            using iterator_type = reverse_broadcast_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order{});
            auto first = iterator_type{walker, shape, strides_div, size};
            auto last = iterator_type{walker, shape, strides_div, index_type{0}};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_random_access_iterator_dereference","[test_iterator]")
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::indexer_iterator;
    using gtensor::reverse_indexer_iterator;
    using gtensor::walker_iterator;
    using gtensor::reverse_walker_iterator;
    using gtensor::broadcast_iterator;
    using gtensor::reverse_broadcast_iterator;
    using gtensor::detail::make_dim;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_adapted_strides;
    using gtensor::detail::make_reset_strides;
    using gtensor::detail::make_strides_div;
    using helpers_for_testing::apply_by_element;

    const auto storage_c = storage_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    const auto storage_f = storage_type{1,13,5,17,9,21,2,14,6,18,10,22,3,15,7,19,11,23,4,16,8,20,12,24};
    const auto shape = shape_type{2,3,4};

    //0elements_order,1traverse_order,2storage,3shape,4command,5forward_expected,6reverse_expected
    auto test_data = std::make_tuple(
        //operator effect on dereference
        //elements in c_order, traverse c_order
        //operator=
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return first = first+11, *first;}, value_type{12}, value_type{13}),
        //operator++
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return ++first, *first;}, value_type{2}, value_type{23}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return *++first;}, value_type{2}, value_type{23}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return first++, *first;}, value_type{2}, value_type{23}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return *first++;}, value_type{1}, value_type{24}),
        //operator--
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return --last, *last;}, value_type{24}, value_type{1}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return *--last;}, value_type{24}, value_type{1}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return last--, *last;}, value_type{24}, value_type{1}),
        //operator+=
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return first+=0, *first;}, value_type{1}, value_type{24}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return first+=1, *first;}, value_type{2}, value_type{23}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return first+=10, *first;}, value_type{11}, value_type{14}),
        //operator-=
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return last-=1, *last;}, value_type{24}, value_type{1}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return last-=10, *last;}, value_type{15}, value_type{10}),
        //operator+
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return *(first+0);}, value_type{1}, value_type{24}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return *(first+1);}, value_type{2}, value_type{23}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return *(first+5);}, value_type{6}, value_type{19}),
        //operator-
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return *(last-1);}, value_type{24}, value_type{1}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return *(last-6);}, value_type{19}, value_type{6}),
        //operator[]
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return first[0];}, value_type{1}, value_type{24}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return first[7];}, value_type{8}, value_type{17}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto first, auto){return first[16];}, value_type{17}, value_type{8}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return last[-8];}, value_type{17}, value_type{8}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto, auto last){return last[-20];}, value_type{5}, value_type{20}),
        //elements in c_order, traverse f_order
        //operator=
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return first = first+11, *first;}, value_type{22}, value_type{3}),
        //operator++
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return ++first, *first;}, value_type{13}, value_type{12}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return *++first;}, value_type{13}, value_type{12}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return first++, *first;}, value_type{13}, value_type{12}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return *first++;}, value_type{1}, value_type{24}),
        //operator--
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return --last, *last;}, value_type{24}, value_type{1}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return *--last;}, value_type{24}, value_type{1}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return last--, *last;}, value_type{24}, value_type{1}),
        //operator+=
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return first+=0, *first;}, value_type{1}, value_type{24}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return first+=1, *first;}, value_type{13}, value_type{12}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return first+=10, *first;}, value_type{10}, value_type{15}),
        //operator-=
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return last-=1, *last;}, value_type{24}, value_type{1}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return last-=10, *last;}, value_type{7}, value_type{18}),
        //operator+
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return *(first+0);}, value_type{1}, value_type{24}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return *(first+1);}, value_type{13}, value_type{12}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return *(first+5);}, value_type{21}, value_type{4}),
        //operator-
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return *(last-1);}, value_type{24}, value_type{1}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return *(last-6);}, value_type{4}, value_type{21}),
        //operator[]
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return first[0];}, value_type{1}, value_type{24}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return first[7];}, value_type{14}, value_type{11}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto first, auto){return first[16];}, value_type{11}, value_type{14}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return last[-8];}, value_type{11}, value_type{14}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto, auto last){return last[-20];}, value_type{9}, value_type{16}),
        //elements in f_order, traverse c_order
        //operator=
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return first = first+11, *first;}, value_type{12}, value_type{13}),
        //operator++
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return ++first, *first;}, value_type{2}, value_type{23}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return *++first;}, value_type{2}, value_type{23}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return first++, *first;}, value_type{2}, value_type{23}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return *first++;}, value_type{1}, value_type{24}),
        //operator--
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return --last, *last;}, value_type{24}, value_type{1}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return *--last;}, value_type{24}, value_type{1}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return last--, *last;}, value_type{24}, value_type{1}),
        //operator+=
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return first+=0, *first;}, value_type{1}, value_type{24}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return first+=1, *first;}, value_type{2}, value_type{23}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return first+=10, *first;}, value_type{11}, value_type{14}),
        //operator-=
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return last-=1, *last;}, value_type{24}, value_type{1}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return last-=10, *last;}, value_type{15}, value_type{10}),
        //operator+
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return *(first+0);}, value_type{1}, value_type{24}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return *(first+1);}, value_type{2}, value_type{23}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return *(first+5);}, value_type{6}, value_type{19}),
        //operator-
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return *(last-1);}, value_type{24}, value_type{1}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return *(last-6);}, value_type{19}, value_type{6}),
        //operator[]
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return first[0];}, value_type{1}, value_type{24}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return first[7];}, value_type{8}, value_type{17}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto first, auto){return first[16];}, value_type{17}, value_type{8}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return last[-8];}, value_type{17}, value_type{8}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto, auto last){return last[-20];}, value_type{5}, value_type{20}),
        //elements in f_order, traverse f_order
        //operator=
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return first = first+11, *first;}, value_type{22}, value_type{3}),
        //operator++
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return ++first, *first;}, value_type{13}, value_type{12}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return *++first;}, value_type{13}, value_type{12}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return first++, *first;}, value_type{13}, value_type{12}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return *first++;}, value_type{1}, value_type{24}),
        //operator--
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return --last, *last;}, value_type{24}, value_type{1}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return *--last;}, value_type{24}, value_type{1}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return last--, *last;}, value_type{24}, value_type{1}),
        //operator+=
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return first+=0, *first;}, value_type{1}, value_type{24}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return first+=1, *first;}, value_type{13}, value_type{12}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return first+=10, *first;}, value_type{10}, value_type{15}),
        //operator-=
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return last-=1, *last;}, value_type{24}, value_type{1}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return last-=10, *last;}, value_type{7}, value_type{18}),
        //operator+
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return *(first+0);}, value_type{1}, value_type{24}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return *(first+1);}, value_type{13}, value_type{12}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return *(first+5);}, value_type{21}, value_type{4}),
        //operator-
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return *(last-1);}, value_type{24}, value_type{1}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return *(last-6);}, value_type{4}, value_type{21}),
        //operator[]
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return first[0];}, value_type{1}, value_type{24}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return first[7];}, value_type{14}, value_type{11}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto first, auto){return first[16];}, value_type{11}, value_type{14}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return last[-8];}, value_type{11}, value_type{14}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto, auto last){return last[-20];}, value_type{9}, value_type{16})
    );
    SECTION("test_indexer_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            if constexpr (std::is_same_v<decltype(elements_order),decltype(traverse_order)>){
                auto storage = std::get<2>(t);
                auto shape = std::get<3>(t);
                auto command = std::get<4>(t);
                auto expected = std::get<5>(t);
                auto size = make_size(shape);
                using iterator_type = indexer_iterator<config_type,indexer_type>;
                auto first = iterator_type{indexer_type{storage},index_type{0}};
                auto last = iterator_type{indexer_type{storage},size};
                auto result = command(first, last);
                REQUIRE(result == expected);
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_indexer_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            if constexpr (std::is_same_v<decltype(elements_order),decltype(traverse_order)>){
                auto storage = std::get<2>(t);
                auto shape = std::get<3>(t);
                auto command = std::get<4>(t);
                auto expected = std::get<6>(t);
                auto size = make_size(shape);
                using iterator_type = reverse_indexer_iterator<config_type,indexer_type>;
                auto first = iterator_type{indexer_type{storage},size};
                auto last = iterator_type{indexer_type{storage},index_type{0}};
                auto result = command(first, last);
                REQUIRE(result == expected);
            }
        };
        apply_by_element(test,test_data);
    }

    SECTION("test_walker_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto storage = std::get<2>(t);
            auto shape = std::get<3>(t);
            auto command = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order);
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using elements_order_type = decltype(elements_order);
            using walker_type = gtensor::indexer_walker<config_type, indexer_type,elements_order_type>;
            walker_type walker{adapted_strides,reset_strides,offset,indexer};
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using iterator_type = walker_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order);
            auto first = iterator_type{walker, shape, strides_div, index_type{0}};
            auto last = iterator_type{walker, shape, strides_div, size};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_walker_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto storage = std::get<2>(t);
            auto shape = std::get<3>(t);
            auto command = std::get<4>(t);
            auto expected = std::get<6>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order);
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using elements_order_type = decltype(elements_order);
            using walker_type = gtensor::indexer_walker<config_type, indexer_type, elements_order_type>;
            walker_type walker{adapted_strides,reset_strides,offset,indexer};
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using iterator_type = reverse_walker_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order);
            auto first = iterator_type{walker, shape, strides_div, size};
            auto last = iterator_type{walker, shape, strides_div, index_type{0}};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }

    SECTION("test_broadcast_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto storage = std::get<2>(t);
            auto shape = std::get<3>(t);
            auto command = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order);
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using elements_order_type = decltype(elements_order);
            using walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, indexer_type, elements_order_type>>;
            walker_type walker{0,adapted_strides,reset_strides,offset,indexer};
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using iterator_type = broadcast_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order);
            auto first = iterator_type{walker, shape, strides_div, index_type{0}};
            auto last = iterator_type{walker, shape, strides_div, size};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_broadcast_iterator_dereference")
    {
        auto test = [](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto storage = std::get<2>(t);
            auto shape = std::get<3>(t);
            auto command = std::get<4>(t);
            auto expected = std::get<6>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order);
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using elements_order_type = decltype(elements_order);
            using walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, indexer_type, elements_order_type>>;
            walker_type walker{0,adapted_strides,reset_strides,offset,indexer};
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using iterator_type = reverse_broadcast_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order);
            auto first = iterator_type{walker, shape, strides_div, size};
            auto last = iterator_type{walker, shape, strides_div, index_type{0}};
            auto result = command(first, last);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

namespace test_random_access_iterator_result_type{

template<typename Config, typename Walker,typename Order> using traverser = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<Config,Walker>>,Order>;

}

TEST_CASE("test_random_access_iterator_result_type","test_iterator")
{
    using value_type = int;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using const_indexer_type = gtensor::basic_indexer<const storage_type&>;
    using walker_type = gtensor::indexer_walker<config_type, indexer_type, c_order>;
    using const_walker_type = gtensor::indexer_walker<config_type, const_indexer_type, c_order>;
    using max_dim_walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, indexer_type, c_order>>;
    using const_max_dim_walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, const_indexer_type, c_order>>;
    using gtensor::indexer_iterator;
    using gtensor::reverse_indexer_iterator;
    using gtensor::walker_iterator;
    using gtensor::reverse_walker_iterator;
    using gtensor::broadcast_iterator;
    using gtensor::reverse_broadcast_iterator;
    using test_random_access_iterator_result_type::traverser;

    REQUIRE(std::is_same_v<decltype(*std::declval<indexer_iterator<config_type,indexer_type>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<indexer_iterator<config_type,const_indexer_type>>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_indexer_iterator<config_type,indexer_type>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_indexer_iterator<config_type,const_indexer_type>>()),const value_type&>);

    REQUIRE(std::is_same_v<decltype(*std::declval<walker_iterator<config_type,traverser<config_type,walker_type,c_order>>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<walker_iterator<config_type,traverser<config_type,const_walker_type,c_order>>>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_walker_iterator<config_type,traverser<config_type,walker_type,c_order>>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_walker_iterator<config_type,traverser<config_type,const_walker_type,c_order>>>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<walker_iterator<config_type,traverser<config_type,walker_type,f_order>>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<walker_iterator<config_type,traverser<config_type,const_walker_type,f_order>>>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_walker_iterator<config_type,traverser<config_type,walker_type,f_order>>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_walker_iterator<config_type,traverser<config_type,const_walker_type,f_order>>>()),const value_type&>);

    REQUIRE(std::is_same_v<decltype(*std::declval<broadcast_iterator<config_type,traverser<config_type,max_dim_walker_type,c_order>>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<broadcast_iterator<config_type,traverser<config_type,const_max_dim_walker_type,c_order>>>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_broadcast_iterator<config_type,traverser<config_type,max_dim_walker_type,c_order>>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_broadcast_iterator<config_type,traverser<config_type,const_max_dim_walker_type,c_order>>>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<broadcast_iterator<config_type,traverser<config_type,max_dim_walker_type,f_order>>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<broadcast_iterator<config_type,traverser<config_type,const_max_dim_walker_type,f_order>>>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_broadcast_iterator<config_type,traverser<config_type,max_dim_walker_type,f_order>>>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<reverse_broadcast_iterator<config_type,traverser<config_type,const_max_dim_walker_type,f_order>>>()),const value_type&>);
}

TEMPLATE_TEST_CASE("test_gtensor_iterator_std_reverse_adapter","[test_iterator]",
    test_config::config_storage_selector_t<std::vector>,
    test_config::config_storage_selector_t<test_iterator_::subscriptable_storage_by_value>
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::indexer_iterator;
    using gtensor::walker_iterator;
    using gtensor::reverse_iterator_generic;
    using gtensor::detail::make_size;
    using helpers_for_testing::apply_by_element;

    using order = gtensor::config::c_order;
    storage_type elements{1,2,3,4,5,6,7,8,9,10,11,12};
    shape_type shape{2,3,2};
    indexer_type indexer{elements};
    auto size = make_size(shape);

    SECTION("test_indexer_iterator_std_reverse_adapter")
    {
        using iterator_type = indexer_iterator<config_type,indexer_type>;
        using reverse_iterator_type = reverse_iterator_generic<iterator_type>;
        reverse_iterator_type rfirst{indexer, size};
        reverse_iterator_type rlast{indexer, index_type{0}};
        auto std_rfirst = std::make_reverse_iterator(iterator_type{indexer, size});
        auto std_rlast = std::make_reverse_iterator(iterator_type{indexer, index_type{0}});
        REQUIRE(std::equal(std_rfirst,std_rlast,rfirst,rlast));
    }
    SECTION("test_walker_iterator_std_reverse_adapter")
    {
        using walker_type = gtensor::indexer_walker<config_type, indexer_type, order>;
        using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,order>;
        using iterator_type = walker_iterator<config_type,traverser_type>;
        using reverse_iterator_type = reverse_iterator_generic<iterator_type>;
        auto strides = gtensor::detail::make_strides(shape, order{});
        auto strides_div = gtensor::detail::make_strides_div<config_type>(shape, order{});
        auto adapted_strides = gtensor::detail::make_adapted_strides(shape, strides);
        auto reset_strides = gtensor::detail::make_reset_strides(shape, strides);
        index_type offset{0};
        walker_type walker{adapted_strides,reset_strides,offset,indexer};
        auto first = iterator_type{walker, shape, strides_div, 0};
        auto last = iterator_type{walker, shape, strides_div, size};
        auto rfirst = reverse_iterator_type{last};
        auto rlast = reverse_iterator_type{first};
        auto std_rfirst = std::make_reverse_iterator(last);
        auto std_rlast = std::make_reverse_iterator(first);
        REQUIRE(std::equal(std_rfirst,std_rlast,rfirst,rlast));
    }
}

TEST_CASE("test_walker_iterator_traverse","[test_iterator]")
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::walker_iterator;
    using gtensor::reverse_walker_iterator;
    using gtensor::detail::make_dim;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_adapted_strides;
    using gtensor::detail::make_reset_strides;
    using gtensor::detail::make_strides_div;
    using helpers_for_testing::apply_by_element;

    const auto storage_c = storage_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
    const auto storage_f = storage_type{1,13,5,17,9,21,2,14,6,18,10,22,3,15,7,19,11,23,4,16,8,20,12,24};

    //0elements_order,1traverse_order,2storage,3shape,4expected
    auto test_data = std::make_tuple(
        //elements in c_order
        std::make_tuple(c_order{}, c_order{}, storage_c, shape_type{24}, storage_c),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape_type{2,3,4}, storage_c),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape_type{24}, storage_c),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape_type{2,3,4}, storage_f),
        //elements in f_order
        std::make_tuple(f_order{}, c_order{}, storage_f, shape_type{24}, storage_f),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape_type{2,3,4}, storage_c),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape_type{24}, storage_f),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape_type{2,3,4}, storage_f)
    );

    auto test_equal = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        REQUIRE(std::equal(res_first,res_last,expected_first,expected_last));
    };
    auto test_backward_traverse = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        std::vector<value_type> result;
        while(res_last!=res_first){
            result.push_back(*--res_last);
        }
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_first,expected_last));
    };
    auto test_subscript = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        std::vector<value_type> result;
        for(index_type i{0}, i_last = res_last-res_first; i!=i_last; ++i){
            result.push_back(res_first[i]);
        }
        REQUIRE(std::equal(result.begin(),result.end(),expected_first,expected_last));
    };

    SECTION("test_walker_iterator")
    {
        auto test = [test_equal,test_backward_traverse,test_subscript](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto storage = std::get<2>(t);
            auto shape = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order);
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using elements_order_type = decltype(elements_order);
            using walker_type = gtensor::indexer_walker<config_type, indexer_type, elements_order_type>;
            walker_type walker{adapted_strides,reset_strides,offset,indexer};
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using iterator_type = walker_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order);
            auto first = iterator_type{walker, shape, strides_div, index_type{0}};
            auto last = iterator_type{walker, shape, strides_div, size};

            test_equal(first,last,expected.begin(),expected.end());
            test_backward_traverse(first,last,expected.begin(),expected.end());
            test_subscript(first,last,expected.begin(),expected.end());
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_walker_iterator")
    {
        auto test = [test_equal,test_backward_traverse,test_subscript](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto storage = std::get<2>(t);
            auto shape = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto size = make_size(shape);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order);
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            using elements_order_type = decltype(elements_order);
            using walker_type = gtensor::indexer_walker<config_type, indexer_type, elements_order_type>;
            walker_type walker{adapted_strides,reset_strides,offset,indexer};
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using iterator_type = reverse_walker_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(shape, traverse_order);
            auto first = iterator_type{walker, shape, strides_div, size};
            auto last = iterator_type{walker, shape, strides_div, index_type{0}};

            test_equal(first,last,expected.rbegin(),expected.rend());
            test_backward_traverse(first,last,expected.rbegin(),expected.rend());
            test_subscript(first,last,expected.rbegin(),expected.rend());
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_broadcast_iterator_traverse","[test_iterator]")
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::template storage<value_type>;
    using indexer_type = gtensor::basic_indexer<storage_type&>;
    using gtensor::broadcast_iterator;
    using gtensor::reverse_broadcast_iterator;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_dim;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_adapted_strides;
    using gtensor::detail::make_reset_strides;
    using gtensor::detail::make_strides_div;
    using helpers_for_testing::apply_by_element;

    //REQUIRE(std::is_same_v<decltype(std::declval<storage_type>()[std::declval<index_type>()]),decltype(*std::declval<broadcast_iterator_type>())>);

    auto storage_c = storage_type{1,2,3,4,5,6};
    auto storage_f = storage_type{1,4,2,5,3,6};

    //0elements_order,1traverse_order,2storage,3shape,4broadcast_shape,5expected
    auto test_data = std::make_tuple(
        //elements in c_order
        //traverse c_order
        std::make_tuple(c_order{}, c_order{}, storage_c, shape_type{6}, shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape_type{6}, shape_type{1,2,6}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape_type{2,3}, shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(c_order{}, c_order{}, storage_c, shape_type{2,3}, shape_type{2,2,3}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        //traverse f_order
        std::make_tuple(c_order{}, f_order{}, storage_c, shape_type{6}, shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape_type{6}, shape_type{1,2,6}, std::vector<value_type>{1,1,2,2,3,3,4,4,5,5,6,6}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape_type{2,3}, shape_type{2,3}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(c_order{}, f_order{}, storage_c, shape_type{2,3}, shape_type{2,2,3}, std::vector<value_type>{1,1,4,4,2,2,5,5,3,3,6,6}),
        //elements in f_order
        //traverse c_order
        std::make_tuple(f_order{}, c_order{}, storage_f, shape_type{6}, shape_type{6}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape_type{6}, shape_type{1,2,6}, std::vector<value_type>{1,4,2,5,3,6,1,4,2,5,3,6}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape_type{2,3}, shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(f_order{}, c_order{}, storage_f, shape_type{2,3}, shape_type{2,2,3}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        // //traverse f_order
        std::make_tuple(f_order{}, f_order{}, storage_f, shape_type{6}, shape_type{6}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape_type{6}, shape_type{1,2,6}, std::vector<value_type>{1,1,4,4,2,2,5,5,3,3,6,6}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape_type{2,3}, shape_type{2,3}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(f_order{}, f_order{}, storage_f, shape_type{2,3}, shape_type{2,2,3}, std::vector<value_type>{1,1,4,4,2,2,5,5,3,3,6,6})
    );

    auto test_equal = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        REQUIRE(std::equal(res_first,res_last,expected_first,expected_last));
    };
    auto test_backward_traverse = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        std::vector<value_type> result;
        while(res_last!=res_first){
            result.push_back(*--res_last);
        }
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_first,expected_last));
    };
    auto test_subscript = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        std::vector<value_type> result;
        for(index_type i{0}, i_last = res_last-res_first; i!=i_last; ++i){
            result.push_back(res_first[i]);
        }
        REQUIRE(std::equal(result.begin(),result.end(),expected_first,expected_last));
    };

    SECTION("test_broadcast_iterator")
    {
        auto test = [test_equal,test_backward_traverse,test_subscript](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto storage = std::get<2>(t);
            auto shape = std::get<3>(t);
            auto broadcast_shape = std::get<4>(t);
            auto expected = std::get<5>(t);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order);
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            auto dim = gtensor::detail::make_dim(shape);
            dim_type max_dim = std::max(gtensor::detail::make_dim(broadcast_shape),dim);
            auto dim_offset = max_dim-dim;
            using elements_order_type = decltype(elements_order);
            using walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, indexer_type, elements_order_type>>;
            walker_type walker{dim_offset,adapted_strides,reset_strides,offset,indexer};
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using iterator_type = broadcast_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(broadcast_shape, traverse_order);
            auto size = make_size(broadcast_shape);
            auto first = iterator_type{walker, broadcast_shape, strides_div, index_type{0}};
            auto last = iterator_type{walker, broadcast_shape, strides_div, size};

            test_equal(first,last,expected.begin(),expected.end());
            test_backward_traverse(first,last,expected.begin(),expected.end());
            test_subscript(first,last,expected.begin(),expected.end());
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_broadcast_iterator")
    {
        auto test = [test_equal,test_backward_traverse,test_subscript](const auto& t){
            auto elements_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto storage = std::get<2>(t);
            auto shape = std::get<3>(t);
            auto broadcast_shape = std::get<4>(t);
            auto expected = std::get<5>(t);
            index_type offset{0};
            auto strides = make_strides(shape, elements_order);
            auto adapted_strides = make_adapted_strides(shape, strides);
            auto reset_strides = make_reset_strides(shape, strides);
            auto indexer = indexer_type{storage};
            auto dim = gtensor::detail::make_dim(shape);
            dim_type max_dim = std::max(gtensor::detail::make_dim(broadcast_shape),dim);
            auto dim_offset = max_dim-dim;
            using elements_order_type = decltype(elements_order);
            using walker_type = gtensor::axes_correction_walker<gtensor::indexer_walker<config_type, indexer_type, elements_order_type>>;
            walker_type walker{dim_offset,adapted_strides,reset_strides,offset,indexer};
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using iterator_type = reverse_broadcast_iterator<config_type,traverser_type>;
            auto strides_div = make_strides_div<config_type>(broadcast_shape, traverse_order);
            auto size = make_size(broadcast_shape);
            auto rfirst = iterator_type{walker, broadcast_shape, strides_div, size};
            auto rlast = iterator_type{walker, broadcast_shape, strides_div, index_type{0}};

            test_equal(rfirst,rlast,expected.rbegin(),expected.rend());
            test_backward_traverse(rfirst,rlast,expected.rbegin(),expected.rend());
            test_subscript(rfirst,rlast,expected.rbegin(),expected.rend());
        };
        apply_by_element(test,test_data);
    }
}

