#include "catch.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

namespace test_walker_traverser{
template<typename Tr, typename Idx>
auto do_next(Tr& tr, Idx n){
    bool result{true};
    while(n!=0){
        result = tr.next();
        --n;
    }
    return result;
};
template<typename Tr, typename Idx>
auto do_prev(Tr& tr, Idx n){
    bool result{true};
    while(n!=0){
        result = tr.prev();
        --n;
    }
    return result;
};
}   //end of namespace test_walker_traverser

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
    using test_walker_traverser::do_next;
    using helpers_for_testing::apply_by_element;
    //0storage,1shape,2mover,3expected_index,4expected_element,5expected_is_next
    auto test_data = std::make_tuple(
        //1-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,0);}, shape_type{0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,1);}, shape_type{1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,2);}, shape_type{2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,3);}, shape_type{3} ,value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,4);}, shape_type{4} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,5);}, shape_type{5} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6);}, shape_type{0} ,value_type{1}, false),
        //3-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,0);}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,1);}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,2);}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,3);}, shape_type{1,0,0} ,value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,4);}, shape_type{1,0,1} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,5);}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6);}, shape_type{0,0,0} ,value_type{1}, false)
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
)
{
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
    using test_walker_traverser::do_next;
    using test_walker_traverser::do_prev;
    using helpers_for_testing::apply_by_element;
    //0storage,1shape,2mover,3expected_index,4expected_element,5expected_is_next
    auto test_data = std::make_tuple(
        //1-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1);}, shape_type{5} ,value_type{6}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),do_next(tr,1);}, shape_type{0}, value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),do_next(tr,2);}, shape_type{1}, value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),do_next(tr,3);}, shape_type{2}, value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),do_next(tr,4);}, shape_type{3}, value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),do_next(tr,5);}, shape_type{4}, value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),do_next(tr,6);}, shape_type{5}, value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),do_next(tr,7);}, shape_type{0}, value_type{1}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6), do_prev(tr,1);},shape_type{5},value_type{6},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6), do_prev(tr,2);},shape_type{4},value_type{5},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6), do_prev(tr,3);},shape_type{3},value_type{4},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6), do_prev(tr,4);},shape_type{2},value_type{3},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6), do_prev(tr,5);},shape_type{1},value_type{2},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6), do_prev(tr,6);},shape_type{0},value_type{1},true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6), do_prev(tr,7);},shape_type{5},value_type{6},false),
        //3-d
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1);}, shape_type{1,0,2} ,value_type{6}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),do_next(tr,1);}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),do_next(tr,2);}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),do_next(tr,3);}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),do_next(tr,4);}, shape_type{1,0,0} ,value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),do_next(tr,5);}, shape_type{1,0,1} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),do_next(tr,6);}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),do_next(tr,7);}, shape_type{0,0,0} ,value_type{1}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),do_prev(tr,1);}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),do_prev(tr,2);}, shape_type{1,0,1} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),do_prev(tr,3);}, shape_type{1,0,0} ,value_type{4}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),do_prev(tr,4);}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),do_prev(tr,5);}, shape_type{0,0,1} ,value_type{2}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),do_prev(tr,6);}, shape_type{0,0,0} ,value_type{1}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),do_prev(tr,7);}, shape_type{1,0,2} ,value_type{6}, false)
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
)
{
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
    using test_walker_traverser::do_next;
    using test_walker_traverser::do_prev;
    using helpers_for_testing::apply_by_element;
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
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),tr.move(5),do_next(tr,1);}, shape_type{0} ,value_type{1}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),tr.move(5),do_prev(tr,1);}, shape_type{4} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_prev(tr,1),tr.move(4),do_next(tr,1);}, shape_type{5} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6),tr.move(0),do_prev(tr,1);}, shape_type{5} ,value_type{6}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6),tr.move(1),do_next(tr,1);}, shape_type{2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{6}, [](auto& tr){return do_next(tr,6),tr.move(1),do_prev(tr,1);}, shape_type{0} ,value_type{1}, true),
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
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),tr.move(5),do_next(tr,1);}, shape_type{0,0,0} ,value_type{1}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),tr.move(5),do_prev(tr,1);}, shape_type{1,0,1} ,value_type{5}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_prev(tr,1),tr.move(4),do_next(tr,1);}, shape_type{1,0,2} ,value_type{6}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),tr.move(0),do_prev(tr,1);}, shape_type{1,0,2} ,value_type{6}, false),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),tr.move(1),do_next(tr,1);}, shape_type{0,0,2} ,value_type{3}, true),
        std::make_tuple(storage_type{1,2,3,4,5,6}, shape_type{2,1,3}, [](auto& tr){return do_next(tr,6),tr.move(1),do_prev(tr,1);}, shape_type{0,0,0} ,value_type{1}, true)
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
