#include <tuple>
#include <vector>
#include <numeric>
#include "catch.hpp"
#include "config.hpp"
#include "broadcast.hpp"
#include "descriptor.hpp"
#include "indexer.hpp"
#include "helpers_for_testing.hpp"

TEMPLATE_TEST_CASE("test_variadic_broadcast_shape","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using result_shape_type = shape_type;
    using helpers_for_testing::apply_by_element;
    using gtensor::detail::broadcast_shape;
    //0shapes,1expected broadcast shape
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{1,2,3}), result_shape_type{1,2,3}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{1}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{0}), result_shape_type{0}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{1}, shape_type{1}, shape_type{1}), result_shape_type{1}),
        std::make_tuple(std::make_tuple(shape_type{5}, shape_type{5}), result_shape_type{5}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{0}), result_shape_type{1,0}),
        std::make_tuple(std::make_tuple(shape_type{0}, shape_type{1,1}), result_shape_type{1,0}),
        std::make_tuple(std::make_tuple(shape_type{2,1}, shape_type{0}), result_shape_type{2,0}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1}), result_shape_type{1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1}, shape_type{1,1,1}, shape_type{1,1}), result_shape_type{1,1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,1}, shape_type{1,1}, shape_type{1,1}), result_shape_type{1,1}),
        std::make_tuple(std::make_tuple(shape_type{1,5}, shape_type{5,1}), result_shape_type{5,5}),
        std::make_tuple(std::make_tuple(shape_type{1,5}, shape_type{5,1}, shape_type{1,5}, shape_type{1,1}), result_shape_type{5,5}),
        std::make_tuple(std::make_tuple(shape_type{1,2,0}, shape_type{3,1,1}), result_shape_type{3,2,0}),
        std::make_tuple(std::make_tuple(shape_type{2,3,4}, shape_type{3,4}), result_shape_type{2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,3,4}, shape_type{3,4}, shape_type{1,1,1,1}, shape_type{5,1,1,1}), result_shape_type{5,2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,1,4}, shape_type{3,1}, shape_type{3,4}), result_shape_type{2,3,4}),
        std::make_tuple(std::make_tuple(shape_type{0,1,4}, shape_type{3,1}, shape_type{0,3,4}, shape_type{1,3,4}), result_shape_type{0,3,4}),
        std::make_tuple(std::make_tuple(shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4}),
        std::make_tuple(std::make_tuple(shape_type{2,1}, shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4})
    );
    auto test = [](const auto& t){
        auto shapes = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto apply_shapes = [](const auto&...shapes_){
            return broadcast_shape<result_shape_type>(shapes_...);
        };
        auto result = std::apply(apply_shapes, shapes);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_variadic_broadcast_shape_exception","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::broadcast_exception;
    using gtensor::detail::broadcast_shape;
    using helpers_for_testing::apply_by_element;
    //0shapes
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{}),
        std::make_tuple(shape_type{}, shape_type{}),
        std::make_tuple(shape_type{1}, shape_type{}),
        std::make_tuple(shape_type{}, shape_type{1}),
        std::make_tuple(shape_type{0}, shape_type{2}),
        std::make_tuple(shape_type{2}, shape_type{0}),
        std::make_tuple(shape_type{3}, shape_type{2}),
        std::make_tuple(shape_type{2}, shape_type{3}),
        std::make_tuple(shape_type{1,2}, shape_type{0}),
        std::make_tuple(shape_type{1,2}, shape_type{3}),
        std::make_tuple(shape_type{1,2}, shape_type{4,3}),
        std::make_tuple(shape_type{3,2}, shape_type{4,2}),
        std::make_tuple(shape_type{5,1,2}, shape_type{4,4,2}),
        std::make_tuple(shape_type{}, shape_type{1}, shape_type{}),
        std::make_tuple(shape_type{3}, shape_type{0}, shape_type{3}),
        std::make_tuple(shape_type{3}, shape_type{3}, shape_type{2}),
        std::make_tuple(shape_type{1,2}, shape_type{3}, shape_type{1}),
        std::make_tuple(shape_type{1,2}, shape_type{1,1}, shape_type{4,4}),
        std::make_tuple(shape_type{5,1,0}, shape_type{2,1}, shape_type{5,2,2}),
        std::make_tuple(shape_type{5,1,2}, shape_type{2,2}, shape_type{4,4,2})
    );
    auto test = [](const auto& shapes){
        auto apply_shapes = [](const auto&...shapes_){
            return broadcast_shape<shape_type>(shapes_...);
        };
        REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), broadcast_exception);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_walker_iterator_adapter", "[test_broadcast]"){
    using config_type = gtensor::config::default_config;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using value_type = int;
    using storage_type = std::vector<value_type>;
    //using indexer_type = gtensor::basic_indexer<index_type, typename storage_type::iterator>;
    using indexer_type = gtensor::basic_indexer<index_type, gtensor::indexable<decltype(std::declval<storage_type>().data()) ,index_type> >;
    using walker_type = gtensor::walker<config_type,indexer_type>;
    using walker_adapter_type = gtensor::walker_iterator_adapter<config_type,walker_type>;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_reset_strides;
    using gtensor::detail::shape_to_str;
    using gtensor::detail::make_dividers;
    using helpers_for_testing::apply_by_element;

    //0shape,1elements,2action_on_adapter,3result_index,4result_walker_deref
    auto test_data = std::make_tuple(
        //first
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){}, shape_type{0,0,0}, value_type{1}),
        //prev first
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.prev();}, shape_type{0,1,2}, value_type{6}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.prev(); a.next();}, shape_type{0,0,0}, value_type{1}),
        //last
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.next(); a.next(); a.next(); a.next(); a.next();}, shape_type{0,1,2}, value_type{6}),
        //past last
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.next(); a.next(); a.next(); a.next(); a.next(); a.next();}, shape_type{0,0,0}, value_type{1}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.next(); a.next(); a.next(); a.next(); a.next(); a.next(); a.prev();}, shape_type{0,1,2}, value_type{6}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.next(); a.next(); a.prev(); a.next(); a.next(); a.prev(); a.prev();}, shape_type{0,0,1}, value_type{2}),
        //move
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(0);}, shape_type{0,0,0}, value_type{1}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(1);}, shape_type{0,0,1}, value_type{2}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(5);}, shape_type{0,1,2}, value_type{6}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(5); a.move(0);}, shape_type{0,0,0}, value_type{1}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(1); a.move(0); a.move(3);}, shape_type{0,1,0}, value_type{4}),
        //next prev move
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(0); a.next();}, shape_type{0,0,1}, value_type{2}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(1); a.prev();}, shape_type{0,0,0}, value_type{1}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(1); a.next();}, shape_type{0,0,2}, value_type{3}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.move(0); a.prev();}, shape_type{0,1,2}, value_type{6}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.prev(); a.move(4);}, shape_type{0,1,1}, value_type{5}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.next(); a.next(); a.move(3);}, shape_type{0,1,0}, value_type{4}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.next(); a.next(); a.next(); a.next(); a.next(); a.next(); a.move(3);}, shape_type{0,1,0}, value_type{4}),
        std::make_tuple(shape_type{1,2,3}, storage_type{1,2,3,4,5,6}, [](auto& a){a.prev(); a.next(); a.move(4); a.prev(); a.next();}, shape_type{0,1,1}, value_type{5})
    );

    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto elements = std::get<1>(t);
        auto action = std::get<2>(t);
        auto result_index = std::get<3>(t);
        auto result_walker_deref = std::get<4>(t);

        shape_type strides = make_strides(shape);
        shape_type reset_strides = make_reset_strides(shape,strides);
        auto strides_div = make_dividers<config_type>(strides);
        walker_type walker{shape, strides, reset_strides, index_type{0}, indexer_type{elements.data()}};
        walker_adapter_type adapter{shape, strides_div, walker};
        action(adapter);
        REQUIRE(adapter.index() == result_index);
        REQUIRE(*adapter.walker() == result_walker_deref);
    };

    apply_by_element(test, test_data);
}
