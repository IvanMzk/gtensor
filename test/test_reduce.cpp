#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEST_CASE("test_check_reduce_args","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::reduce_exception;
    using gtensor::detail::check_reduce_args;

    REQUIRE_NOTHROW(check_reduce_args(shape_type{1},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{10},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},dim_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{2}));

    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},dim_type{0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},dim_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{1,0},dim_type{2}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},dim_type{3}), reduce_exception);
}

TEST_CASE("test_make_reduce_shape","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_reduce_shape;
    //0pshape,1direction,2expected
    using test_type = std::tuple<shape_type,dim_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{0},dim_type{0},shape_type{}},
        test_type{shape_type{1},dim_type{0},shape_type{}},
        test_type{shape_type{10},dim_type{0},shape_type{}},
        test_type{shape_type{2,3,0},dim_type{0},shape_type{3,0}},
        test_type{shape_type{2,3,0},dim_type{1},shape_type{2,0}},
        test_type{shape_type{2,3,0},dim_type{2},shape_type{2,3}},
        test_type{shape_type{2,3,4},dim_type{0},shape_type{3,4}},
        test_type{shape_type{2,3,4},dim_type{1},shape_type{2,4}},
        test_type{shape_type{2,3,4},dim_type{2},shape_type{2,3}}
    );
    auto pshape = std::get<0>(test_data);
    auto direction = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = make_reduce_shape(pshape,direction);
    REQUIRE(result == expected);
}

namespace test_reduce_{

struct max
{
    template<typename T>
    auto operator()(const T& u, const T& v){
        return std::max(u,v);
    }
};
struct min
{
    template<typename T>
    auto operator()(const T& u, const T& v){
        return std::min(u,v);
    }
};

}   //end of namespace test_reduce_

TEST_CASE("test_reduce","[test_reduce]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::max;
    using test_reduce_::min;
    using gtensor::reduce;
    using helpers_for_testing::apply_by_element;
    //0tensor,1direction,2functor,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, std::plus{}, tensor_type(value_type{})),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, std::plus{}, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, std::plus{}, tensor_type{value_type{}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, std::plus{}, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, std::plus{}, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, std::plus{}, tensor_type{{value_type{},value_type{},value_type{}},{value_type{},value_type{},value_type{}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, std::plus{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, std::plus{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, std::plus{}, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, std::plus{}, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, std::plus{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, std::plus{}, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, std::plus{}, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, std::multiplies{}, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, tensor_type{4,6,7,9}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, tensor_type{1,0}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, std::plus{}, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, std::plus{}, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, std::plus{}, tensor_type{{1,5},{9,13}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto direction = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = reduce(tensor, direction, functor);
        REQUIRE(result == expected);
        auto result1 = reduce(tensor, direction, functor);
        REQUIRE(result1 == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_reduce_ecxeption","[test_reduce]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using gtensor::reduce_exception;
    using gtensor::reduce;
    using helpers_for_testing::apply_by_element;


    //0tensor,1direction,2functor
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{1}, std::plus{}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{2}, std::plus{}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{2}, std::plus{}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{4}, std::plus{}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{3}, std::plus{})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto direction = std::get<1>(t);
        auto functor = std::get<2>(t);
        REQUIRE_THROWS_AS(reduce(tensor, direction, functor), reduce_exception);
    };
    apply_by_element(test, test_data);
}

