#include "catch.hpp"
#include "manipulation.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

//test helpers
TEST_CASE("test_check_stack_args_nothrow","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::check_stack_variadic_args;
    using gtensor::detail::check_stack_container_args;
    using helpers_for_testing::apply_by_element;
    //0axis,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0}, shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}))
    );
    SECTION("test_check_stack_variadic_args_nothrow")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto apply_shapes = [&axis](const auto&...shapes_){
                check_stack_variadic_args(axis, shapes_...);
            };
            REQUIRE_NOTHROW(std::apply(apply_shapes, shapes));
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_stack_container_args_nothrow")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            REQUIRE_NOTHROW(check_stack_container_args(axis,container));
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_check_stack_args_exception","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::value_error;
    using gtensor::detail::check_stack_container_args;
    using gtensor::detail::check_stack_variadic_args;
    using helpers_for_testing::apply_by_element;
    //0axis,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{4}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1}, shape_type{1})),
        std::make_tuple(dim_type{4}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1}, shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{2,2,3}, shape_type{1,2,3}))
    );
    SECTION("test_check_stack_variadic_args_exception")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto apply_shapes = [&axis](const auto&...shapes_){
                check_stack_variadic_args(axis, shapes_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_shapes, shapes), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_stack_container_args_exception")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            REQUIRE_THROWS_AS(check_stack_container_args(axis,container), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_check_concatenate_args_nothrow","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::check_concatenate_variadic_args;
    using gtensor::detail::check_concatenate_container_args;
    using helpers_for_testing::apply_by_element;

    //0axis,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1}, shape_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{2,0}, shape_type{3,0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{2,0}, shape_type{2,0})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{2,2}, shape_type{1,2})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{2,2}, shape_type{2,1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{5,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,20,3}, shape_type{1,10,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{1,2,3}, shape_type{5,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,22,3}, shape_type{1,12,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,13}, shape_type{1,2,33}))
    );
    SECTION("test_check_concatenate_variadic_args_nothrow")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            REQUIRE_NOTHROW(check_concatenate_variadic_args(axis, shapes));
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_concatenate_container_args_nothrow")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            REQUIRE_NOTHROW(check_concatenate_container_args(axis,container));
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_check_concatenate_args_exception","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::value_error;
    using gtensor::detail::check_concatenate_variadic_args;
    using gtensor::detail::check_concatenate_container_args;
    using helpers_for_testing::apply_by_element;

    //0axis,1shapes
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0}, shape_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1}, shape_type{1})),
        std::make_tuple(dim_type{3}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0,1}, shape_type{1,0})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,0}, shape_type{0,1})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,20,3}, shape_type{1,20,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{10,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,30}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{10,2,3}, shape_type{1,2,3})),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,12,3}, shape_type{1,2,3}))
    );
    SECTION("test_check_concatenate_variadic_args_exception")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            REQUIRE_THROWS_AS(check_concatenate_variadic_args(axis, shapes), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_concatenate_container_args_exception")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            REQUIRE_THROWS_AS(check_concatenate_container_args(axis,container), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_make_stack_shape","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_stack_shape;
    using test_type = std::tuple<shape_type,dim_type,index_type,shape_type>;
    //0shape,1axis,2tensors_number,3expected
    auto test_data = GENERATE(
        test_type{shape_type{0},dim_type{0},index_type{1},shape_type{1,0}},
        test_type{shape_type{0},dim_type{1},index_type{1},shape_type{0,1}},
        test_type{shape_type{5},dim_type{0},index_type{1},shape_type{1,5}},
        test_type{shape_type{5},dim_type{1},index_type{1},shape_type{5,1}},
        test_type{shape_type{3,4},dim_type{0},index_type{7},shape_type{7,3,4}},
        test_type{shape_type{3,4},dim_type{1},index_type{7},shape_type{3,7,4}},
        test_type{shape_type{3,4},dim_type{2},index_type{7},shape_type{3,4,7}},
        test_type{shape_type{3,4,5},dim_type{0},index_type{7},shape_type{7,3,4,5}},
        test_type{shape_type{3,4,5},dim_type{1},index_type{7},shape_type{3,7,4,5}},
        test_type{shape_type{3,4,5},dim_type{2},index_type{7},shape_type{3,4,7,5}},
        test_type{shape_type{3,4,5},dim_type{3},index_type{7},shape_type{3,4,5,7}}
    );
    auto shape = std::get<0>(test_data);
    auto axis = std::get<1>(test_data);
    auto tensors_number = std::get<2>(test_data);
    auto expected = std::get<3>(test_data);
    auto result = make_stack_shape(axis,shape,tensors_number);
    REQUIRE(result == expected);
}

TEST_CASE("test_make_concatenate_shape","[test_manipulation]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::detail::make_concatenate_variadic_shape;
    using gtensor::detail::make_concatenate_container_shape;
    using helpers_for_testing::apply_by_element;
    //0axis,1shapes,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}), shape_type{0}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{0}), shape_type{0}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{3}, shape_type{0}), shape_type{3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{3}), shape_type{3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0}, shape_type{1}, shape_type{2}, shape_type{0}), shape_type{3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0,3}, shape_type{2,3}), shape_type{2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{2,3}, shape_type{0,3}), shape_type{2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{0,2}, shape_type{0,2}), shape_type{0,2}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{0,2}, shape_type{0,2}), shape_type{0,4}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{2,0}, shape_type{2,3}), shape_type{2,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{2,3}, shape_type{2,0}), shape_type{2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}), shape_type{1,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{2,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{5,2,3}), shape_type{15,2,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,4,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,20,3}, shape_type{1,10,3}), shape_type{1,30,3}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,2,6}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,30}, shape_type{1,2,3}), shape_type{1,2,33}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{3,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(shape_type{10,2,3}, shape_type{1,2,3}, shape_type{5,2,3}), shape_type{16,2,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,6,3}),
        std::make_tuple(dim_type{1}, std::make_tuple(shape_type{1,2,3}, shape_type{1,22,3}, shape_type{1,12,3}), shape_type{1,36,3}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,3}, shape_type{1,2,3}), shape_type{1,2,9}),
        std::make_tuple(dim_type{2}, std::make_tuple(shape_type{1,2,3}, shape_type{1,2,13}, shape_type{1,2,33}), shape_type{1,2,49})
    );

    SECTION("test_make_concatenate_variadic_shape")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = make_concatenate_variadic_shape(axis, shapes);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_make_concatenate_container_shape")
    {
        using container_type = typename config_type::template container<shape_type>;
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto shapes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto make_shapes_container = [](const auto&...shapes_){
                return container_type{shapes_...};
            };
            auto container = std::apply(make_shapes_container, shapes);
            auto result = make_concatenate_container_shape(axis,container);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_nested_tuple_depth", "[test_manipulation]"){
    using gtensor::detail::nested_tuple_depth_v;
    REQUIRE(nested_tuple_depth_v<std::tuple<int>> == 1);
    REQUIRE(nested_tuple_depth_v<std::tuple<int,int>> == 1);
    REQUIRE(nested_tuple_depth_v<std::tuple<int,int,int>> == 1);
    REQUIRE(nested_tuple_depth_v<std::tuple<std::tuple<int>,std::tuple<int>,std::tuple<int>>> == 2);
    REQUIRE(nested_tuple_depth_v<std::tuple<std::tuple<int,int>,std::tuple<int>,std::tuple<int,int,int>>> == 2);
    REQUIRE(nested_tuple_depth_v<std::tuple<std::tuple<std::tuple<int>, std::tuple<int>>,std::tuple<std::tuple<int>>,std::tuple<std::tuple<int>>>> == 3);
}

TEST_CASE("test_is_tensor_nested_tuple", "[test_manipulation]"){
    using tensor_int_type = gtensor::tensor<int>;
    using tensor_double_type = gtensor::tensor<double>;
    using gtensor::detail::is_tensor_nested_tuple_v;
    REQUIRE(!is_tensor_nested_tuple_v<int>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>, int>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>, tensor_int_type>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>, std::tuple<tensor_int_type,int>>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<std::tuple<tensor_int_type>>, std::tuple<tensor_int_type>>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type,std::tuple<tensor_double_type>>,std::tuple<tensor_int_type,tensor_double_type>>>);
    REQUIRE(!is_tensor_nested_tuple_v<std::tuple<std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type,tensor_int_type>>,std::tuple<tensor_int_type>>>);
    REQUIRE(!is_tensor_nested_tuple_v<
        std::tuple<
            std::tuple<std::tuple<tensor_int_type,tensor_int_type>,std::tuple<tensor_int_type>>,
            std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type>,std::tuple<tensor_int_type>>,
            tensor_int_type>
        >
    );
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<tensor_int_type>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<tensor_int_type,tensor_double_type>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<tensor_int_type,tensor_int_type,tensor_double_type>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type,tensor_double_type>,std::tuple<tensor_double_type,tensor_int_type>>>);
    REQUIRE(is_tensor_nested_tuple_v<std::tuple<
        std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_int_type,tensor_int_type>>,
        std::tuple<std::tuple<tensor_int_type>,std::tuple<tensor_double_type>>,
        std::tuple<std::tuple<tensor_double_type,tensor_int_type,tensor_int_type>>>>
    );
}

TEST_CASE("test_widen_shape", "[test_manipulation]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using dim_type = config_type::dim_type;
    using gtensor::detail::widen_shape;
    //0shape,1new_dim,2expected
    using test_type = std::tuple<shape_type,dim_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{0},dim_type{1},shape_type{0}},
        test_type{shape_type{0},dim_type{2},shape_type{1,0}},
        test_type{shape_type{0},dim_type{3},shape_type{1,1,0}},
        test_type{shape_type{2,3},dim_type{1},shape_type{2,3}},
        test_type{shape_type{2,3},dim_type{2},shape_type{2,3}},
        test_type{shape_type{2,3},dim_type{3},shape_type{1,2,3}},
        test_type{shape_type{2,3},dim_type{4},shape_type{1,1,2,3}}
    );
    auto shape = std::get<0>(test_data);
    auto new_dim = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = widen_shape(shape,new_dim);
    REQUIRE(result == expected);
}

//test module routines
TEMPLATE_TEST_CASE("test_stack_nothrow","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using dim_type = typename tensor_type::dim_type;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}), tensor_type{}.reshape(1,0)),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{}), tensor_type{}.reshape(1,0)),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}), tensor_type{}.reshape(0,1)),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}.reshape(3,0)),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}.reshape(0,3)),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(3,1,0)),
        std::make_tuple(dim_type{-2}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(3,1,0)),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(1,3,0)),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(1,3,0)),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0), tensor_type{}.reshape(1,0)), tensor_type{}.reshape(1,0,3)),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1},{2},{3}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1,2,3}}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1,2,3,4},tensor_type{5,6,7,8},tensor_type{9,10,11,12}), tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1,2,3,4},tensor_type{5,6,7,8},tensor_type{9,10,11,12}), tensor_type{{1,5,9},{2,6,10},{3,7,11},{4,8,12}}),
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{-2},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{7,8,9},{13,14,15},{19,20,21}},{{4,5,6},{10,11,12},{16,17,18},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{-1},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,2,3},{7,8,9},{13,14,15},{19,20,21}},{{4,5,6},{10,11,12},{16,17,18},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13,14,15},{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{{1,7,13,19},{2,8,14,20},{3,9,15,21}},{{4,10,16,22},{5,11,17,23},{6,12,18,24}}}
        )
    );
    SECTION("test_stack_variadic_nothrow")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);

            auto apply_tensors = [&axis](const auto&...tensors_){
                return stack(axis, tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_stack_container_nothrow")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(order{}))>;
        auto test_concatenate_container = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(order{})...};}, tensors);
            auto result = stack(axis, container);
            REQUIRE(result == expected);
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEST_CASE("test_stack_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0axis,1tensors
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{})),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{1})),
        std::make_tuple(dim_type{4}, std::make_tuple(tensor_type({1,2,3},value_type{}))),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{1}, tensor_type{1})),
        std::make_tuple(dim_type{4}, std::make_tuple(tensor_type({1,2,3},value_type{}), tensor_type({1,2,3},value_type{}))),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}, tensor_type{1})),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1}, tensor_type{})),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type({1,2,3},value_type{}), tensor_type{})),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}, tensor_type({1,2,3},value_type{}))),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type({1,2,3},value_type{}), tensor_type({1,2,3},value_type{}), tensor_type({1,1,2,3},value_type{}))),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type({1,2,3},value_type{}), tensor_type({2,2,3},value_type{}), tensor_type({1,2,3},value_type{})))
    );
    SECTION("test_stack_variadic_exception")
    {
        auto test = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto apply_tensors = [&axis](const auto&...tensors_){
                return stack(axis, tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_stack_container_exception")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy())>;
        auto test_concatenate_container = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy()...};}, tensors);
            REQUIRE_THROWS_AS(stack(axis, container), value_error);
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEST_CASE("test_stack_common_type","[test_manipulation]")
{
    using tensor_int32_type = gtensor::tensor<int>;
    using tensor_int64_type = gtensor::tensor<std::int64_t>;
    using tensor_double_type = gtensor::tensor<double>;
    using dim_type = std::common_type_t<typename tensor_int32_type::dim_type, typename tensor_int64_type::dim_type, typename tensor_double_type::dim_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_int32_type{1},tensor_int32_type{2},tensor_int64_type{3}), tensor_int64_type{{1},{2},{3}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_int32_type{1},tensor_double_type{2},tensor_int64_type{3}), tensor_double_type{{1,2,3}})
    );
    auto test = [](const auto& t){
        auto axis = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&axis](const auto&...tensors_){
            return stack(axis, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_stack_common_order","[test_manipulation]")
{

    using value_type = int;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using dim_type = typename tensor<value_type>::dim_type;
    using traverse_order = typename tensor<value_type>::config_type::order;
    using helpers_for_testing::apply_by_element;
    using gtensor::stack;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(
            dim_type{0},
            std::make_tuple(
                tensor<value_type, c_order>{{1,2,3},{4,5,6}},
                tensor<value_type, c_order>{{7,8,9},{10,11,12}},
                tensor<value_type, c_order>{{13,14,15},{16,17,18}},
                tensor<value_type, c_order>{{19,20,21},{22,23,24}}
            ),
            tensor<value_type, c_order>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(
                tensor<value_type, f_order>{{1,2,3},{4,5,6}},
                tensor<value_type, f_order>{{7,8,9},{10,11,12}},
                tensor<value_type, f_order>{{13,14,15},{16,17,18}},
                tensor<value_type, f_order>{{19,20,21},{22,23,24}}
            ),
            tensor<value_type, f_order>{{{1,2,3},{7,8,9},{13,14,15},{19,20,21}},{{4,5,6},{10,11,12},{16,17,18},{22,23,24}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(
                tensor<value_type, f_order>{{1,2,3},{4,5,6}},
                tensor<value_type, c_order>{{7,8,9},{10,11,12}},
                tensor<value_type, c_order>{{13,14,15},{16,17,18}},
                tensor<value_type, f_order>{{19,20,21},{22,23,24}}),
            tensor<value_type, traverse_order>{{{1,7,13,19},{2,8,14,20},{3,9,15,21}},{{4,10,16,22},{5,11,17,23},{6,12,18,24}}}
        )
    );
    auto test = [](const auto& t){
        auto axis = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&axis](const auto&...tensors_){
            return stack(axis, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_concatenate","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using dim_type = typename tensor_type::dim_type;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1}), tensor_type{1}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{1,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{},tensor_type{1},tensor_type{},tensor_type{2},tensor_type{3},tensor_type{}), tensor_type{1,2,3}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{1},tensor_type{2,3},tensor_type{4,5,6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{1},tensor_type{2,3},tensor_type{4,5,6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}.reshape(1,0),tensor_type{}.reshape(2,0)), tensor_type{}.reshape(3,0)),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}.reshape(0,1),tensor_type{}.reshape(0,2)), tensor_type{}.reshape(0,3)),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(dim_type{-2}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}.reshape(0,2),tensor_type{{1,2},{3,4}},tensor_type{}.reshape(0,2),tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}.reshape(2,0),tensor_type{{1,2},{3,4}},tensor_type{}.reshape(2,0),tensor_type{{5},{6}}), tensor_type{{1,2,5},{3,4,6}}),
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{}.reshape(0,2,2),tensor_type{{{9,10},{11,12}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{}.reshape(2,0,2),tensor_type{{{9,10}},{{11,12}}}),
            tensor_type{{{1,2},{3,4},{9,10}},{{5,6},{7,8},{11,12}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{}.reshape(2,2,0),tensor_type{{{9},{10}},{{11},{12}}}),
            tensor_type{{{1,2,9},{3,4,10}},{{5,6,11},{7,8,12}}}
        ),
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12},{13,14,15}},tensor_type{{16,17,18}},tensor_type{{19,20,21},{22,23,24}}),
            tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15},{16,17,18},{19,20,21},{22,23,24}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5},{6}},tensor_type{{7,8,9},{10,11,12}},tensor_type{{13},{14}}),
            tensor_type{{1,2,5,7,8,9,13},{3,4,6,10,11,12,14}}
        ),
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10},{11,12}}},tensor_type{{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}
        ),
        std::make_tuple(
            dim_type{-3},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10},{11,12}}},tensor_type{{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10}},{{11,12}}},tensor_type{{{13,14},{15,16},{17,18}},{{19,20},{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4},{9,10},{13,14},{15,16},{17,18}},{{5,6},{7,8},{11,12},{19,20},{21,22},{23,24}}}
        ),
        std::make_tuple(
            dim_type{-2},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10}},{{11,12}}},tensor_type{{{13,14},{15,16},{17,18}},{{19,20},{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4},{9,10},{13,14},{15,16},{17,18}},{{5,6},{7,8},{11,12},{19,20},{21,22},{23,24}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9},{10}},{{11},{12}}},tensor_type{{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}),
            tensor_type{{{1,2,9,13,14,15},{3,4,10,16,17,18}},{{5,6,11,19,20,21},{7,8,12,22,23,24}}}
        ),
        std::make_tuple(
            dim_type{-1},
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9},{10}},{{11},{12}}},tensor_type{{{13,14,15},{16,17,18}},{{19,20,21},{22,23,24}}}),
            tensor_type{{{1,2,9,13,14,15},{3,4,10,16,17,18}},{{5,6,11,19,20,21},{7,8,12,22,23,24}}}
        ),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}), tensor_type{{{1,3}},{{2,4}}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{{{1,2}},{{3,4}}}, tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}), tensor_type{{{1,2},{5,6},{7,8}},{{3,4},{9,10},{11,12}}}),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}, tensor_type{{{13,14},{15,16},{17,18}}}),
            tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    SECTION("test_concatenate_variadic")
    {
        auto test_concatenate_variadic = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);

            auto apply_tensors = [&axis](const auto&...tensors_){
                return concatenate(axis, tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result == expected);
        };
        apply_by_element(test_concatenate_variadic, test_data);
    }
    SECTION("test_concatenate_container")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(order{}))>;
        auto test_concatenate_container = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(order{})...};}, tensors);
            auto result = concatenate(axis, container);
            REQUIRE(result == expected);
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEST_CASE("test_concatenate_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using shape_type = typename tensor_type::shape_type;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0axis,1tensors
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{})),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1})),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{{1,2,3},{4,5,6}})),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{0}, tensor_type{0})),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{1}, tensor_type{1})),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{1,2,3},{4,5,6}})),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{}.reshape(0,1), tensor_type{}.reshape(1,0))),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{}.reshape(1,0), tensor_type{}.reshape(0,1))),
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{2,4,4}}, tensor_type{shape_type{2,3,4}})),
        std::make_tuple(dim_type{-3}, std::make_tuple(tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{2,4,4}}, tensor_type{shape_type{2,3,4}})),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_type{shape_type{2,3,5}}, tensor_type{shape_type{2,4,4}}, tensor_type{shape_type{2,3,4}})),
        std::make_tuple(dim_type{-2}, std::make_tuple(tensor_type{shape_type{2,3,5}}, tensor_type{shape_type{2,4,4}}, tensor_type{shape_type{2,3,4}})),
        std::make_tuple(dim_type{2}, std::make_tuple(tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{3,3,4}})),
        std::make_tuple(dim_type{-1}, std::make_tuple(tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{2,3,4}}, tensor_type{shape_type{3,3,4}}))
    );
    SECTION("test_concatenate_variadic_exception")
    {
        auto test_concatenate_variadic = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto apply_tensors = [&axis](const auto&...tensors_){
                return concatenate(axis, tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), value_error);
        };
        apply_by_element(test_concatenate_variadic, test_data);
    }
    SECTION("test_concatenate_container")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy())>;
        auto test_concatenate_container = [](const auto& t){
            auto axis = std::get<0>(t);
            auto tensors = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy()...};}, tensors);
            REQUIRE_THROWS_AS(concatenate(axis, container), value_error);
        };
        apply_by_element(test_concatenate_container, test_data);
    }
}

TEST_CASE("test_concatenate_common_type","[test_manipulation]")
{
    using tensor_int32_type = gtensor::tensor<int>;
    using tensor_int64_type = gtensor::tensor<std::int64_t>;
    using tensor_double_type = gtensor::tensor<double>;
    using dim_type = std::common_type_t<typename tensor_int32_type::dim_type, typename tensor_int64_type::dim_type, typename tensor_double_type::dim_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{0}, std::make_tuple(tensor_int32_type{{1,2},{3,4}},tensor_int64_type{{5,6}}), tensor_int64_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(dim_type{1}, std::make_tuple(tensor_int64_type{{1,2},{3,4}},tensor_double_type{{5},{6}}), tensor_double_type{{1,2,5},{3,4,6}})
    );
    auto test = [](const auto& t){
        auto axis = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&axis](const auto&...tensors_){
            return concatenate(axis, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_concatenate_common_order","[test_manipulation]")
{
    using value_type = int;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using dim_type = typename tensor<value_type>::dim_type;
    using traverse_order = typename tensor<value_type>::config_type::order;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    //0axis,1tensors,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(
            dim_type{0},
            std::make_tuple(tensor<value_type, c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},tensor<value_type, c_order>{}.reshape(0,2,2),tensor<value_type, c_order>{{{9,10},{11,12}}}),
            tensor<value_type, c_order>{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}
        ),
        std::make_tuple(
            dim_type{1},
            std::make_tuple(tensor<value_type, f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},tensor<value_type, f_order>{}.reshape(2,0,2),tensor<value_type, f_order>{{{9,10}},{{11,12}}}),
            tensor<value_type, f_order>{{{1,2},{3,4},{9,10}},{{5,6},{7,8},{11,12}}}
        ),
        std::make_tuple(
            dim_type{2},
            std::make_tuple(tensor<value_type, f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},tensor<value_type, c_order>{}.reshape(2,2,0),tensor<value_type, c_order>{{{9},{10}},{{11},{12}}}),
            tensor<value_type, traverse_order>{{{1,2,9},{3,4,10}},{{5,6,11},{7,8,12}}}
        )
    );
    auto test = [](const auto& t){
        auto axis = std::get<0>(t);
        auto tensors = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_tensors = [&axis](const auto&...tensors_){
            return concatenate(axis, tensors_...);
        };
        auto result = std::apply(apply_tensors, tensors);
        REQUIRE(std::is_same_v<typename decltype(result)::value_type, typename decltype(expected)::value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_vstack","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using helpers_for_testing::apply_by_element;
    using gtensor::vstack;
    //0tensors,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}), tensor_type{}.reshape(1,0)),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}.reshape(3,0)),
        std::make_tuple(std::make_tuple(tensor_type{}.reshape(2,0), tensor_type{}.reshape(2,0), tensor_type{}.reshape(2,0)), tensor_type{}.reshape(6,0)),
        std::make_tuple(std::make_tuple(tensor_type{1}), tensor_type{{1}}),
        std::make_tuple(std::make_tuple(tensor_type{1},tensor_type{2},tensor_type{3}), tensor_type{{1},{2},{3}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3},tensor_type{4,5,6},tensor_type{7,8,9}), tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}), tensor_type{{1,2},{3,4}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{5,6}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{{5,6}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}},{{5}}}, tensor_type{{{6}}}), tensor_type{{{1}},{{2}},{{3}},{{4}},{{5}},{{6}}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2}},{{3,4}}}, tensor_type{{{5,6}},{{7,8}}}, tensor_type{{{9,10}}}), tensor_type{{{1,2}},{{3,4}},{{5,6}},{{7,8}},{{9,10}}}),
        std::make_tuple(
            std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{7,8,9},{10,11,12},{13,14,15}},tensor_type{{16,17,18}},tensor_type{19,20,21}),
            tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15},{16,17,18},{19,20,21}}
        ),
        std::make_tuple(
            std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{9,10},{11,12}}},tensor_type{{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}},{{17,18},{19,20}},{{21,22},{23,24}}}
        ),
        std::make_tuple(std::make_tuple(tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}, tensor_type{{{13,14},{15,16},{17,18}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    SECTION("test_vstack_variadic")
    {
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return vstack(tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_vstack_container")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(order{}))>;
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(order{})...};}, tensors);
            auto result = vstack(container);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_vstack_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::value_error;
    using gtensor::vstack;
    //0tensors
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{1}, tensor_type{1,1})),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{5,6,7})),
        std::make_tuple(std::make_tuple(tensor_type{{{1}}}, tensor_type{1})),
        std::make_tuple(std::make_tuple(tensor_type{{{1}}}, tensor_type{{1}})),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, tensor_type{{1,2},{3,4}}))
    );
    SECTION("test_vstack_variadic_exception")
    {
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return vstack(tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_vstack_container_exception")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy())>;
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy()...};}, tensors);
            REQUIRE_THROWS_AS(vstack(container), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_hstack","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using helpers_for_testing::apply_by_element;
    using gtensor::hstack;
    //0tensors,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{1,2,3}, tensor_type{}, tensor_type{4,5,6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7},{8}}, tensor_type{}.reshape(2,0), tensor_type{{9,10},{11,12}}), tensor_type{{1,2,3,7,9,10},{4,5,6,8,11,12}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, tensor_type{{{9,10}},{{11,12}}}, tensor_type{}.reshape(2,0,2)), tensor_type{{{1,2},{3,4},{9,10}},{{5,6},{7,8},{11,12}}})
    );
    SECTION("test_hstack_variadic")
    {
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return hstack(tensors_...);
            };
            auto result = std::apply(apply_tensors, tensors);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_hstack_container")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy(order{}))>;
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy(order{})...};}, tensors);
            auto result = hstack(container);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_hstack_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::value_error;
    using gtensor::hstack;
    //0tensors
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}.reshape(1,0))),
        std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7},{8},{9}})),
        std::make_tuple(std::make_tuple(tensor_type{}.reshape(2,2,0), tensor_type{}.reshape(3,2,0))),
        std::make_tuple(std::make_tuple(tensor_type{}.reshape(0,2,2), tensor_type{}.reshape(0,2,3)))
    );
    SECTION("test_hstack_variadic_exception")
    {
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto apply_tensors = [](const auto&...tensors_){
                return hstack(tensors_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_tensors, tensors), value_error);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_hstack_container_exception")
    {
        using container_type = std::vector<decltype(std::declval<tensor_type>().copy())>;
        auto test = [](const auto& t){
            auto tensors = std::get<0>(t);
            auto container = std::apply([](const auto&...ts){return container_type{ts.copy()...};}, tensors);
            REQUIRE_THROWS_AS(hstack(container), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_block_tuple","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;

    //0blocks,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{}, tensor_type{}), tensor_type{}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{1,2,3}, tensor_type{}, tensor_type{4,5}, tensor_type{}), tensor_type{1,2,3,4,5}),
        std::make_tuple(std::make_tuple(tensor_type{}, tensor_type{{1,2,3}}, tensor_type{}, tensor_type{{{4,5}}}, tensor_type{}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{})), tensor_type{}.reshape(1,0)),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{}), std::make_tuple(tensor_type{})), tensor_type{}.reshape(2,0)),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{},tensor_type{}),std::make_tuple(tensor_type{},tensor_type{})), tensor_type{}.reshape(2,0)),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{},tensor_type{1,2,3}),std::make_tuple(tensor_type{4,5,6},tensor_type{})), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5}), tensor_type{1,2,3,4,5}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7,8},{9,10}}), tensor_type{{1,2,3,7,8},{4,5,6,9,10}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5})), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2,3,4,5}))), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}})), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{4,5,6}}))), tensor_type{{{1,2,3},{4,5,6}}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6})), tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2}), std::make_tuple(tensor_type{3,4}), std::make_tuple(tensor_type{5,6})), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1}, tensor_type{2,3}), std::make_tuple(tensor_type{4,5,6})), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}}})),
            std::make_tuple(std::make_tuple(tensor_type{{{3}},{{4}},{{5}}}))),
            tensor_type{{{1}},{{2}},{{3}},{{4}},{{5}}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor_type{1,2})),std::make_tuple(std::make_tuple(tensor_type{3,4})),std::make_tuple(std::make_tuple(tensor_type{5,6}))),
            tensor_type{{{1,2}},{{3,4}},{{5,6}}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor_type{1,2}),std::make_tuple(tensor_type{3,4})),std::make_tuple(std::make_tuple(tensor_type{5,6}),std::make_tuple(tensor_type{7,8}))),
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}
        ),
        std::make_tuple(std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2}, tensor_type{3,4,5}, tensor_type{6,7,8,9}))), tensor_type{{{1,2,3,4,5,6,7,8,9}}}),
        std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}},{{3}}}, tensor_type{{{4}},{{5}},{{6}}}), tensor_type{{{1,4}},{{2,5}},{{3,6}}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2}, tensor_type{{3,4,5}}), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(std::make_tuple(tensor_type{{3,4,5}}, tensor_type{1,2}), tensor_type{{3,4,5,1,2}}),
        std::make_tuple(std::make_tuple(tensor_type{1,2}, tensor_type{{{3,4,5}}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}, tensor_type{{11},{12}}), tensor_type{{1,2,5,6,7,11},{3,4,8,9,10,12}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{1,2}), std::make_tuple(tensor_type{{3,4},{5,6}})), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{{1,2,3},{3,4,5}}, tensor_type{{6,7,8,9},{10,11,12,13}})), tensor_type{{1,2,3,6,7,8,9},{3,4,5,10,11,12,13}}),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{{1,2},{3,4}}), std::make_tuple(tensor_type{{7,8},{9,10},{11,12}})), tensor_type{{1,2},{3,4},{7,8},{9,10},{11,12}}),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}), std::make_tuple(tensor_type{{11,12,13,14,15}})),
            tensor_type{{1,2,5,6,7},{3,4,8,9,10},{11,12,13,14,15}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor_type{{1,2},{3,4}}, tensor_type{{5,6},{7,8}}), std::make_tuple(tensor_type{{9},{10}}, tensor_type{{11,12,13},{14,15,16}})),
            tensor_type{{1,2,5,6},{3,4,7,8},{9,11,12,13},{10,14,15,16}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor_type{{1,2},{3,4}}), std::make_tuple(tensor_type{{5,6},{7,8}}), std::make_tuple(tensor_type{{9},{10}}, tensor_type{{11},{12}})),
            tensor_type{{1,2},{3,4},{5,6},{7,8},{9,11},{10,12}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}), std::make_tuple(tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}})),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}), std::make_tuple(tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}})),
            std::make_tuple(std::make_tuple(tensor_type{13,14}), std::make_tuple(tensor_type{15,16}), std::make_tuple(tensor_type{17,18}))),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    auto test = [](const auto& t){
        auto blocks = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = block(blocks);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_block_tuple_common_order","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;

    //0blocks,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor<value_type,c_order>{{1,2},{3,4}}, tensor<value_type,f_order>{{5,6,7},{8,9,10}}), std::make_tuple(tensor<value_type,c_order>{{11,12,13,14,15}})),
            tensor_type{{1,2,5,6,7},{3,4,8,9,10},{11,12,13,14,15}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor<value_type,f_order>{{1,2},{3,4}}, tensor<value_type,f_order>{{5,6},{7,8}}), std::make_tuple(tensor<value_type,c_order>{{9},{10}}, tensor<value_type,c_order>{{11,12,13},{14,15,16}})),
            tensor_type{{1,2,5,6},{3,4,7,8},{9,11,12,13},{10,14,15,16}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor<value_type,c_order>{{1,2},{3,4}}), std::make_tuple(tensor<value_type,f_order>{{5,6},{7,8}}), std::make_tuple(tensor<value_type,c_order>{{9},{10}}, tensor<value_type,f_order>{{11},{12}})),
            tensor_type{{1,2},{3,4},{5,6},{7,8},{9,11},{10,12}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(tensor<value_type,f_order>{{{1}},{{2}}}, tensor<value_type,c_order>{{{3}},{{4}}}), std::make_tuple(tensor<value_type,c_order>{{{5,6},{7,8}},{{9,10},{11,12}}})),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}
        ),
        std::make_tuple(std::make_tuple(
            std::make_tuple(std::make_tuple(tensor<value_type,c_order>{{{1}},{{2}}}, tensor<value_type,f_order>{{{3}},{{4}}}), std::make_tuple(tensor<value_type,c_order>{{{5,6},{7,8}},{{9,10},{11,12}}})),
            std::make_tuple(std::make_tuple(tensor<value_type,f_order>{13,14}), std::make_tuple(tensor<value_type,c_order>{15,16}), std::make_tuple(tensor<value_type,f_order>{17,18}))),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}}
        )
    );
    auto test = [](const auto& t){
        auto blocks = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = block(blocks);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_block_tuple_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;
    //blocks
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2},tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{1},{2},{3}},tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{3,4},{5,6}}, tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}},{{5}}}),
        std::make_tuple(std::make_tuple(tensor_type{},tensor_type{1,2,3}),std::make_tuple(tensor_type{})),
        std::make_tuple(std::make_tuple(tensor_type{1,2}),std::make_tuple(tensor_type{})),
        std::make_tuple(std::make_tuple(tensor_type{1,2}),std::make_tuple(tensor_type{3,4,5})),
        std::make_tuple(std::make_tuple(std::make_tuple(tensor_type{},tensor_type{}),std::make_tuple(tensor_type{})), std::make_tuple(std::make_tuple(tensor_type{})))
    );
    auto test = [](const auto& blocks){
        REQUIRE_THROWS_AS(block(blocks),value_error);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_block_init_list","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;
    //0result,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{}}), tensor_type{}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{}, tensor_type{}, tensor_type{}}), tensor_type{}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{}, tensor_type{1,2,3}, tensor_type{}, tensor_type{4,5}, tensor_type{}}), tensor_type{1,2,3,4,5}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{}, tensor_type{{1,2,3}}, tensor_type{}, tensor_type{{{4,5}}}, tensor_type{}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{}}}), tensor_type{}.reshape(1,0)),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{},tensor_type{}},{tensor_type{},tensor_type{}}}), tensor_type{}.reshape(2,0)),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{},tensor_type{1,2,3}},{tensor_type{4,5,6},tensor_type{}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2,3,4,5}}), tensor_type{1,2,3,4,5}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{1,2,3},{4,5,6}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}), tensor_type{1,2,3,4,5,6}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{1,2,3},{4,5,6}}, tensor_type{{7,8},{9,10}}}), tensor_type{{1,2,3,7,8},{4,5,6,9,10}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1,2,3,4,5}}}), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{1,2,3,4,5}}}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2,3},{4,5,6}}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{{1,2,3},{4,5,6}}}}}), tensor_type{{{1,2,3},{4,5,6}}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{6}}}), tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1,2}}, {tensor_type{3,4}}, {tensor_type{5,6}}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1}, tensor_type{2,3}}, {tensor_type{4,5,6}}}), tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{1,2}}},{{tensor_type{3,4}}},{{tensor_type{5,6}}}}), tensor_type{{{1,2}},{{3,4}},{{5,6}}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{1,2}},{tensor_type{3,4}}},{{tensor_type{5,6}},{tensor_type{7,8}}}}), tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{{{tensor_type{1,2}, tensor_type{3,4,5}, tensor_type{6,7,8,9}}}}), tensor_type{{{1,2,3,4,5,6,7,8,9}}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{{1}},{{2}},{{3}}}, tensor_type{{{4}},{{5}},{{6}}}}), tensor_type{{{1,4}},{{2,5}},{{3,6}}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2}, tensor_type{{3,4,5}}}), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{3,4,5}}, tensor_type{1,2}}), tensor_type{{3,4,5,1,2}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{1,2}, tensor_type{{{3,4,5}}}}), tensor_type{{{1,2,3,4,5}}}),
        std::make_tuple(block(nested_init_list1<tensor_type>{tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}, tensor_type{{11},{12}}}), tensor_type{{1,2,5,6,7,11},{3,4,8,9,10,12}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{1,2}}, {tensor_type{{3,4},{5,6}}}}), tensor_type{{1,2},{3,4},{5,6}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2,3},{3,4,5}}, tensor_type{{6,7,8,9},{10,11,12,13}}}}), tensor_type{{1,2,3,6,7,8,9},{3,4,5,10,11,12,13}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2},{3,4}}}, {tensor_type{{7,8},{9,10},{11,12}}}}), tensor_type{{1,2},{3,4},{7,8},{9,10},{11,12}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2},{3,4}}, tensor_type{{5,6,7},{8,9,10}}}, {tensor_type{{11,12,13,14,15}}}}), tensor_type{{1,2,5,6,7},{3,4,8,9,10},{11,12,13,14,15}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2},{3,4}}, tensor_type{{5,6},{7,8}}}, {tensor_type{{9},{10}}, tensor_type{{11,12,13},{14,15,16}}}}), tensor_type{{1,2,5,6},{3,4,7,8},{9,11,12,13},{10,14,15,16}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{{tensor_type{{1,2},{3,4}}}, {tensor_type{{5,6},{7,8}}}, {tensor_type{{9},{10}}, tensor_type{{11},{12}}}}), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,11},{10,12}}),
        std::make_tuple(block(nested_init_list2<tensor_type>{
            {tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}},
            {tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}}}),
        std::make_tuple(block(nested_init_list3<tensor_type>{
            {{tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}}}}, {tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}}},
            {{tensor_type{13,14}}, {tensor_type{15,16}}, {tensor_type{17,18}}}}),
            tensor_type{{{1,3},{5,6},{7,8}},{{2,4},{9,10},{11,12}},{{13,14},{15,16},{17,18}}})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result == expected);
    };

    apply_by_element(test, test_data);
}

TEST_CASE("test_block_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::block;
    using helpers_for_testing::apply_by_element;
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{1,2},tensor_type{{3,4},{5,6}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{{1},{2},{3}},tensor_type{{3,4},{5,6}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{{3,4},{5,6}}, tensor_type{}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{},tensor_type{{3,4},{5,6}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list1<tensor_type>{tensor_type{{{1}},{{2}}}, tensor_type{{{3}},{{4}},{{5}}} }), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list2<tensor_type>{{tensor_type{},tensor_type{1,2,3}},{tensor_type{}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list2<tensor_type>{{tensor_type{1,2}},{tensor_type{}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list2<tensor_type>{{tensor_type{1,2}},{tensor_type{3,4,5}}}), value_error);
    REQUIRE_THROWS_AS(block(nested_init_list3<tensor_type>{{{tensor_type{},tensor_type{}},{tensor_type{}}}, {{tensor_type{}}}}), value_error);
}

TEMPLATE_TEST_CASE("test_split_split_points","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::split;
    using helpers_for_testing::apply_by_element;

    //0ten,1split_points,2axis,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, std::vector<int>{}, dim_type{0}, result_type{tensor_type{}}),
        std::make_tuple(tensor_type{}, std::vector<int>{0}, dim_type{0}, result_type{tensor_type{},tensor_type{}}),
        std::make_tuple(tensor_type{}, std::vector<int>{0,1}, dim_type{0}, result_type{tensor_type{},tensor_type{},tensor_type{}}),
        std::make_tuple(tensor_type{}.reshape(5,0), std::vector<int>{}, dim_type{0}, result_type{tensor_type{}.reshape(5,0).copy(order{})}),
        std::make_tuple(tensor_type{}.reshape(5,0), std::vector<int>{0}, dim_type{0}, result_type{tensor_type{}.reshape(0,0).copy(order{}), tensor_type{}.reshape(5,0).copy(order{})}),
        std::make_tuple(
            tensor_type{}.reshape(5,0),
            std::vector<int>{0,1},
            dim_type{0},
            result_type{tensor_type{}.reshape(0,0).copy(order{}), tensor_type{}.reshape(1,0).copy(order{}), tensor_type{}.reshape(4,0).copy(order{})}
        ),
        std::make_tuple(tensor_type{1}, std::vector<int>{}, dim_type{0}, result_type{tensor_type{1}}),
        std::make_tuple(tensor_type{1}, std::vector<int>{0}, dim_type{0}, result_type{tensor_type{},tensor_type{1}}),
        std::make_tuple(tensor_type{1}, std::vector<int>{1}, dim_type{0}, result_type{tensor_type{1},tensor_type{}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{}, dim_type{0}, result_type{tensor_type{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{2}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{2,4}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{0,3,5}, dim_type{0}, result_type{tensor_type{}, tensor_type{1,2,3}, tensor_type{4,5}, tensor_type{}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::vector<int>{1,2,3,4}, dim_type{0}, result_type{tensor_type{1}, tensor_type{2}, tensor_type{3}, tensor_type{4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, std::initializer_list<int>{2,4}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, gtensor::tensor<int>{2,4}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5}}),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            std::initializer_list<index_type>{0,2},
            dim_type{0},
            result_type{tensor_type{}.reshape(0,4).copy(order{}), tensor_type{{1,2,3,4},{5,6,7,8}}, tensor_type{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            std::initializer_list<index_type>{1,2},
            dim_type{0},
            result_type{tensor_type{{1,2,3,4}}, tensor_type{{5,6,7,8}}, tensor_type{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            std::initializer_list<int>{1,2},
            dim_type{1},
            result_type{tensor_type{{1},{5},{9},{13}}, tensor_type{{2},{6},{10},{14}}, tensor_type{{3,4},{7,8},{11,12},{15,16}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            std::vector<int>{1,2},
            dim_type{0},
            result_type{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            std::vector<int>{1,2},
            dim_type{-3},
            result_type{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            gtensor::tensor<index_type>{1},
            dim_type{1},
            result_type{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            gtensor::tensor<index_type>{1},
            dim_type{-2},
            result_type{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            std::initializer_list<std::size_t>{1,2},
            dim_type{2},
            result_type{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            std::initializer_list<std::size_t>{1,2},
            dim_type{-1},
            result_type{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto split_points = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = split(ten, split_points, axis);
        REQUIRE(expected.size() == result.size());
        auto result_it = result.begin();
        for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
            REQUIRE(*result_it == *expected_it);
        }
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_split_equal_parts","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::split;
    using helpers_for_testing::apply_by_element;

    //0ten,1parts_number,2axis,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, index_type{1}, dim_type{0}, result_type{tensor_type{}}),
        std::make_tuple(tensor_type{}, index_type{2}, dim_type{0}, result_type{tensor_type{},tensor_type{}}),
        std::make_tuple(tensor_type{}, index_type{3}, dim_type{0}, result_type{tensor_type{},tensor_type{},tensor_type{}}),
        std::make_tuple(
            tensor_type{}.reshape(3,0),
            index_type{3},
            dim_type{0},
            result_type{tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{})}
        ),
        std::make_tuple(tensor_type{1}, index_type{1}, dim_type{0}, result_type{tensor_type{1}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, index_type{1}, dim_type{0}, result_type{tensor_type{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, int{5}, dim_type{0}, result_type{tensor_type{1}, tensor_type{2}, tensor_type{3}, tensor_type{4}, tensor_type{5}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, index_type{2}, dim_type{0}, result_type{tensor_type{1,2,3}, tensor_type{4,5,6}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::size_t{3}, dim_type{0}, result_type{tensor_type{1,2}, tensor_type{3,4}, tensor_type{5,6}}),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            index_type{2},
            dim_type{0},
            result_type{tensor_type{{1,2,3,4},{5,6,7,8}}, tensor_type{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},
            index_type{2},
            dim_type{1},
            result_type{tensor_type{{1,2},{5,6},{9,10},{13,14}}, tensor_type{{3,4},{7,8},{11,12},{15,16}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            index_type{3},
            dim_type{0},
            result_type{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},
            index_type{3},
            dim_type{-3},
            result_type{tensor_type{{{1,2},{3,4}}},  tensor_type{{{5,6},{7,8}}}, tensor_type{{{9,10},{11,12}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{2},
            dim_type{1},
            result_type{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{2},
            dim_type{-2},
            result_type{tensor_type{{{1,2,3}},{{7,8,9}},{{13,14,15}}}, tensor_type{{{4,5,6}},{{10,11,12}},{{16,17,18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{3},
            dim_type{2},
            result_type{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}},
            index_type{3},
            dim_type{-1},
            result_type{tensor_type{{{1},{4}},{{7},{10}},{{13},{16}}},  tensor_type{{{2},{5}},{{8},{11}},{{14},{17}}}, tensor_type{{{3},{6}},{{9},{12}},{{15},{18}}}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto parts_number = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = split(ten, parts_number, axis);
        REQUIRE(expected.size() == result.size());
        auto result_it = result.begin();
        for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
            REQUIRE(*result_it == *expected_it);
        }
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_split_exception","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::split;
    using helpers_for_testing::apply_by_element;

    //0tensor,1split_arg,2axis
    auto test_data = std::make_tuple(
        //split by points
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{1},1),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::vector<int>{1},2),
        //split by equal parts
        std::make_tuple(tensor_type{1,2,3,4,5},1,1),
        std::make_tuple(tensor_type{1,2,3,4,5},0,0),
        std::make_tuple(tensor_type{1,2,3,4,5},2,0),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},1,2),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},0,0),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},4,0)
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto split_arg = std::get<1>(t);
        auto axis = std::get<2>(t);
        REQUIRE_THROWS_AS(split(ten,split_arg,axis), value_error);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_vsplit","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::vsplit;
    using helpers_for_testing::apply_by_element;

    SECTION("test_vsplit_nothrow")
    {
        //0tensor,1split_arg,2expected
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::vector<int>{2},result_type{tensor_type{{1,2},{3,4}},tensor_type{{5,6},{7,8},{9,10}}}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::vector<int>{1,3},result_type{tensor_type{{1,2}},tensor_type{{3,4},{5,6}},tensor_type{{7,8},{9,10}}}),
            std::make_tuple(
                tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},
                std::initializer_list<int>{1,2,3,4},
                result_type{tensor_type{{1,2}},tensor_type{{3,4}},tensor_type{{5,6}},tensor_type{{7,8}},tensor_type{{9,10}}}
            ),
            std::make_tuple(
                tensor_type{{{1},{2}},{{3},{4}},{{5},{6}},{{7},{8}},{{9},{10}}},
                gtensor::tensor<std::size_t>{1,3},
                result_type{tensor_type{{{1},{2}}},tensor_type{{{3},{4}},{{5},{6}}},tensor_type{{{7},{8}},{{9},{10}}}}
            ),
            //split by equal parts
            std::make_tuple(tensor_type{}.reshape(1,0),1,result_type{tensor_type{}.reshape(1,0).copy(order{})}),
            std::make_tuple(tensor_type{}.reshape(3,0),3,result_type{tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{})}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},1,result_type{tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}}}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},5,result_type{tensor_type{{1,2}},tensor_type{{3,4}},tensor_type{{5,6}},tensor_type{{7,8}},tensor_type{{9,10}}}),
            std::make_tuple(
                tensor_type{{{1},{2}},{{3},{4}},{{5},{6}},{{7},{8}}},
                2,
                result_type{tensor_type{{{1},{2}},{{3},{4}}},tensor_type{{{5},{6}},{{7},{8}}}}
            )
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = vsplit(ten, split_arg);
            REQUIRE(expected.size() == result.size());
            auto result_it = result.begin();
            for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
                REQUIRE(*result_it == *expected_it);
            }
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_vsplit_exception")
    {
        using gtensor::value_error;
        //0tensor,1split_arg
        auto test_data = std::make_tuple(
            std::make_tuple(tensor_type{1,2,3},std::vector<int>{1}),
            std::make_tuple(tensor_type{1,2,3},3),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},2)
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            REQUIRE_THROWS_AS(vsplit(ten, split_arg), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEMPLATE_TEST_CASE("test_hsplit","[test_manipulation]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type, order>;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::hsplit;
    using helpers_for_testing::apply_by_element;

    SECTION("test_hsplit_nothrow")
    {
        //0tensor,1split_arg,2expected
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{2},result_type{tensor_type{1,2},tensor_type{3,4,5}}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},std::vector<int>{1},result_type{tensor_type{{1},{5},{9}},tensor_type{{2,3,4},{6,7,8},{10,11,12}}}),
            std::make_tuple(
                tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},
                std::vector<int>{1,3},
                result_type{tensor_type{{1},{5},{9}},tensor_type{{2,3},{6,7},{10,11}},tensor_type{{4},{8},{12}}}
            ),
            std::make_tuple(
                tensor_type{{{1},{2},{3},{4}},{{5},{6},{7},{8}},{{9},{10},{11},{12}}},
                gtensor::tensor<std::size_t>{1,3},
                result_type{tensor_type{{{1}},{{5}},{{9}}},tensor_type{{{2},{3}},{{6},{7}},{{10},{11}}},tensor_type{{{4}},{{8}},{{12}}}}
            ),
            //split by equal parts
            std::make_tuple(tensor_type{}.reshape(0,3),3,result_type{tensor_type{}.reshape(0,1).copy(order{}),tensor_type{}.reshape(0,1).copy(order{}),tensor_type{}.reshape(0,1).copy(order{})}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1,result_type{tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}}}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},2,result_type{tensor_type{{1,2},{5,6},{9,10}},tensor_type{{3,4},{7,8},{11,12}}}),
            std::make_tuple(
                tensor_type{{{1},{2},{3},{4}},{{5},{6},{7},{8}},{{9},{10},{11},{12}}},
                2,
                result_type{tensor_type{{{1},{2}},{{5},{6}},{{9},{10}}},tensor_type{{{3},{4}},{{7},{8}},{{11},{12}}}}
            )
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = hsplit(ten, split_arg);
            REQUIRE(expected.size() == result.size());
            auto result_it = result.begin();
            for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
                REQUIRE(*result_it == *expected_it);
            }
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_hsplit_exception")
    {
        using gtensor::value_error;
        //0tensor,1split_arg
        auto test_data = std::make_tuple(
            std::make_tuple(tensor_type{1},2),
            std::make_tuple(tensor_type{1,2,3},2),
            std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},2)
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            REQUIRE_THROWS_AS(hsplit(ten, split_arg), value_error);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_hsplit_hstack","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using order = typename tensor_type::order;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::hsplit;
    using gtensor::hstack;
    using helpers_for_testing::apply_by_element;

    SECTION("test_hsplit_hstack")
    {
        //0tensor,1split_arg
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{2}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},std::vector<int>{1}),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},std::vector<int>{1,3}),
            std::make_tuple(tensor_type{{{1},{2},{3},{4}},{{5},{6},{7},{8}},{{9},{10},{11},{12}}},gtensor::tensor<std::size_t>{1,3}),
            //split by equal parts
            std::make_tuple(tensor_type{}.reshape(0,3),3),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1),
            std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},2),
            std::make_tuple(tensor_type{{{1},{2},{3},{4}},{{5},{6},{7},{8}},{{9},{10},{11},{12}}},2)
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            auto expected = ten;
            auto result = hstack(hsplit(ten, split_arg));
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_hstack_hsplit")
    {
        //0split_arg,1parts
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(std::vector<int>{2},result_type{tensor_type{1,2},tensor_type{3,4,5}}),
            std::make_tuple(std::vector<int>{1},result_type{tensor_type{{1},{5},{9}},tensor_type{{2,3,4},{6,7,8},{10,11,12}}}),
            std::make_tuple(std::vector<int>{1,3},result_type{tensor_type{{1},{5},{9}},tensor_type{{2,3},{6,7},{10,11}},tensor_type{{4},{8},{12}}}),
            std::make_tuple(gtensor::tensor<std::size_t>{1,3},result_type{tensor_type{{{1}},{{5}},{{9}}},tensor_type{{{2},{3}},{{6},{7}},{{10},{11}}},tensor_type{{{4}},{{8}},{{12}}}}),
            //split by equal parts
            std::make_tuple(3,result_type{tensor_type{}.reshape(0,1).copy(order{}),tensor_type{}.reshape(0,1).copy(order{}),tensor_type{}.reshape(0,1).copy(order{})}),
            std::make_tuple(1,result_type{tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}}}),
            std::make_tuple(2,result_type{tensor_type{{1,2},{5,6},{9,10}},tensor_type{{3,4},{7,8},{11,12}}}),
            std::make_tuple(2,result_type{tensor_type{{{1},{2}},{{5},{6}},{{9},{10}}},tensor_type{{{3},{4}},{{7},{8}},{{11},{12}}}})
        );
        auto test = [](const auto& t){
            auto split_arg = std::get<0>(t);
            auto parts = std::get<1>(t);
            auto expected = parts;
            auto result = hsplit(hstack(parts), split_arg);
            REQUIRE(expected.size() == result.size());
            auto result_it = result.begin();
            for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
                REQUIRE(*result_it == *expected_it);
            }
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_vsplit_vstack","[test_manipulation]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using order = typename tensor_type::order;
    using config_type = typename tensor_type::config_type;
    using result_type = typename config_type::template container<tensor_type>;
    using gtensor::vstack;
    using gtensor::vsplit;
    using helpers_for_testing::apply_by_element;

    SECTION("test_vsplit_vstack")
    {
        //0tensor,1split_arg
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::vector<int>{2}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::vector<int>{1,3}),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},std::initializer_list<int>{1,2,3,4}),
            std::make_tuple(tensor_type{{{1},{2}},{{3},{4}},{{5},{6}},{{7},{8}},{{9},{10}}},gtensor::tensor<std::size_t>{1,3}),
            //split by equal parts
            std::make_tuple(tensor_type{}.reshape(1,0),1),
            std::make_tuple(tensor_type{}.reshape(3,0),3),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},1),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}},5),
            std::make_tuple(tensor_type{{{1},{2}},{{3},{4}},{{5},{6}},{{7},{8}}},2)
        );
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto split_arg = std::get<1>(t);
            auto expected = ten;
            auto result = vstack(vsplit(ten, split_arg));
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_vstack_vsplit")
    {
        //0split_arg,1parts
        auto test_data = std::make_tuple(
            //split by points
            std::make_tuple(std::vector<int>{2},result_type{tensor_type{{1,2},{3,4}},tensor_type{{5,6},{7,8},{9,10}}}),
            std::make_tuple(std::vector<int>{1,3},result_type{tensor_type{{1,2}},tensor_type{{3,4},{5,6}},tensor_type{{7,8},{9,10}}}),
            std::make_tuple(std::initializer_list<int>{1,2,3,4},result_type{tensor_type{{1,2}},tensor_type{{3,4}},tensor_type{{5,6}},tensor_type{{7,8}},tensor_type{{9,10}}}),
            std::make_tuple(gtensor::tensor<std::size_t>{1,3},result_type{tensor_type{{{1},{2}}},tensor_type{{{3},{4}},{{5},{6}}},tensor_type{{{7},{8}},{{9},{10}}}}),
            //split by equal parts
            std::make_tuple(1,result_type{tensor_type{}.reshape(1,0).copy(order{})}),
            std::make_tuple(3,result_type{tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{}),tensor_type{}.reshape(1,0).copy(order{})}),
            std::make_tuple(1,result_type{tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10}}}),
            std::make_tuple(5,result_type{tensor_type{{1,2}},tensor_type{{3,4}},tensor_type{{5,6}},tensor_type{{7,8}},tensor_type{{9,10}}}),
            std::make_tuple(2,result_type{tensor_type{{{1},{2}},{{3},{4}}},tensor_type{{{5},{6}},{{7},{8}}}})
        );
        auto test = [](const auto& t){
            auto split_arg = std::get<0>(t);
            auto parts = std::get<1>(t);
            auto expected = parts;
            auto result = vsplit(vstack(parts), split_arg);
            REQUIRE(expected.size() == result.size());
            auto result_it = result.begin();
            for (auto expected_it = expected.begin(); expected_it!=expected.end(); ++expected_it, ++result_it){
                REQUIRE(*result_it == *expected_it);
            }
        };
        apply_by_element(test, test_data);
    }
}

//tensor of tensors variadic arg should be treated by combine routines as basic_tensor not as container of tensors
TEST_CASE("test_tensor_of_tensors_variadic_arg","[test_manipulation]")
{
    using gtensor::tensor;
    using value_type = tensor<double>;
    using tensor_type = tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    using gtensor::concatenate;
    using gtensor::vstack;
    using gtensor::hstack;
    SECTION("stack")
    {
        using gtensor::stack;
        REQUIRE(std::is_same_v<decltype(stack(0,tensor_type{value_type{1}})),tensor_type>);
        REQUIRE(stack(0,tensor_type{value_type{1}}) == tensor_type{{value_type{1}}});
        REQUIRE(stack(0,tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}}) == tensor_type{{value_type{1},value_type{2,3},value_type{4,5,6}}});
    }
    SECTION("concatenate")
    {
        using gtensor::concatenate;
        REQUIRE(std::is_same_v<decltype(concatenate(0,tensor_type{value_type{1}})),tensor_type>);
        REQUIRE(concatenate(0,tensor_type{value_type{1}}) == tensor_type{value_type{1}});
        REQUIRE(concatenate(0,tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}}) == tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}});
    }
    SECTION("vstack")
    {
        using gtensor::vstack;
        REQUIRE(std::is_same_v<decltype(vstack(tensor_type{value_type{1}})),tensor_type>);
        REQUIRE(vstack(tensor_type{value_type{1}}) == tensor_type{{value_type{1}}});
        REQUIRE(vstack(tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}}) == tensor_type{{value_type{1},value_type{2,3},value_type{4,5,6}}});
    }
    SECTION("hstack")
    {
        using gtensor::hstack;
        REQUIRE(std::is_same_v<decltype(hstack(tensor_type{value_type{1}})),tensor_type>);
        REQUIRE(hstack(tensor_type{value_type{1}}) == tensor_type{value_type{1}});
        REQUIRE(hstack(tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}}) == tensor_type{value_type{1},value_type{2,3},value_type{4,5,6}});
    }
}
