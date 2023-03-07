#include <tuple>
#include <vector>
#include "catch.hpp"
#include "test_config.hpp"
#include "view_factory.hpp"
#include "gtensor.hpp"
#include "helpers_for_testing.hpp"

namespace test_view_factory{

template<typename T>
struct test_tensor : public T{
    test_tensor(const T& base):
        T{base}
    {}
    using T::descriptor;
    using T::engine;
    using T::impl;
};

template<template<typename> typename TestT = test_tensor, typename T>
auto make_test_tensor(T&& t){return TestT<std::decay_t<T>>{t};}

}   //end of namespace test_view_factory


TEMPLATE_TEST_CASE("test_make_view_slice_shape","[test_view_factory]", std::vector<std::int64_t>){
    using index_type = typename TestType::value_type;
    using shape_type = TestType;
    using slice_type = gtensor::slice<index_type>;
    using slices_collection_type = std::vector<slice_type>;
    using gtensor::detail::make_view_slice_shape;
    using test_type = std::tuple<shape_type, slices_collection_type, shape_type>;
    //0parent_shape,1slices_collection,2expected_shape
    auto test_data = GENERATE(
        test_type{shape_type{11},{}, shape_type{11}},
        test_type{shape_type{11},{slice_type{0,11,1}}, shape_type{11}},
        test_type{shape_type{11},{slice_type{0,11,2}}, shape_type{6}},
        test_type{shape_type{11},{slice_type{3,11,1}}, shape_type{8}},
        test_type{shape_type{11},{slice_type{3,9,3}}, shape_type{2}},
        test_type{shape_type{11},{slice_type{3,11,2}}, shape_type{4}},
        test_type{shape_type{11},{slice_type{5,5,1}}, shape_type{0}},
        test_type{shape_type{11},{slice_type{5,5,2}}, shape_type{0}},
        test_type{shape_type{2,4,3},{}, shape_type{2,4,3}},
        test_type{shape_type{2,4,3},{slice_type{0,2},slice_type{0,4}}, shape_type{2,4,3}},
        test_type{shape_type{2,4,3},{slice_type{1},slice_type{2}}, shape_type{1,2,3}},
        test_type{shape_type{2,4,3},{slice_type{0,2},slice_type{0,4},slice_type{0,3}}, shape_type{2,4,3}},
        test_type{shape_type{1,10,10},{slice_type{0,1,1},slice_type{9,-1,-1}}, shape_type{1,10,10}},
        test_type{shape_type{10,1,10},{slice_type{9,3,-2}}, shape_type{3,1,10}}
    );
    auto parent_shape = std::get<0>(test_data);
    auto slices_collection = std::get<1>(test_data);
    auto expected_shape = std::get<2>(test_data);
    REQUIRE(make_view_slice_shape(parent_shape, slices_collection) == expected_shape);
}

TEMPLATE_TEST_CASE("test_make_view_subdim_shape","[test_view_factory]", std::vector<std::int64_t>){
    using shape_type = TestType;
    using gtensor::detail::make_view_subdim_shape;
    using test_type = std::tuple<shape_type, shape_type, shape_type>;
    //0parent_shape,1subs,2expected_shape
    auto test_data = GENERATE(
        test_type{shape_type{11,1},shape_type{}, shape_type{11,1}},
        test_type{shape_type{11,1},shape_type{0}, shape_type{1}},
        test_type{shape_type{1,11},shape_type{0}, shape_type{11}},
        test_type{shape_type{3,4,10,2},shape_type{1,3}, shape_type{10,2}}
    );
    auto parent_shape = std::get<0>(test_data);
    auto subs = std::get<1>(test_data);
    auto expected_shape = std::get<2>(test_data);
    REQUIRE(make_view_subdim_shape(parent_shape, subs) == expected_shape);
}

TEMPLATE_TEST_CASE("test_make_view_reshape_shape","[test_view_factory]", std::vector<std::int64_t>){
    using shape_type = TestType;
    using gtensor::detail::make_view_reshape_shape;
    using test_type = std::tuple<shape_type, shape_type, shape_type>;
    //0parent_shape,1subs,2expected_shape
    auto test_data = GENERATE(
        test_type{shape_type{11,1},shape_type{}, shape_type{11,1}},
        test_type{shape_type{11,1},shape_type{11}, shape_type{11}},
        test_type{shape_type{1,11},shape_type{11,1}, shape_type{11,1}},
        test_type{shape_type{3,4,10,2},shape_type{}, shape_type{3,4,10,2}},
        test_type{shape_type{3,4,10,2},shape_type{20,12}, shape_type{20,12}}
    );
    auto parent_shape = std::get<0>(test_data);
    auto subs = std::get<1>(test_data);
    auto expected_shape = std::get<2>(test_data);
    REQUIRE(make_view_reshape_shape(parent_shape, subs) == expected_shape);
}

TEMPLATE_TEST_CASE("test_make_view_slice_offset","[test_view_factory]", std::vector<std::int64_t>){
    using index_type = typename TestType::value_type;
    using shape_type = TestType;
    using slice_type = gtensor::slice<index_type>;
    using slices_collection_type = std::vector<slice_type>;
    using gtensor::detail::make_view_slice_offset;
    using test_type = std::tuple<shape_type, slices_collection_type, index_type>;
    //0parent_strides,1slices_collection,2expected_offset
    auto test_data = GENERATE(
        test_type{shape_type{1}, {}, 0},
        test_type{shape_type{1}, {slice_type{0,10,2}}, 0},
        test_type{shape_type{1}, {slice_type{4,10,2}}, 4},
        test_type{shape_type{1,5}, {}, 0},
        test_type{shape_type{1,5}, {slice_type{1,5,1}, slice_type{3,5,1}}, 16},
        test_type{shape_type{1,5,20}, {slice_type{2,4,1}, slice_type{1,3,1}}, 7}
    );

    auto parent_strides = std::get<0>(test_data);
    auto slices_collection = std::get<1>(test_data);
    auto expected_offset = std::get<2>(test_data);
    REQUIRE(make_view_slice_offset(parent_strides, slices_collection) == expected_offset);
}

TEMPLATE_TEST_CASE("test_make_view_subdim_offset","[test_view_factory]", std::vector<std::int64_t>){
    using index_type = typename TestType::value_type;
    using shape_type = TestType;
    using gtensor::detail::make_view_subdim_offset;
    using test_type = std::tuple<shape_type, shape_type, index_type>;
    //0parent_strides,1subs,2expected_offset
    auto test_data = GENERATE(
        test_type{shape_type{11,1},shape_type{}, 0},
        test_type{shape_type{11,1},shape_type{0}, 0},
        test_type{shape_type{11,1},shape_type{}, 0},
        test_type{shape_type{11,1},shape_type{4}, 44},
        test_type{shape_type{1,4,20,100},shape_type{}, 0},
        test_type{shape_type{1,4,20,100},shape_type{2,3}, 14}
    );

    auto parent_strides = std::get<0>(test_data);
    auto subs = std::get<1>(test_data);
    auto expected_offset = std::get<2>(test_data);
    REQUIRE(make_view_subdim_offset(parent_strides, subs) == expected_offset);
}

TEMPLATE_TEST_CASE("test_make_view_slice_cstrides","[test_view_factory]", std::vector<std::int64_t>){
    using index_type = typename TestType::value_type;
    using shape_type = TestType;
    using slice_type = gtensor::slice<index_type>;
    using gtensor::detail::make_view_slice_cstrides;
    using slices_collection_type = std::vector<slice_type>;
    using test_type = std::tuple<shape_type, slices_collection_type, shape_type>;
    //0parent_strides,1slices_collection,2expected_cstrides
    auto test_data = GENERATE(
        test_type{shape_type{1},{slice_type{0,10,1}}, shape_type{1}},
        test_type{shape_type{1},{slice_type{2,8,3}}, shape_type{3}},
        test_type{shape_type{1},{slice_type{10,-1,-1}}, shape_type{-1}},
        test_type{shape_type{1,5},{slice_type{0,5,1}}, shape_type{1,5}},
        test_type{shape_type{1,5},{slice_type{0,5,-1}}, shape_type{-1,5}},
        test_type{shape_type{1,5},{slice_type{0,5,2}}, shape_type{2,5}},
        test_type{shape_type{1,5,20},{slice_type{0,10,2}, slice_type{0,5,2},slice_type{4,-1,-1}}, shape_type{2,10,-20}},
        test_type{shape_type{1,5,20},{slice_type{0,5,2},slice_type{3,-1,-1}}, shape_type{2,-5,20}}
    );

    auto parent_strides = std::get<0>(test_data);
    auto slices_collection = std::get<1>(test_data);
    auto expected_cstrides = std::get<2>(test_data);
    REQUIRE(make_view_slice_cstrides(parent_strides, slices_collection) == expected_cstrides);

}

TEMPLATE_TEST_CASE("test_transpose","[test_view_factory]", std::vector<std::int64_t>){
    using shape_type = TestType;
    using gtensor::detail::transpose;
    using test_type = std::tuple<shape_type, shape_type, shape_type>;
    //0source,1indeces,2expected_transposed
    auto test_data = GENERATE(
        test_type{shape_type{3},shape_type{}, shape_type{3}},
        test_type{shape_type{3},shape_type{0}, shape_type{3}},
        test_type{shape_type{3,2},shape_type{0,1}, shape_type{3,2}},
        test_type{shape_type{3,2},shape_type{}, shape_type{2,3}},
        test_type{shape_type{3,2},shape_type{1,0}, shape_type{2,3}},
        test_type{shape_type{4,3,2,2},shape_type{}, shape_type{2,2,3,4}},
        test_type{shape_type{4,3,2,2},shape_type{3,1,0,2}, shape_type{2,3,4,2}}
    );

    auto source = std::get<0>(test_data);
    auto indeces = std::get<1>(test_data);
    auto expected_transposed = std::get<2>(test_data);
    REQUIRE(transpose(source, indeces) == expected_transposed);
}

TEMPLATE_TEST_CASE("test_make_index_mapping_view_shape","[test_view_factory]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using size_type = typename config_type::size_type;
    using gtensor::subscript_exception;
    using gtensor::detail::make_index_mapping_view_shape;
    using gtensor::detail::broadcast_shape;
    using test_type = std::tuple<shape_type,shape_type>;
    //0result shape,1expected shape
    auto test_data = GENERATE(
        test_type{make_index_mapping_view_shape(shape_type{10},broadcast_shape<shape_type>(shape_type{4}),size_type{1}), shape_type{4}},
        test_type{make_index_mapping_view_shape(shape_type{10},broadcast_shape<shape_type>(shape_type{2,2}),size_type{1}), shape_type{2,2}},
        test_type{make_index_mapping_view_shape(shape_type{5,4,3,2},broadcast_shape<shape_type>(shape_type{8},shape_type{8}),size_type{2}), shape_type{8,3,2}},
        test_type{make_index_mapping_view_shape(shape_type{5,4,3,2},broadcast_shape<shape_type>(shape_type{3,4},shape_type{1},shape_type{3,1}),size_type{3}), shape_type{3,4,2}},
        test_type{make_index_mapping_view_shape(shape_type{5,4,3,2},broadcast_shape<shape_type>(shape_type{3,4},shape_type{1},shape_type{3,1}, shape_type{1,4}),size_type{4}), shape_type{3,4}}
    );

    auto result_shape = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    REQUIRE(result_shape == expected_shape);
}

TEMPLATE_TEST_CASE("test_make_index_mapping_view_shape_exception","[test_view_factory]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::subscript_exception;
    using gtensor::detail::make_index_mapping_view_shape;
    using gtensor::detail::broadcast_shape;

    REQUIRE_THROWS_AS(make_index_mapping_view_shape(shape_type{10},broadcast_shape<shape_type>(shape_type{4}, shape_type{4}),size_type{2}), subscript_exception);
    REQUIRE_THROWS_AS(make_index_mapping_view_shape(shape_type{3,4},broadcast_shape<shape_type>(shape_type{4}, shape_type{4}, shape_type{1}),size_type{3}), subscript_exception);
}

TEMPLATE_TEST_CASE("test_fill_index_mapping_view","[test_view_factory]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using config_type = TestType;
    using value_type = int;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using index_tensor_type = gtensor::tensor<index_type, config_type>;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using test_view_factory::test_tensor;
    using gtensor::walker_bidirectional_adapter;
    using test_view_factory::make_test_tensor;
    using gtensor::detail::fill_index_mapping_view;
    using gtensor::detail::make_index_mapping_view_shape;
    using gtensor::detail::broadcast_shape;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_strides_div;
    using gtensor::detail::broadcast_shape;
    using helpers_for_testing::apply_by_element;

    //0pshape,1subs,2expected_map
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, std::make_tuple(index_tensor_type{0,1,2}),std::vector<value_type>{}),
        std::make_tuple(tensor_type{0}, std::make_tuple(index_tensor_type{0,0,0}),std::vector<value_type>{0,0,0}),
        std::make_tuple(tensor_type{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{0,2,0,1,0}),std::vector<value_type>{0,2,0,1,0}),
        std::make_tuple(tensor_type{0,1,2,3,4}, std::make_tuple(index_tensor_type{4,3,2,1,0}),std::vector<value_type>{4,3,2,1,0}),
        std::make_tuple(tensor_type{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{{0,2,4,6,8},{1,3,5,7,9}}),std::vector<value_type>{0,2,4,6,8,1,3,5,7,9}),
        std::make_tuple(tensor_type{0,1,2,3,4}, std::make_tuple(index_tensor_type{{0,2,4,0,2},{1,3,1,3,1}}),std::vector<value_type>{0,2,4,0,2,1,3,1,3,1}),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{1,3,0,1}),std::vector<value_type>{3,4,5,9,10,11,0,1,2,3,4,5}),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1},{3}}),std::vector<value_type>{3,4,5,9,10,11}),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{0,1,2}),std::vector<value_type>{0,4,8}),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{0,1}}, index_tensor_type{{0,1},{1,2}}),std::vector<value_type>{3,7,1,5}),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{1}),std::vector<value_type>{1,4,7}),
        std::make_tuple(
            tensor_type{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1}, index_tensor_type{1}),
            std::vector<value_type>{3,4,5,15,16,17}
        ),
        std::make_tuple(
            tensor_type{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1}, index_tensor_type{3}, index_tensor_type{2}),
            std::vector<value_type>{11,23}
        )
    );

    auto test = [](const auto& t){
        auto parent = make_test_tensor(std::get<0>(t));
        auto subs = std::get<1>(t);
        auto expected_elements = std::get<2>(t);
        auto broadcast_shape_maker = [](const auto&...subs){
            return broadcast_shape<shape_type>(subs.shape()...);
        };
        auto subs_shape = std::apply(broadcast_shape_maker, subs);
        auto subs_size = make_size(subs_shape);
        auto view_shape = make_index_mapping_view_shape(parent.shape(), subs_shape, std::tuple_size_v<std::decay_t<decltype(subs)>>);
        auto expected_view_size = make_size(view_shape);
        std::vector<value_type> result_elements(expected_view_size);
        auto elements_filler = [&result_elements,&parent,&subs_size,&subs_shape](const auto&...subs){
            return fill_index_mapping_view(
                result_elements.begin(),
                parent.begin(),
                parent.shape(),
                parent.descriptor().strides(),
                subs_size,
                walker_bidirectional_adapter<config_type, decltype(make_test_tensor(subs).engine().create_walker())>{subs_shape, make_test_tensor(subs).engine().create_walker()}...
            );
        };
        auto result_view_size = std::apply(elements_filler, subs);
        REQUIRE(result_view_size == expected_view_size);
        REQUIRE(result_elements == expected_elements);
    };

    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_fill_index_mapping_view_exception","[test_view_factory]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using config_type = TestType;
    using value_type = int;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using index_tensor_type = gtensor::tensor<index_type, config_type>;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using test_view_factory::test_tensor;
    using gtensor::walker_bidirectional_adapter;
    using gtensor::subscript_exception;
    using test_view_factory::make_test_tensor;
    using gtensor::detail::fill_index_mapping_view;
    using gtensor::detail::make_index_mapping_view_shape;
    using gtensor::detail::broadcast_shape;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_strides_div;
    using gtensor::detail::broadcast_shape;
    using helpers_for_testing::apply_by_element;

    //0pshape,1subs
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{0}, std::make_tuple(index_tensor_type{0,0,3})),
        std::make_tuple(tensor_type{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{0,20,0,1,0})),
        std::make_tuple(tensor_type{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{{0,2,4,16,8},{1,3,5,7,9}})),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{1,3,0,4})),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{11},{3}})),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{5,1}}, index_tensor_type{{0,1},{1,2}})),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{0,1}}, index_tensor_type{{0,1},{1,4}})),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,5}, index_tensor_type{1})),
        std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{3})),
        std::make_tuple(
            tensor_type{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{4}, index_tensor_type{2})
        )
    );

    auto test = [](const auto& t){
        auto parent = make_test_tensor(std::get<0>(t));
        auto subs = std::get<1>(t);
        auto broadcast_shape_maker = [](const auto&...subs){
            return broadcast_shape<shape_type>(subs.shape()...);
        };
        auto subs_shape = std::apply(broadcast_shape_maker, subs);
        auto subs_size = make_size(subs_shape);
        auto view_shape = make_index_mapping_view_shape(parent.shape(), subs_shape, std::tuple_size_v<std::decay_t<decltype(subs)>>);
        auto expected_view_size = make_size(view_shape);
        std::vector<value_type> result_elements(expected_view_size);
        auto elements_filler = [&result_elements,&parent,&subs_size,&subs_shape](const auto&...subs){
            return fill_index_mapping_view(
                result_elements.begin(),
                parent.begin(),
                parent.shape(),
                parent.descriptor().strides(),
                subs_size,
                walker_bidirectional_adapter<config_type, decltype(make_test_tensor(subs).engine().create_walker())>{subs_shape, make_test_tensor(subs).engine().create_walker()}...
            );
        };
        REQUIRE_THROWS_AS(std::apply(elements_filler, subs), subscript_exception);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_check_bool_mapping_view_subs","[test_view_factory]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using config_type = TestType;
    using shape_type = typename config_type::shape_type;
    using gtensor::subscript_exception;
    using gtensor::detail::check_bool_mapping_view_subs;

    REQUIRE_NOTHROW(check_bool_mapping_view_subs(shape_type{1}, shape_type{1}));
    REQUIRE_NOTHROW(check_bool_mapping_view_subs(shape_type{10}, shape_type{10}));
    REQUIRE_NOTHROW(check_bool_mapping_view_subs(shape_type{10}, shape_type{5}));
    REQUIRE_NOTHROW(check_bool_mapping_view_subs(shape_type{4,3,2}, shape_type{4,3,2}));
    REQUIRE_NOTHROW(check_bool_mapping_view_subs(shape_type{4,3,2}, shape_type{2,2,2}));
    REQUIRE_NOTHROW(check_bool_mapping_view_subs(shape_type{4,3,2}, shape_type{2,1}));
    REQUIRE_NOTHROW(check_bool_mapping_view_subs(shape_type{4,3,2}, shape_type{2}));

    //subs dim > parent dim
    REQUIRE_THROWS_AS(check_bool_mapping_view_subs(shape_type{10}, shape_type{10,10}) ,subscript_exception);
    REQUIRE_THROWS_AS(check_bool_mapping_view_subs(shape_type{3,2,4}, shape_type{2,2,2,2}) ,subscript_exception);
    //subs direction size > parent direction size
    REQUIRE_THROWS_AS(check_bool_mapping_view_subs(shape_type{10}, shape_type{20}) ,subscript_exception);
    REQUIRE_THROWS_AS(check_bool_mapping_view_subs(shape_type{3,2,4}, shape_type{3,3}) ,subscript_exception);
}

TEMPLATE_TEST_CASE("test_bool_mapping_view_block_size","[test_view_factory]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using config_type = TestType;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using index_tensor_type = gtensor::tensor<bool, config_type>;
    using gtensor::detail::bool_mapping_view_block_size;
    using test_type = std::tuple<shape_type,index_tensor_type,index_type>;
    //0pshape,1subs,2expected_block_size
    auto test_data = GENERATE(
        test_type{shape_type{},index_tensor_type{},index_type{0}},
        test_type{shape_type{},index_tensor_type{true,false},index_type{0}},
        test_type{shape_type{5},index_tensor_type{},index_type{0}},
        test_type{shape_type{5},index_tensor_type{true,true,false,false,true},index_type{1}},
        test_type{shape_type{5},index_tensor_type{true,true,false},index_type{1}},
        test_type{shape_type{5},index_tensor_type{true,true,false},index_type{1}},
        test_type{shape_type{5,4},index_tensor_type{true,true,false},index_type{4}},
        test_type{shape_type{5,10},index_tensor_type{true,true,false},index_type{10}}
    );

    auto pshape = std::get<0>(test_data);
    auto subs = std::get<1>(test_data);
    auto expected_block_size = std::get<2>(test_data);
    auto result_block_size = bool_mapping_view_block_size(pshape, subs);
    REQUIRE(result_block_size == expected_block_size);
}

TEMPLATE_TEST_CASE("test_make_bool_mapping_view_shape","[test_view_factory]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using config_type = TestType;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_bool_mapping_view_shape;
    using test_type = std::tuple<shape_type,shape_type,index_type,index_type,shape_type>;
    //0pshape,1subs_shape,2block_size,3filled_map_size,4expected_shape
    auto test_data = GENERATE(
        test_type{shape_type{10},shape_type{10},index_type{1},index_type{10},shape_type{10}},
        test_type{shape_type{10},shape_type{10},index_type{1},index_type{3},shape_type{3}},
        test_type{shape_type{10},shape_type{10},index_type{1},index_type{0},shape_type{}},
        test_type{shape_type{10},shape_type{5},index_type{1},index_type{5},shape_type{5}},
        test_type{shape_type{10},shape_type{5},index_type{1},index_type{2},shape_type{2}},
        test_type{shape_type{10},shape_type{5},index_type{1},index_type{0},shape_type{}},
        test_type{shape_type{3,4,5},shape_type{3,4,5},index_type{1},index_type{60},shape_type{60}},
        test_type{shape_type{3,4,5},shape_type{3,4,5},index_type{1},index_type{10},shape_type{10}},
        test_type{shape_type{3,4,5},shape_type{3,4,5},index_type{1},index_type{0},shape_type{}},
        test_type{shape_type{3,4,5},shape_type{3,2,1},index_type{1},index_type{6},shape_type{6}},
        test_type{shape_type{3,4,5},shape_type{3,2,1},index_type{1},index_type{2},shape_type{2}},
        test_type{shape_type{3,4,5},shape_type{3,4},index_type{5},index_type{60},shape_type{12,5}},
        test_type{shape_type{3,4,5},shape_type{3,4},index_type{5},index_type{50},shape_type{10,5}},
        test_type{shape_type{3,4,5},shape_type{3,4},index_type{5},index_type{0},shape_type{}},
        test_type{shape_type{3,4,5},shape_type{3,4},index_type{5},index_type{0},shape_type{}},
        test_type{shape_type{3,4,5},shape_type{2,2},index_type{5},index_type{20},shape_type{4,5}},
        test_type{shape_type{3,4,5},shape_type{2,2},index_type{5},index_type{10},shape_type{2,5}},
        test_type{shape_type{3,4,5},shape_type{2,2},index_type{5},index_type{0},shape_type{}},
        test_type{shape_type{3,4,5},shape_type{3},index_type{20},index_type{60},shape_type{3,4,5}},
        test_type{shape_type{3,4,5},shape_type{3},index_type{20},index_type{20},shape_type{1,4,5}},
        test_type{shape_type{3,4,5},shape_type{1},index_type{20},index_type{20},shape_type{1,4,5}},
        test_type{shape_type{3,4,5},shape_type{1},index_type{20},index_type{0},shape_type{}}
    );

    auto pshape = std::get<0>(test_data);
    auto subs_shape = std::get<1>(test_data);
    auto block_size = std::get<2>(test_data);
    auto filled_map_size = std::get<3>(test_data);
    auto expected_shape = std::get<4>(test_data);
    auto result_shape = make_bool_mapping_view_shape(pshape,subs_shape,block_size,filled_map_size);
    REQUIRE(result_shape == expected_shape);
}

TEMPLATE_TEST_CASE("test_fill_bool_mapping_view","[test_view_factory]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using config_type = TestType;
    using value_type = float;
    using index_tensor_type = gtensor::tensor<bool, config_type>;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using gtensor::walker_bidirectional_adapter;
    using test_view_factory::make_test_tensor;
    using gtensor::detail::fill_bool_mapping_view;
    using gtensor::detail::bool_mapping_view_block_size;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;

    using test_type = std::tuple<tensor_type,index_tensor_type,std::vector<value_type>>;
    //0parent_tensor,1subs,2expected_map
    auto test_data = GENERATE(
        test_type{tensor_type{0}, index_tensor_type{}, std::vector<value_type>{}},
        test_type{tensor_type{0}, index_tensor_type{true}, std::vector<value_type>{0}},
        test_type{tensor_type{0}, index_tensor_type{false}, std::vector<value_type>{}},
        test_type{tensor_type{0,1,2,3,4,5}, index_tensor_type{false,false,false,false,false}, std::vector<value_type>{}},
        test_type{tensor_type{0,1,2,3,4,5}, index_tensor_type{false,true,false,true,false}, std::vector<value_type>{1,3}},
        test_type{tensor_type{0,1,2,3,4,5,6,7,8,9}, index_tensor_type{true,false,true,false,false}, std::vector<value_type>{0,2}},
        test_type{tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{true,false,true,false}, std::vector<value_type>{0,1,2,6,7,8}},
        test_type{tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{{false,true,true}}, std::vector<value_type>{1,2}},
        test_type{tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{{false},{true},{true}}, std::vector<value_type>{3,6}},
        test_type{tensor_type{{0,1,2},{3,4,5},{6,7,8}}, index_tensor_type{{true,true,true},{true,true,true},{true,true,true}}, std::vector<value_type>{0,1,2,3,4,5,6,7,8}},
        test_type{tensor_type{{0,1,2},{3,4,5},{6,7,8}}, index_tensor_type{{false,false,false},{false,false,false},{false,false,false}}, std::vector<value_type>{}},
        test_type{tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{{false,true},{true,false}}, std::vector<value_type>{1,3}},
        test_type{tensor_type{{0,1,2,3},{4,5,6,7},{8,9,10,11}}, index_tensor_type{{false,false,true},{false,false,true},{false,true,false}}, std::vector<value_type>{2,6,9}},
        test_type{tensor_type{{0,1,2,3},{4,5,6,7},{8,9,10,11}}, index_tensor_type{{false,false,true,false},{false,false,true,true},{false,true,false,true}}, std::vector<value_type>{2,6,7,9,11}},
        test_type{
            tensor_type{{{0,1},{2,3},{4,5},{6,7}},{{8,9},{10,11},{12,13},{14,15}},{{16,17},{18,19},{20,21},{22,23}}},
            index_tensor_type{{false,false,true,false},{false,false,false,true},{false,true,false,true}},
            std::vector<value_type>{4,5,14,15,18,19,22,23}
        },
        test_type{
            tensor_type{{{0,1},{2,3},{4,5},{6,7}},{{8,9},{10,11},{12,13},{14,15}},{{16,17},{18,19},{20,21},{22,23}}},
            index_tensor_type{{{true,false},{false,true},{false,false}}},
            std::vector<value_type>{0,3}
        }
    );

    auto parent = make_test_tensor(std::get<0>(test_data));
    auto subs = make_test_tensor(std::get<1>(test_data));
    auto expected_elements = std::get<2>(test_data);
    auto expected_size = expected_elements.size();
    using walker_adapter_type = walker_bidirectional_adapter<config_type,decltype(subs.engine().create_walker())>;
    auto block_size = bool_mapping_view_block_size(parent.shape(),subs);
    auto subs_size = subs.size();
    std::vector<value_type> result_elements(parent.size());
    auto result_size = fill_bool_mapping_view(
        result_elements.begin(),
        parent.begin(),
        parent.descriptor().strides(),
        block_size,
        subs_size,
        walker_adapter_type{subs.shape(), subs.engine().create_walker()}
    );
    result_elements.resize(result_size);
    REQUIRE(result_size == expected_size);
    REQUIRE(result_elements == expected_elements);
}
