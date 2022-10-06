#include <list>
#include <iostream>
#include "catch.hpp"
#include "descriptor.hpp"
#include "test_config.hpp"



TEMPLATE_PRODUCT_TEST_CASE("test_make_strides","[test_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
    using shape_type = TestType;
    using test_data_type = typename std::tuple<shape_type,shape_type>;
    using gtensor::detail::make_strides;
    //shape, expected strides
    auto test_data = GENERATE(
        test_data_type(),
        test_data_type({1},{1}),
        test_data_type({5},{1}),
        test_data_type({1,1},{1,1}),
        test_data_type({5,1},{1,1}),
        test_data_type({1,5},{5,1}),
        test_data_type({2,3,4},{12,4,1})
    );

    shape_type shape{std::get<0>(test_data)};
    shape_type strides_expected{std::get<1>(test_data)};
    auto strides_result = make_strides(shape);
    REQUIRE(strides_result == strides_expected);
}

TEMPLATE_PRODUCT_TEST_CASE("test_make_reset_strides","[test_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
    using shape_type = TestType;
    using test_data_type = typename std::tuple<shape_type,shape_type,shape_type>;
    using gtensor::detail::make_reset_strides;
    //0shape,1strides,2expected reset strides
    auto test_data = GENERATE(
        test_data_type(),
        test_data_type({1},{1},{0}),
        test_data_type({5},{1},{4}),
        test_data_type({1,1},{1,1},{0,0}),
        test_data_type({5,1},{1,1},{4,0}),
        test_data_type({1,5},{5,1},{0,4}),
        test_data_type({2,3,4},{12,4,1},{12,8,3})
    );

    shape_type shape{std::get<0>(test_data)};
    shape_type strides{std::get<1>(test_data)};
    shape_type reset_strides_expected{std::get<2>(test_data)};
    auto reset_strides_result = make_reset_strides(shape,strides);
    REQUIRE(reset_strides_result == reset_strides_expected);
}

TEMPLATE_PRODUCT_TEST_CASE("test_make_size_using_strides","[test_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
    using shape_type = TestType;
    using index_type = typename TestType::value_type;
    using test_data_type = typename std::tuple<shape_type,shape_type, index_type>;
    using gtensor::detail::make_size;
    //shape,stride,expected size
    auto test_data = GENERATE(
        test_data_type(shape_type{}, shape_type{}, 0),
        test_data_type({1}, {1},1),
        test_data_type({5}, {1}, 5),
        test_data_type({1,1}, {1,1}, 1),
        test_data_type({5,1}, {1,1}, 5),
        test_data_type({1,5}, {5,1}, 5),
        test_data_type({2,3,4}, {12,4,1}, 24)
    );

    shape_type shape{std::get<0>(test_data)};
    shape_type strides{std::get<1>(test_data)};
    index_type size_expected{std::get<2>(test_data)};
    index_type size_result{make_size(shape,strides)};
    REQUIRE(size_result == size_expected);
}

TEST_CASE("test_make_shape_of_type","[test_descriptor]"){
    using gtensor::detail::make_shape_of_type;
    using shape_type = std::vector<int>;
    auto s = shape_type{1,2,3};
    auto l = std::list<int>{1,2,3};
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(shape_type{1,2,3})), shape_type&&>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(s)), shape_type&>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(std::list{1,2,3})), shape_type>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>(l)), shape_type>);
    REQUIRE(std::is_same_v<decltype(make_shape_of_type<shape_type>({1,2,3})), shape_type>);

    REQUIRE(make_shape_of_type<shape_type>(shape_type{1,2,3}) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(s) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(std::list{1,2,3}) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>(l) == shape_type{1,2,3});
    REQUIRE(make_shape_of_type<shape_type>({1,2,3}) == shape_type{1,2,3});
}

TEST_CASE("test_vmax","test_descriptor"){
    using gtensor::detail::vmax;
    using index_type = std::int64_t;
    using test_type = std::tuple<index_type,index_type>;
    auto test_data = GENERATE(
        test_type{vmax(index_type(1)),index_type(1)},
        test_type{vmax(index_type(0), index_type(1)),index_type(1)},
        test_type{vmax(index_type(1), index_type(0)),index_type(1)},
        test_type{vmax(index_type(1), index_type(1)),index_type(1)},
        test_type{vmax(index_type(1), index_type(1),index_type(2),index_type(0)),index_type(2)},
        test_type{vmax(index_type(3), index_type(1),index_type(2),index_type(0)),index_type(3)},
        test_type{vmax(index_type(3), index_type(1),index_type(2),index_type(3)),index_type(3)},
        test_type{vmax(index_type(3), index_type(1),index_type(2),index_type(4)),index_type(4)},
        test_type{vmax(index_type(3), index_type(5),index_type(5),index_type(4)),index_type(5)},
        test_type{vmax(index_type(3), index_type(3),index_type(3),index_type(3)),index_type(3)}
    );
    auto result = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    REQUIRE(result == expected);
}
TEST_CASE("test_vmin","test_descriptor"){
    using gtensor::detail::vmin;
    using index_type = std::int64_t;
    using test_type = std::tuple<index_type,index_type>;
    auto test_data = GENERATE(
        test_type{vmin(index_type(1)),index_type(1)},
        test_type{vmin(index_type(0), index_type(1)),index_type(0)},
        test_type{vmin(index_type(1), index_type(0)),index_type(0)},
        test_type{vmin(index_type(1), index_type(1)),index_type(1)},
        test_type{vmin(index_type(1), index_type(1),index_type(2),index_type(0)),index_type(0)},
        test_type{vmin(index_type(3), index_type(1),index_type(2),index_type(0)),index_type(0)},
        test_type{vmin(index_type(3), index_type(1),index_type(2),index_type(3)),index_type(1)},
        test_type{vmin(index_type(3), index_type(1),index_type(2),index_type(4)),index_type(1)},
        test_type{vmin(index_type(3), index_type(5),index_type(5),index_type(4)),index_type(3)},
        test_type{vmin(index_type(3), index_type(3),index_type(3),index_type(3)),index_type(3)}
    );
    auto result = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    REQUIRE(result == expected);
}

TEMPLATE_PRODUCT_TEST_CASE("test_broadcast_shape","[test_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
    using shape_type = TestType;
    using test_type = std::tuple<shape_type, shape_type, shape_type>;
    using gtensor::detail::broadcast;
    //shape1,shape2, expected broadcast shape
    auto test_data = GENERATE(
        test_type(shape_type{1}, shape_type{1}, shape_type{1}),
        test_type(shape_type{5}, shape_type{5}, shape_type{5}),
        test_type(shape_type{1,1}, shape_type{1}, shape_type{1,1}),
        test_type(shape_type{1}, shape_type{1,1}, shape_type{1,1}),
        test_type(shape_type{1,1}, shape_type{1,1}, shape_type{1,1}),
        test_type(shape_type{1,5}, shape_type{5,1}, shape_type{5,5}),
        test_type(shape_type{2,3,4}, shape_type{3,4}, shape_type{2,3,4}),
        test_type(shape_type{2,1,4}, shape_type{3,1}, shape_type{2,3,4}),
        test_type(shape_type{2,4}, shape_type{3,1,4}, shape_type{3,2,4})
    );

    auto shape1 = std::get<0>(test_data);
    auto shape2 = std::get<1>(test_data);
    auto expected_broadcast_shape = std::get<2>(test_data);
    REQUIRE(broadcast(shape1, shape2) == expected_broadcast_shape);
}

TEMPLATE_PRODUCT_TEST_CASE("test_broadcast_shape_exception","[test_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
    using shape_type = TestType;
    using test_type = std::tuple<shape_type, shape_type>;
    using gtensor::detail::broadcast;
    using gtensor::broadcast_exception;
    //shape1,shape2
    auto test_data = GENERATE(
        test_type(shape_type{}, shape_type{}),
        test_type(shape_type{1}, shape_type{}),
        test_type(shape_type{}, shape_type{1}),
        test_type(shape_type{3}, shape_type{2}),
        test_type(shape_type{2}, shape_type{3}),
        test_type(shape_type{1,2}, shape_type{3}),
        test_type(shape_type{1,2}, shape_type{4,3}),
        test_type(shape_type{3,2}, shape_type{4,2}),
        test_type(shape_type{5,1,2}, shape_type{4,4,2})
    );

    auto shape1 = std::get<0>(test_data);
    auto shape2 = std::get<1>(test_data);
    REQUIRE_THROWS_AS(broadcast(shape1, shape2), broadcast_exception);
}

TEMPLATE_PRODUCT_TEST_CASE("test_variadic_broadcast_shape","[test_descriptor]", (std::vector,trivial_type_vector::uvector),(std::int64_t)){
    using shape_type = TestType;
    using result_shape_type = std::vector<std::int64_t>;
    using test_type = std::tuple<result_shape_type, result_shape_type>;
    using gtensor::detail::broadcast_shape;
    //0result broadcast shape,1expected broadcast shape
    auto test_data = GENERATE(
        test_type(broadcast_shape<result_shape_type>(shape_type{1}), result_shape_type{1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,2,3}), result_shape_type{1,2,3}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1}, shape_type{1}), result_shape_type{1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1}, shape_type{1}, shape_type{1}), result_shape_type{1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{5}, shape_type{5}), result_shape_type{5}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,1}, shape_type{1}), result_shape_type{1,1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,1}, shape_type{1}, shape_type{1,1,1}, shape_type{1,1}), result_shape_type{1,1,1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,1}, shape_type{1,1}, shape_type{1,1}), result_shape_type{1,1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,5}, shape_type{5,1}), result_shape_type{5,5}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,5}, shape_type{5,1}, shape_type{1,5}, shape_type{1,1}), result_shape_type{5,5}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,3,4}, shape_type{3,4}), result_shape_type{2,3,4}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,3,4}, shape_type{3,4}, shape_type{1,1,1,1}, shape_type{5,1,1,1}), result_shape_type{5,2,3,4}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,1,4}, shape_type{3,1}, shape_type{3,4}), result_shape_type{2,3,4}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,1}, shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4})
    );

    auto result_broadcast_shape = std::get<0>(test_data);
    auto expected_broadcast_shape = std::get<1>(test_data);
    REQUIRE(result_broadcast_shape.size() == expected_broadcast_shape.size());
    REQUIRE(std::equal(expected_broadcast_shape.begin(),expected_broadcast_shape.end(),result_broadcast_shape.begin()));
}

TEMPLATE_PRODUCT_TEST_CASE("test_variadic_broadcast_shape_exception","[test_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
    using shape_type = TestType;
    using gtensor::detail::broadcast_shape;
    using gtensor::broadcast_exception;

    SECTION("1-shape"){
        //shape1,shape2
        REQUIRE_THROWS_AS(broadcast_shape<shape_type>(shape_type{}), broadcast_exception);
    }
    SECTION("2-shapes"){
        //shape1,shape2
        using test_type = std::tuple<shape_type, shape_type>;
        auto test_data = GENERATE(
            test_type(shape_type{}, shape_type{}),
            test_type(shape_type{1}, shape_type{}),
            test_type(shape_type{}, shape_type{1}),
            test_type(shape_type{3}, shape_type{2}),
            test_type(shape_type{2}, shape_type{3}),
            test_type(shape_type{1,2}, shape_type{3}),
            test_type(shape_type{1,2}, shape_type{4,3}),
            test_type(shape_type{3,2}, shape_type{4,2}),
            test_type(shape_type{5,1,2}, shape_type{4,4,2})
        );
        auto shape1 = std::get<0>(test_data);
        auto shape2 = std::get<1>(test_data);
        REQUIRE_THROWS_AS(broadcast_shape<shape_type>(shape1, shape2), broadcast_exception);
    }
    SECTION("3-shapes"){
        //shape1,shape2,shape3
        using test_type = std::tuple<shape_type, shape_type, shape_type>;
        auto test_data = GENERATE(
            test_type(shape_type{}, shape_type{1}, shape_type{}),
            test_type(shape_type{3}, shape_type{3}, shape_type{2}),
            test_type(shape_type{1,2}, shape_type{3}, shape_type{1}),
            test_type(shape_type{1,2}, shape_type{1,1}, shape_type{4,4}),
            test_type(shape_type{5,1,2}, shape_type{2,2}, shape_type{4,4,2})
        );
        auto shape1 = std::get<0>(test_data);
        auto shape2 = std::get<1>(test_data);
        auto shape3 = std::get<2>(test_data);
        REQUIRE_THROWS_AS(broadcast_shape<shape_type>(shape1, shape2, shape3), broadcast_exception);
    }
}

TEST_CASE("test_basic_descriptor","[test_descriptor]"){
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::basic_descriptor<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, shape_type, index_type, index_type>;
    //0descriptor,1expected shape,2expected strides,3expected reset_strides,4expected size,5expected dim
    auto test_data = GENERATE(
        test_type(descriptor_type(),shape_type{},shape_type{},shape_type{},0,0),
        test_type(descriptor_type{shape_type{}},shape_type{},shape_type{},shape_type{},0,0),
        test_type(descriptor_type{shape_type{1}},shape_type{1},shape_type{1},shape_type{0},1,1),
        test_type(descriptor_type{shape_type{5}},shape_type{5},shape_type{1},shape_type{4},5,1),
        test_type(descriptor_type{shape_type{1,1}},shape_type{1,1},shape_type{1,1},shape_type{0,0},1,2),
        test_type(descriptor_type{shape_type{1,5}},shape_type{1,5},shape_type{5,1},shape_type{0,4},5,2),
        test_type(descriptor_type{shape_type{5,1}},shape_type{5,1},shape_type{1,1},shape_type{4,0},5,2),
        test_type(descriptor_type{shape_type{5,4,3}},shape_type{5,4,3},shape_type{12,3,1},shape_type{48,9,2},60,3)
    );
    auto descriptor = std::get<0>(test_data);
    auto dd = descriptor;
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_reset_strides = std::get<3>(test_data);
    auto expected_size = std::get<4>(test_data);
    auto expected_dim = std::get<5>(test_data);
    auto expected_offset = index_type{0};
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_strides);
    REQUIRE(descriptor.reset_strides() == expected_reset_strides);
    REQUIRE(descriptor.reset_cstrides() == expected_reset_strides);
    REQUIRE(descriptor.offset() == expected_offset);
}

TEST_CASE("test_basic_descriptor_convert", "[test_descriptor]"){
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::basic_descriptor<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type>;
    //descriptor{shape,offset}
    //0descriptor,1multi_idx,2flat_idx,3converted_multi_idx,4converted_flat_idx
    auto test_data = GENERATE(
        test_type{descriptor_type{shape_type{15}},shape_type{0},0,0,0},
        test_type{descriptor_type{shape_type{1,15}},shape_type{0,7},7,7,7},
        test_type{descriptor_type{shape_type{3,3,5}},shape_type{1,0,4},22,19,22}     //(3,3,5) (15,5,1)  22->(1,1,2)->22
    );
    auto descriptor = std::get<0>(test_data);
    auto multi_idx = std::get<1>(test_data);
    auto flat_idx = std::get<2>(test_data);
    auto expected_convert_multi_idx = std::get<3>(test_data);
    auto expected_convert_flat_idx = std::get<4>(test_data);

    REQUIRE(descriptor.convert(multi_idx) == expected_convert_multi_idx);
    REQUIRE(descriptor.convert(flat_idx) == expected_convert_flat_idx);
}

TEST_CASE("test_mapping_descriptor_convert", "[test_descriptor]"){
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::mapping_descriptor<config_type>;
    using map_type = typename descriptor_type::map_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type>;
    //0descriptor,1multi_idx,2flat_idx,3converted_multi_idx,4converted_flat_idx
    auto test_data = GENERATE(
        test_type{descriptor_type{shape_type{5}, map_type{5,6,7,8,9}},shape_type{0},0,5,5},
        test_type{descriptor_type{shape_type{5}, map_type{5,6,7,8,9}},shape_type{3},3,8,8},
        test_type{descriptor_type{shape_type{1,5}, map_type{5,6,7,8,9}},shape_type{0,3},3,8,8},
        test_type{descriptor_type{shape_type{5,1}, map_type{5,6,7,8,9}},shape_type{3,0},3,8,8},
        test_type{descriptor_type{shape_type{2,3,2},map_type{0,1,2,3,4,5,12,13,14,15,16,17}},shape_type{1,1,1},10,15,16},
        test_type{descriptor_type{shape_type{4,2},map_type{0,1,4,5,16,17,18,19}},shape_type{2,1},3,17,5}
    );
    auto descriptor = std::get<0>(test_data);
    auto multi_idx = std::get<1>(test_data);
    auto flat_idx = std::get<2>(test_data);
    auto expected_convert_multi_idx = std::get<3>(test_data);
    auto expected_convert_flat_idx = std::get<4>(test_data);

    REQUIRE(descriptor.convert(multi_idx) == expected_convert_multi_idx);
    REQUIRE(descriptor.convert(flat_idx) == expected_convert_flat_idx);
}

TEST_CASE("test_descriptor_with_offset","[test_descriptor]"){
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::descriptor_with_offset<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, index_type, index_type, index_type>;
    //0descriptor,1expected shape,2expected strides,3expected size,4expected dim,5expected offset
    auto test_data = GENERATE(
        test_type(descriptor_type(),shape_type{},shape_type{},0,0,0),
        test_type({shape_type{},0},shape_type{},shape_type{},0,0,0),
        test_type({shape_type{1},0},shape_type{1},shape_type{1},1,1,0),
        test_type({shape_type{5},10},shape_type{5},shape_type{1},5,1,10),
        test_type({shape_type{1,1},0},shape_type{1,1},shape_type{1,1},1,2,0),
        test_type({shape_type{1,5},0},shape_type{1,5},shape_type{5,1},5,2,0),
        test_type({shape_type{5,1},4},shape_type{5,1},shape_type{1,1},5,2,4),
        test_type({shape_type{5,4,3},100},shape_type{5,4,3},shape_type{12,3,1},60,3,100)
    );
    auto descriptor = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_size = std::get<3>(test_data);
    auto expected_dim = std::get<4>(test_data);
    auto expected_offset = std::get<5>(test_data);
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_strides);
    REQUIRE(descriptor.offset() == expected_offset);
}

TEST_CASE("test_descriptor_with_offset_convert", "[test_descriptor]"){
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::descriptor_with_offset<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type>;
    //descriptor{shape,offset}
    //0descriptor,1multi_idx,2flat_idx,3converted_multi_idx,4converted_flat_idx
    auto test_data = GENERATE(
        test_type{descriptor_type{shape_type{15},0},shape_type{0},0,0,0},
        test_type{descriptor_type{shape_type{1,15},0},shape_type{0,7},7,7,7},
        test_type{descriptor_type{shape_type{3,3,5},5},shape_type{1,0,4},22,24,27}     //(3,3,5) (15,5,1)  22->(1,1,2)->22
    );
    auto descriptor = std::get<0>(test_data);
    auto multi_idx = std::get<1>(test_data);
    auto flat_idx = std::get<2>(test_data);
    auto expected_convert_multi_idx = std::get<3>(test_data);
    auto expected_convert_flat_idx = std::get<4>(test_data);

    REQUIRE(descriptor.convert(multi_idx) == expected_convert_multi_idx);
    REQUIRE(descriptor.convert(flat_idx) == expected_convert_flat_idx);
}

TEMPLATE_TEST_CASE("test_descriptor_with_libdivide","[test_descriptor]", gtensor::config::mode_div_libdivide, gtensor::config::mode_div_native){
    using config_type = test_config::config_div_mode_selector<TestType>::config_type;
    using descriptor_type = gtensor::descriptor_with_libdivide<config_type>;
    using descriptor_base_type = gtensor::descriptor_base<config_type>;
    using strides_libdivide_type = gtensor::detail::libdiv_strides_traits<config_type>::type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_dividers;

    using test_type = std::tuple<descriptor_type, shape_type, shape_type, strides_libdivide_type, index_type, index_type>;
    //0descriptor,1expected shape,2expected strides,3expected_strides_libdivide, 4expected size,5expected dim
    auto test_data = GENERATE(
        test_type(descriptor_type(),shape_type{},shape_type{}, make_dividers<config_type>(shape_type{}),0,0),
        test_type(descriptor_type{shape_type{}},shape_type{},shape_type{},make_dividers<config_type>(shape_type{}),0,0),
        test_type(descriptor_type{shape_type{1}},shape_type{1},shape_type{1},make_dividers<config_type>(shape_type{1}),1,1),
        test_type(descriptor_type{shape_type{5}},shape_type{5},shape_type{1},make_dividers<config_type>(shape_type{1}),5,1),
        test_type(descriptor_type{shape_type{1,1}},shape_type{1,1},shape_type{1,1},make_dividers<config_type>(shape_type{1,1}),1,2),
        test_type(descriptor_type{shape_type{1,5}},shape_type{1,5},shape_type{5,1},make_dividers<config_type>(shape_type{5,1}),5,2),
        test_type(descriptor_type{shape_type{5,1}},shape_type{5,1},shape_type{1,1},make_dividers<config_type>(shape_type{1,1}),5,2),
        test_type(descriptor_type{shape_type{5,4,3}},shape_type{5,4,3},shape_type{12,3,1},make_dividers<config_type>(shape_type{12,3,1}),60,3)
    );
    auto descriptor = std::get<0>(test_data);
    const descriptor_base_type& descriptor_base = descriptor;
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_strides_libdivide = std::get<3>(test_data);
    auto expected_size = std::get<4>(test_data);
    auto expected_dim = std::get<5>(test_data);
    auto expected_offset = index_type{0};
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_strides);
    REQUIRE(descriptor.strides_libdivide() == expected_strides_libdivide);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.offset() == expected_offset);
    REQUIRE(descriptor_base.as_descriptor_with_libdivide() == &descriptor);
    REQUIRE(descriptor_base.as_descriptor_with_libdivide()->strides_libdivide() == expected_strides_libdivide);
}

TEMPLATE_TEST_CASE("test_converting_descriptor_getters", "[test_descriptor]", gtensor::config::mode_div_libdivide, gtensor::config::mode_div_native){
    using config_type = test_config::config_div_mode_selector<TestType>::config_type;
    using descriptor_type = gtensor::converting_descriptor<config_type>;
    using descriptor_base_type = gtensor::descriptor_base<config_type>;
    using descriptor_with_libdivide_type = gtensor::descriptor_with_libdivide<config_type>;
    using strides_libdivide_type = gtensor::detail::libdiv_strides_traits<config_type>::type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_dividers;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, shape_type, index_type, index_type,index_type,shape_type, strides_libdivide_type>;
    //0descriptor, 1expected_shape, 2expected_strides, 3expected_cstrides, 4expected_dim, 5expected_size, 6expected_offset 7expected_reset_cstrides 8expected_strides_libdivide
    auto test_data = GENERATE(
        test_type{descriptor_type{shape_type{15},shape_type{-1},14},shape_type{15},shape_type{1},shape_type{-1},1,15,14, shape_type{-14}, make_dividers<config_type>(shape_type{1})},
        test_type{descriptor_type{shape_type{3,1,7},shape_type{7,7,1},0},shape_type{3,1,7},shape_type{7,7,1},shape_type{7,7,1},3,21,0, shape_type{14,0,6}, make_dividers<config_type>(shape_type{7,7,1})},
        test_type{descriptor_type{shape_type{3,1,7},shape_type{7,7,-1},6},shape_type{3,1,7},shape_type{7,7,1},shape_type{7,7,-1},3,21,6, shape_type{14,0,-6}, make_dividers<config_type>(shape_type{7,7,1})}
    );
    auto descriptor = std::get<0>(test_data);
    const descriptor_base_type& descriptor_base= descriptor;
    const descriptor_with_libdivide_type& descriptor_with_libdivide = descriptor;
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_cstrides = std::get<3>(test_data);
    auto expected_dim = std::get<4>(test_data);
    auto expected_size = std::get<5>(test_data);
    auto expected_offset = std::get<6>(test_data);
    auto expected_reset_cstrides = std::get<7>(test_data);
    auto expected_strides_libdivide = std::get<8>(test_data);
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_cstrides);
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.offset() == expected_offset);
    REQUIRE(descriptor.reset_cstrides() == expected_reset_cstrides);
    REQUIRE(descriptor.strides_libdivide() == expected_strides_libdivide);
    REQUIRE(descriptor_base.as_descriptor_with_libdivide() == &descriptor_with_libdivide);
    REQUIRE(descriptor_base.as_descriptor_with_libdivide()->strides_libdivide() == expected_strides_libdivide);
}

TEMPLATE_TEST_CASE("test_converting_descriptor_convert", "[test_descriptor]", gtensor::config::mode_div_native, gtensor::config::mode_div_libdivide){
    using config_type = test_config::config_div_mode_selector<TestType>::config_type;
    using descriptor_type = gtensor::converting_descriptor<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type,index_type>;
    //descriptor{shape,cstrides,offset}
    //0descriptor,1multi_idx,2flat_idx,3converted_multi_idx,4converted_flat_idx,5converted_by_prev
    auto test_data = GENERATE(
                                test_type{descriptor_type{shape_type{15},shape_type{1},0},shape_type{0},0,0,0,0},
                                test_type{descriptor_type{shape_type{15},shape_type{1},0},shape_type{7},7,7,7,7},
                                test_type{descriptor_type{shape_type{3,3,5},shape_type{15,5,1},5},shape_type{1,0,4},22,24,27,22},     //(3,3,5) (15,5,1)  22->(1,1,2)->22
                                test_type{descriptor_type{shape_type{3,2,5},shape_type{-20,10,1},40},shape_type{1,0,4},22,24,2,22}   //(3,2,5) (10,5,1)  22->(2,0,2)->-38
    );
    auto descriptor = std::get<0>(test_data);
    auto multi_idx = std::get<1>(test_data);
    auto flat_idx = std::get<2>(test_data);
    auto expected_convert_multi_idx = std::get<3>(test_data);
    auto expected_convert_flat_idx = std::get<4>(test_data);
    auto expected_convert_by_prev_flat_idx = std::get<5>(test_data);

    REQUIRE(descriptor.convert(multi_idx) == expected_convert_multi_idx);
    REQUIRE(descriptor.convert(flat_idx) == expected_convert_flat_idx);
}



