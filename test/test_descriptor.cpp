#include <list>
#include <iostream>
#include "catch.hpp"
#include "descriptor.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_make_strides","[test_descriptor]",
    std::vector<std::int64_t>,
    std::vector<std::size_t>
)
{
    using shape_type = TestType;
    using gtensor::detail::make_strides;
    //0shape,1expected strides
    using test_type = typename std::tuple<shape_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},shape_type{}},
        test_type{shape_type{0},shape_type{1}},
        test_type{shape_type{1},shape_type{1}},
        test_type{shape_type{5},shape_type{1}},
        test_type{shape_type{0,0},shape_type{1,1}},
        test_type{shape_type{1,0},shape_type{1,1}},
        test_type{shape_type{0,1},shape_type{1,1}},
        test_type{shape_type{5,0},shape_type{1,1}},
        test_type{shape_type{0,5},shape_type{5,1}},
        test_type{shape_type{1,1},shape_type{1,1}},
        test_type{shape_type{5,1},shape_type{1,1}},
        test_type{shape_type{1,5},shape_type{5,1}},
        test_type{shape_type{2,3,4},shape_type{12,4,1}},
        test_type{shape_type{0,0,0},shape_type{1,1,1}},
        test_type{shape_type{2,2,0,2},shape_type{4,2,2,1}},
        test_type{shape_type{4,3,2,0},shape_type{6,2,1,1}},
        test_type{shape_type{0,3,2,1},shape_type{6,2,1,1}}
    );
    auto shape = std::get<0>(test_data);
    auto strides_expected = std::get<1>(test_data);
    auto strides_result = make_strides(shape);
    REQUIRE(strides_result == strides_expected);
}
TEMPLATE_TEST_CASE("test_make_strides_div","[test_descriptor]",
    gtensor::config::mode_div_libdivide,
    gtensor::config::mode_div_native
)
{
    using config_type = gtensor::config::extend_config_t<typename test_config::config_div_mode_selector<TestType>::config_type, int>;
    using shape_type = typename config_type::shape_type;
    using strides_div_type = typename gtensor::detail::strides_div_traits<config_type>::type;
    using divider_type = typename strides_div_type::value_type;
    using test_type = typename std::tuple<shape_type,strides_div_type>;
    using gtensor::detail::make_strides_div;
    //0shape,1expected_strides_div
    auto test_data = GENERATE(
        test_type{shape_type{},strides_div_type{}},
        test_type{shape_type{0},strides_div_type{divider_type(1)}},
        test_type{shape_type{1},strides_div_type{divider_type(1)}},
        test_type{shape_type{5},strides_div_type{divider_type(1)}},
        test_type{shape_type{0,0},strides_div_type{divider_type(1),divider_type(1)}},
        test_type{shape_type{1,0},strides_div_type{divider_type(1),divider_type(1)}},
        test_type{shape_type{5,0},strides_div_type{divider_type(1),divider_type(1)}},
        test_type{shape_type{0,1},strides_div_type{divider_type(1),divider_type(1)}},
        test_type{shape_type{0,5},strides_div_type{divider_type(5),divider_type(1)}},
        test_type{shape_type{1,1},strides_div_type{divider_type(1),divider_type(1)}},
        test_type{shape_type{5,1},strides_div_type{divider_type(1),divider_type(1)}},
        test_type{shape_type{1,5},strides_div_type{divider_type(5),divider_type(1)}},
        test_type{shape_type{0,0,0},strides_div_type{divider_type(1),divider_type(1),divider_type(1)}},
        test_type{shape_type{2,3,4},strides_div_type{divider_type(12),divider_type(4),divider_type(1)}},
        test_type{shape_type{2,2,0,2},strides_div_type{divider_type(4),divider_type(2),divider_type(2),divider_type(1)}},
        test_type{shape_type{4,3,2,0},strides_div_type{divider_type(6),divider_type(2),divider_type(1),divider_type(1)}},
        test_type{shape_type{0,3,2,1},strides_div_type{divider_type(6),divider_type(2),divider_type(1),divider_type(1)}}
    );

    auto shape = std::get<0>(test_data);
    auto strides_expected = std::get<1>(test_data);
    auto strides_result = make_strides_div<config_type>(shape);
    REQUIRE(strides_result == strides_expected);
}

TEMPLATE_TEST_CASE("test_make_reset_strides","[test_descriptor]",
    std::vector<std::int64_t>,
    std::vector<std::size_t>
)
{
    using shape_type = TestType;
    using gtensor::detail::make_reset_strides;
    //0shape,1strides,2expected reset strides
    using test_type = typename std::tuple<shape_type,shape_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},shape_type{},shape_type{}},
        test_type{shape_type{0},shape_type{1},shape_type{0}},
        test_type{shape_type{1},shape_type{1},shape_type{0}},
        test_type{shape_type{5},shape_type{1},shape_type{4}},
        test_type{shape_type{0,0},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{1,0},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{0,1},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{5,0},shape_type{1,1},shape_type{4,0}},
        test_type{shape_type{0,5},shape_type{5,1},shape_type{0,4}},
        test_type{shape_type{1,1},shape_type{1,1},shape_type{0,0}},
        test_type{shape_type{5,1},shape_type{1,1},shape_type{4,0}},
        test_type{shape_type{1,5},shape_type{5,1},shape_type{0,4}},
        test_type{shape_type{0,0,0},shape_type{1,1,1},shape_type{0,0,0}},
        test_type{shape_type{2,3,4},shape_type{12,4,1},{12,8,3}},
        test_type{shape_type{2,2,0,2},shape_type{4,2,2,1},shape_type{4,2,0,1}},
        test_type{shape_type{4,3,2,0},shape_type{6,2,1,1},shape_type{18,4,1,0}},
        test_type{shape_type{0,3,2,1},shape_type{6,2,1,1},shape_type{0,4,1,0}}
    );
    auto shape = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto reset_strides_expected = std::get<2>(test_data);
    auto reset_strides_result = make_reset_strides(shape,strides);
    REQUIRE(reset_strides_result == reset_strides_expected);
}

TEMPLATE_TEST_CASE("test_make_size","[test_descriptor]",
    std::vector<std::int64_t>,
    std::vector<std::size_t>
)
{
    using shape_type = TestType;
    using index_type = typename TestType::value_type;
    using gtensor::detail::make_size;
    //shape,expected
    using test_type = typename std::tuple<shape_type,index_type>;
    auto test_data = GENERATE(
        test_type{shape_type{},index_type{1}},
        test_type{shape_type{0},index_type{0}},
        test_type{shape_type{1},index_type{1}},
        test_type{shape_type{5},index_type{5}},
        test_type{shape_type{0,0},index_type{0}},
        test_type{shape_type{1,0},index_type{0}},
        test_type{shape_type{0,1},index_type{0}},
        test_type{shape_type{0,5},index_type{0}},
        test_type{shape_type{5,0},index_type{0}},
        test_type{shape_type{1,1},index_type{1}},
        test_type{shape_type{5,1},index_type{5}},
        test_type{shape_type{1,5},index_type{5}},
        test_type{shape_type{0,0,0},index_type{0}},
        test_type{shape_type{2,3,4},index_type{24}},
        test_type{shape_type{2,2,0,2},index_type{0}},
        test_type{shape_type{4,3,2,0},index_type{0}},
        test_type{shape_type{0,3,2,1},index_type{0}}
    );
    auto shape = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    auto result = make_size(shape);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_flat_to_multi", "[test_descriptor]",
    gtensor::config::mode_div_native,
    gtensor::config::mode_div_libdivide
)
{
    using config_type = gtensor::config::extend_config_t<typename test_config::config_div_mode_selector<TestType>::config_type, int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_dividers;
    using gtensor::detail::flat_to_multi;
    using test_type = std::tuple<index_type, decltype(make_dividers<config_type>(std::declval<shape_type>())), shape_type>;
    //0flat_idx,1strides,2expected_multi_idx
    auto test_data = GENERATE(
                                test_type{0,make_dividers<config_type>(shape_type{1}),shape_type{0}},
                                test_type{5,make_dividers<config_type>(shape_type{1}),shape_type{5}},
                                test_type{5,make_dividers<config_type>(shape_type{1,1}),shape_type{5,0}},
                                test_type{5,make_dividers<config_type>(shape_type{3,1}),shape_type{1,2}},
                                test_type{34,make_dividers<config_type>(shape_type{12,3,1}),shape_type{2,3,1}}
    );

    auto flat_idx = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto expected_multi_idx = std::get<2>(test_data);
    REQUIRE(flat_to_multi<shape_type>(strides, flat_idx) == expected_multi_idx);
}

TEMPLATE_TEST_CASE("test_flat_to_flat", "[test_descriptor]",
    gtensor::config::mode_div_native,
    gtensor::config::mode_div_libdivide
)
{
    using config_type = gtensor::config::extend_config_t<typename test_config::config_div_mode_selector<TestType>::config_type, int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_dividers;
    using gtensor::detail::flat_to_flat;
    using test_type = std::tuple<index_type, decltype(make_dividers<config_type>(std::declval<shape_type>())), shape_type, index_type, index_type>;
    //0flat_idx,1strides,2cstrides,3offset,4expected_flat_idx
    auto test_data = GENERATE(
                                test_type{0,make_dividers<config_type>(shape_type{1}),shape_type{1},0,0},
                                test_type{5,make_dividers<config_type>(shape_type{1}),shape_type{1},0,5},
                                test_type{5,make_dividers<config_type>(shape_type{1,1}),shape_type{1,1},10,15},
                                test_type{5,make_dividers<config_type>(shape_type{3,1}),shape_type{2,1},10,14},
                                test_type{34,make_dividers<config_type>(shape_type{12,3,1}),shape_type{6,3,1},0,22}
    );

    auto flat_idx = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto cstrides = std::get<2>(test_data);
    auto offset = std::get<3>(test_data);
    auto expected_flat_idx = std::get<4>(test_data);
    REQUIRE(flat_to_flat(strides, cstrides, offset, flat_idx) == expected_flat_idx);
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

TEST_CASE("test_basic_descriptor","[test_descriptor]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using descriptor_type = gtensor::basic_descriptor<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, shape_type, index_type, dim_type>;
    //0descriptor,1expected shape,2expected strides,3expected reset_strides,4expected size,5expected dim
    auto test_data = GENERATE(
        test_type(descriptor_type{shape_type{}},shape_type{},shape_type{},shape_type{},1,0),
        test_type(descriptor_type{shape_type{0}},shape_type{0},shape_type{1},shape_type{0},0,1),
        test_type(descriptor_type{shape_type{1}},shape_type{1},shape_type{1},shape_type{0},1,1),
        test_type(descriptor_type{shape_type{5}},shape_type{5},shape_type{1},shape_type{4},5,1),
        test_type(descriptor_type{shape_type{0,0}},shape_type{0,0},shape_type{1,1},shape_type{0,0},0,2),
        test_type(descriptor_type{shape_type{1,0}},shape_type{1,0},shape_type{1,1},shape_type{0,0},0,2),
        test_type(descriptor_type{shape_type{0,1}},shape_type{0,1},shape_type{1,1},shape_type{0,0},0,2),
        test_type(descriptor_type{shape_type{5,0}},shape_type{5,0},shape_type{1,1},shape_type{4,0},0,2),
        test_type(descriptor_type{shape_type{0,5}},shape_type{0,5},shape_type{5,1},shape_type{0,4},0,2),
        test_type(descriptor_type{shape_type{1,1}},shape_type{1,1},shape_type{1,1},shape_type{0,0},1,2),
        test_type(descriptor_type{shape_type{1,5}},shape_type{1,5},shape_type{5,1},shape_type{0,4},5,2),
        test_type(descriptor_type{shape_type{5,1}},shape_type{5,1},shape_type{1,1},shape_type{4,0},5,2),
        test_type(descriptor_type{shape_type{5,4,3}},shape_type{5,4,3},shape_type{12,3,1},shape_type{48,9,2},60,3),
        test_type(descriptor_type{shape_type{2,2,0,2}},shape_type{2,2,0,2},shape_type{4,2,2,1},shape_type{4,2,0,1},0,4),
        test_type(descriptor_type{shape_type{4,3,2,0}},shape_type{4,3,2,0},shape_type{6,2,1,1},shape_type{18,4,1,0},0,4),
        test_type(descriptor_type{shape_type{0,3,2,1}},shape_type{0,3,2,1},shape_type{6,2,1,1},shape_type{0,4,1,0},0,4)
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
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using descriptor_type = gtensor::basic_descriptor<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type>;
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

TEST_CASE("test_descriptor_with_offset","[test_descriptor]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using descriptor_type = gtensor::descriptor_with_offset<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, index_type, dim_type, index_type>;
    //0descriptor,1expected shape,2expected strides,3expected size,4expected dim,5expected offset
    auto test_data = GENERATE(
        test_type({shape_type{},0},shape_type{},shape_type{},1,0,0),
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
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using descriptor_type = gtensor::descriptor_with_offset<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type>;
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

TEMPLATE_TEST_CASE("test_converting_descriptor", "[test_descriptor]",
    gtensor::config::mode_div_libdivide,
    gtensor::config::mode_div_native
)
{
    using config_type = gtensor::config::extend_config_t<typename test_config::config_div_mode_selector<TestType>::config_type, int>;
    using descriptor_type = gtensor::converting_descriptor<config_type>;
    using strides_div_type = typename gtensor::detail::strides_div_traits<config_type>::type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using gtensor::detail::make_dividers;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, shape_type, dim_type, index_type,index_type,shape_type, strides_div_type>;
    //0descriptor, 1expected_shape, 2expected_strides, 3expected_cstrides, 4expected_dim, 5expected_size, 6expected_offset 7expected_reset_cstrides 8expected_strides_libdivide
    auto test_data = GENERATE(
        test_type{descriptor_type{shape_type{15},shape_type{-1},14},shape_type{15},shape_type{1},shape_type{-1},1,15,14, shape_type{-14}, make_dividers<config_type>(shape_type{1})},
        test_type{descriptor_type{shape_type{3,1,7},shape_type{7,7,1},0},shape_type{3,1,7},shape_type{7,7,1},shape_type{7,7,1},3,21,0, shape_type{14,0,6}, make_dividers<config_type>(shape_type{7,7,1})},
        test_type{descriptor_type{shape_type{3,1,7},shape_type{7,7,-1},6},shape_type{3,1,7},shape_type{7,7,1},shape_type{7,7,-1},3,21,6, shape_type{14,0,-6}, make_dividers<config_type>(shape_type{7,7,1})}
    );
    auto descriptor = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_cstrides = std::get<3>(test_data);
    auto expected_dim = std::get<4>(test_data);
    auto expected_size = std::get<5>(test_data);
    auto expected_offset = std::get<6>(test_data);
    auto expected_reset_cstrides = std::get<7>(test_data);
    auto expected_strides_div = std::get<8>(test_data);
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_cstrides);
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.offset() == expected_offset);
    REQUIRE(descriptor.reset_cstrides() == expected_reset_cstrides);
    REQUIRE(descriptor.strides_div() == expected_strides_div);
}

TEMPLATE_TEST_CASE("test_converting_descriptor_convert", "[test_descriptor]",
    gtensor::config::mode_div_native,
    gtensor::config::mode_div_libdivide
)
{
    using config_type = gtensor::config::extend_config_t<typename test_config::config_div_mode_selector<TestType>::config_type, int>;
    using descriptor_type = gtensor::converting_descriptor<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type>;
    //0descriptor,1multi_idx,2flat_idx,3converted_multi_idx,4converted_flat_idx
    auto test_data = GENERATE(
                                test_type{descriptor_type{shape_type{15},shape_type{1},0},shape_type{0},0,0,0},
                                test_type{descriptor_type{shape_type{15},shape_type{1},0},shape_type{7},7,7,7},
                                test_type{descriptor_type{shape_type{3,3,5},shape_type{15,5,1},5},shape_type{1,0,4},22,24,27},     //(3,3,5) (15,5,1)  22->(1,1,2)->22
                                test_type{descriptor_type{shape_type{3,2,5},shape_type{-20,10,1},40},shape_type{1,0,4},22,24,2}   //(3,2,5) (10,5,1)  22->(2,0,2)->-38
    );
    auto descriptor = std::get<0>(test_data);
    auto multi_idx = std::get<1>(test_data);
    auto flat_idx = std::get<2>(test_data);
    auto expected_convert_multi_idx = std::get<3>(test_data);
    auto expected_convert_flat_idx = std::get<4>(test_data);

    REQUIRE(descriptor.convert(multi_idx) == expected_convert_multi_idx);
    REQUIRE(descriptor.convert(flat_idx) == expected_convert_flat_idx);
}

