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

TEST_CASE("test_basic_descriptor","[test_descriptor]"){
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::basic_descriptor<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, index_type, index_type>;
    //descriptor,expected shape,expected strides,expected size,expected dim
    auto test_data = GENERATE(                                
                                test_type(descriptor_type(),shape_type{},shape_type{},0,0),
                                test_type({shape_type{}},shape_type{},shape_type{},0,0),
                                test_type({shape_type{1}},shape_type{1},shape_type{1},1,1),
                                test_type({shape_type{5}},shape_type{5},shape_type{1},5,1),
                                test_type({shape_type{1,1}},shape_type{1,1},shape_type{1,1},1,2),
                                test_type({shape_type{1,5}},shape_type{1,5},shape_type{5,1},5,2),
                                test_type({shape_type{5,1}},shape_type{5,1},shape_type{1,1},5,2),
                                test_type({shape_type{5,4,3}},shape_type{5,4,3},shape_type{12,3,1},60,3)
                            );
    auto descriptor = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_size = std::get<3>(test_data);
    auto expected_dim = std::get<4>(test_data);
    auto expected_offset = index_type{0};
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.size() == expected_size);    
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_strides);
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

TEMPLATE_TEST_CASE("test_descriptor_with_libdivide_native_div","[test_descriptor]", gtensor::config::mode_div_native){
    using config_type = test_config::config_div_mode_selector<TestType>::config_type;
    using descriptor_type = gtensor::descriptor_with_libdivide<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, index_type, index_type>;
    //descriptor,expected shape,expected strides,expected size,expected dim
    auto test_data = GENERATE(                                
                                test_type(descriptor_type(),shape_type{},shape_type{},0,0),
                                test_type({shape_type{}},shape_type{},shape_type{},0,0),
                                test_type({shape_type{1}},shape_type{1},shape_type{1},1,1),
                                test_type({shape_type{5}},shape_type{5},shape_type{1},5,1),
                                test_type({shape_type{1,1}},shape_type{1,1},shape_type{1,1},1,2),
                                test_type({shape_type{1,5}},shape_type{1,5},shape_type{5,1},5,2),
                                test_type({shape_type{5,1}},shape_type{5,1},shape_type{1,1},5,2),
                                test_type({shape_type{5,4,3}},shape_type{5,4,3},shape_type{12,3,1},60,3)
                            );
    auto descriptor = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_size = std::get<3>(test_data);
    auto expected_dim = std::get<4>(test_data);
    auto expected_offset = index_type{0};
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_strides);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.offset() == expected_offset);
}

TEMPLATE_TEST_CASE("test_descriptor_with_libdivide_libdivide_div","[test_descriptor]", gtensor::config::mode_div_libdivide){
    using config_type = test_config::config_div_mode_selector<TestType>::config_type;
    using descriptor_type = gtensor::descriptor_with_libdivide<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using libdivide_vector_type = gtensor::detail::libdivide_vector<index_type>;
    using libdivide_divider_type = gtensor::detail::libdivide_divider<index_type>;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, index_type, index_type, libdivide_vector_type>;
    //0descriptor,1expected shape,2expected strides,3expected size,4expected dim,5expected_strides_libdivide
    auto test_data = GENERATE(                                
                                test_type(descriptor_type(),shape_type{},shape_type{},0,0,{}),
                                test_type({shape_type{}},shape_type{},shape_type{},0,0,libdivide_vector_type{}),
                                test_type({shape_type{1}},shape_type{1},shape_type{1},1,1,{libdivide_divider_type{1}}),
                                test_type({shape_type{5}},shape_type{5},shape_type{1},5,1,{libdivide_divider_type{1}}),
                                test_type({shape_type{1,1}},shape_type{1,1},shape_type{1,1},1,2,{libdivide_divider_type{1},libdivide_divider_type{1}}),
                                test_type({shape_type{1,5}},shape_type{1,5},shape_type{5,1},5,2,{libdivide_divider_type{5},libdivide_divider_type{1}}),
                                test_type({shape_type{5,1}},shape_type{5,1},shape_type{1,1},5,2,{libdivide_divider_type{1},libdivide_divider_type{1}}),
                                test_type({shape_type{5,4,3}},shape_type{5,4,3},shape_type{12,3,1},60,3,{libdivide_divider_type{12},libdivide_divider_type{3},libdivide_divider_type{1}})
                            );
    auto descriptor = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_size = std::get<3>(test_data);
    auto expected_dim = std::get<4>(test_data);
    auto expected_strides_libdivide = std::get<5>(test_data);
    auto expected_offset = index_type{0};
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_strides);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.strides_libdivide() == expected_strides_libdivide);
    REQUIRE(descriptor.offset() == expected_offset);
}

TEST_CASE("test_converting_descriptor_getters", "[test_descriptor]"){
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::converting_descriptor<config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, shape_type, index_type, index_type,index_type>;
    //0descriptor, 1expected_shape, 2expected_strides, 3expected_cstrides, 4expected_dim, 5expected_size, 6expected_offset
    auto test_data = GENERATE(
                                test_type{descriptor_type{shape_type{15},shape_type{-1},14},shape_type{15},shape_type{1},shape_type{-1},1,15,14},
                                test_type{descriptor_type{shape_type{3,1,7},shape_type{7,7,1},0},shape_type{3,1,7},shape_type{7,7,1},shape_type{7,7,1},3,21,0},
                                test_type{descriptor_type{shape_type{3,1,7},shape_type{7,7,-1},6},shape_type{3,1,7},shape_type{7,7,1},shape_type{7,7,-1},3,21,6}
    );
    auto descriptor = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_strides = std::get<2>(test_data);
    auto expected_cstrides = std::get<3>(test_data);
    auto expected_dim = std::get<4>(test_data);
    auto expected_size = std::get<5>(test_data);
    auto expected_offset = std::get<6>(test_data);
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.cstrides() == expected_cstrides);
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.offset() == expected_offset);
    //gtensor::descriptor_base<value_type,gtensor::config::default_config>* pd = &descriptor;
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


