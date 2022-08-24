#include <tuple>
#include "catch.hpp"
#include "view_slice_descriptor.hpp"
#include "view_subdim_descriptor.hpp"
#include "test_config.hpp"

TEST_CASE("test_view_slice_descriptor_getters", "[test_view_descriptor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::view_slice_descriptor<value_type, gtensor::config::default_config>;
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

TEMPLATE_TEST_CASE("test_view_slice_descriptor_convert", "[test_view_descriptor]", gtensor::config::mode_div_native, gtensor::config::mode_div_libdivide){
    using value_type = float;
    using config_type = test_config::config_div_mode_selector<TestType>::config_type;
    using descriptor_type = gtensor::view_slice_descriptor<value_type, config_type>;
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

TEST_CASE("test_view_subdim_descriptor_getters", "[test_view_descriptor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using descriptor_type = gtensor::view_subdim_descriptor<value_type, gtensor::config::default_config>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, shape_type, shape_type, index_type, index_type,index_type>;
    //0descriptor, 1expected_shape, 2expected_strides, 3expected_cstrides, 4expected_dim, 5expected_size, 6expected_offset
    auto test_data = GENERATE(
                                test_type{descriptor_type{shape_type{15},0},shape_type{15},shape_type{1},shape_type{1},1,15,0},
                                test_type{descriptor_type{shape_type{3,1,7},0},shape_type{3,1,7},shape_type{7,7,1},shape_type{7,7,1},3,21,0},
                                test_type{descriptor_type{shape_type{4,5},40},shape_type{4,5},shape_type{5,1},shape_type{5,1},2,20,40}
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
}

TEMPLATE_TEST_CASE("test_view_subdim_descriptor_convert", "[test_view_descriptor]", gtensor::config::mode_div_native, gtensor::config::mode_div_libdivide){
    using value_type = float;
    using config_type = test_config::config_div_mode_selector<TestType>::config_type;
    using descriptor_type = gtensor::view_subdim_descriptor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type,index_type>;
    //descriptor{shape,offset}
    //0descriptor,1multi_idx,2flat_idx,3converted_multi_idx,4converted_flat_idx,5converted_by_prev
    auto test_data = GENERATE(
                                test_type{descriptor_type{shape_type{15},0},shape_type{0},0,0,0,0},
                                test_type{descriptor_type{shape_type{1,15},0},shape_type{0,7},7,7,7,7},
                                test_type{descriptor_type{shape_type{3,3,5},5},shape_type{1,0,4},22,24,27,22}     //(3,3,5) (15,5,1)  22->(1,1,2)->22                                
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