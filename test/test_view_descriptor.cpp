#include <tuple>
#include "catch.hpp"
#include "view_descriptor.hpp"
#include "test_config.hpp"


TEMPLATE_TEST_CASE("test_view_slice_descriptor", "[test_view_slice_descriptor]", gtensor::config::mode_div_native, gtensor::config::mode_div_libdivide){
    using value_type = float;
    using config_type = test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl<value_type>;
    using descriptor_type = gtensor::view_slice_descriptor<value_type, typename test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type>;
    //descriptor{shape,cstrides,offset}
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

TEMPLATE_TEST_CASE("test_view_slice_descriptor_prev_descriptor", "[test_view_slice_descriptor]", gtensor::config::mode_div_native, gtensor::config::mode_div_libdivide){
    using value_type = float;
    using config_type = test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl<value_type>;
    using prev_descriptor_type = gtensor::view_slice_descriptor<value_type, typename test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using descriptor_type = gtensor::view_slice_descriptor<value_type, typename test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl, prev_descriptor_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<descriptor_type, shape_type, index_type, index_type, index_type>;
    //descriptor{shape,cstrides,offset,prev_descriptor}
    //0descriptor,1multi_idx,2flat_idx,3converted_multi_idx,4converted_flat_idx
    auto test_data = GENERATE(
                                test_type{descriptor_type{shape_type{7},shape_type{1},7,prev_descriptor_type{shape_type{15},shape_type{1},0}},shape_type{0},0,7,7},
                                test_type{descriptor_type{shape_type{7},shape_type{1},7,prev_descriptor_type{shape_type{15},shape_type{1},0}},shape_type{3},6,10,13},
                                test_type{descriptor_type{shape_type{2,3,5},shape_type{15,-5,1},15, prev_descriptor_type{shape_type{3,3,5},shape_type{15,5,1},5}},shape_type{1,0,4},22,39,32}
    );
    auto descriptor = std::get<0>(test_data);
    auto multi_idx = std::get<1>(test_data);
    auto flat_idx = std::get<2>(test_data);
    auto expected_convert_multi_idx = std::get<3>(test_data);
    auto expected_convert_flat_idx = std::get<4>(test_data);

    REQUIRE(descriptor.convert(multi_idx) == expected_convert_multi_idx);
    REQUIRE(descriptor.convert(flat_idx) == expected_convert_flat_idx);
}