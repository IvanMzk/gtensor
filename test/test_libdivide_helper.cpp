#include "catch.hpp"
#include <iostream>
#include "libdivide_helper.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_flat_to_multi_mode_native", "[test_flat_to_multi]", gtensor::config::mode_div_native){
    using config_type = typename test_config::config_div_mode_selector<TestType>::config_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<index_type, shape_type, shape_type>;
    using gtensor::detail::flat_to_multi;
    //0flat_idx,1strides,2expected_multi_idx
    auto test_data = GENERATE(
                                test_type{0,shape_type{1},shape_type{0}},
                                test_type{5,shape_type{1},shape_type{5}},
                                test_type{5,shape_type{1,1},shape_type{5,0}},
                                test_type{5,shape_type{3,1},shape_type{1,2}},
                                test_type{34,shape_type{12,3,1},shape_type{2,3,1}}
    );

    auto flat_idx = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto expected_multi_idx = std::get<2>(test_data);
    REQUIRE(flat_to_multi<shape_type>(strides, flat_idx) == expected_multi_idx);

}
TEMPLATE_TEST_CASE("test_flat_to_multi_mode_libdivide", "[test_flat_to_multi]", gtensor::config::mode_div_libdivide){
    using config_type = typename test_config::config_div_mode_selector<TestType>::config_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_libdivide_vector;
    using gtensor::detail::flat_to_multi;
    using libdiv_vector_type = decltype(make_libdivide_vector(std::declval<shape_type>()));
    using test_type = std::tuple<index_type, libdiv_vector_type, shape_type>;
    //0flat_idx,1strides_libdiv,2expected_multi_idx
    auto test_data = GENERATE(
                                test_type{0,make_libdivide_vector(shape_type{1}),shape_type{0}},
                                test_type{5,make_libdivide_vector(shape_type{1}),shape_type{5}},
                                test_type{5,make_libdivide_vector(shape_type{1,1}),shape_type{5,0}},
                                test_type{5,make_libdivide_vector(shape_type{3,1}),shape_type{1,2}},
                                test_type{34,make_libdivide_vector(shape_type{12,3,1}),shape_type{2,3,1}}
    );

    auto flat_idx = std::get<0>(test_data);
    auto strides_libdiv = std::get<1>(test_data);
    auto expected_multi_idx = std::get<2>(test_data);
    REQUIRE(flat_to_multi<shape_type>(strides_libdiv, flat_idx) == expected_multi_idx);
}
TEMPLATE_TEST_CASE("test_flat_to_flat_mode_native", "[test_flat_to_flat]", gtensor::config::mode_div_native){
    using config_type = typename test_config::config_div_mode_selector<TestType>::config_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<index_type, shape_type, shape_type, index_type, index_type>;
    using gtensor::detail::flat_to_flat;
    //0flat_idx,1strides,2cstrides,3offset,4expected_flat_idx
    auto test_data = GENERATE(
                                test_type{0,shape_type{1},shape_type{1},0,0},
                                test_type{5,shape_type{1},shape_type{1},0,5},
                                test_type{5,shape_type{1,1},shape_type{1,1},10,15},
                                test_type{5,shape_type{3,1},shape_type{2,1},10,14},
                                test_type{34,shape_type{12,3,1},shape_type{6,3,1},0,22}
    );

    auto flat_idx = std::get<0>(test_data);
    auto strides = std::get<1>(test_data);
    auto cstrides = std::get<2>(test_data);
    auto offset = std::get<3>(test_data);
    auto expected_flat_idx = std::get<4>(test_data);
    REQUIRE(flat_to_flat(strides, cstrides, offset, flat_idx) == expected_flat_idx);

}
TEMPLATE_TEST_CASE("test_flat_to_flat_mode_libdivide", "[test_flat_to_flat]", gtensor::config::mode_div_libdivide){
    using config_type = typename test_config::config_div_mode_selector<TestType>::config_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_libdivide_vector;
    using gtensor::detail::flat_to_flat;
    using libdiv_vector_type = decltype(make_libdivide_vector(std::declval<shape_type>()));
    using test_type = std::tuple<index_type, libdiv_vector_type, shape_type, index_type, index_type>;
    //0flat_idx,1strides_libdivide,2cstrides,3offset,4expected_flat_idx
    auto test_data = GENERATE(
                                test_type{0,make_libdivide_vector(shape_type{1}),shape_type{1},0,0},
                                test_type{5,make_libdivide_vector(shape_type{1}),shape_type{1},0,5},
                                test_type{5,make_libdivide_vector(shape_type{1,1}),shape_type{1,1},10,15},
                                test_type{5,make_libdivide_vector(shape_type{3,1}),shape_type{2,1},10,14},
                                test_type{34,make_libdivide_vector(shape_type{12,3,1}),shape_type{6,3,1},0,22}
    );

    auto flat_idx = std::get<0>(test_data);
    auto strides_libdivide = std::get<1>(test_data);
    auto cstrides = std::get<2>(test_data);
    auto offset = std::get<3>(test_data);
    auto expected_flat_idx = std::get<4>(test_data);
    REQUIRE(flat_to_flat(strides_libdivide, cstrides, offset, flat_idx) == expected_flat_idx);
}