#include "catch.hpp"
#include <iostream>
#include "libdivide_helper.hpp"
#include "test_config.hpp"


TEMPLATE_TEST_CASE("test_is_libdivide_div","[test_libdivide]", std::int64_t, std::uint64_t){
    using index_type = TestType;
    using branchfull_divider = libdivide::divider<index_type>;
    using branchfree_divider = libdivide::branchfree_divider<index_type>;
    

    REQUIRE(gtensor::detail::is_libdivide_div<index_type> == false);
    REQUIRE(gtensor::detail::is_libdivide_div<branchfull_divider> == true);
    REQUIRE(gtensor::detail::is_libdivide_div<branchfree_divider> == true);
}

TEMPLATE_TEST_CASE("test_flat_to_multi_mode_native", "[test_flat_to_multi]", gtensor::config::mode_div_native){
    using value_type = float;
    using config_type = test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl<value_type>;    
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
    REQUIRE(flat_to_multi(strides, flat_idx) == expected_multi_idx);
    
}
TEMPLATE_TEST_CASE("test_flat_to_multi_mode_libdivide", "[test_flat_to_multi]", gtensor::config::mode_div_libdivide){
    using value_type = float;
    using config_type = test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl<value_type>;    
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_libdive_vector;
    using gtensor::detail::flat_to_multi;
    using libdiv_vector_type = decltype(make_libdive_vector<config_type>(std::declval<shape_type>()));
    using test_type = std::tuple<index_type, libdiv_vector_type, shape_type>;
    //0flat_idx,1strides_libdiv,2expected_multi_idx
    auto test_data = GENERATE(
                                test_type{0,make_libdive_vector<config_type>(shape_type{1}),shape_type{0}},
                                test_type{5,make_libdive_vector<config_type>(shape_type{1}),shape_type{5}},
                                test_type{5,make_libdive_vector<config_type>(shape_type{1,1}),shape_type{5,0}},
                                test_type{5,make_libdive_vector<config_type>(shape_type{3,1}),shape_type{1,2}},
                                test_type{34,make_libdive_vector<config_type>(shape_type{12,3,1}),shape_type{2,3,1}}
    );

    auto flat_idx = std::get<0>(test_data);
    auto strides_libdiv = std::get<1>(test_data);
    auto expected_multi_idx = std::get<2>(test_data);
    REQUIRE(flat_to_multi<shape_type>(strides_libdiv, flat_idx) == expected_multi_idx);
}