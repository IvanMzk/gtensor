#include <iostream>
#include "catch.hpp"
#include "stensor_descriptor.hpp"
#include "test_config.hpp"



TEMPLATE_PRODUCT_TEST_CASE("test_make_strides","[test_stensor_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
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

TEMPLATE_PRODUCT_TEST_CASE("test_make_size_using_strides","[test_stensor_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::int64_t)){
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

TEMPLATE_TEST_CASE("test_stensor_descriptor_native_div","[test_stensor_descriptor]", gtensor::config::mode_div_native){
    using value_type = float;
    using gtensor::stensor_descriptor;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using descriptor_type = stensor_descriptor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
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
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.dim() == expected_dim);
}

TEMPLATE_TEST_CASE("test_stensor_descriptor_libdiv_div","[test_stensor_descriptor]", gtensor::config::mode_div_libdivide){
    using value_type = float;
    using gtensor::stensor_descriptor;
    using config_type = test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl<value_type>;
    using descriptor_type = stensor_descriptor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
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
    REQUIRE(descriptor.shape() == expected_shape);
    REQUIRE(descriptor.strides() == expected_strides);
    REQUIRE(descriptor.size() == expected_size);
    REQUIRE(descriptor.dim() == expected_dim);
    REQUIRE(descriptor.strides_libdivide() == expected_strides_libdivide);
}

// TEMPLATE_PRODUCT_TEST_CASE("test_make_shape","[test_stensor_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::uint32_t, int)){
//     using shape_type = TestType;
//     using index_type = typename shape_type::value_type;
//     using tensor::detail::make_shape;
//     index_type d{3};
//     REQUIRE(make_shape<shape_type>() == shape_type{});
//     REQUIRE(make_shape<shape_type>(1) == shape_type{1});
//     REQUIRE(make_shape<shape_type>(d) == shape_type{d});
//     REQUIRE(make_shape<shape_type>(1,5) == shape_type{5,1});
//     REQUIRE(make_shape<shape_type>(5,1) == shape_type{1,5});
//     REQUIRE(make_shape<shape_type>(4,3,d,2) == shape_type{2,d,3,4});
// }

// TEMPLATE_PRODUCT_TEST_CASE("test_make_size","[test_stensor_descriptor]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::uint32_t, int)){
//     using shape_type = TestType;
//     using index_type = typename shape_type::value_type;
//     using tensor::detail::make_size;
//     REQUIRE(make_size(shape_type{}) == index_type(0));
//     REQUIRE(make_size(shape_type{1}) == index_type(1));
//     REQUIRE(make_size(shape_type{1,1}) == index_type(1));
//     REQUIRE(make_size(shape_type{3,1}) == index_type(3));
//     REQUIRE(make_size(shape_type{3,4,2}) == index_type(24));
// }

// TEST_CASE("test_libdivide_strides_mode_native","[test_stensor_descriptor]"){
//     using descriptor_type = typename tensor::TensorSlice<float, test_config::config_tmpl_div_mode_selector<tensor::config::mode_div_native>::config_tmpl>;
//     descriptor_type desc{3,4,5};
//     REQUIRE(desc.strides_div_().size() == 0);
// }
// TEST_CASE("test_libdivide_strides_mode_libdivide","[test_stensor_descriptor]"){
//     using descriptor_type = typename tensor::TensorSlice<float, test_config::config_tmpl_div_mode_selector<tensor::config::mode_div_libdivide>::config_tmpl>;
//     using shape_type = typename descriptor_type::shape_type;
//     using index_type = typename descriptor_type::index_type;
//     descriptor_type desc{3,4,5};
//     REQUIRE(desc.strides_div_().size() == 3);
//     REQUIRE(desc.strides_() == shape_type{1,5,20});
//     index_type numerator{7};
//     REQUIRE(numerator/desc.strides_()[0] == numerator/desc.strides_div_()[0]);
//     REQUIRE(numerator/desc.strides_()[1] == numerator/desc.strides_div_()[1]);
//     REQUIRE(numerator/desc.strides_()[2] == numerator/desc.strides_div_()[2]);
// }