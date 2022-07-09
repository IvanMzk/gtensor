#include "catch.hpp"
#include <tuple>
#include "view_factory.hpp"

TEMPLATE_TEST_CASE("test_make_view_slice_shape","[test_view_factory]", trivial_type_vector::uvector<std::int64_t>, std::vector<std::int64_t>){
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

    // SECTION("index_type_args"){
    //     REQUIRE(make_view_shape(shape_type{11,1},index_type(0)) == shape_type{11});
    //     REQUIRE(make_view_shape(shape_type{1,11},index_type(0)) == shape_type{1});
    //     REQUIRE(make_view_shape(shape_type{3,4,10,2},index_type(1), index_type(3)) == shape_type{3,4});
    // }    
}


TEMPLATE_TEST_CASE("test_make_view_slice_offset","[test_view_factory]", trivial_type_vector::uvector<std::int64_t>, std::vector<std::int64_t>){
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
    
    
    // SECTION("index_type_args"){
    //     REQUIRE(make_view_offset(shape_type{1,11},index_type(0)) == index_type(0));
    //     REQUIRE(make_view_offset(shape_type{1,11},index_type(4)) == index_type(44));    //psh(11,4)
    //     REQUIRE(make_view_offset(shape_type{1,4,20,100},index_type(2),index_type(3)) == index_type(260));   //psh(4,5,5,3)
    // }
}

TEMPLATE_TEST_CASE("test_make_view_slice_cstrides","[test_view_factory]", trivial_type_vector::uvector<std::int64_t>){
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

// TEMPLATE_PRODUCT_TEST_CASE("test_transpose","[test_view_factory]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::uint32_t, int)){
//     using shape_type = TestType;
//     using tensor::detail::transpose;    
//     REQUIRE(transpose(shape_type{3},0) == shape_type{3});
//     REQUIRE(transpose(shape_type{3,2},0,1) == shape_type{3,2});
//     REQUIRE(transpose(shape_type{3,2},1,0) == shape_type{2,3});
//     REQUIRE(transpose(shape_type{4,3,2,2},3,1,0,2) == shape_type{3,2,2,4});
    
//     REQUIRE(transpose(shape_type{}) == shape_type{});
//     REQUIRE(transpose(shape_type{3}) == shape_type{3});
//     REQUIRE(transpose(shape_type{3,2}) == shape_type{2,3});
//     REQUIRE(transpose(shape_type{4,3,2,2}) == shape_type{2,2,3,4});
// }
