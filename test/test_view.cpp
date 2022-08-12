#include <tuple>
#include "catch.hpp"
#include "impl_stensor.hpp"

TEST_CASE("test_view_of_stensor","[test_view]"){
    using value_type = float;
    using gtensor::storage_tensor;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using stensor_type = storage_tensor<value_type, default_config>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using slices_collection_type = typename config_type::slices_collection_type;
    using slice_type = typename config_type::slice_type;
    using view_ptr_type = std::shared_ptr<gtensor::tensor_impl_base<value_type,gtensor::config::default_config>>;
    SECTION("slices_collection_subs"){
        //0view_ptr,1expected_shape,2expected size,3expected dim
        using test_type = std::tuple<view_ptr_type, shape_type, index_type, index_type>;
        //slices must be filled        
        auto test_data = GENERATE(
            test_type{stensor_type{1,2,3,4,5}.create_view_slice(slices_collection_type{slice_type{0,5,1}}), shape_type{5}, 5 ,1},
            test_type{stensor_type{1,2,3,4,5}.create_view_slice(slices_collection_type{slice_type{4,-1,-1}}), shape_type{5}, 5 ,1},
            test_type{stensor_type{1,2,3,4,5}.create_view_slice(slices_collection_type{slice_type{0,5,2}}), shape_type{3}, 3 ,1},
            test_type{stensor_type{{1,2,3,4},{4,5,6,7},{7,8,9,10}}.create_view_slice(slices_collection_type{slice_type{0,3,2},slice_type{0,4,2}}), shape_type{2,2}, 4 ,2},
            test_type{stensor_type{{1,2,3,4},{4,5,6,7},{7,8,9,10}}.create_view_slice(slices_collection_type{slice_type{2,3,1},slice_type{0,4,1}}), shape_type{1,4}, 4 ,2}
        );
        
        auto v = std::get<0>(test_data);
        auto expected_shape = std::get<1>(test_data);
        auto expected_size = std::get<2>(test_data);
        auto expected_dim = std::get<3>(test_data);
        REQUIRE(v->shape() == expected_shape);
        REQUIRE(v->size() == expected_size);
        REQUIRE(v->dim() == expected_dim);
    }
    SECTION("shape_subs"){
        //0view_ptr,1expected_shape,2expected size,3expected dim
        using test_type = std::tuple<view_ptr_type, shape_type, index_type, index_type>;
        //slices must be filled        
        auto test_data = GENERATE(
            test_type{stensor_type{1,2,3,4,5}.create_view_transpose(shape_type{}), shape_type{5}, 5 ,1},
            test_type{stensor_type{{1},{2},{3},{4},{5}}.create_view_transpose(shape_type{}), shape_type{1,5}, 5 ,2},
            test_type{stensor_type{{{1,1},{2,2},{3,3}},{{2,2},{3,3},{4,4}},{{3,3},{4,4},{5,5}}}.create_view_transpose(shape_type{}), shape_type{2,3,3}, 18 ,3},
            test_type{stensor_type{{{1,1},{2,2},{3,3}},{{2,2},{3,3},{4,4}},{{3,3},{4,4},{5,5}}}.create_view_transpose(shape_type{0,2,1}), shape_type{3,2,3}, 18 ,3},
            test_type{stensor_type{1,2,3,4,5}.create_view_subdim(shape_type{}), shape_type{5}, 5 ,1},
            test_type{stensor_type{{1},{2},{3},{4},{5}}.create_view_subdim(shape_type{}), shape_type{5,1}, 5 ,2},
            test_type{stensor_type{{1},{2},{3},{4},{5}}.create_view_subdim(shape_type{0}), shape_type{1}, 1 ,1},
            test_type{stensor_type{{{1,1},{2,2},{3,3}},{{2,2},{3,3},{4,4}},{{3,3},{4,4},{5,5}}}.create_view_subdim(shape_type{}), shape_type{3,3,2}, 18 ,3},
            test_type{stensor_type{{{1,1},{2,2},{3,3}},{{2,2},{3,3},{4,4}},{{3,3},{4,4},{5,5}}}.create_view_subdim(shape_type{2}), shape_type{3,2}, 6 ,2},
            test_type{stensor_type{{{1,1},{2,2},{3,3}},{{2,2},{3,3},{4,4}},{{3,3},{4,4},{5,5}}}.create_view_subdim(shape_type{2,0}), shape_type{2}, 2 ,1},
            test_type{stensor_type{1,2,3,4,5}.create_view_reshape(shape_type{}), shape_type{5}, 5 ,1},
            test_type{stensor_type{1,2,3,4,5}.create_view_reshape(shape_type{5,1}), shape_type{5,1}, 5 ,2},
            test_type{stensor_type{{{1,1},{2,2},{3,3}},{{2,2},{3,3},{4,4}},{{3,3},{4,4},{5,5}}}.create_view_reshape(shape_type{}), shape_type{3,3,2}, 18 ,3},
            test_type{stensor_type{{{1,1},{2,2},{3,3}},{{2,2},{3,3},{4,4}},{{3,3},{4,4},{5,5}}}.create_view_reshape(shape_type{18}), shape_type{18}, 18 ,1},
            test_type{stensor_type{{{1,1},{2,2},{3,3}},{{2,2},{3,3},{4,4}},{{3,3},{4,4},{5,5}}}.create_view_reshape(shape_type{9,2}), shape_type{9,2}, 18 ,2}
        );
        
        auto v = std::get<0>(test_data);
        auto expected_shape = std::get<1>(test_data);
        auto expected_size = std::get<2>(test_data);
        auto expected_dim = std::get<3>(test_data);
        REQUIRE(v->shape() == expected_shape);
        REQUIRE(v->size() == expected_size);
        REQUIRE(v->dim() == expected_dim);
    }    
}