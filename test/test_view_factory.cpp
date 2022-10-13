#include <tuple>
#include <vector>
#include "catch.hpp"
#include "test_config.hpp"
#include "view_factory.hpp"
#include "gtensor.hpp"

namespace test_view_factory{

template<typename T>
struct test_tensor : public T{
    test_tensor(const T& base):
        T{base}
    {}
    using T::engine;
    using T::impl;
};

template<template<typename> typename TestT = test_tensor, typename T>
auto make_test_tensor(T&& t){return TestT<std::decay_t<T>>{t};}

}   //end of namespace test_view_factory


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
}

TEMPLATE_TEST_CASE("test_make_view_subdim_shape","[test_view_factory]", trivial_type_vector::uvector<std::int64_t>, std::vector<std::int64_t>){
    using index_type = typename TestType::value_type;
    using shape_type = TestType;
    using slice_type = gtensor::slice<index_type>;
    using gtensor::detail::make_view_subdim_shape;
    using test_type = std::tuple<shape_type, shape_type, shape_type>;
    //0parent_shape,1subs,2expected_shape
    auto test_data = GENERATE(
        test_type{shape_type{11,1},shape_type{}, shape_type{11,1}},
        test_type{shape_type{11,1},shape_type{0}, shape_type{1}},
        test_type{shape_type{1,11},shape_type{0}, shape_type{11}},
        test_type{shape_type{3,4,10,2},shape_type{1,3}, shape_type{10,2}}
    );
    auto parent_shape = std::get<0>(test_data);
    auto subs = std::get<1>(test_data);
    auto expected_shape = std::get<2>(test_data);
    REQUIRE(make_view_subdim_shape(parent_shape, subs) == expected_shape);
}

TEMPLATE_TEST_CASE("test_make_view_reshape_shape","[test_view_factory]", trivial_type_vector::uvector<std::int64_t>, std::vector<std::int64_t>){
    using index_type = typename TestType::value_type;
    using shape_type = TestType;
    using slice_type = gtensor::slice<index_type>;
    using gtensor::detail::make_view_reshape_shape;
    using test_type = std::tuple<shape_type, shape_type, shape_type>;
    //0parent_shape,1subs,2expected_shape
    auto test_data = GENERATE(
        test_type{shape_type{11,1},shape_type{}, shape_type{11,1}},
        test_type{shape_type{11,1},shape_type{11}, shape_type{11}},
        test_type{shape_type{1,11},shape_type{11,1}, shape_type{11,1}},
        test_type{shape_type{3,4,10,2},shape_type{}, shape_type{3,4,10,2}},
        test_type{shape_type{3,4,10,2},shape_type{20,12}, shape_type{20,12}}
    );
    auto parent_shape = std::get<0>(test_data);
    auto subs = std::get<1>(test_data);
    auto expected_shape = std::get<2>(test_data);
    REQUIRE(make_view_reshape_shape(parent_shape, subs) == expected_shape);
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
}

TEMPLATE_TEST_CASE("test_make_view_subdim_offset","[test_view_factory]", trivial_type_vector::uvector<std::int64_t>, std::vector<std::int64_t>){
    using index_type = typename TestType::value_type;
    using shape_type = TestType;
    using slice_type = gtensor::slice<index_type>;
    using slices_collection_type = std::vector<slice_type>;
    using gtensor::detail::make_view_subdim_offset;
    using test_type = std::tuple<shape_type, shape_type, index_type>;
    //0parent_strides,1subs,2expected_offset
    auto test_data = GENERATE(
        test_type{shape_type{11,1},shape_type{}, 0},
        test_type{shape_type{11,1},shape_type{0}, 0},
        test_type{shape_type{11,1},shape_type{}, 0},
        test_type{shape_type{11,1},shape_type{4}, 44},
        test_type{shape_type{1,4,20,100},shape_type{}, 0},
        test_type{shape_type{1,4,20,100},shape_type{2,3}, 14}
    );

    auto parent_strides = std::get<0>(test_data);
    auto subs = std::get<1>(test_data);
    auto expected_offset = std::get<2>(test_data);
    REQUIRE(make_view_subdim_offset(parent_strides, subs) == expected_offset);
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

TEMPLATE_PRODUCT_TEST_CASE("test_transpose","[test_view_factory]", (std::vector,trivial_type_vector::uvector),(std::size_t, std::uint32_t, int)){
    using shape_type = TestType;
    using gtensor::detail::transpose;
    using test_type = std::tuple<shape_type, shape_type, shape_type>;
    //0source,1indeces,2expected_transposed
    auto test_data = GENERATE(
        test_type{shape_type{3},shape_type{}, shape_type{3}},
        test_type{shape_type{3},shape_type{0}, shape_type{3}},
        test_type{shape_type{3,2},shape_type{0,1}, shape_type{3,2}},
        test_type{shape_type{3,2},shape_type{}, shape_type{2,3}},
        test_type{shape_type{3,2},shape_type{1,0}, shape_type{2,3}},
        test_type{shape_type{4,3,2,2},shape_type{}, shape_type{2,2,3,4}},
        test_type{shape_type{4,3,2,2},shape_type{3,1,0,2}, shape_type{2,3,4,2}}
    );

    auto source = std::get<0>(test_data);
    auto indeces = std::get<1>(test_data);
    auto expected_transposed = std::get<2>(test_data);
    REQUIRE(transpose(source, indeces) == expected_transposed);
}

TEMPLATE_TEST_CASE("test_make_shape_index_tensor","[test_view_factory]",
    typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using test_config_type = TestType;
    using index_type = typename test_config_type::index_type;
    using shape_type = typename test_config_type::shape_type;
    using gtensor::subscript_exception;
    using gtensor::detail::make_shape_index_tensor;
    using gtensor::detail::broadcast_shape;
    using test_type = std::tuple<shape_type,shape_type>;
    //0result shape,1expected shape
    auto test_data = GENERATE(
        test_type{make_shape_index_tensor(shape_type{10},broadcast_shape<shape_type>(shape_type{4}),index_type{1}), shape_type{4}},
        test_type{make_shape_index_tensor(shape_type{10},broadcast_shape<shape_type>(shape_type{2,2}),index_type{1}), shape_type{2,2}},
        test_type{make_shape_index_tensor(shape_type{5,4,3,2},broadcast_shape<shape_type>(shape_type{8},shape_type{8}),index_type{2}), shape_type{8,3,2}},
        test_type{make_shape_index_tensor(shape_type{5,4,3,2},broadcast_shape<shape_type>(shape_type{3,4},shape_type{1},shape_type{3,1}),index_type{3}), shape_type{3,4,2}},
        test_type{make_shape_index_tensor(shape_type{5,4,3,2},broadcast_shape<shape_type>(shape_type{3,4},shape_type{1},shape_type{3,1}, shape_type{1,4}),index_type{4}), shape_type{3,4}}
    );

    auto result_shape = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    REQUIRE(result_shape == expected_shape);
}

TEMPLATE_TEST_CASE("test_make_shape_index_tensor_exception","[test_view_factory]",
    typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using test_config_type = TestType;
    using index_type = typename test_config_type::index_type;
    using shape_type = typename test_config_type::shape_type;
    using gtensor::subscript_exception;
    using gtensor::detail::make_shape_index_tensor;
    using gtensor::detail::broadcast_shape;

    REQUIRE_THROWS_AS(make_shape_index_tensor(shape_type{10},broadcast_shape<shape_type>(shape_type{4}, shape_type{4}),index_type{2}), subscript_exception);
    REQUIRE_THROWS_AS(make_shape_index_tensor(shape_type{3,4},broadcast_shape<shape_type>(shape_type{4}, shape_type{4}, shape_type{1}),index_type{3}), subscript_exception);
}

TEMPLATE_TEST_CASE("test_make_map_index_tensor","[test_view_factory]",
    typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using test_config_type = TestType;
    using index_type = typename test_config_type::index_type;
    using shape_type = typename test_config_type::shape_type;
    using tensor_type = gtensor::tensor<index_type, test_config_type>;
    using test_view_factory::test_tensor;
    using test_view_factory::make_test_tensor;
    using gtensor::detail::make_map_index_tensor;
    using gtensor::detail::broadcast_shape;
    using gtensor::detail::make_size;
    using map_type = std::vector<index_type>;
    using test_type = std::tuple<map_type,map_type>;
    //0result map,1expected map
    auto test_data = GENERATE(
        test_type{
            [](){
            auto sub1 = make_test_tensor(tensor_type{0,0,0});
            auto index_shape = broadcast_shape<shape_type>(sub1.impl()->shape());
            auto index_size = make_size(index_shape);
            return make_map_index_tensor<map_type>(shape_type{1},shape_type{1},index_size,sub1.engine().begin_broadcast(index_shape));
            }(),
            map_type{0,0,0}
        },
        test_type{
            [](){
            auto sub1 = make_test_tensor(tensor_type{2});
            auto index_shape = broadcast_shape<shape_type>(sub1.impl()->shape());
            auto index_size = make_size(index_shape);
            return make_map_index_tensor<map_type>(shape_type{10},shape_type{1},index_size,sub1.engine().begin_broadcast(index_shape));
            }(),
            map_type{2}
        },
        test_type{
            [](){
            auto sub1 = make_test_tensor(tensor_type{0,1,2});
            auto sub2 = make_test_tensor(tensor_type{0,1,2});
            auto index_shape = broadcast_shape<shape_type>(sub1.impl()->shape(),sub2.impl()->shape());
            auto index_size = make_size(index_shape);
            return make_map_index_tensor<map_type>(shape_type{4,3},shape_type{3,1},index_size,sub1.engine().begin_broadcast(index_shape), sub2.engine().begin_broadcast(index_shape));
            }(),
            map_type{0,4,8}
        },
        test_type{
            [](){
                auto sub1 = make_test_tensor(tensor_type{0,1,2});
                auto sub2 = make_test_tensor(tensor_type{1});
                auto index_shape = broadcast_shape<shape_type>(sub1.impl()->shape(),sub2.impl()->shape());
                auto index_size = make_size(index_shape);
                return make_map_index_tensor<map_type>(shape_type{4,3},shape_type{3,1},index_size,sub1.engine().begin_broadcast(index_shape), sub2.engine().begin_broadcast(index_shape));
            }(),
            map_type{1,4,7}
        },
        test_type{
            [](){
                auto sub1 = make_test_tensor(tensor_type{1,3});
                auto index_shape = broadcast_shape<shape_type>(sub1.impl()->shape());
                auto index_size = make_size(index_shape);
                return make_map_index_tensor<map_type>(shape_type{4,3},shape_type{3,1},index_size,sub1.engine().begin_broadcast(index_shape));
            }(),
            map_type{3,4,5,9,10,11}
        },
        test_type{
            [](){
                auto sub1 = make_test_tensor(tensor_type{1,3,0,1});
                auto index_shape = broadcast_shape<shape_type>(sub1.impl()->shape());
                auto index_size = make_size(index_shape);
                return make_map_index_tensor<map_type>(shape_type{4,3},shape_type{3,1},index_size,sub1.engine().begin_broadcast(index_shape));
            }(),
            map_type{3,4,5,9,10,11,0,1,2,3,4,5}
        }
    );

    auto result_map = std::get<0>(test_data);
    auto expected_map = std::get<1>(test_data);
    REQUIRE(result_map == expected_map);
}

TEMPLATE_TEST_CASE("test_make_map_index_tensor_exception","[test_view_factory]",
    typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using test_config_type = TestType;
    using index_type = typename test_config_type::index_type;
    using shape_type = typename test_config_type::shape_type;
    using tensor_type = gtensor::tensor<index_type, test_config_type>;
    using test_view_factory::test_tensor;
    using test_view_factory::make_test_tensor;
    using gtensor::detail::make_map_index_tensor;
    using gtensor::detail::broadcast_shape;
    using gtensor::detail::make_size;
    using gtensor::subscript_exception;
    using map_type = std::vector<index_type>;

    SECTION("two_subscripts"){
        using test_type = std::tuple<tensor_type,tensor_type>;
        //sub0,sub1
        auto test_data = GENERATE(
            test_type{tensor_type{0,1,2}, tensor_type{0,1,3}},
            test_type{tensor_type{0,5,2}, tensor_type{0,1,2}},
            test_type{tensor_type{0,1,2}, tensor_type{4}},
            test_type{tensor_type{4,4,4}, tensor_type{2}}
        );

        auto sub1 = make_test_tensor(std::get<0>(test_data));
        auto sub2 = make_test_tensor(std::get<1>(test_data));
        auto index_shape = broadcast_shape<shape_type>(sub1.impl()->shape(),sub2.impl()->shape());
        auto index_size = make_size(index_shape);
        REQUIRE_THROWS_AS(
            make_map_index_tensor<map_type>(shape_type{4,3},shape_type{3,1},index_size, sub1.engine().begin_broadcast(index_shape), sub2.engine().begin_broadcast(index_shape)),
            subscript_exception
        );
    }
    SECTION("one_subscript"){
        using test_type = std::tuple<tensor_type>;
        //sub0
        auto test_data = GENERATE(
            test_type{tensor_type{5,3,0,0}}
        );

        auto sub1 = make_test_tensor(std::get<0>(test_data));
        auto index_shape = broadcast_shape<shape_type>(sub1.impl()->shape());
        auto index_size = make_size(index_shape);
        REQUIRE_THROWS_AS(
            make_map_index_tensor<map_type>(shape_type{4,3},shape_type{3,1},index_size, sub1.engine().begin_broadcast(index_shape)),
            subscript_exception
        );
    }
}

TEMPLATE_TEST_CASE("test_make_shape_bool_tensor","[test_view_factory]",
    typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using test_config_type = TestType;
    using index_type = typename test_config_type::index_type;
    using shape_type = typename test_config_type::shape_type;
    using index_tensor_type = gtensor::tensor<bool, test_config_type>;
    using gtensor::subscript_exception;
    using gtensor::detail::make_shape_bool_tensor;
    using test_type = std::tuple<shape_type,shape_type>;
    //0result shape,1expected shape
    auto test_data = GENERATE(
        test_type{
            [](){
                auto idx = index_tensor_type{true,false,false,true,false};
                return make_shape_bool_tensor(shape_type{5},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{2}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{true};
                return make_shape_bool_tensor(shape_type{5},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{1}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{false,true,true};
                return make_shape_bool_tensor(shape_type{5},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{2}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{false,false,false,false,false};
                return make_shape_bool_tensor(shape_type{5},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{false,true,false};
                return make_shape_bool_tensor(shape_type{3,2,4},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{1,2,4}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{true};
                return make_shape_bool_tensor(shape_type{3,2,4},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{1,2,4}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{{false,true},{true,true},{false,false}};
                return make_shape_bool_tensor(shape_type{3,2,4},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{3,4}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{{false},{true}};
                return make_shape_bool_tensor(shape_type{3,2,4},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{1,4}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{{false,true},{true,true},{true,false}};
                return make_shape_bool_tensor(shape_type{3,2},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{4}
        },
        test_type{
            [](){
                auto idx = index_tensor_type{{false,false},{false,false},{false,false}};
                return make_shape_bool_tensor(shape_type{3,2},idx.shape(), idx.begin(), idx.end());
            }(),
            shape_type{}
        }
    );

    auto result_shape = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    REQUIRE(result_shape == expected_shape);
}

TEMPLATE_TEST_CASE("test_make_shape_bool_tensor_exception","[test_view_factory]",
    typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using test_config_type = TestType;
    using index_type = typename test_config_type::index_type;
    using shape_type = typename test_config_type::shape_type;
    using index_tensor_type = gtensor::tensor<bool, test_config_type>;
    using gtensor::subscript_exception;
    using gtensor::detail::make_shape_bool_tensor;


    REQUIRE_THROWS_AS(
        ([](){
            auto idx = index_tensor_type{true,false,false,true,false,false};
            return make_shape_bool_tensor(shape_type{5},idx.shape(), idx.begin(), idx.end());
        }()),
        subscript_exception
    );
    REQUIRE_THROWS_AS(
        ([](){
            auto idx = index_tensor_type{{true},{false}};
            return make_shape_bool_tensor(shape_type{5},idx.shape(), idx.begin(), idx.end());
        }()),
        subscript_exception
    );
    REQUIRE_THROWS_AS(
        ([](){
            auto idx = index_tensor_type{{false,true,false},{true,true,false},{false,false,false}};
            return make_shape_bool_tensor(shape_type{3,2,4},idx.shape(), idx.begin(), idx.end());
        }()),
        subscript_exception
    );
}

TEMPLATE_TEST_CASE("test_make_map_bool_tensor","[test_view_factory]",
    typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type
){
    using test_config_type = TestType;
    using index_type = typename test_config_type::index_type;
    using shape_type = typename test_config_type::shape_type;
    using index_tensor_type = gtensor::tensor<bool, test_config_type>;
    using test_view_factory::test_tensor;
    using test_view_factory::make_test_tensor;
    using gtensor::detail::make_shape_bool_tensor;
    using gtensor::detail::make_map_bool_tensor;
    using gtensor::detail::make_size;
    using map_type = std::vector<index_type>;
    using test_type = std::tuple<map_type,map_type>;
    //0result map,1expected map
    auto test_data = GENERATE(
        test_type{
            [](){
            auto sub1 = index_tensor_type{true,false,true,false,false};
            auto view_shape = make_shape_bool_tensor(shape_type{5},sub1.shape(),sub1.begin(),sub1.end());
            return make_map_bool_tensor<map_type>(shape_type{5},shape_type{1},make_size(view_shape),sub1.dim(),sub1.begin(),sub1.end());
            }(),
            map_type{0,2}
        },
        test_type{
            [](){
            auto sub1 = index_tensor_type{true,false,true};
            auto view_shape = make_shape_bool_tensor(shape_type{5},sub1.shape(),sub1.begin(),sub1.end());
            return make_map_bool_tensor<map_type>(shape_type{5},shape_type{1},make_size(view_shape),sub1.dim(),sub1.begin(),sub1.end());
            }(),
            map_type{0,2}
        },
        test_type{
            [](){
            auto sub1 = index_tensor_type{true,false,true,false};
            auto view_shape = make_shape_bool_tensor(shape_type{4,3,2},sub1.shape(),sub1.begin(),sub1.end());
            return make_map_bool_tensor<map_type>(shape_type{4,3,2},shape_type{6,2,1},make_size(view_shape),sub1.dim(),sub1.begin(),sub1.end());
            }(),
            map_type{0,1,2,3,4,5,12,13,14,15,16,17}
        },
        test_type{
            [](){
            auto sub1 = index_tensor_type{false,false,true};
            auto view_shape = make_shape_bool_tensor(shape_type{4,3,2},sub1.shape(),sub1.begin(),sub1.end());
            return make_map_bool_tensor<map_type>(shape_type{4,3,2},shape_type{6,2,1},make_size(view_shape),sub1.dim(),sub1.begin(),sub1.end());
            }(),
            map_type{12,13,14,15,16,17}
        },
        test_type{
            [](){
            auto sub1 = index_tensor_type{{false,false,true},{false,false,true},{false,false,false},{false,true,false}};
            auto view_shape = make_shape_bool_tensor(shape_type{4,3,2},sub1.shape(),sub1.begin(),sub1.end());
            return make_map_bool_tensor<map_type>(shape_type{4,3,2},shape_type{6,2,1},make_size(view_shape),sub1.dim(),sub1.begin(),sub1.end());
            }(),
            map_type{4,5,10,11,20,21}
        },
        test_type{
            [](){
            auto sub1 = index_tensor_type{{true,false},{false,true}};
            auto view_shape = make_shape_bool_tensor(shape_type{4,3,2},sub1.shape(),sub1.begin(),sub1.end());
            return make_map_bool_tensor<map_type>(shape_type{4,3,2},shape_type{6,2,1},make_size(view_shape),sub1.dim(),sub1.begin(),sub1.end());
            }(),
            map_type{0,1,6,7}
        },
        test_type{
            [](){
            auto sub1 = index_tensor_type{{false,false,true},{false,false,true},{false,false,false},{false,true,false}};
            auto view_shape = make_shape_bool_tensor(shape_type{4,3},sub1.shape(),sub1.begin(),sub1.end());
            return make_map_bool_tensor<map_type>(shape_type{4,3},shape_type{3,1},make_size(view_shape),sub1.dim(),sub1.begin(),sub1.end());
            }(),
            map_type{2,5,10}
        }
    );

    auto result_map = std::get<0>(test_data);
    auto expected_map = std::get<1>(test_data);
    REQUIRE(result_map == expected_map);
}
