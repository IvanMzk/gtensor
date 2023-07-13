#include <tuple>
#include <vector>
#include <iomanip>
#include "catch.hpp"
#include "indexing.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_indexing_make_take_shape","[test_indexing]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_take_shape;
    using helpers_for_testing::apply_by_element;

    //0shape,1indexes_shape,2axis,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{0},shape_type{0},0,shape_type{0}),
        std::make_tuple(shape_type{0,2,3},shape_type{0},0,shape_type{0,2,3}),
        std::make_tuple(shape_type{0,2,3},shape_type{0},1,shape_type{0,0,3}),
        std::make_tuple(shape_type{0,2,3},shape_type{0},2,shape_type{0,2,0}),
        std::make_tuple(shape_type{3,2,4},shape_type{0},0,shape_type{0,2,4}),
        std::make_tuple(shape_type{3,2,4},shape_type{0,2,3},0,shape_type{0,2,3,2,4}),
        std::make_tuple(shape_type{3,2,4},shape_type{0,2,3},1,shape_type{3,0,2,3,4}),
        std::make_tuple(shape_type{3,2,4},shape_type{0,2,3},2,shape_type{3,2,0,2,3})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto indexes_shape = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = make_take_shape(shape,indexes_shape,axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_indexing_take","[test_indexing]",
    //input order, indexes order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using value_type = double;
    using input_order = std::tuple_element_t<0,TestType>;
    using indexes_order = std::tuple_element_t<1,TestType>;
    using tensor_type = gtensor::tensor<value_type,input_order>;
    using index_tensor_type = gtensor::tensor<int,indexes_order>;
    using gtensor::take;
    using helpers_for_testing::apply_by_element;

    //0tensor,1indexes,2axis,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},index_tensor_type{},0,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),index_tensor_type{},0,tensor_type{}.reshape(0,2,3)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),index_tensor_type{},1,tensor_type{}.reshape(0,0,3)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),index_tensor_type{},2,tensor_type{}.reshape(0,2,0)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),index_tensor_type{}.reshape(0,5),1,tensor_type{}.reshape(0,0,5,3)),
        std::make_tuple(tensor_type(1),index_tensor_type{},0,tensor_type{}),
        std::make_tuple(tensor_type(1),index_tensor_type{}.reshape(0,2,3),0,tensor_type{}.reshape(0,2,3)),
        std::make_tuple(tensor_type(2),index_tensor_type(0),0,tensor_type(2)),
        std::make_tuple(tensor_type(3),index_tensor_type{0},0,tensor_type{3}),
        std::make_tuple(tensor_type(3),index_tensor_type{0,0,0,0},0,tensor_type{3,3,3,3}),
        std::make_tuple(tensor_type(3),index_tensor_type{{0,0,0},{0,0,0}},0,tensor_type{{3,3,3},{3,3,3}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},index_tensor_type{},0,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{},0,tensor_type{}.reshape(0,3)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{},1,tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{}.reshape(0,5),1,tensor_type{}.reshape(2,0,5)),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10,11,12},index_tensor_type{1,0,3,5,2,6,11},0,tensor_type{2,1,4,6,3,7,12}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10,11,12},index_tensor_type{{1,0,3},{5,2,1},{3,11,4}},0,tensor_type{{2,1,4},{6,3,2},{4,12,5}}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},index_tensor_type{2,2,0,3,1},0,tensor_type{{9,10,11,12},{9,10,11,12},{1,2,3,4},{13,14,15,16},{5,6,7,8}}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},index_tensor_type{2,2,0,3,1},1,tensor_type{{3,3,1,4,2},{7,7,5,8,6},{11,11,9,12,10},{15,15,13,16,14}}),
        std::make_tuple(
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}},
            index_tensor_type{0,2,0,1},
            0,
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{17,18,19,20},{21,22,23,24}},{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}},
            index_tensor_type{0,1,1,0},
            1,
            tensor_type{{{1,2,3,4},{5,6,7,8},{5,6,7,8},{1,2,3,4}},{{9,10,11,12},{13,14,15,16},{13,14,15,16},{9,10,11,12}},{{17,18,19,20},{21,22,23,24},{21,22,23,24},{17,18,19,20}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}},
            index_tensor_type{0,3,0,2,1,3},
            2,
            tensor_type{{{1,4,1,3,2,4},{5,8,5,7,6,8}},{{9,12,9,11,10,12},{13,16,13,15,14,16}},{{17,20,17,19,18,20},{21,24,21,23,22,24}}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto indexes = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = take(ten,indexes,axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

