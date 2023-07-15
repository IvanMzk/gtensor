#include <tuple>
#include <vector>
#include "catch.hpp"
#include "indexing.hpp"
#include "builder.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_indexing_traverse_predicate","[test_indexing]")
{
    using gtensor::detail::make_traverse_predicate;
    using gtensor::detail::traverse_predicate;
    using helpers_for_testing::apply_by_element;

    //0axes,1inverse,2axis,3expected
    auto test_data = std::make_tuple(
        //no inverse
        //axes scalar
        std::make_tuple(0,std::false_type{},0,true),
        std::make_tuple(0,std::false_type{},1,false),
        std::make_tuple(0,std::false_type{},2,false),
        std::make_tuple(2,std::false_type{},0,false),
        std::make_tuple(2,std::false_type{},1,false),
        std::make_tuple(2,std::false_type{},2,true),
        //axes container
        std::make_tuple(std::vector<std::size_t>{},std::false_type{},0,false),
        std::make_tuple(std::vector<std::size_t>{},std::false_type{},1,false),
        std::make_tuple(std::vector<std::size_t>{},std::false_type{},2,false),
        std::make_tuple(std::vector<std::size_t>{1},std::false_type{},0,false),
        std::make_tuple(std::vector<std::size_t>{1},std::false_type{},1,true),
        std::make_tuple(std::vector<std::size_t>{1},std::false_type{},2,false),
        std::make_tuple(std::vector<std::size_t>{1,3},std::false_type{},0,false),
        std::make_tuple(std::vector<std::size_t>{1,3},std::false_type{},1,true),
        std::make_tuple(std::vector<std::size_t>{1,3},std::false_type{},2,false),
        std::make_tuple(std::vector<std::size_t>{1,3},std::false_type{},3,true),
        std::make_tuple(std::vector<std::size_t>{1,3},std::false_type{},4,false),
        //inverse
        //axes scalar
        std::make_tuple(0,std::true_type{},0,false),
        std::make_tuple(0,std::true_type{},1,true),
        std::make_tuple(0,std::true_type{},2,true),
        std::make_tuple(2,std::true_type{},0,true),
        std::make_tuple(2,std::true_type{},1,true),
        std::make_tuple(2,std::true_type{},2,false),
        //axes container
        std::make_tuple(std::vector<std::size_t>{},std::true_type{},0,true),
        std::make_tuple(std::vector<std::size_t>{},std::true_type{},1,true),
        std::make_tuple(std::vector<std::size_t>{},std::true_type{},2,true),
        std::make_tuple(std::vector<std::size_t>{1},std::true_type{},0,true),
        std::make_tuple(std::vector<std::size_t>{1},std::true_type{},1,false),
        std::make_tuple(std::vector<std::size_t>{1},std::true_type{},2,true),
        std::make_tuple(std::vector<std::size_t>{1,3},std::true_type{},0,true),
        std::make_tuple(std::vector<std::size_t>{1,3},std::true_type{},1,false),
        std::make_tuple(std::vector<std::size_t>{1,3},std::true_type{},2,true),
        std::make_tuple(std::vector<std::size_t>{1,3},std::true_type{},3,false),
        std::make_tuple(std::vector<std::size_t>{1,3},std::true_type{},4,true)
    );
    auto test = [](const auto& t){
        auto axes = std::get<0>(t);
        auto inverse = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto predicate = make_traverse_predicate(axes,inverse);

        auto result = predicate(axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

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
        std::make_tuple(shape_type{3,2,4},shape_type{0,2,3},2,shape_type{3,2,0,2,3}),
        std::make_tuple(shape_type{2,3,4},shape_type{4,2},1,shape_type{2,4,2,4})
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
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},index_tensor_type{{2,1},{0,3},{1,2}},0,tensor_type{{{9,10,11,12},{5,6,7,8}},{{1,2,3,4},{13,14,15,16}},{{5,6,7,8},{9,10,11,12}}}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},index_tensor_type{{2,1},{0,3},{1,2}},1,tensor_type{{{3,2},{1,4},{2,3}},{{7,6},{5,8},{6,7}},{{11,10},{9,12},{10,11}},{{15,14},{13,16},{14,15}}}),
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
            index_tensor_type{{0,1},{1,1},{1,0}},
            1,
            tensor_type{{{{1,2,3,4},{5,6,7,8}},{{5,6,7,8},{5,6,7,8}},{{5,6,7,8},{1,2,3,4}}},{{{9,10,11,12},{13,14,15,16}},{{13,14,15,16},{13,14,15,16}},{{13,14,15,16},{9,10,11,12}}},{{{17,18,19,20},{21,22,23,24}},{{21,22,23,24},{21,22,23,24}},{{21,22,23,24},{17,18,19,20}}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}},
            index_tensor_type{0,3,0,2,1,3},
            2,
            tensor_type{{{1,4,1,3,2,4},{5,8,5,7,6,8}},{{9,12,9,11,10,12},{13,16,13,15,14,16}},{{17,20,17,19,18,20},{21,24,21,23,22,24}}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}},
            index_tensor_type{{{0,2,1},{3,1,2}},{{1,2,0},{2,0,3}}},
            2,
            tensor_type{{{{{1,3,2},{4,2,3}},{{2,3,1},{3,1,4}}},{{{5,7,6},{8,6,7}},{{6,7,5},{7,5,8}}}},{{{{9,11,10},{12,10,11}},{{10,11,9},{11,9,12}}},{{{13,15,14},{16,14,15}},{{14,15,13},{15,13,16}}}},{{{{17,19,18},{20,18,19}},{{18,19,17},{19,17,20}}},{{{21,23,22},{24,22,23}},{{22,23,21},{23,21,24}}}}}
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

TEMPLATE_TEST_CASE("test_indexing_take_like_over_flatten","[test_indexing]",
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

    //0tensor,1indexes,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},index_tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),index_tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),index_tensor_type{}.reshape(0,5),tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type(1),index_tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type(1),index_tensor_type{}.reshape(0,2,3),tensor_type{}.reshape(0,2,3)),
        std::make_tuple(tensor_type(2),index_tensor_type(0),tensor_type(2)),
        std::make_tuple(tensor_type(3),index_tensor_type{0},tensor_type{3}),
        std::make_tuple(tensor_type(3),index_tensor_type{0,0,0,0},tensor_type{3,3,3,3}),
        std::make_tuple(tensor_type(3),index_tensor_type{{0,0,0},{0,0,0}},tensor_type{{3,3,3},{3,3,3}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},index_tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{}.reshape(0,5),tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10,11,12},index_tensor_type{1,0,3,5,2,6,11},tensor_type{2,1,4,6,3,7,12}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10,11,12},index_tensor_type{{1,0,3},{5,2,1},{3,11,4}},tensor_type{{2,1,4},{6,3,2},{4,12,5}}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},index_tensor_type{2,12,0,3,1},tensor_type{3,13,1,4,2}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},index_tensor_type{{1,0,3},{5,2,1},{3,11,4}},tensor_type{{2,1,4},{6,3,2},{4,12,5}}),
        std::make_tuple(
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}},
            index_tensor_type{0,2,0,1,6,3,10,4,8,22},
            tensor_type{1,3,1,2,7,4,11,5,9,23}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}},
            index_tensor_type{{2,1,5},{10,0,2},{6,6,8}},
            tensor_type{{3,2,6},{11,1,3},{7,7,9}}
        ),
        std::make_tuple(
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}},
            index_tensor_type{{{2,1,5},{10,0,2}},{{6,6,8},{4,3,14}}},
            tensor_type{{{3,2,6},{11,1,3}},{{7,7,9},{5,4,15}}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto indexes = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto result = take(ten,indexes);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_indexing_take_exception","[test_indexing]",
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
    using gtensor::indexing_exception;
    using helpers_for_testing::apply_by_element;

    //take from empty
    REQUIRE_THROWS_AS(take(tensor_type{},index_tensor_type{0,0,0},0),indexing_exception);
    //axis out of bounds
    REQUIRE_THROWS_AS(take(tensor_type{},index_tensor_type{},1),indexing_exception);
    REQUIRE_THROWS_AS(take(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{},2),indexing_exception);
    REQUIRE_THROWS_AS(take(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{},2),indexing_exception);
    REQUIRE_THROWS_AS(take(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{0,0,0},2),indexing_exception);
    //indexes out of bounds
    REQUIRE_THROWS_AS(take(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{0,1,2},0),indexing_exception);
    REQUIRE_THROWS_AS(take(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{0,1,2,3},1),indexing_exception);
    REQUIRE_THROWS_AS(take(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{0,6,2,3}),indexing_exception);
}

