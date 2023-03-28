#include <tuple>
#include <vector>
#include "catch.hpp"
#include "gtensor.hpp"
#include "helpers_for_testing.hpp"
#include "integral_type.hpp"

TEST_CASE("test_is_iterator","[test_tensor]"){
    using gtensor::detail::is_iterator;
    REQUIRE(!is_iterator<int>);
    REQUIRE(!is_iterator<std::vector<int>>);
    REQUIRE(is_iterator<std::vector<int>::iterator>);
}

TEST_CASE("test_is_tensor","[test_tensor]"){
    using gtensor::tensor;
    using gtensor::integral;
    using gtensor::detail::is_index_tensor_v;
    using gtensor::detail::is_bool_tensor_v;

    REQUIRE(is_index_tensor_v<tensor<int>,int>);
    REQUIRE(is_index_tensor_v<tensor<integral<std::int64_t>>,integral<std::int64_t>>);
    REQUIRE(is_index_tensor_v<tensor<int>,integral<std::int64_t>>);
    REQUIRE(is_index_tensor_v<tensor<std::size_t>,std::int64_t>);
    REQUIRE(is_index_tensor_v<tensor<bool>,std::int64_t>);
    REQUIRE(is_index_tensor_v<tensor<bool>,int>);

    REQUIRE(!is_index_tensor_v<tensor<float>,integral<std::int64_t>>);

    REQUIRE(!is_index_tensor_v<tensor<integral<std::int64_t>>,std::int64_t>);
    REQUIRE(!is_index_tensor_v<std::vector<int>,int>);
    REQUIRE(!is_index_tensor_v<std::string,int>);
    REQUIRE(!is_index_tensor_v<std::vector<bool>,int>);

    REQUIRE(is_bool_tensor_v<tensor<bool>>);
    REQUIRE(!is_bool_tensor_v<tensor<int>>);
    REQUIRE(!is_bool_tensor_v<tensor<float>>);
    REQUIRE(!is_bool_tensor_v<std::vector<int>>);
    REQUIRE(!is_bool_tensor_v<std::string>);
    REQUIRE(!is_bool_tensor_v<std::vector<bool>>);
}

TEST_CASE("test_tensor_default_constructor","[test_tensor]"){
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = tensor_type::config_type;
    using size_type = config_type::size_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;

    auto test_data = GENERATE(
        tensor_type(),
        tensor_type{}
    );
    REQUIRE(test_data.size() == index_type{0});
    REQUIRE(test_data.dim() == size_type{1});
    REQUIRE(test_data.shape() == shape_type{0,});
}

TEST_CASE("test_tensor_constructor_from_list","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using size_type = typename config_type::size_type;
    using test_type = std::tuple<tensor_type, shape_type, index_type, size_type>;
    //tensor,expected_shape,expected size,expected dim
    auto test_data = GENERATE(
        test_type(tensor_type{1}, shape_type{1}, 1 , 1),
        test_type(tensor_type{1,2,3}, shape_type{3}, 3 , 1),
        test_type(tensor_type{{1}}, shape_type{1,1}, 1 , 2),
        test_type(tensor_type{{1,2,3}}, shape_type{1,3}, 3 , 2),
        test_type(tensor_type{{1,2,3},{4,5,6}}, shape_type{2,3}, 6 , 2),
        test_type(tensor_type{{{1,2,3,4}}}, shape_type{1,1,4}, 4 , 3),
        test_type(tensor_type{{{1},{2},{3},{4}}}, shape_type{1,4,1}, 4 , 3),
        test_type(tensor_type{{{1,2,3},{2,3,4},{3,4,5},{4,5,6}}}, shape_type{1,4,3}, 12 , 3)
    );

    auto t = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_size = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
}

TEST_CASE("test_tensor_constructor_shape_value","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using size_type = typename config_type::size_type;
    using test_type = std::tuple<tensor_type, shape_type, index_type, size_type>;
    //tensor,expected_shape,expected size,expected dim
    auto test_data = GENERATE(
        test_type(tensor_type(shape_type{},1.0f), shape_type{}, 0 , 0),
        test_type(tensor_type(shape_type{1},1.0f), shape_type{1}, 1 , 1),
        test_type(tensor_type({1},1.0f), shape_type{1}, 1 , 1),
        test_type(tensor_type(shape_type{10},1.0f), shape_type{10}, 10 , 1),
        test_type(tensor_type(shape_type{1,1},1.0f), shape_type{1,1}, 1 , 2),
        test_type(tensor_type(shape_type{1,3},1.0f), shape_type{1,3}, 3 , 2),
        test_type(tensor_type(std::vector<std::size_t>{2,3},1.0f), shape_type{2,3}, 6 , 2),
        test_type(tensor_type(std::vector<int>{1,1,4},1.0f), shape_type{1,1,4}, 4 , 3),
        test_type(tensor_type(shape_type{1,4,1},1.0f), shape_type{1,4,1}, 4 , 3),
        test_type(tensor_type(shape_type{1,4,3},1.0f), shape_type{1,4,3}, 12 , 3),
        test_type(tensor_type({1,4,3},1.0f), shape_type{1,4,3}, 12 , 3),
        test_type(tensor_type(std::initializer_list<std::size_t>{1,4,3},1.0f), shape_type{1,4,3}, 12 , 3)

    );

    auto t = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_size = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
}

TEST_CASE("test_tensor_constructor_container_shape_range","[test_tensor]"){
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = typename tensor_type::config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    using helpers_for_testing::cmp_equal;
    using gtensor::detail::shape_to_str;
    //0shape,1src_elements,2expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(std::vector<std::size_t>{2,2,2}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10}, std::vector<value_type>{1,2,3,4,5,6,7,8})
    );
    auto test = [](auto& t){
        auto shape = std::get<0>(t);
        auto src_elements = std::get<1>(t);
        auto expected_elements = std::get<2>(t);
        tensor_type result_tensor{shape, src_elements.begin(), src_elements.end()};
        REQUIRE(cmp_equal(std::distance(result_tensor.begin(),result_tensor.end()), expected_elements.size()));
        REQUIRE(std::equal(result_tensor.begin(),result_tensor.end(),expected_elements.begin()));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_init_list_shape_range","[test_tensor]"){
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    using helpers_for_testing::cmp_equal;
    using gtensor::detail::shape_to_str;

    std::vector<value_type> elements{1,2,3,4,5,6};
    std::vector<value_type> expected_elements(elements);
    tensor_type result_tensor(std::initializer_list<int>{2,3}, elements.begin(), elements.end());
    REQUIRE(cmp_equal(std::distance(result_tensor.begin(),result_tensor.end()), expected_elements.size()));
    REQUIRE(std::equal(result_tensor.begin(),result_tensor.end(),expected_elements.begin()));
}

TEST_CASE("test_tensor_construct_using_operator","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using size_type = typename config_type::size_type;
    using index_type = typename config_type::index_type;
    using helpers_for_testing::apply_by_element;
    //0tensor,1expected_shape,2expected_size,3expected_dim
    auto test_data = std::make_tuple(
        std::make_tuple((tensor_type{1}+tensor_type{1}), shape_type{1}, 1 , 1),
        std::make_tuple((tensor_type{1}+tensor_type{1,2,3}), shape_type{3}, 3 , 1),
        std::make_tuple((tensor_type{1,2,3}+tensor_type{1,2,3}), shape_type{3}, 3 , 1),
        std::make_tuple((tensor_type{{1,2,3}}+tensor_type{1,2,3}), shape_type{1,3}, 3 , 2),
        std::make_tuple((tensor_type{{1,2,3}}+tensor_type{{1},{2},{3}}), shape_type{3,3}, 9 , 2),
        std::make_tuple((tensor_type{{1,2,3},{4,5,6}}+tensor_type{{1},{2}}), shape_type{2,3}, 6 , 2),
        std::make_tuple((tensor_type{{{1,2,3},{4,5,6}}}+tensor_type{1,2,3}), shape_type{1,2,3}, 6 , 3),
        std::make_tuple((tensor_type{1}+tensor_type{1}+tensor_type{1}), shape_type{1}, 1 , 1),
        std::make_tuple((tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{1}), shape_type{3}, 3 , 1),
        std::make_tuple((tensor_type{1,2,3}+tensor_type{{1},{2},{3}}+tensor_type{1,2,3}), shape_type{3,3}, 9 , 2),
        std::make_tuple(((tensor_type{1,2,3}+(tensor_type{{1},{2},{3}})+(tensor_type{1,2,3})+tensor_type{1})), shape_type{3,3}, 9 , 2)
    );

    auto test = [](auto& t){
        auto ten = std::get<0>(t);
        auto expected_shape = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_dim = std::get<3>(t);
        REQUIRE(ten.shape() == expected_shape);
        REQUIRE(ten.size() == static_cast<index_type>(expected_size));
        REQUIRE(ten.dim() == static_cast<size_type>(expected_dim));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_view_making_interface","[test_tensor]"){
    using value_type = double;
    using gtensor::tensor;
    using config_type = gtensor::config::default_config;
    using tensor_type = tensor<value_type, config_type>;
    using nop_type = typename gtensor::slice_traits<config_type>::nop_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using slices_container_type = typename gtensor::slice_traits<config_type>::slices_container_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using size_type = typename config_type::size_type;
    using gtensor::subscript_exception;
    using gtensor::broadcast_exception;
    nop_type nop;
    SECTION("test_subscripts_correctenes_check"){
        SECTION("view_slice_slice_direction_interface"){
            REQUIRE_NOTHROW(tensor_type{1}({},0));
            REQUIRE_THROWS_AS(tensor_type{1}({},1),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1}({0,1},0));
            REQUIRE_THROWS_AS(tensor_type{1}({nop,2},0),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1,2,3,4,5}({},0));
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}({0,4,-1},0)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}({0,0},0)),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}({1,-1},0));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}({1,-1},1)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}({1,-1},2)),subscript_exception);
        }
        SECTION("view_slice_init_list_interface"){
            REQUIRE_NOTHROW(tensor_type{1}({}));
            REQUIRE_THROWS_AS((tensor_type{1}({{},{}})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1}({{0,1}}));
            REQUIRE_THROWS_AS((tensor_type{1}({{nop,2}})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1,2,3,4,5}({}));
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}({{0,4,-1}})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}({{0,0}})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}({{1,-1}}));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}({{1,-1},{1,-1}})),subscript_exception);
        }
        SECTION("view_slice_variadic_interface"){
            REQUIRE_NOTHROW(tensor_type{1}(slice_type{}));
            REQUIRE_THROWS_AS((tensor_type{1}(slice_type{},slice_type{})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1}(slice_type{0,1}));
            REQUIRE_THROWS_AS((tensor_type{1}(slice_type{nop,2})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1,2,3,4,5}(slice_type{}));
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}(slice_type{0,4,-1})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}(slice_type{0,0})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}(slice_type{1,-1}));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}(slice_type{1,-1},slice_type{1,-1})),subscript_exception);
        }
        SECTION("view_slice_container_interface"){
            REQUIRE_NOTHROW(tensor_type{1}(slices_container_type{slice_type{}}));
            REQUIRE_THROWS_AS((tensor_type{1}(slices_container_type{slice_type{},slice_type{}})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1}(slices_container_type{slice_type{0,1}}));
            REQUIRE_THROWS_AS((tensor_type{1}(slices_container_type{slice_type{nop,2}})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{1,2,3,4,5}(slices_container_type{slice_type{}}));
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}(slices_container_type{slice_type{0,4,-1}})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}(slices_container_type{slice_type{0,0}})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}(slices_container_type{slice_type{1,-1}}));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}(slices_container_type{slice_type{1,-1},slice_type{1,-1}})),subscript_exception);
        }
        SECTION("view_transpose"){
            REQUIRE_NOTHROW(tensor_type{1}.transpose());
            REQUIRE_NOTHROW(tensor_type{1}.transpose(0));
            REQUIRE_THROWS_AS((tensor_type{1}.transpose(1)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1}.transpose(0,1)),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4}}.transpose());
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4}}.transpose(0,1));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4}}.transpose(1,0));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4}}.transpose(0)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4}}.transpose(1,1)),subscript_exception);
        }
        SECTION("view_subdim"){
            REQUIRE_NOTHROW(tensor_type{1}());
            REQUIRE_THROWS_AS((tensor_type{1}(0)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1,2,3,4,5}(0)),subscript_exception);
            REQUIRE_NOTHROW((tensor_type{{{1,2},{3,4},{5,6}}}(0)));
            REQUIRE_NOTHROW((tensor_type{{{1,2},{3,4},{5,6}}}(0,0)));
            REQUIRE_NOTHROW((tensor_type{{{1,2},{3,4},{5,6}}}(0,2)));
            REQUIRE_THROWS_AS((tensor_type{{{1,2},{3,4},{5,6}}}(1)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{{1,2},{3,4},{5,6}}}(0,3)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{{1,2},{3,4},{5,6}}}(0,2,0)),subscript_exception);
        }
        SECTION("view_reshape_variadic"){
            REQUIRE_NOTHROW(tensor_type{1}.reshape());
            REQUIRE_NOTHROW(tensor_type{1}.reshape(1));
            REQUIRE_THROWS_AS((tensor_type{1}.reshape(0)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1}.reshape(2)),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape());
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(6));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(6,1));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(1,6,1));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(2,3));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(3,2));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}.reshape(10)),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}.reshape(3,3)),subscript_exception);
        }
        SECTION("view_reshape_container"){
            REQUIRE_NOTHROW(tensor_type{1}.reshape(std::vector<int>{}));
            REQUIRE_NOTHROW(tensor_type{1}.reshape(std::vector<int>{1}));
            REQUIRE_THROWS_AS((tensor_type{1}.reshape(std::vector<int>{0})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{1}.reshape(std::vector<std::size_t>{2})),subscript_exception);
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape({}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape({6}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape({6,1}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(std::vector<int>{1,6,1}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(std::vector<int>{2,3}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}.reshape(std::vector<int>{3,2}));
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}.reshape(std::vector<int>{10})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}.reshape(std::vector<int>{3,3})),subscript_exception);
        }
        SECTION("mapping_view_index_tensor"){
            using index_tensor_type = tensor<index_type, config_type>;
            //nothrow, empty parent, empty subs
            REQUIRE_NOTHROW(tensor_type{}(index_tensor_type{}));
            REQUIRE_NOTHROW(tensor_type{}(index_tensor_type{}.reshape(2,3,0)));
            REQUIRE_NOTHROW(tensor_type{}.reshape(1,0)(index_tensor_type{0},index_tensor_type{}));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{1},index_tensor_type{2},index_tensor_type{}.reshape(0,2,2)));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{0,1},index_tensor_type{0,2},index_tensor_type{}.reshape(0,2,2)));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{{0,1}},index_tensor_type{{0,2}},index_tensor_type{}.reshape(0,2,2)));
            //nothrow, empty parent, not empty subs
            REQUIRE_NOTHROW(tensor_type{}.reshape(1,0)(index_tensor_type{0}));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{1}));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{{0,1},{1,0}}));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{1},index_tensor_type{0,1,2,0,1,2}));
            //empty result, no fill, no exception even if subs out of bounds
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{1},index_tensor_type{4},index_tensor_type{}));
            //nothrow, not empty parent, empty subs
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{}));
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{}.reshape(2,3,0)));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1}, index_tensor_type{}.reshape(0,2,2)));
            //nothrow, not empty parent, not empty subs
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{0}));
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{0,0,0}));
            REQUIRE_NOTHROW(tensor_type{1,2}(index_tensor_type{{0,1},{1,0}}));
            REQUIRE_NOTHROW(tensor_type{1,2}(index_tensor_type{{{1,0},{0,0}},{{0,1},{1,1}}}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,0,1}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1}, index_tensor_type{1,2}));
            //exception, parent zero size direction and non zero size subs
            REQUIRE_THROWS_AS(tensor_type{}(index_tensor_type{1}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{}.reshape(2,3,0)(index_tensor_type{1},index_tensor_type{2},index_tensor_type{0}),subscript_exception);
            //exception, subs number more than parent dim
            REQUIRE_THROWS_AS(tensor_type{}(index_tensor_type{},index_tensor_type{}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{0},index_tensor_type{0,0,0}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{0,1},index_tensor_type{0,1}),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1},index_tensor_type{1,1},index_tensor_type{})),subscript_exception);
            //exception, subs shapes not broadcast
            REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,0},index_tensor_type{0,0,0})),broadcast_exception);
            //exception, subs out of bounds
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{0,4,0}),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{3})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0},index_tensor_type{1,2,3})),subscript_exception);
        }
        SECTION("mapping_view_bool_tensor"){
            using index_tensor_type = tensor<bool,config_type>;
            REQUIRE_NOTHROW(tensor_type{}(index_tensor_type{}));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{}));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{true,true}));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{{true,true,true},{false,false,false}}));
            REQUIRE_NOTHROW(tensor_type{}.reshape(2,3,0)(index_tensor_type{}.reshape(2,3,0)));
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{}.reshape(2,0)));
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{true}));
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{false}));
            REQUIRE_NOTHROW(tensor_type{1,2,3,4,5}(index_tensor_type{false,false,true,true,false}));
            REQUIRE_NOTHROW(tensor_type{1,2,3,4,5}(index_tensor_type{true}));
            REQUIRE_NOTHROW(tensor_type{1,2,3,4,5}(index_tensor_type{false,false,true}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{true}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{true,false,true}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{{true}}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{{true,false},{false,true}}));
            REQUIRE_NOTHROW(tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{{true,false}}));
            //exception, subs dim > parent dim
            REQUIRE_THROWS_AS(tensor_type{}(index_tensor_type{}.reshape(0,0)),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{}.reshape(2,3,0)(index_tensor_type{}.reshape(1,2,3,0)),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{{true}}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{{false}}),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{{{true}}})),subscript_exception);
            //exception, subs out of bounds
            REQUIRE_THROWS_AS(tensor_type{}(index_tensor_type{true}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{}.reshape(1,0)(index_tensor_type{{true}}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{}.reshape(2,3,0)(index_tensor_type{}.reshape(3,3,0)),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{true,true}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{false,false}),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{{true,false,true}})),subscript_exception);
        }
    }
    SECTION("test_slices_filling_and_result_view"){
        using index_tensor_type = tensor<index_type, config_type>;
        using bool_tensor_type = tensor<bool, config_type>;
        using helpers_for_testing::apply_by_element;
        //0view,1expected_shape,2expected size,3expected dim
        auto test_data = std::make_tuple(
            //view slice slice-direction interface
            std::make_tuple(tensor_type{1}({},0),shape_type{1},1,1),
            std::make_tuple(tensor_type{1}({0,1},0),shape_type{1},1,1),
            std::make_tuple(tensor_type{1,2,3,4,5}({},0),shape_type{5},5,1),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6}}({1,-1},0),shape_type{1,2},2,2),
            std::make_tuple(tensor_type{{1,2},{3,4},{5,6}}({{},{},-1},0),shape_type{3,2},6,2),
            //view slice init_list interface
            std::make_tuple((tensor_type{1}({{}})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}({{nop,nop,-1}})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}({{}})),shape_type{5}, 5, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}({{nop,nop,2}})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}({{nop,nop,-2}})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}({{},{},{0,-1}})),shape_type{1,3,1}, 3, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}({{},{1,-1},{0,-1}})),shape_type{1,1,1}, 1, 3),
            //view slice variadic interface
            std::make_tuple((tensor_type{1}(slice_type{})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}(slice_type{nop,nop,-1})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}(slice_type{})),shape_type{5}, 5, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}(slice_type{nop,nop,2})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}(slice_type{nop,nop,-2})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}(slice_type{},slice_type{},slice_type{0,-1})),shape_type{1,3,1}, 3, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}(slice_type{},slice_type{1,-1},slice_type{0,-1})),shape_type{1,1,1}, 1, 3),
            //view slice container interface
            std::make_tuple((tensor_type{1}(slices_container_type{slice_type{}})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}(slices_container_type{slice_type{nop,nop,-1}})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}(slices_container_type{slice_type{}})),shape_type{5}, 5, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}(slices_container_type{slice_type{nop,nop,2}})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}(slices_container_type{slice_type{nop,nop,-2}})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}(slices_container_type{slice_type{},slice_type{},slice_type{0,-1}})),shape_type{1,3,1}, 3, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}(slices_container_type{slice_type{},slice_type{1,-1},slice_type{0,-1}})),shape_type{1,1,1}, 1, 3),
            //view transpose
            std::make_tuple((tensor_type{1}.transpose()),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}.transpose(0)),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}.transpose()),shape_type{5}, 5, 1),
            std::make_tuple((tensor_type{{1,2,3,4,5}}.transpose()),shape_type{5,1}, 5, 2),
            std::make_tuple((tensor_type{{1,2,3,4,5}}.transpose(1,0)),shape_type{5,1}, 5, 2),
            std::make_tuple((tensor_type{{1,2,3,4,5}}.transpose(1,0)),shape_type{5,1}, 5, 2),
            std::make_tuple((tensor_type{{1,2,3,4,5}}.transpose(0,1)),shape_type{1,5}, 5, 2),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.transpose()),shape_type{2,3,1}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.transpose(0,2,1)),shape_type{1,2,3}, 6, 3),
            //view subdim
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}()),shape_type{1,3,2}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}(0)),shape_type{3,2}, 6, 2),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}(0,1)),shape_type{2}, 2, 1),
            //view reshape variadic
            std::make_tuple((tensor_type{1}.reshape()),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}.reshape(1)),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape()),shape_type{1,3,2}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape(6)),shape_type{6}, 6, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape(2,1,3)),shape_type{2,1,3}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape(6,1)),shape_type{6,1}, 6, 2),
            //view reshape container
            std::make_tuple((tensor_type{1}.reshape(std::vector<int>{})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}.reshape({1})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape({})),shape_type{1,3,2}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape(std::vector<int>{6})),shape_type{6}, 6, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape({2,1,3})),shape_type{2,1,3}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape(std::vector<size_t>{6,1})),shape_type{6,1}, 6, 2),
            //mapping view index tensor
            std::make_tuple((tensor_type{}(index_tensor_type{})),shape_type{0}, 0, 1),
            std::make_tuple((tensor_type{}(index_tensor_type{}.reshape(2,3,0))),shape_type{2,3,0}, 0, 3),
            std::make_tuple((tensor_type{}.reshape(1,0)(index_tensor_type{0})),shape_type{1,0}, 0, 2),
            std::make_tuple((tensor_type{}.reshape(1,0)(index_tensor_type{0,0,0})),shape_type{3,0}, 0, 2),
            std::make_tuple((tensor_type{}.reshape(1,0)(index_tensor_type{0},index_tensor_type{})),shape_type{0}, 0, 1),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(index_tensor_type{0,1,0,1,0})),shape_type{5,3,0}, 0, 3),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(index_tensor_type{0,1},index_tensor_type{2})),shape_type{2,0}, 0, 2),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(index_tensor_type{0,1},index_tensor_type{2},index_tensor_type{}.reshape(0,3,1))),shape_type{0,3,2}, 0, 3),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(index_tensor_type{{0,1}},index_tensor_type{{0,2}},index_tensor_type{}.reshape(0,3,1))),shape_type{0,3,2}, 0, 3),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(index_tensor_type{{0,1}},index_tensor_type{4},index_tensor_type{}.reshape(0,3,1))),shape_type{0,3,2}, 0, 3),
            std::make_tuple((tensor_type{1}(index_tensor_type{0})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}(index_tensor_type{0,0,0})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0})),shape_type{1,3}, 3, 2),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1,1,0})),shape_type{4,3}, 12, 2),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1}, index_tensor_type{0})),shape_type{2}, 2, 1),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{{0,0},{1,1}}, index_tensor_type{{0,2},{0,2}})),shape_type{2,2}, 4, 2),
            //mapping view bool tensor
            std::make_tuple((tensor_type{}(bool_tensor_type{})),shape_type{0}, 0, 1),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(bool_tensor_type{})),shape_type{0,3,0}, 0, 3),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(bool_tensor_type{}.reshape(2,0))),shape_type{0,0}, 0, 2),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(bool_tensor_type{}.reshape(2,3,0))),shape_type{0}, 0, 1),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(bool_tensor_type{false,false})),shape_type{0,3,0}, 0, 3),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(bool_tensor_type{true,false})),shape_type{1,3,0}, 0, 3),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(bool_tensor_type{true,true})),shape_type{2,3,0}, 0, 3),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(bool_tensor_type{{true,true,false},{false,true,true}})),shape_type{4,0}, 0, 2),
            std::make_tuple((tensor_type{}.reshape(2,3,0)(bool_tensor_type{}.reshape(2,3,0))),shape_type{0}, 0, 1),
            std::make_tuple((tensor_type{1}(bool_tensor_type{})),shape_type{0}, 0, 1),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(bool_tensor_type{})),shape_type{0,3}, 0, 2),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(bool_tensor_type{}.reshape(2,0))),shape_type{0}, 0, 1),
            std::make_tuple((tensor_type{1}(bool_tensor_type{false})),shape_type{0}, 0, 1),
            std::make_tuple((tensor_type{1}(bool_tensor_type{true})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}(bool_tensor_type{false,true,false,true,false})),shape_type{2}, 2, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}(bool_tensor_type{true,true,true,true})),shape_type{4}, 4, 1),
            std::make_tuple((tensor_type{{1,2},{3,4},{5,6}}(bool_tensor_type{true})),shape_type{1,2}, 2, 2),
            std::make_tuple((tensor_type{{1,2},{3,4},{5,6}}(bool_tensor_type{{true}})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{{1,2},{3,4},{5,6}}(bool_tensor_type{true,true})),shape_type{2,2}, 4, 2),
            std::make_tuple((tensor_type{{1,2},{3,4},{5,6}}(bool_tensor_type{{true},{true}})),shape_type{2}, 2, 1),
            std::make_tuple((tensor_type{{1,2},{3,4},{5,6}}(bool_tensor_type{{false,true},{false,false},{false,true}})),shape_type{2}, 2, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{true,true})),shape_type{2,2,2}, 8, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{{true},{true}})),shape_type{2,2}, 4, 2)
        );
        auto test = [](auto& t){
            auto view = std::get<0>(t);
            auto expected_shape = std::get<1>(t);
            auto expected_size = std::get<2>(t);
            auto expected_dim = std::get<3>(t);
            REQUIRE(view.shape() == expected_shape);
            REQUIRE(view.size() == static_cast<index_type>(expected_size));
            REQUIRE(view.dim() == static_cast<size_type>(expected_dim));
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_tensor_equals","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using test_type = std::tuple<bool,bool,bool>;
    using gtensor::equals;
    auto t = tensor_type{{1,2,3},{4,5,6}};
    auto v = t(1);
    //0result equals,1result equals, 2expected equals
    auto test_data = GENERATE_COPY(
        test_type{equals(tensor_type{1,2,3},tensor_type{1,2,3}),tensor_type{1,2,3}.equals(tensor_type{1,2,3}),true},
        test_type{equals(t,t),t.equals(t),true},
        test_type{equals(v,v),v.equals(v),true},
        test_type{equals(v,t),v.equals(t),false},
        test_type{equals(tensor_type{{1,2,3}},tensor_type{1,2,3}),tensor_type{{1,2,3}}.equals(tensor_type{1,2,3}),false},
        test_type{equals(tensor_type{1,2,3},tensor_type{{1,2,3}}),tensor_type{1,2,3}.equals(tensor_type{{1,2,3}}),false},
        test_type{equals(tensor_type{1,2,3},tensor_type{0,1,2}),tensor_type{1,2,3}.equals(tensor_type{0,1,2}),false},
        test_type{equals(tensor_type{1,2,3}({{}}),tensor_type{1,2,3}),tensor_type{1,2,3}({{}}).equals(tensor_type{1,2,3}),true},
        test_type{equals(tensor_type{1,2,3}({{{},{},-1}}),tensor_type{1,2,3}),tensor_type{1,2,3}({{{},{},-1}}).equals(tensor_type{1,2,3}),false},
        test_type{equals(tensor_type{1,2,3}({{{},{},2}}).reshape(1,2).transpose(),tensor_type{{1},{3}}),tensor_type{1,2,3}({{{},{},2}}).reshape(1,2).transpose().equals(tensor_type{{1},{3}}),true}
    );

    auto result_equals = std::get<0>(test_data);
    auto result_equals_member = std::get<1>(test_data);
    auto expected_equals = std::get<2>(test_data);
    REQUIRE(result_equals == expected_equals);
    REQUIRE(result_equals_member == expected_equals);
}

TEST_CASE("test_tensor_copy","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using test_type = std::tuple<tensor_type,tensor_type>;
    //0result tensor,expected tensor
    auto test_data = GENERATE(
        test_type{tensor_type{{1,2,3},{4,5,6}}.copy(),tensor_type{{1,2,3},{4,5,6}}},
        test_type{(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1} + tensor_type{0,1,2}).copy(),tensor_type{{1,3,5},{4,6,8}}},
        test_type{tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).copy(),tensor_type{{5,6},{2,3}}},
        test_type{tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose().copy(),tensor_type{{5,2},{6,3}}},
        test_type{(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1})({{{},{},-1},{1}}).transpose().copy(),tensor_type{{5,2},{6,3}}},
        test_type{(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1}).copy(),tensor_type{{5,2},{6,3}}},
        test_type{((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1})).reshape(4).copy(),tensor_type{5,2,6,3}}
    );

    auto result_tensor = std::get<0>(test_data);
    auto expected_tensor = std::get<1>(test_data);
    REQUIRE(result_tensor.equals(expected_tensor));
}
