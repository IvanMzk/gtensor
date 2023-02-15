#include <tuple>
#include <vector>
#include "catch.hpp"
#include "gtensor.hpp"
#include "helpers_for_testing.hpp"

namespace test_tensor_{

    class C{};

    template<typename ValT, typename CfgT>
    class test_tensor : public gtensor::tensor<ValT, CfgT>
    {
        using base_tensor = gtensor::tensor<ValT, CfgT>;
        using base_tensor::tensor;
    };
}

TEST_CASE("test_is_iterator","[test_tensor]"){
    using gtensor::detail::is_iterator;
    REQUIRE(!is_iterator<int>);
    REQUIRE(!is_iterator<std::vector<int>>);
    REQUIRE(is_iterator<std::vector<int>::iterator>);
}

TEST_CASE("test_is_tensor","[test_tensor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using test_tensor_type = test_tensor_::test_tensor<value_type,config_type>;
    using gtensor::detail::is_index;
    using gtensor::detail::is_bool;
    using gtensor::detail::is_index_tensor;
    using gtensor::detail::is_bool_tensor;
    using gtensor::detail::is_tensor;

    REQUIRE(is_tensor<tensor_type>::value);
    REQUIRE(is_tensor<decltype(tensor_type{1,2,3} + tensor_type{1,2,3})>::value);
    REQUIRE(is_tensor<test_tensor_type>::value);
    REQUIRE(is_tensor<decltype(test_tensor_type{1,2,3} + test_tensor_type{1,2,3})>::value);
    REQUIRE(!is_tensor<std::string>::value);
    REQUIRE(!is_tensor<std::vector<int>>::value);

    REQUIRE(is_index<int,int>::value);
    REQUIRE(is_index<std::size_t,int>::value);
    REQUIRE(is_index<float,int>::value);
    REQUIRE(!is_index<bool,int>::value);
    REQUIRE(!is_index<bool,std::int64_t>::value);
    REQUIRE(is_bool<bool>::value);
    REQUIRE(!is_bool<int>::value);
    REQUIRE(!is_bool<std::vector<bool>>::value);

    REQUIRE(is_index_tensor<gtensor::tensor<int, config_type>,int>::value);
    REQUIRE(is_index_tensor<gtensor::tensor<std::size_t, config_type>,std::int64_t>::value);
    REQUIRE(is_index_tensor<gtensor::tensor<float, config_type>,std::int64_t>::value);
    REQUIRE(is_index_tensor<test_tensor_::test_tensor<int, config_type>,int>::value);
    REQUIRE(is_index_tensor<test_tensor_::test_tensor<std::size_t, config_type>,std::int64_t>::value);
    REQUIRE(is_index_tensor<test_tensor_::test_tensor<float, config_type>,std::int64_t>::value);
    REQUIRE(!is_index_tensor<test_tensor_::test_tensor<bool, config_type>,std::int64_t>::value);
    REQUIRE(!is_index_tensor<test_tensor_::C,std::int64_t>::value);
    REQUIRE(!is_index_tensor<gtensor::tensor<bool, config_type>,std::int64_t>::value);
    REQUIRE(!is_index_tensor<gtensor::tensor<bool, config_type>,int>::value);
    REQUIRE(!is_index_tensor<std::vector<int>,int>::value);
    REQUIRE(!is_index_tensor<std::string,int>::value);
    REQUIRE(!is_index_tensor<std::vector<bool>,int>::value);

    REQUIRE(is_bool_tensor<gtensor::tensor<bool, config_type>>::value);
    REQUIRE(is_bool_tensor<test_tensor_::test_tensor<bool, config_type>>::value);
    REQUIRE(!is_bool_tensor<gtensor::tensor<int, config_type>>::value);
    REQUIRE(!is_bool_tensor<gtensor::tensor<float, config_type>>::value);
    REQUIRE(!is_bool_tensor<test_tensor_::test_tensor<int, config_type>>::value);
    REQUIRE(!is_bool_tensor<test_tensor_::test_tensor<float, config_type>>::value);
    REQUIRE(!is_bool_tensor<test_tensor_::C>::value);
    REQUIRE(!is_bool_tensor<std::vector<int>>::value);
    REQUIRE(!is_bool_tensor<std::string>::value);
    REQUIRE(!is_bool_tensor<std::vector<bool>>::value);
}

TEST_CASE("test_tensor_default_constructor","[test_tensor]"){
    using value_type = float;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = typename tensor_type::config_type::shape_type;

    auto test_data = GENERATE(
        tensor_type(),
        tensor_type{}
    );
    REQUIRE(test_data.size() == 0);
    REQUIRE(test_data.shape() == shape_type{});
}

TEST_CASE("test_tensor_constructor_from_list","[test_tensor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<tensor_type, shape_type, index_type, index_type>;
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
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<tensor_type, shape_type, index_type, index_type>;
    //tensor,expected_shape,expected size,expected dim
    auto test_data = GENERATE(
        test_type(tensor_type(shape_type{},1.0f), shape_type{}, 0 , 0),
        test_type(tensor_type(shape_type{1},1.0f), shape_type{1}, 1 , 1),
        test_type(tensor_type({1},1.0f), shape_type{1}, 1 , 1),
        test_type(tensor_type(shape_type{10},1.0f), shape_type{10}, 10 , 1),
        test_type(tensor_type(shape_type{1,1},1.0f), shape_type{1,1}, 1 , 2),
        test_type(tensor_type(shape_type{1,3},1.0f), shape_type{1,3}, 3 , 2),
        test_type(tensor_type(shape_type{2,3},1.0f), shape_type{2,3}, 6 , 2),
        test_type(tensor_type(std::vector<int>{1,1,4},1.0f), shape_type{1,1,4}, 4 , 3),
        test_type(tensor_type(shape_type{1,4,1},1.0f), shape_type{1,4,1}, 4 , 3),
        test_type(tensor_type(shape_type{1,4,3},1.0f), shape_type{1,4,3}, 12 , 3),
        test_type(tensor_type({1,4,3},1.0f), shape_type{1,4,3}, 12 , 3),
        test_type(tensor_type({1,4,3},1.0f), shape_type{1,4,3}, 12 , 3)

    );

    auto t = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_size = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
}

TEST_CASE("test_tensor_constructor_shape_range","[test_tensor]"){
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = typename tensor_type::config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0shape,1src_elements,2expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(std::initializer_list<int>{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(std::vector<std::size_t>{2,2,2}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10}, std::vector<value_type>{1,2,3,4,5,6,7,8})
    );
    auto test = [](auto& t){
        auto shape = std::get<0>(t);
        auto src_elements = std::get<1>(t);
        auto expected_elements = std::get<2>(t);
        tensor_type result_tensor{shape, src_elements.begin(), src_elements.end()};
        //REQUIRE(result_tensor.shape() == shape);
        REQUIRE(std::distance(result_tensor.begin(),result_tensor.end()) == expected_elements.size());
        REQUIRE(std::equal(result_tensor.begin(),result_tensor.end(),expected_elements.begin()));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_construct_using_operator","[test_tensor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
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
        REQUIRE(ten.size() == expected_size);
        REQUIRE(ten.dim() == expected_dim);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_view_making_interface","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using config_type = gtensor::config::default_config;
    using tensor_type = tensor<value_type, config_type>;
    using slice_type = typename config_type::slice_type;
    using nop_type = typename config_type::nop_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::subscript_exception;
    using gtensor::broadcast_exception;
    nop_type nop;
    SECTION("test_subscripts_correctenes_check"){
        SECTION("view_slice"){
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
        SECTION("view_reshape"){
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
        SECTION("mapping_view_index_tensor"){
            using index_tensor_type = tensor<index_type, config_type>;
            using index_tensor_type1 = tensor<std::size_t, config_type>;
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{0}));
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type1{0}));
            REQUIRE_NOTHROW(tensor_type{1}(tensor_type{0}));
            REQUIRE_NOTHROW(tensor_type{1}(index_tensor_type{0,0,0}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,0,1}));
            REQUIRE_NOTHROW(tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1}, index_tensor_type{1,2}));
            REQUIRE_THROWS_AS(tensor_type{1}(tensor_type{0,0,0},tensor_type{0,0}),broadcast_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(tensor_type{0,1},tensor_type{0,1}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(tensor_type{0,1},tensor_type{1}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(tensor_type{0,4,0}),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{3})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0},index_tensor_type{1,2,3})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0},index_tensor_type{1,2,3},index_tensor_type{0})),subscript_exception);
        }
        SECTION("mapping_view_bool_tensor"){
            using index_tensor_type = tensor<bool,config_type>;
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
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{{true}}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{{false}}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{true,true}),subscript_exception);
            REQUIRE_THROWS_AS(tensor_type{1}(index_tensor_type{false,false}),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{{{true}}})),subscript_exception);
            REQUIRE_THROWS_AS((tensor_type{{1,2},{3,4},{5,6}}(index_tensor_type{{true,false,true}})),subscript_exception);
        }
    }
    SECTION("test_slices_filling_and_result_view"){
        using index_tensor_type = tensor<index_type, config_type>;
        using bool_tensor_type = tensor<bool, config_type>;
        using helpers_for_testing::apply_by_element;
        //0view,1expected_shape,2expected size,3expected dim
        auto test_data = std::make_tuple(
            //view slice
            std::make_tuple((tensor_type{1}({{}})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}({{nop,nop,-1}})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}({{}})),shape_type{5}, 5, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}({{nop,nop,2}})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}({{nop,nop,-2}})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}({{},{},{0,-1}})),shape_type{1,3,1}, 3, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}({{},{1,-1},{0,-1}})),shape_type{1,1,1}, 1, 3),
            //view transpo)e
            std::make_tuple((tensor_type{1}.transpose()),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}.transpose(0)),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1,2,3,4,5}.transpose()),shape_type{5}, 5, 1),
            std::make_tuple((tensor_type{{1,2,3,4,5}}.transpose()),shape_type{5,1}, 5, 2),
            std::make_tuple((tensor_type{{1,2,3,4,5}}.transpose(1,0)),shape_type{5,1}, 5, 2),
            std::make_tuple((tensor_type{{1,2,3,4,5}}.transpose(1,0)),shape_type{5,1}, 5, 2),
            std::make_tuple((tensor_type{{1,2,3,4,5}}.transpose(0,1)),shape_type{1,5}, 5, 2),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.transpose()),shape_type{2,3,1}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.transpose(0,2,1)),shape_type{1,2,3}, 6, 3),
            //view subd)m
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}()),shape_type{1,3,2}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}(0)),shape_type{3,2}, 6, 2),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}(0,1)),shape_type{2}, 2, 1),
            //view resha)e
            std::make_tuple((tensor_type{1}.reshape()),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}.reshape(1)),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape()),shape_type{1,3,2}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape(6)),shape_type{6}, 6, 1),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape(2,1,3)),shape_type{2,1,3}, 6, 3),
            std::make_tuple((tensor_type{{{1,2},{3,4},{5,6}}}.reshape(6,1)),shape_type{6,1}, 6, 2),
            //mapping view index tens)r
            std::make_tuple((tensor_type{1}(index_tensor_type{0})),shape_type{1}, 1, 1),
            std::make_tuple((tensor_type{1}(index_tensor_type{0,0,0})),shape_type{3}, 3, 1),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0})),shape_type{1,3}, 3, 2),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1,1,0})),shape_type{4,3}, 12, 2),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{0,1}, index_tensor_type{0})),shape_type{2}, 2, 1),
            std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(index_tensor_type{{0,0},{1,1}}, index_tensor_type{{0,2},{0,2}})),shape_type{2,2}, 4, 2),
            //mapping view bool tens)r
            std::make_tuple((tensor_type{1}(bool_tensor_type{false})),shape_type{}, 0, 0),
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
            REQUIRE(view.size() == expected_size);
            REQUIRE(view.dim() == expected_dim);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_tensor_equals","[test_tensor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
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
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
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

