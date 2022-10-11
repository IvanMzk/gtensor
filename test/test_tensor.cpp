#include "catch.hpp"
#include "gtensor.hpp"
#include <tuple>
#include <vector>

namespace test_tensor_{

    class C{};

    template<typename ValT, typename CfgT>
    class test_tensor : public gtensor::tensor<ValT, CfgT>
    {
        using base_tensor = gtensor::tensor<ValT, CfgT>;
        using base_tensor::tensor;
    };
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

TEST_CASE("test_tensor_construct_from_list","[test_tensor]"){
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
    REQUIRE(t.as_htensor().as_htensor().shape() == expected_shape);
    REQUIRE(t.as_htensor().size() == expected_size);
    REQUIRE(t.as_htensor().dim() == expected_dim);
    REQUIRE(static_cast<typename tensor_type::htensor_type>(t).shape() == expected_shape);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
}

TEST_CASE("test_tensor_construct_given_shape","[test_tensor]"){
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
        test_type(tensor_type(shape_type{10},1.0f), shape_type{10}, 10 , 1),
        test_type(tensor_type(shape_type{1,1},1.0f), shape_type{1,1}, 1 , 2),
        test_type(tensor_type(shape_type{1,3},1.0f), shape_type{1,3}, 3 , 2),
        test_type(tensor_type(shape_type{2,3},1.0f), shape_type{2,3}, 6 , 2),
        test_type(tensor_type(std::vector<int>{1,1,4},1.0f), shape_type{1,1,4}, 4 , 3),
        test_type(tensor_type(shape_type{1,4,1},1.0f), shape_type{1,4,1}, 4 , 3),
        test_type(tensor_type(shape_type{1,4,3},1.0f), shape_type{1,4,3}, 12 , 3)
    );

    auto t = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_size = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
    REQUIRE(t.as_htensor().shape() == expected_shape);
    REQUIRE(t.as_htensor().size() == expected_size);
    REQUIRE(t.as_htensor().dim() == expected_dim);
}

TEST_CASE("test_tensor_construct_using_operator","[test_tensor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using htensor_type = typename tensor_type::htensor_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<htensor_type, shape_type, index_type, index_type>;
    //0htensor,1expected_shape,2expected size,3expected dim
    auto test_data = GENERATE(
        test_type((tensor_type{1}+tensor_type{1}).as_htensor(), shape_type{1}, 1 , 1),
        test_type(static_cast<htensor_type>(tensor_type{1}+tensor_type{1,2,3}), shape_type{3}, 3 , 1),
        test_type(static_cast<htensor_type>(tensor_type{1,2,3}+tensor_type{1,2,3}), shape_type{3}, 3 , 1),
        test_type(static_cast<htensor_type>(tensor_type{{1,2,3}}+tensor_type{1,2,3}), shape_type{1,3}, 3 , 2),
        test_type(static_cast<htensor_type>(tensor_type{{1,2,3}}+tensor_type{{1},{2},{3}}), shape_type{3,3}, 9 , 2),
        test_type(static_cast<htensor_type>(tensor_type{{1,2,3},{4,5,6}}+tensor_type{{1},{2}}), shape_type{2,3}, 6 , 2),
        test_type(static_cast<htensor_type>(tensor_type{{{1,2,3},{4,5,6}}}+tensor_type{1,2,3}), shape_type{1,2,3}, 6 , 3),
        test_type(static_cast<htensor_type>(tensor_type{1}+tensor_type{1}+tensor_type{1}) ,shape_type{1}, 1 , 1),
        test_type(static_cast<htensor_type>(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{1}) ,shape_type{3}, 3 , 1),
        test_type(static_cast<htensor_type>(tensor_type{1,2,3}+tensor_type{{1},{2},{3}}+tensor_type{1,2,3}) ,shape_type{3,3}, 9 , 2),
        test_type(static_cast<htensor_type>((tensor_type{1,2,3}+(tensor_type{{1},{2},{3}})+(tensor_type{1,2,3})+tensor_type{1})) ,shape_type{3,3}, 9 , 2)
    );

    auto t = std::get<0>(test_data);
    auto expected_shape = std::get<1>(test_data);
    auto expected_size = std::get<2>(test_data);
    auto expected_dim = std::get<3>(test_data);
    REQUIRE(t.shape() == expected_shape);
    REQUIRE(t.size() == expected_size);
    REQUIRE(t.dim() == expected_dim);
}

TEST_CASE("test_view_making_interface","[test_tensor]"){
    using value_type = float;
    using gtensor::tensor;
    using config_type = gtensor::config::default_config;
    using tensor_type = tensor<value_type, config_type>;
    using htensor_type = typename tensor_type::htensor_type;
    using slice_type = typename config_type::slice_type;
    using nop_type = typename config_type::nop_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::subscript_exception;
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
    }
    SECTION("test_slices_filling_and_result_view"){
        using test_type = std::tuple<htensor_type, shape_type, index_type, index_type>;
        //0view,1expected_shape,2expected size,3expected dim
        auto test_data = GENERATE_COPY(
            test_type{static_cast<htensor_type>(tensor_type{1}({{}})),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1}({{nop,nop,-1}})),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1,2,3,4,5}({{}})),shape_type{5}, 5, 1},
            test_type{static_cast<htensor_type>(tensor_type{1,2,3,4,5}({{nop,nop,2}})),shape_type{3}, 3, 1},
            test_type{static_cast<htensor_type>(tensor_type{1,2,3,4,5}({{nop,nop,-2}})),shape_type{3}, 3, 1},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}({{},{},{0,-1}})),shape_type{1,3,1}, 3, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}({{},{1,-1},{0,-1}})),shape_type{1,1,1}, 1, 3},
            test_type{static_cast<htensor_type>(tensor_type{1}.transpose()),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1}.transpose(0)),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1,2,3,4,5}.transpose()),shape_type{5}, 5, 1},
            test_type{static_cast<htensor_type>(tensor_type{{1,2,3,4,5}}.transpose()),shape_type{5,1}, 5, 2},
            test_type{static_cast<htensor_type>(tensor_type{{1,2,3,4,5}}.transpose(1,0)),shape_type{5,1}, 5, 2},
            test_type{static_cast<htensor_type>(tensor_type{{1,2,3,4,5}}.transpose(1,0)),shape_type{5,1}, 5, 2},
            test_type{static_cast<htensor_type>(tensor_type{{1,2,3,4,5}}.transpose(0,1)),shape_type{1,5}, 5, 2},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.transpose()),shape_type{2,3,1}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(0,2,1)),shape_type{1,2,3}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}()),shape_type{1,3,2}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}(0)),shape_type{3,2}, 6, 2},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}(0,1)),shape_type{2}, 2, 1},
            test_type{static_cast<htensor_type>(tensor_type{1}.reshape()),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{1}.reshape(1)),shape_type{1}, 1, 1},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.reshape()),shape_type{1,3,2}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.reshape(6)),shape_type{6}, 6, 1},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.reshape(2,1,3)),shape_type{2,1,3}, 6, 3},
            test_type{static_cast<htensor_type>(tensor_type{{{1,2},{3,4},{5,6}}}.reshape(6,1)),shape_type{6,1}, 6, 2}
        );
        auto view = std::get<0>(test_data);
        auto expected_shape = std::get<1>(test_data);
        auto expected_size = std::get<2>(test_data);
        auto expected_dim = std::get<3>(test_data);
        REQUIRE(view.shape() == expected_shape);
        REQUIRE(view.size() == expected_size);
        REQUIRE(view.dim() == expected_dim);
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

TEST_CASE("test_tensor_construct_from_other_tensor","[test_tensor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<tensor_type,tensor_type>;
    //0result tensor,expected tensor
    auto test_data = GENERATE(
        test_type{tensor_type(tensor_type{{1,2,3},{4,5,6}}),tensor_type{{1,2,3},{4,5,6}}},
        test_type{tensor_type(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1} + tensor_type{0,1,2}),tensor_type{{1,3,5},{4,6,8}}},
        test_type{tensor_type(tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}})),tensor_type{{5,6},{2,3}}},
        test_type{tensor_type(tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose()),tensor_type{{5,2},{6,3}}},
        test_type{tensor_type((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1})({{{},{},-1},{1}}).transpose()),tensor_type{{5,2},{6,3}}},
        test_type{tensor_type((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1})),tensor_type{{5,2},{6,3}}},
        test_type{tensor_type(((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1})).reshape(4)),tensor_type{5,2,6,3}}
    );

    auto result_tensor = std::get<0>(test_data);
    auto expected_tensor = std::get<1>(test_data);
    REQUIRE(result_tensor.equals(expected_tensor));
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

TEST_CASE("test_tensor_combine_different_storages","[test_tensor]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using gtensor::detail::storage_engine_traits;
    using gtensor::storage_tensor;
    using trivial_type_vector::uvector;
    using tensor_vec_type = gtensor::tensor<value_type, config_type, storage_tensor<typename storage_engine_traits<typename config_type::engine,config_type,std::vector<value_type>>::type>>;
    using tensor_uvec_type = gtensor::tensor<value_type, config_type, storage_tensor<typename storage_engine_traits<typename config_type::engine,config_type,uvector<value_type>>::type>>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using test_type = std::tuple<tensor_uvec_type,tensor_vec_type>;
    //0result tensor,expected tensor
    auto test_data = GENERATE(
        test_type{tensor_uvec_type(tensor_vec_type{{1,2,3},{4,5,6}}),tensor_vec_type{{1,2,3},{4,5,6}}},
        test_type{tensor_uvec_type(tensor_uvec_type{-1} + tensor_uvec_type{{1,2,3},{4,5,6}} + tensor_vec_type{1} + tensor_vec_type{0,1,2}),tensor_uvec_type{{1,3,5},{4,6,8}}},
        test_type{tensor_vec_type(tensor_uvec_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}})),tensor_uvec_type{{5,6},{2,3}}},
        test_type{tensor_vec_type(tensor_uvec_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose()),tensor_vec_type{{5,2},{6,3}}},
        test_type{tensor_vec_type((tensor_vec_type{-1} + tensor_vec_type{{1,2,3},{4,5,6}} + tensor_uvec_type{1})({{{},{},-1},{1}}).transpose()),tensor_vec_type{{5,2},{6,3}}},
        test_type{tensor_uvec_type((tensor_vec_type{-1} + tensor_uvec_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_vec_type{1})),tensor_uvec_type{{5,2},{6,3}}},
        test_type{tensor_uvec_type(((tensor_vec_type{-1} + tensor_vec_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_vec_type{1})).reshape(4)),tensor_vec_type{5,2,6,3}}
    );

    auto result_tensor = std::get<0>(test_data);
    auto expected_tensor = std::get<1>(test_data);
    REQUIRE(result_tensor.equals(expected_tensor));
}