#include <tuple>
#include <vector>
#include "catch.hpp"
#include "gtensor.hpp"
#include "helpers_for_testing.hpp"
#include "integral_type.hpp"

TEST_CASE("test_tensor_default_constructor","[test_tensor]"){
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = tensor_type::config_type;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;

    auto test_data = GENERATE(
        tensor_type(),
        tensor_type{}
    );
    REQUIRE(test_data.size() == index_type{0});
    REQUIRE(test_data.dim() == dim_type{1});
    REQUIRE(test_data.shape() == shape_type{0,});
}

TEST_CASE("test_0-dim_tensor_constructor","[test_tensor]"){
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = tensor_type::config_type;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;

    auto test_data = GENERATE(
        tensor_type(0),
        tensor_type(0.0f),
        tensor_type(std::size_t{1}),
        tensor_type(std::int64_t{1})
    );
    REQUIRE(test_data.size() == index_type{1});
    REQUIRE(test_data.dim() == dim_type{0});
    REQUIRE(test_data.shape() == shape_type{});
}

TEST_CASE("test_tensor_constructor_from_list","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0result,1expected_shape,2expected_size,3expected_dim,4expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, shape_type{0}, index_type{0} , dim_type{1}, std::vector<value_type>{}),
        std::make_tuple(tensor_type{1}, shape_type{1}, index_type{1} , dim_type{1}, std::vector<value_type>{1}),
        std::make_tuple(tensor_type{1,2,3}, shape_type{3}, index_type{3} , dim_type{1}, std::vector<value_type>{1,2,3}),
        std::make_tuple(tensor_type{{1}}, shape_type{1,1}, index_type{1} , dim_type{2}, std::vector<value_type>{1}),
        std::make_tuple(tensor_type{{1,2,3}}, shape_type{1,3}, index_type{3} , dim_type{2}, std::vector<value_type>{1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, shape_type{2,3}, index_type{6} , dim_type{2}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{{1,2,3,4}}}, shape_type{1,1,4}, index_type{4} , dim_type{3}, std::vector<value_type>{1,2,3,4}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4}}}, shape_type{1,4,1}, index_type{4} , dim_type{3}, std::vector<value_type>{1,2,3,4}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}}, shape_type{1,4,3}, index_type{12} , dim_type{3}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected_shape = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_dim = std::get<3>(t);
        auto expected_elements = std::get<4>(t);
        REQUIRE(result.shape() == expected_shape);
        REQUIRE(result.size() == expected_size);
        REQUIRE(result.dim() == expected_dim);
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin()));
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_constructor_shape","[test_tensor]"){
    using value_type = int;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0shape,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{}, tensor_type(value_type{})),
        std::make_tuple(shape_type{0}, tensor_type()),
        std::make_tuple(std::vector<int>{}, tensor_type(value_type{})),
        std::make_tuple(std::vector<int>{1}, tensor_type{value_type{}}),
        std::make_tuple(shape_type{3}, tensor_type{value_type{},value_type{},value_type{}}),
        std::make_tuple(std::vector<int>{2,3}, tensor_type{{value_type{},value_type{},value_type{}},{value_type{},value_type{},value_type{}}})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto expected = std::get<1>(t);
        tensor_type result(shape);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_constructor_shape_container_value","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0shape,1value,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{},1,tensor_type(1)),
        std::make_tuple(shape_type{},-1,tensor_type(-1)),
        std::make_tuple(shape_type{0},1,tensor_type{}),
        std::make_tuple(shape_type{1},1,tensor_type{1}),
        std::make_tuple(shape_type{5},2,tensor_type{2,2,2,2,2}),
        std::make_tuple(shape_type{1,1},2,tensor_type{{2}}),
        std::make_tuple(shape_type{1,3},0,tensor_type{{0,0,0}}),
        std::make_tuple(std::vector<std::size_t>{2,3},0,tensor_type{{0,0,0},{0,0,0}}),
        std::make_tuple(std::vector<int>{1,2,3},3,tensor_type{{{3,3,3},{3,3,3}}})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto value = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = tensor_type{shape, value};
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_shape_container_range","[test_tensor]"){
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = typename tensor_type::config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    using helpers_for_testing::cmp_equal;
    using gtensor::detail::shape_to_str;
    //0shape,1elements,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{}, std::vector<value_type>{}, tensor_type(value_type{})),
        std::make_tuple(shape_type{}, std::vector<value_type>{1,2,3,4,5}, tensor_type(value_type{1})),
        std::make_tuple(shape_type{0}, std::vector<value_type>{}, tensor_type{}),
        std::make_tuple(shape_type{0}, std::vector<value_type>{1,2,3,4,5}, tensor_type{}),
        std::make_tuple(shape_type{3}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3}),
        std::make_tuple(shape_type{8}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3,4,5,value_type{},value_type{},value_type{}}),
        std::make_tuple(shape_type{5}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3,4,5}),
        std::make_tuple(std::vector<int>{2,3}, std::vector<value_type>{1,2,3,4,5}, tensor_type{{1,2,3},{4,5,value_type{}}}),
        std::make_tuple(std::vector<std::size_t>{2,2}, std::vector<value_type>{1,2,3,4,5}, tensor_type{{1,2},{3,4}})
    );
    auto test = [](auto& t){
        auto shape = std::get<0>(t);
        auto elements = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = tensor_type{shape, elements.begin(), elements.end()};
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_shape_init_list_value","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using helpers_for_testing::apply_by_element;
    //0result,1expected
    auto test_data = std::make_tuple(
        std::make_tuple([](){return tensor_type({},1);}(), tensor_type(value_type{1})),
        std::make_tuple([](){return tensor_type({},-1);}(), tensor_type(value_type{-1})),
        std::make_tuple([](){return tensor_type({0},1);}(), tensor_type{}),
        std::make_tuple([](){return tensor_type({1},1);}(), tensor_type{1}),
        std::make_tuple([](){return tensor_type({5},2);}(), tensor_type{2,2,2,2,2}),
        std::make_tuple([](){return tensor_type({1,1},2);}(), tensor_type{{2}}),
        std::make_tuple([](){return tensor_type({1,3},0);}(), tensor_type{{0,0,0}}),
        std::make_tuple([](){return tensor_type({2,3},1);}(), tensor_type{{1,1,1},{1,1,1}}),
        std::make_tuple([](){return tensor_type({1,2,3},1);}(), tensor_type{{{1,1,1},{1,1,1}}})
    );
    auto test = [](auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_shape_init_list_range","[test_tensor]"){
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //0result,1expected
    auto test_data = std::make_tuple(
        std::make_tuple([](){std::vector<value_type> elements{}; return tensor_type{{},elements.begin(),elements.end()};}(), tensor_type(value_type{})),
        std::make_tuple([](){std::vector<value_type> elements{3,2,1}; return tensor_type({},elements.begin(),elements.end());}(), tensor_type(value_type{3})),
        std::make_tuple([](){std::vector<value_type> elements{}; return tensor_type{{0},elements.begin(),elements.end()};}(), tensor_type{}),
        std::make_tuple([](){std::vector<value_type> elements{1,2,3}; return tensor_type({0},elements.begin(),elements.end());}(), tensor_type{}),
        std::make_tuple([](){std::vector<value_type> elements{1,2,3,4,5}; return tensor_type{{3},elements.begin(),elements.end()};}(), tensor_type{1,2,3}),
        std::make_tuple([](){std::vector<value_type> elements{1,2,3}; return tensor_type{{5},elements.begin(),elements.end()};}(), tensor_type{1,2,3,value_type{},value_type{}}),
        std::make_tuple([](){std::vector<value_type> elements{1,2,3,4,5}; return tensor_type{{5},elements.begin(),elements.end()};}(), tensor_type{1,2,3,4,5}),
        std::make_tuple([](){std::vector<value_type> elements{1,2,3,4,5}; return tensor_type({2,3},elements.begin(),elements.end());}(), tensor_type{{1,2,3},{4,5,value_type{}}}),
        std::make_tuple([](){std::vector<value_type> elements{1,2,3,4,5}; return tensor_type{{2,2},elements.begin(),elements.end()};}(), tensor_type{{1,2},{3,4}})
    );
    auto test = [](auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_construct_using_operator","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using shape_type = typename config_type::shape_type;
    using dim_type = typename config_type::dim_type;
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
        REQUIRE(ten.dim() == static_cast<dim_type>(expected_dim));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_equals","[test_tensor]"){
    using value_type = double;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using gtensor::equals;
    using helpers_for_testing::apply_by_element;
    //0tensor0,1tensor1,2expected
    auto test_data = std::make_tuple(
        //equal
        std::make_tuple(tensor_type{},tensor_type{},true),
        std::make_tuple(tensor_type{1},tensor_type{1},true),
        std::make_tuple(tensor_type{1,2,3},tensor_type{1,2,3},true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}},true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},true),
        std::make_tuple(tensor_type{1}.reshape(1,1),tensor_type{{1}},true),
        std::make_tuple(tensor_type{}.reshape(1,0),tensor_type{}.reshape(1,0),true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{},{1,-1}}),tensor_type{{2},{5}},true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),tensor_type{1,4,2,5,3,6}.reshape(3,2),true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(1,4,2),tensor_type{{{1,2},{3,4},{5,6},{7,8}}},true),
        //not equal
        std::make_tuple(tensor_type{},tensor_type{}.reshape(1,0),false),
        std::make_tuple(tensor_type{1},tensor_type{2},false),
        std::make_tuple(tensor_type{1,2,3},tensor_type{{1,2,3}},false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,2},{4,5,6}},false)
    );
    SECTION("ten0_equals_ten0")
    {
        auto test = [](const auto& t){
            auto ten0 = std::get<0>(t);
            auto ten1 = std::get<1>(t);
            auto expected = true;
            auto result = ten0.equals(ten0);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("ten0_equals_ten1")
    {
        auto test = [](const auto& t){
            auto ten0 = std::get<0>(t);
            auto ten1 = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = ten0.equals(ten1);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
    SECTION("ten1_equals_ten0")
    {
        auto test = [](const auto& t){
            auto ten0 = std::get<0>(t);
            auto ten1 = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto result = ten1.equals(ten0);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
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
