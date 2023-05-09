#include <tuple>
#include <vector>
#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "integral_type.hpp"

TEST_CASE("test_tensor_default_constructor","[test_tensor]")
{
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

TEST_CASE("test_0-dim_tensor_constructor","[test_tensor]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = tensor_type::config_type;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0value
    auto test_data = std::make_tuple(
        0,
        1.0f,
        std::size_t{2},
        std::int64_t{3}
    );
    auto test = [](const auto& value){
        const shape_type expected_shape{};
        const index_type expected_size{1};
        const dim_type expected_dim{0};
        const value_type expected_value = static_cast<value_type>(value);
        auto result_tensor = tensor_type(value);
        auto result_shape = result_tensor.shape();
        auto result_size = result_tensor.size();
        auto result_dim = result_tensor.dim();
        auto result_value = *result_tensor.begin();
        REQUIRE(result_shape == expected_shape);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_value == expected_value);
        REQUIRE(std::next(result_tensor.begin()) == result_tensor.end());
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_constructor_from_list","[test_tensor]")
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
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
        REQUIRE(static_cast<index_type>(std::distance(result.begin(),result.end())) == static_cast<index_type>(std::distance(expected_elements.begin(),expected_elements.end())));
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin()));
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_operator==","[test_tensor]")
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using helpers_for_testing::apply_by_element;
    //0tensor0,1tensor1,2expected
    auto test_data = std::make_tuple(
        //equal
        std::make_tuple(tensor_type(2),tensor_type(2),true),
        std::make_tuple(tensor_type{},tensor_type{},true),
        std::make_tuple(tensor_type{1},tensor_type{1},true),
        std::make_tuple(tensor_type{1,2,3},tensor_type{1,2,3},true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}},true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},true),
        //not equal
        std::make_tuple(tensor_type(1),tensor_type(2),false),
        std::make_tuple(tensor_type{},tensor_type{1},false),
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
            auto result = ten0 == ten0;
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
            auto result = ten0 == ten1;
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
            auto result = ten1 == ten0;
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_tensor_constructor_shape","[test_tensor]")
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
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
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_tensor_constructor_shape_container_value","[test_tensor]")
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
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
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_shape_container_range","[test_tensor]")
{
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
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_shape_init_list_value","[test_tensor]")
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
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
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_constructor_shape_init_list_range","[test_tensor]")
{
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
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}


TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_result","[test_tensor]",
    //0lhs_value_type,1rhs_value_type
    (std::tuple<int,int>),
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        //rhs scalar
        std::make_tuple(lhs_tensor_type{},1,lhs_tensor_type(1),1),
        std::make_tuple(lhs_tensor_type(1),2,lhs_tensor_type(2),2),
        std::make_tuple(lhs_tensor_type{1,2,3},4,lhs_tensor_type(4),4),
        //rhs tensor
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(3)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{4,5},{6,7}}),
        //rhs view
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}.reshape(2,3,0),lhs_tensor_type{}.reshape(2,3,0),rhs_tensor_type{}.reshape(2,3,0)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(4).transpose(),lhs_tensor_type(4),rhs_tensor_type(4)),
        std::make_tuple(lhs_tensor_type{{1,2},{3,4}},rhs_tensor_type{{5,6},{7,8}}(1,1),lhs_tensor_type(8),rhs_tensor_type(8)),
        //rhs expression
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}+rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{1,2,3}+rhs_tensor_type{4,5,6}+rhs_tensor_type(1),lhs_tensor_type{6,8,10},rhs_tensor_type{6,8,10})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto& result = lhs = rhs;
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_value_semantic","[test_tensor]",
//0lhs_value_type,1rhs_value_type
    (std::tuple<int,int>),
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    //0lhs,1rhs,2expected_lhs
    auto test_data = std::make_tuple(
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        lhs = rhs;
        *rhs.begin() = -1;
        REQUIRE(lhs == expected_lhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_move_assignment_result","[test_tensor]",
//0lhs_value_type,1rhs_value_type
    (std::tuple<int,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    REQUIRE(std::is_same_v<lhs_value_type,rhs_value_type>);
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{1}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{1}),
        std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{{1},{2},{3}}})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto& result = lhs = std::move(rhs);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_converting_move_assignment_result","[test_tensor]",
//0lhs_value_type,1rhs_value_type
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    REQUIRE(!std::is_same_v<lhs_value_type,rhs_value_type>);
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(3)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
        std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{4,5},{6,7}}),
        //rhs view
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}.reshape(2,3,0),lhs_tensor_type{}.reshape(2,3,0),rhs_tensor_type{}.reshape(2,3,0)),
        std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(4).transpose(),lhs_tensor_type(4),rhs_tensor_type(4)),
        std::make_tuple(lhs_tensor_type{{1,2},{3,4}},rhs_tensor_type{{5,6},{7,8}}(1,1),lhs_tensor_type(8),rhs_tensor_type(8)),
        //rhs expression
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}+rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type{},rhs_tensor_type{1,2,3}+rhs_tensor_type{4,5,6}+rhs_tensor_type(1),lhs_tensor_type{6,8,10},rhs_tensor_type{6,8,10})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto& result = lhs = std::move(rhs);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_assignment_corner_cases","[test_tensor]")
{
    using helpers_for_testing::apply_by_element;

    struct assign_exception{};
    struct throw_on_assign{
        throw_on_assign() = default;
        throw_on_assign(const throw_on_assign&) = default;
        throw_on_assign& operator=(const throw_on_assign&){
            throw assign_exception{};
        }
    };
    REQUIRE_THROWS(throw_on_assign{} = throw_on_assign{});
    SECTION("copy_assign_to_same_tensor_view")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        auto test_data = std::make_tuple(
            tensor_type({10},value_type{}),
            tensor_type({10},value_type{}).transpose()
        );
        auto test = [](auto lhs){
            const auto ptr_to_first_expected = &(*lhs.begin());
            REQUIRE_NOTHROW(lhs.operator=(lhs));    //no assignment
            const auto ptr_to_first_result = &(*lhs.begin());
            REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
        };
        apply_by_element(test,test_data);
    }
    SECTION("move_assign_to_same_tensor")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        auto test_data = std::make_tuple(
            tensor_type({10},value_type{}),
            tensor_type({10},value_type{}).transpose()
        );
        auto test = [](auto lhs){
            const auto ptr_to_first_expected = &(*lhs.begin());
            REQUIRE_NOTHROW(lhs.operator=(std::move(lhs)));    //no assignment
            const auto ptr_to_first_result = &(*lhs.begin());
            REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
        };
        apply_by_element(test,test_data);
    }
    SECTION("move_assign")
    {
        using value_type = throw_on_assign;
        using tensor_type = gtensor::tensor<value_type>;
        tensor_type lhs({10},value_type{});
        tensor_type rhs({10},value_type{});
        const auto ptr_to_first_expected = &(*rhs.begin());
        REQUIRE_NOTHROW(lhs.operator=(std::move(rhs)));    //no assignment
        const auto ptr_to_first_result = &(*lhs.begin());
        REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //swap impl
    }
    SECTION("copy_assign_convert_copy_assign_to_tensor_same_shape")
    {
        using gtensor::tensor;
        //0lhs,1rhs,2expected_lhs,3expected_rhs
        auto test_data = std::make_tuple(
            std::make_tuple(tensor<int>(1),tensor<int>(2),tensor<int>(2),tensor<int>(2)),
            std::make_tuple(tensor<double>(1),tensor<int>(2),tensor<double>(2),tensor<int>(2)),
            std::make_tuple(tensor<int>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}}),
            std::make_tuple(tensor<double>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}}+0,tensor<double>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}})
        );
        auto test = [](const auto& t){
            auto lhs = std::get<0>(t);
            auto rhs = std::get<1>(t);
            auto expected_lhs = std::get<2>(t);
            auto expected_rhs = std::get<3>(t);
            const auto ptr_to_first_expected = &(*lhs.begin());
            lhs.operator=(rhs);
            const auto ptr_to_first_result = &(*lhs.begin());
            REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
            REQUIRE(lhs == expected_lhs);
            REQUIRE(rhs == expected_rhs);
        };
        apply_by_element(test,test_data);
    }
    SECTION("convert_move_assign_to_tensor_same_shape")
    {
        using gtensor::tensor;
        //0lhs,1rhs,2expected_lhs,3expected_rhs
        auto test_data = std::make_tuple(
            std::make_tuple(tensor<double>(1),tensor<int>(2),tensor<double>(2),tensor<int>(2)),
            std::make_tuple(tensor<double>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}}+0,tensor<double>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}})
        );
        auto test = [](const auto& t){
            auto lhs = std::get<0>(t);
            auto rhs = std::get<1>(t);
            auto expected_lhs = std::get<2>(t);
            auto expected_rhs = std::get<3>(t);
            const auto ptr_to_first_expected = &(*lhs.begin());
            lhs.operator=(std::move(rhs));
            const auto ptr_to_first_result = &(*lhs.begin());
            REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
            REQUIRE(lhs == expected_lhs);
            REQUIRE(rhs == expected_rhs);
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_lhs_is_view","[test_tensor]",
    (std::tuple<int,int>),
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using helpers_for_testing::apply_by_element;
    //0parent,1lhs_view_maker,2rhs,3expected_parent,4expected_lhs,5expected_rhs
    auto test_data = std::make_tuple(
        //rhs scalar
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},1,lhs_tensor_type{},lhs_tensor_type{},1),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},2,lhs_tensor_type(2),lhs_tensor_type(2),2),
        std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},7,lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7},7),
        //rhs 0-dim
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type(1),lhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type(1)),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type(2),lhs_tensor_type(2),lhs_tensor_type(2),rhs_tensor_type(2)),
        std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},rhs_tensor_type(7),lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7},rhs_tensor_type(7)),
        //rhs n-dim
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type{1},lhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{1}),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type{2},lhs_tensor_type(2),lhs_tensor_type(2),rhs_tensor_type{2}),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1},
            lhs_tensor_type{{{1,2},{-1,1}},{{5,6},{-1,1}}},
            lhs_tensor_type{{{-1,1}},{{-1,1}}},
            rhs_tensor_type{-1,1}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1},
            lhs_tensor_type{{{1,5},{-1,1}},{{2,6},{-1,1}}},
            lhs_tensor_type{{{-1,1}},{{-1,1}}},
            rhs_tensor_type{-1,1}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1} + rhs_tensor_type{{1},{2}},
            lhs_tensor_type{{{1,5},{1,3}},{{2,6},{1,3}}},
            lhs_tensor_type{{{1,3}},{{1,3}}},
            rhs_tensor_type{{0,2},{1,3}}
        )
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto lhs_view_maker = std::get<1>(t);
        auto rhs = std::get<2>(t);
        auto expected_parent = std::get<3>(t);
        auto expected_lhs = std::get<4>(t);
        auto expected_rhs = std::get<5>(t);
        auto lhs = lhs_view_maker(parent);
        auto& result = lhs = rhs;
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(parent == expected_parent);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_move_assignment_converting_move_assignment_lhs_is_view","[test_tensor]",
    (std::tuple<int,int>),
    (std::tuple<double,int>)
)
{
    using lhs_value_type = std::tuple_element_t<0,TestType>;
    using rhs_value_type = std::tuple_element_t<1,TestType>;
    using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
    using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
    using gtensor::assign;
    using helpers_for_testing::apply_by_element;
    //0parent,1lhs_view_maker,2rhs,3expected_parent,4expected_lhs
    auto test_data = std::make_tuple(
        //rhs 0-dim
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type(1),lhs_tensor_type{},lhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type(2),lhs_tensor_type(2),lhs_tensor_type(2)),
        std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},rhs_tensor_type(7),lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7}),
        //rhs n-dim
        std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type{1},lhs_tensor_type{},lhs_tensor_type{}),
        std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type{2},lhs_tensor_type(2),lhs_tensor_type(2)),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1},
            lhs_tensor_type{{{1,2},{-1,1}},{{5,6},{-1,1}}},
            lhs_tensor_type{{{-1,1}},{{-1,1}}}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1},
            lhs_tensor_type{{{1,5},{-1,1}},{{2,6},{-1,1}}},
            lhs_tensor_type{{{-1,1}},{{-1,1}}}
        ),
        std::make_tuple(
            lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            rhs_tensor_type{-1,1} + rhs_tensor_type{{1},{2}},
            lhs_tensor_type{{{1,5},{1,3}},{{2,6},{1,3}}},
            lhs_tensor_type{{{1,3}},{{1,3}}}
        )
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto lhs_view_maker = std::get<1>(t);
        auto rhs = std::get<2>(t);
        auto expected_parent = std::get<3>(t);
        auto expected_lhs = std::get<4>(t);
        auto lhs = lhs_view_maker(parent);
        auto& result = lhs = std::move(rhs);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(parent == expected_parent);
        REQUIRE(rhs.empty());
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_is_same","[test_tensor]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;

    const auto t = tensor_type{1,2,3};
    const auto e = tensor_type{1,2,3}+tensor_type{0,0,0};
    const auto v = tensor_type{1,2,3}.transpose().transpose();
    SECTION("test_not_same_tensor")
    {
        auto test_data = std::make_tuple(
            std::make_tuple(t,e,v),
            std::make_tuple(e,t,v),
            std::make_tuple(v,e,t)
        );
        auto test = [](const auto& t){
            auto first = std::get<0>(t);
            auto second = std::get<1>(t);
            auto third = std::get<2>(t);
            REQUIRE(!first.is_same(second));
            REQUIRE(!first.is_same(third));
            REQUIRE(!second.is_same(first));
            REQUIRE(!second.is_same(third));
            REQUIRE(!third.is_same(first));
            REQUIRE(!third.is_same(second));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_not_same_other"){
        REQUIRE(!t.is_same(1));
        REQUIRE(!e.is_same(1.0));
        REQUIRE(!v.is_same(std::string{}));
    }
    SECTION("test_same_tensor")
    {
        auto test_data = std::make_tuple(t,e,v);
        auto test = [](const auto& t){
            REQUIRE(t.is_same(t));
            auto t_ref_copy = t;
            REQUIRE(&t != &t_ref_copy);
            REQUIRE(t.is_same(t_ref_copy));
            auto t_ref_move = std::move(t_ref_copy);
            REQUIRE(t.is_same(t_ref_move));
            REQUIRE(!t.is_same(t_ref_copy));
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_tensor_resize_to_not_bigger","[test_tensor]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //0tensor,1new_shape,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},std::vector<int>{0},tensor_type{}),
        std::make_tuple(tensor_type{},std::array<int,3>{0,2,3},tensor_type{}.reshape(0,2,3)),
        std::make_tuple(tensor_type(1),std::vector<int>{0},tensor_type{}),
        std::make_tuple(tensor_type(2),std::vector<int>{},tensor_type(2)),
        std::make_tuple(tensor_type(1),std::vector<int>{1,1},tensor_type{{1}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{0},tensor_type{}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{},tensor_type(1)),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{3},tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,3},tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,2,2},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto new_shape = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto ten_size = ten.size();
        auto expected_size = expected.size();
        REQUIRE(ten_size >= expected_size);
        ten.resize(new_shape);
        REQUIRE(ten == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_resize_to_bigger","[test_tensor]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using helpers_for_testing::apply_by_element;
    const value_type any{-1};
    //0tensor,1size,2new_shape,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},index_type{0},std::vector<int>{1},tensor_type{any}),
        std::make_tuple(tensor_type{},index_type{0},std::vector<int>{5},tensor_type{any,any,any,any,any}),
        std::make_tuple(tensor_type{},index_type{0},std::vector<int>{2,3},tensor_type{{any,any,any},{any,any,any}}),
        std::make_tuple(tensor_type(3),index_type{1},std::vector<int>{5},tensor_type{3,any,any,any,any}),
        std::make_tuple(tensor_type(3),index_type{1},std::vector<int>{3,2},tensor_type{{3,any},{any,any},{any,any}}),
        std::make_tuple(tensor_type{4},index_type{1},std::vector<int>{5},tensor_type{4,any,any,any,any}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},index_type{6},std::vector<int>{2,2,2},tensor_type{{{1,2},{3,4}},{{5,6},{any,any}}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto ten_size = std::get<1>(t);
        auto new_shape = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto expected_size = expected.size();
        auto expected_shape = expected.shape();
        REQUIRE(ten_size == ten.size());
        REQUIRE(expected_size > ten_size);
        ten.resize(new_shape);
        auto result_shape = ten.shape();
        REQUIRE(result_shape == expected_shape);
        REQUIRE(std::equal(ten.begin(),ten.begin()+ten_size,expected.begin()));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_resize_init_list_interface","[test_tensor]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    tensor_type t0{{1,2,3},{4,5,6}};
    t0.resize({2,2});
    REQUIRE(t0 == tensor_type{{1,2},{3,4}});
    //t0.transpose().resize({2,2}); //can't resize view
}

// TEST_CASE("test_tensor_copy","[test_tensor]"){
//     using value_type = double;
//     using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
//     using tensor_type = gtensor::tensor<value_type, config_type>;
//     using test_type = std::tuple<tensor_type,tensor_type>;
//     //0result tensor,expected tensor
//     auto test_data = GENERATE(
//         test_type{tensor_type{{1,2,3},{4,5,6}}.copy(),tensor_type{{1,2,3},{4,5,6}}},
//         test_type{(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1} + tensor_type{0,1,2}).copy(),tensor_type{{1,3,5},{4,6,8}}},
//         test_type{tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).copy(),tensor_type{{5,6},{2,3}}},
//         test_type{tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose().copy(),tensor_type{{5,2},{6,3}}},
//         test_type{(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1})({{{},{},-1},{1}}).transpose().copy(),tensor_type{{5,2},{6,3}}},
//         test_type{(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1}).copy(),tensor_type{{5,2},{6,3}}},
//         test_type{((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1})).reshape(4).copy(),tensor_type{5,2,6,3}}
//     );

//     auto result_tensor = std::get<0>(test_data);
//     auto expected_tensor = std::get<1>(test_data);
//     REQUIRE(result_tensor.equals(expected_tensor));
// }
