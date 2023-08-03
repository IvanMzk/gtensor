#include <tuple>
#include <vector>
#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

// //test tensor constructors
// TEST_CASE("test_tensor_default_constructor","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = tensor_type::config_type;
//     using dim_type = config_type::dim_type;
//     using index_type = config_type::index_type;
//     using shape_type = config_type::shape_type;

//     auto test_data = GENERATE(
//         tensor_type(),
//         tensor_type{}
//     );
//     REQUIRE(test_data.size() == index_type{0});
//     REQUIRE(test_data.dim() == dim_type{1});
//     REQUIRE(test_data.shape() == shape_type{0,});
// }

// TEST_CASE("test_0-dim_tensor_constructor","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = tensor_type::config_type;
//     using dim_type = config_type::dim_type;
//     using index_type = config_type::index_type;
//     using shape_type = config_type::shape_type;
//     using helpers_for_testing::apply_by_element;
//     //0value
//     auto test_data = std::make_tuple(
//         0,
//         1.0f
//     );
//     auto test = [](const auto& value){
//         const shape_type expected_shape{};
//         const index_type expected_size{1};
//         const dim_type expected_dim{0};
//         const value_type expected_value = static_cast<value_type>(value);
//         auto result_tensor = tensor_type(value);
//         auto result_shape = result_tensor.shape();
//         auto result_size = result_tensor.size();
//         auto result_dim = result_tensor.dim();
//         auto result_value = *result_tensor.begin();
//         REQUIRE(result_shape == expected_shape);
//         REQUIRE(result_size == expected_size);
//         REQUIRE(result_dim == expected_dim);
//         REQUIRE(result_value == expected_value);
//         auto result_first = result_tensor.begin();
//         auto result_last = result_tensor.end();
//         REQUIRE(result_first+1 == result_last);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_tensor_constructor_from_list","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using index_type = typename tensor_type::index_type;
//     using dim_type = typename tensor_type::dim_type;
//     using shape_type = typename tensor_type::shape_type;
//     using helpers_for_testing::apply_by_element;
//     //0result,1expected_shape,2expected_size,3expected_dim,4expected_elements
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{}, shape_type{0}, index_type{0} , dim_type{1}, std::vector<value_type>{}),
//         std::make_tuple(tensor_type{1}, shape_type{1}, index_type{1} , dim_type{1}, std::vector<value_type>{1}),
//         std::make_tuple(tensor_type{1,2,3}, shape_type{3}, index_type{3} , dim_type{1}, std::vector<value_type>{1,2,3}),
//         std::make_tuple(tensor_type{{1}}, shape_type{1,1}, index_type{1} , dim_type{2}, std::vector<value_type>{1}),
//         std::make_tuple(tensor_type{{1,2,3}}, shape_type{1,3}, index_type{3} , dim_type{2}, std::vector<value_type>{1,2,3}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, shape_type{2,3}, index_type{6} , dim_type{2}, std::vector<value_type>{1,2,3,4,5,6}),
//         std::make_tuple(tensor_type{{{1,2,3,4}}}, shape_type{1,1,4}, index_type{4} , dim_type{3}, std::vector<value_type>{1,2,3,4}),
//         std::make_tuple(tensor_type{{{1},{2},{3},{4}}}, shape_type{1,4,1}, index_type{4} , dim_type{3}, std::vector<value_type>{1,2,3,4}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}}, shape_type{1,4,3}, index_type{12} , dim_type{3}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected_shape = std::get<1>(t);
//         auto expected_size = std::get<2>(t);
//         auto expected_dim = std::get<3>(t);
//         auto expected_elements = std::get<4>(t);
//         REQUIRE(result.shape() == expected_shape);
//         REQUIRE(result.size() == expected_size);
//         REQUIRE(result.dim() == expected_dim);
//         REQUIRE(static_cast<index_type>(std::distance(result.begin(),result.end())) == static_cast<index_type>(std::distance(expected_elements.begin(),expected_elements.end())));
//         REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin()));
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_tensor_constructor_shape","[test_tensor]")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using shape_type = typename tensor_type::shape_type;
//     using helpers_for_testing::apply_by_element;
//     const value_type any{std::numeric_limits<value_type>::max()};
//     //0shape,1expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(shape_type{}, tensor_type(any)),
//         std::make_tuple(shape_type{0}, tensor_type()),
//         std::make_tuple(std::vector<int>{}, tensor_type(any)),
//         std::make_tuple(std::vector<int>{1}, tensor_type{any}),
//         std::make_tuple(shape_type{3}, tensor_type{any,any,any}),
//         std::make_tuple(std::vector<int>{2,3}, tensor_type{{any,any,any},{any,any,any}})
//     );
//     auto test = [](const auto& t){
//         auto shape = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         tensor_type result(shape);
//         REQUIRE(result.shape() == expected.shape());
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_tensor_constructor_shape_container_value","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using shape_type = typename tensor_type::shape_type;
//     using helpers_for_testing::apply_by_element;
//     //0shape,1value,2expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(shape_type{},value_type{1},tensor_type(1)),
//         std::make_tuple(shape_type{},value_type{-1},tensor_type(-1)),
//         std::make_tuple(shape_type{0},value_type{1},tensor_type{}),
//         std::make_tuple(shape_type{1},value_type{1},tensor_type{1}),
//         std::make_tuple(shape_type{5},value_type{2},tensor_type{2,2,2,2,2}),
//         std::make_tuple(shape_type{1,1},value_type{2},tensor_type{{2}}),
//         std::make_tuple(shape_type{1,3},value_type{0},tensor_type{{0,0,0}}),
//         std::make_tuple(std::vector<std::size_t>{2,3},value_type{0},tensor_type{{0,0,0},{0,0,0}}),
//         std::make_tuple(std::vector<int>{1,2,3},value_type{3},tensor_type{{{3,3,3},{3,3,3}}})
//     );
//     auto test = [](const auto& t){
//         auto shape = std::get<0>(t);
//         auto value = std::get<1>(t);
//         auto expected = std::get<2>(t);
//         auto result = tensor_type{shape, value};
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_constructor_shape_init_list_value","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using index_type = tensor_type::index_type;
//     using helpers_for_testing::apply_by_element;
//     auto make_result = [](std::initializer_list<index_type> shape, const value_type& v){
//         return tensor_type(shape,v);
//     };
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(make_result({},1), tensor_type(value_type{1})),
//         std::make_tuple(make_result({},-1), tensor_type(value_type{-1})),
//         std::make_tuple(make_result({0},1), tensor_type{}),
//         std::make_tuple(make_result({1},1), tensor_type{1}),
//         std::make_tuple(make_result({5},2), tensor_type{2,2,2,2,2}),
//         std::make_tuple(make_result({1,1},2), tensor_type{{2}}),
//         std::make_tuple(make_result({1,3},0), tensor_type{{0,0,0}}),
//         std::make_tuple(make_result({2,3},1), tensor_type{{1,1,1},{1,1,1}}),
//         std::make_tuple(make_result({1,2,3},1), tensor_type{{{1,1,1},{1,1,1}}})
//     );
//     auto test = [](auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_constructor_shape_container_range","[test_tensor]")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using shape_type = tensor_type::shape_type;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using helpers_for_testing::apply_by_element;
//     SECTION("range_>=_size")
//     {
//         //0layout,1shape,2elements,3expected
//         auto test_data = std::make_tuple(
//             //c_order layout
//             //0-dim
//             std::make_tuple(c_order{},shape_type{}, std::vector<value_type>{1,2,3,4,5}, tensor_type(value_type{1})),
//             //n-dim
//             std::make_tuple(c_order{},shape_type{0}, std::vector<value_type>{}, tensor_type{}),
//             std::make_tuple(c_order{},shape_type{0}, std::vector<value_type>{1,2,3,4,5}, tensor_type{}),
//             std::make_tuple(c_order{},shape_type{3}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3}),
//             std::make_tuple(c_order{},shape_type{5}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3,4,5}),
//             std::make_tuple(c_order{},std::vector<std::size_t>{2,2}, std::vector<value_type>{1,2,3,4,5}, tensor_type{{1,2},{3,4}}),
//             //f_order layout
//             //0-dim
//             std::make_tuple(f_order{},shape_type{}, std::vector<value_type>{1,2,3,4,5}, tensor_type(value_type{1})),
//             //n-dim
//             std::make_tuple(f_order{},shape_type{0}, std::vector<value_type>{}, tensor_type{}),
//             std::make_tuple(f_order{},shape_type{0}, std::vector<value_type>{1,2,3,4,5}, tensor_type{}),
//             std::make_tuple(f_order{},shape_type{3}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3}),
//             std::make_tuple(f_order{},shape_type{5}, std::vector<value_type>{1,2,3,4,5}, tensor_type{1,2,3,4,5}),
//             std::make_tuple(f_order{},std::vector<std::size_t>{2,2}, std::vector<value_type>{1,2,3,4,5}, tensor_type{{1,3},{2,4}})
//         );
//         auto test = [](auto& t){
//             auto layout = std::get<0>(t);
//             auto shape = std::get<1>(t);
//             auto elements = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             using layout_type = decltype(layout);
//             using result_tensor_type = gtensor::tensor<value_type,layout_type>;
//             auto result = result_tensor_type{shape, elements.begin(), elements.end()};
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("range_<_size")
//     {
//         using index_type = tensor_type::index_type;
//         const value_type any{std::numeric_limits<value_type>::max()};

//         //0layout,1shape,2elements,3range_size,4expected
//         auto test_data = std::make_tuple(
//             //c_order layout
//             //0-dim
//             std::make_tuple(c_order{},shape_type{}, std::vector<value_type>{}, index_type{0}, tensor_type(any)),
//             //n-dim
//             std::make_tuple(c_order{},shape_type{8}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
//             std::make_tuple(c_order{},std::vector<int>{2,3}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{{1,2,3},{4,5,any}}),
//             //f_order layout
//             //0-dim
//             std::make_tuple(f_order{},shape_type{}, std::vector<value_type>{}, index_type{0}, tensor_type(any)),
//             //n-dim
//             std::make_tuple(f_order{},shape_type{8}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
//             std::make_tuple(f_order{},std::vector<int>{2,3}, std::vector<value_type>{1,2,3,4,5}, index_type{5}, tensor_type{{1,3,5},{2,4,any}})
//         );
//         auto test = [](auto& t){
//             auto layout = std::get<0>(t);
//             auto shape = std::get<1>(t);
//             auto elements = std::get<2>(t);
//             auto range_size = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto expected_shape = expected.shape();
//             using layout_type = decltype(layout);
//             using result_tensor_type = gtensor::tensor<value_type,layout_type>;
//             auto result = result_tensor_type{shape, elements.begin(), elements.end()};
//             auto result_size = result.size();
//             auto result_shape = result.shape();
//             REQUIRE(range_size < result_size);
//             REQUIRE(result_shape == expected_shape);
//             REQUIRE(std::equal(result.begin(),result.begin()+range_size,expected.begin()));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEST_CASE("test_tensor_constructor_shape_init_list_range","[test_tensor]")
// {
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using index_type = tensor_type::index_type;
//     using helpers_for_testing::apply_by_element;
//     auto make_result = [](auto layout, std::initializer_list<index_type> shape, const std::vector<value_type>& elements){
//         using layout_type = decltype(layout);
//         using result_tensor_type = gtensor::tensor<value_type,layout_type>;
//         return result_tensor_type(shape,elements.begin(),elements.end());
//     };
//     SECTION("range_>=_size")
//     {
//         //0result,1expected
//         auto test_data = std::make_tuple(
//             //c_order layout
//             //0-dim
//             std::make_tuple(make_result(c_order{},{},std::vector<value_type>{1,2,3,4,5}), tensor_type(value_type{1})),
//             //n-dim
//             std::make_tuple(make_result(c_order{},{0},std::vector<value_type>{}), tensor_type{}),
//             std::make_tuple(make_result(c_order{},{0},std::vector<value_type>{1,2,3,4,5}), tensor_type{}),
//             std::make_tuple(make_result(c_order{},{3},std::vector<value_type>{1,2,3,4,5}), tensor_type{1,2,3}),
//             std::make_tuple(make_result(c_order{},{5},std::vector<value_type>{1,2,3,4,5}), tensor_type{1,2,3,4,5}),
//             std::make_tuple(make_result(c_order{},{2,2},std::vector<value_type>{1,2,3,4,5}), tensor_type{{1,2},{3,4}}),
//             //f_order layout
//             //0-dim
//             std::make_tuple(make_result(f_order{},{},std::vector<value_type>{1,2,3,4,5}), tensor_type(value_type{1})),
//             //n-dim
//             std::make_tuple(make_result(f_order{},{0},std::vector<value_type>{}), tensor_type{}),
//             std::make_tuple(make_result(f_order{},{0},std::vector<value_type>{1,2,3,4,5}), tensor_type{}),
//             std::make_tuple(make_result(f_order{},{3},std::vector<value_type>{1,2,3,4,5}), tensor_type{1,2,3}),
//             std::make_tuple(make_result(f_order{},{5},std::vector<value_type>{1,2,3,4,5}), tensor_type{1,2,3,4,5}),
//             std::make_tuple(make_result(f_order{},{2,2},std::vector<value_type>{1,2,3,4,5}), tensor_type{{1,3},{2,4}})
//         );
//         auto test = [](auto& t){
//             auto result = std::get<0>(t);
//             auto expected = std::get<1>(t);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("range_<_size")
//     {
//         using index_type = typename tensor_type::index_type;
//         const value_type any{-1};
//         //0result,1range_size,2expected
//         auto test_data = std::make_tuple(
//             //c_order layout
//             //0-dim
//             std::make_tuple(make_result(c_order{},{},std::vector<value_type>{}), index_type{0}, tensor_type(any)),
//             //n-dim
//             std::make_tuple(make_result(c_order{},{8},std::vector<value_type>{1,2,3,4,5}), index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
//             std::make_tuple(make_result(c_order{},{2,3},std::vector<value_type>{1,2,3,4,5}), index_type{5}, tensor_type{{1,2,3},{4,5,any}}),
//             //f_order layout
//             //0-dim
//             std::make_tuple(make_result(f_order{},{},std::vector<value_type>{}), index_type{0}, tensor_type(any)),
//             //n-dim
//             std::make_tuple(make_result(f_order{},{8},std::vector<value_type>{1,2,3,4,5}), index_type{5}, tensor_type{1,2,3,4,5,any,any,any}),
//             std::make_tuple(make_result(f_order{},{2,3},std::vector<value_type>{1,2,3,4,5}), index_type{5}, tensor_type{{1,3,5},{2,4,any}})
//         );
//         auto test = [](auto& t){
//             auto result = std::get<0>(t);
//             auto range_size = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto expected_shape = expected.shape();
//             auto result_size = result.size();
//             auto result_shape = result.shape();
//             REQUIRE(range_size < result_size);
//             REQUIRE(result_shape == expected_shape);
//             REQUIRE(std::equal(result.begin(),result.begin()+range_size,expected.begin()));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// //test tensor operator==
// TEST_CASE("test_tensor_operator==","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using helpers_for_testing::apply_by_element;
//     //0tensor0,1tensor1,2expected
//     auto test_data = std::make_tuple(
//         //equal
//         std::make_tuple(tensor_type(2),tensor_type(2),true),
//         std::make_tuple(tensor_type{},tensor_type{},true),
//         std::make_tuple(tensor_type{1},tensor_type{1},true),
//         std::make_tuple(tensor_type{1,2,3},tensor_type{1,2,3},true),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}},true),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},true),
//         //not equal
//         std::make_tuple(tensor_type(1),tensor_type(2),false),
//         std::make_tuple(tensor_type{},tensor_type{1},false),
//         std::make_tuple(tensor_type{1},tensor_type{2},false),
//         std::make_tuple(tensor_type{1,2,3},tensor_type{{1,2,3}},false),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,2},{4,5,6}},false)
//     );
//     SECTION("ten0_equals_ten0")
//     {
//         auto test = [](const auto& t){
//             auto ten0 = std::get<0>(t);
//             auto ten1 = std::get<1>(t);
//             auto expected = true;
//             auto result = ten0 == ten0;
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test, test_data);
//     }
//     SECTION("ten0_equals_ten1")
//     {
//         auto test = [](const auto& t){
//             auto ten0 = std::get<0>(t);
//             auto ten1 = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto result = ten0 == ten1;
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test, test_data);
//     }
//     SECTION("ten1_equals_ten0")
//     {
//         auto test = [](const auto& t){
//             auto ten0 = std::get<0>(t);
//             auto ten1 = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto result = ten1 == ten0;
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test, test_data);
//     }
// }

// //value assignment operator=
// TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_result","[test_tensor]",
//     //0lhs_value_type,1rhs_value_type
//     (std::tuple<int,int>),
//     (std::tuple<double,int>)
// )
// {
//     using lhs_value_type = std::tuple_element_t<0,TestType>;
//     using rhs_value_type = std::tuple_element_t<1,TestType>;
//     using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
//     using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
//     using helpers_for_testing::apply_by_element;
//     //0lhs,1rhs,2expected_lhs,3expected_rhs
//     auto test_data = std::make_tuple(
//         //rhs scalar
//         std::make_tuple(lhs_tensor_type{},1,lhs_tensor_type(1),1),
//         std::make_tuple(lhs_tensor_type(1),2,lhs_tensor_type(2),2),
//         std::make_tuple(lhs_tensor_type{1,2,3},4,lhs_tensor_type(4),4),
//         //rhs tensor
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type(1)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(3)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
//         std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{4,5},{6,7}}),
//         //rhs view
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}.reshape(2,3,0),lhs_tensor_type{}.reshape(2,3,0),rhs_tensor_type{}.reshape(2,3,0)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(4).transpose(),lhs_tensor_type(4),rhs_tensor_type(4)),
//         std::make_tuple(lhs_tensor_type{{1,2},{3,4}},rhs_tensor_type{{5,6},{7,8}}(1,1),lhs_tensor_type(8),rhs_tensor_type(8)),
//         //rhs expression
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}+rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{1,2,3}+rhs_tensor_type{4,5,6}+rhs_tensor_type(1),lhs_tensor_type{6,8,10},rhs_tensor_type{6,8,10})
//     );
//     auto test = [](const auto& t){
//         auto lhs = std::get<0>(t);
//         auto rhs = std::get<1>(t);
//         auto expected_lhs = std::get<2>(t);
//         auto expected_rhs = std::get<3>(t);
//         auto& result = lhs = rhs;
//         REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
//         REQUIRE(&result == &lhs);
//         REQUIRE(result == expected_lhs);
//         REQUIRE(rhs == expected_rhs);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_value_semantic","[test_tensor]",
// //0lhs_value_type,1rhs_value_type
//     (std::tuple<int,int>),
//     (std::tuple<double,int>)
// )
// {
//     using lhs_value_type = std::tuple_element_t<0,TestType>;
//     using rhs_value_type = std::tuple_element_t<1,TestType>;
//     using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
//     using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
//     using helpers_for_testing::apply_by_element;
//     //0lhs,1rhs,2expected_lhs
//     auto test_data = std::make_tuple(
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2}),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2}),
//         std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}})
//     );
//     auto test = [](const auto& t){
//         auto lhs = std::get<0>(t);
//         auto rhs = std::get<1>(t);
//         auto expected_lhs = std::get<2>(t);
//         lhs = rhs;
//         *rhs.begin() = -1;
//         REQUIRE(lhs == expected_lhs);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_tensor_move_assignment_result","[test_tensor]",
// //0lhs_value_type,1rhs_value_type
//     (std::tuple<int,int>)
// )
// {
//     using lhs_value_type = std::tuple_element_t<0,TestType>;
//     using rhs_value_type = std::tuple_element_t<1,TestType>;
//     using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
//     using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
//     using helpers_for_testing::apply_by_element;
//     REQUIRE(std::is_same_v<lhs_value_type,rhs_value_type>);
//     //0lhs,1rhs,2expected_lhs,3expected_rhs
//     auto test_data = std::make_tuple(
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type(1)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(1)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type(1)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type(1)),
//         std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{1}),
//         std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{1}),
//         std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{{1},{2},{3}}})
//     );
//     auto test = [](const auto& t){
//         auto lhs = std::get<0>(t);
//         auto rhs = std::get<1>(t);
//         auto expected_lhs = std::get<2>(t);
//         auto expected_rhs = std::get<3>(t);
//         auto& result = lhs = std::move(rhs);
//         REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
//         REQUIRE(&result == &lhs);
//         REQUIRE(result == expected_lhs);
//         REQUIRE(rhs == expected_rhs);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_tensor_converting_move_assignment_result","[test_tensor]",
// //0lhs_value_type,1rhs_value_type
//     (std::tuple<double,int>)
// )
// {
//     using lhs_value_type = std::tuple_element_t<0,TestType>;
//     using rhs_value_type = std::tuple_element_t<1,TestType>;
//     using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
//     using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
//     using helpers_for_testing::apply_by_element;
//     REQUIRE(!std::is_same_v<lhs_value_type,rhs_value_type>);
//     //0lhs,1rhs,2expected_lhs,3expected_rhs
//     auto test_data = std::make_tuple(
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type(1),lhs_tensor_type(1),rhs_tensor_type(1)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(3),lhs_tensor_type(3),rhs_tensor_type(3)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{2},lhs_tensor_type{2},rhs_tensor_type{2}),
//         std::make_tuple(lhs_tensor_type{1},rhs_tensor_type{{1,2,3},{4,5,6}},lhs_tensor_type{{1,2,3},{4,5,6}},rhs_tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(lhs_tensor_type{{{1},{2},{3}}},rhs_tensor_type{{4,5},{6,7}},lhs_tensor_type{{4,5},{6,7}},rhs_tensor_type{{4,5},{6,7}}),
//         //rhs view
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}.reshape(2,3,0),lhs_tensor_type{}.reshape(2,3,0),rhs_tensor_type{}.reshape(2,3,0)),
//         std::make_tuple(lhs_tensor_type(1),rhs_tensor_type(4).transpose(),lhs_tensor_type(4),rhs_tensor_type(4)),
//         std::make_tuple(lhs_tensor_type{{1,2},{3,4}},rhs_tensor_type{{5,6},{7,8}}(1,1),lhs_tensor_type(8),rhs_tensor_type(8)),
//         //rhs expression
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{}+rhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type{},rhs_tensor_type{1,2,3}+rhs_tensor_type{4,5,6}+rhs_tensor_type(1),lhs_tensor_type{6,8,10},rhs_tensor_type{6,8,10})
//     );
//     auto test = [](const auto& t){
//         auto lhs = std::get<0>(t);
//         auto rhs = std::get<1>(t);
//         auto expected_lhs = std::get<2>(t);
//         auto expected_rhs = std::get<3>(t);
//         auto& result = lhs = std::move(rhs);
//         REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
//         REQUIRE(&result == &lhs);
//         REQUIRE(result == expected_lhs);
//         REQUIRE(rhs == expected_rhs);
//     };
//     apply_by_element(test,test_data);
// }

// //broadcast elementwise assignment operator=
// TEMPLATE_TEST_CASE("test_tensor_copy_assignment_converting_copy_assignment_lhs_is_rvalue_view","[test_tensor]",
//     (std::tuple<int,int>),
//     (std::tuple<double,int>)
// )
// {
//     using lhs_value_type = std::tuple_element_t<0,TestType>;
//     using rhs_value_type = std::tuple_element_t<1,TestType>;
//     using gtensor::tensor;
//     using lhs_tensor_type = tensor<lhs_value_type>;
//     using rhs_tensor_type = tensor<rhs_value_type>;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1lhs_view_maker,2rhs,3expected_parent,4expected_lhs,5expected_rhs
//     auto test_data = std::make_tuple(
//         //rhs scalar
//         std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},1,lhs_tensor_type{},lhs_tensor_type{},1),
//         std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},2,lhs_tensor_type(2),lhs_tensor_type(2),2),
//         std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},7,lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7},7),
//         std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t(tensor<int>{3,4,1});},7,lhs_tensor_type{1,7,3,7,7,6},lhs_tensor_type{7,7,7},7),
//         std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t(tensor<bool>{true,true,false,true});},7,lhs_tensor_type{7,7,3,7,5,6},lhs_tensor_type{7,7,7},7),
//         //rhs 0-dim
//         std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type(1),lhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type(1)),
//         std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type(2),lhs_tensor_type(2),lhs_tensor_type(2),rhs_tensor_type(2)),
//         std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},rhs_tensor_type(7),lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7},rhs_tensor_type(7)),
//         std::make_tuple(lhs_tensor_type{{1,2,3},{4,5,6}},[](const auto& t){return t(tensor<int>(1),tensor<int>{0,0,1});},rhs_tensor_type(7),lhs_tensor_type{{1,2,3},{7,7,6}},lhs_tensor_type{7,7,7},rhs_tensor_type(7)),
//         std::make_tuple(lhs_tensor_type{{1,2,3},{4,5,6}},[](const auto& t){return t(tensor<bool>{{false,true},{true,true}});},rhs_tensor_type(7),lhs_tensor_type{{1,7,3},{7,7,6}},lhs_tensor_type{7,7,7},rhs_tensor_type(7)),
//         //rhs n-dim
//         std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type{1},lhs_tensor_type{},lhs_tensor_type{},rhs_tensor_type{1}),
//         std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type{2},lhs_tensor_type(2),lhs_tensor_type(2),rhs_tensor_type{2}),
//         std::make_tuple(
//             lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
//             [](const auto& t){return t({{},{1}});},
//             rhs_tensor_type{-1,1},
//             lhs_tensor_type{{{1,2},{-1,1}},{{5,6},{-1,1}}},
//             lhs_tensor_type{{{-1,1}},{{-1,1}}},
//             rhs_tensor_type{-1,1}
//         ),
//         std::make_tuple(
//             lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
//             [](const auto& t){return t({{},{1}});},
//             rhs_tensor_type{-1,1},
//             lhs_tensor_type{{{1,5},{-1,1}},{{2,6},{-1,1}}},
//             lhs_tensor_type{{{-1,1}},{{-1,1}}},
//             rhs_tensor_type{-1,1}
//         ),
//         std::make_tuple(
//             lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
//             [](const auto& t){return t({{},{1}});},
//             rhs_tensor_type{-1,1} + rhs_tensor_type{{1},{2}},
//             lhs_tensor_type{{{1,5},{1,3}},{{2,6},{1,3}}},
//             lhs_tensor_type{{{1,3}},{{1,3}}},
//             rhs_tensor_type{{0,2},{1,3}}
//         ),
//         std::make_tuple(
//             lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
//             [](const auto& t){return t(tensor<int>{0,1},tensor<int>(1),tensor<int>(0));},
//             rhs_tensor_type{9,10},
//             lhs_tensor_type{{{1,2},{9,4}},{{5,6},{10,8}}},
//             lhs_tensor_type{9,10},
//             rhs_tensor_type{9,10}
//         )
//     );
//     auto test = [](const auto& t){
//         auto parent = std::get<0>(t);
//         auto lhs_view_maker = std::get<1>(t);
//         auto rhs = std::get<2>(t);
//         auto expected_parent = std::get<3>(t);
//         auto expected_lhs = std::get<4>(t);
//         auto expected_rhs = std::get<5>(t);
//         auto lhs = lhs_view_maker(parent);
//         auto& result = std::move(lhs) = rhs;
//         REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
//         REQUIRE(&result == &lhs);
//         REQUIRE(result == expected_lhs);
//         REQUIRE(parent == expected_parent);
//         REQUIRE(rhs == expected_rhs);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_tensor_move_assignment_converting_move_assignment_lhs_is_rvalue_view","[test_tensor]",
//     (std::tuple<int,int>),
//     (std::tuple<double,int>)
// )
// {
//     using lhs_value_type = std::tuple_element_t<0,TestType>;
//     using rhs_value_type = std::tuple_element_t<1,TestType>;
//     using lhs_tensor_type = gtensor::tensor<lhs_value_type>;
//     using rhs_tensor_type = gtensor::tensor<rhs_value_type>;
//     using gtensor::assign;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1lhs_view_maker,2rhs,3expected_parent,4expected_lhs
//     auto test_data = std::make_tuple(
//         //rhs 0-dim
//         std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type(1),lhs_tensor_type{},lhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type(2),lhs_tensor_type(2),lhs_tensor_type(2)),
//         std::make_tuple(lhs_tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},rhs_tensor_type(7),lhs_tensor_type{1,7,7,7,7,6},lhs_tensor_type{7,7,7,7}),
//         //rhs n-dim
//         std::make_tuple(lhs_tensor_type{},[](const auto& t){return t();},rhs_tensor_type{1},lhs_tensor_type{},lhs_tensor_type{}),
//         std::make_tuple(lhs_tensor_type(1),[](const auto& t){return t.transpose();},rhs_tensor_type{2},lhs_tensor_type(2),lhs_tensor_type(2)),
//         std::make_tuple(
//             lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
//             [](const auto& t){return t({{},{1}});},
//             rhs_tensor_type{-1,1},
//             lhs_tensor_type{{{1,2},{-1,1}},{{5,6},{-1,1}}},
//             lhs_tensor_type{{{-1,1}},{{-1,1}}}
//         ),
//         std::make_tuple(
//             lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
//             [](const auto& t){return t({{},{1}});},
//             rhs_tensor_type{-1,1},
//             lhs_tensor_type{{{1,5},{-1,1}},{{2,6},{-1,1}}},
//             lhs_tensor_type{{{-1,1}},{{-1,1}}}
//         ),
//         std::make_tuple(
//             lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
//             [](const auto& t){return t({{},{1}});},
//             rhs_tensor_type{-1,1} + rhs_tensor_type{{1},{2}},
//             lhs_tensor_type{{{1,5},{1,3}},{{2,6},{1,3}}},
//             lhs_tensor_type{{{1,3}},{{1,3}}}
//         )
//     );
//     auto test = [](const auto& t){
//         auto parent = std::get<0>(t);
//         auto lhs_view_maker = std::get<1>(t);
//         auto rhs = std::get<2>(t);
//         auto expected_parent = std::get<3>(t);
//         auto expected_lhs = std::get<4>(t);
//         auto lhs = lhs_view_maker(parent);
//         auto& result = std::move(lhs) = std::move(rhs);
//         REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
//         REQUIRE(&result == &lhs);
//         REQUIRE(result == expected_lhs);
//         REQUIRE(parent == expected_parent);
//         REQUIRE(rhs.empty());
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_assign_to_rvalue_view_or_tensor","[test_tensor]")
// {
//     using gtensor::tensor;
//     tensor<double> t{1,2,3,4,5,6,7,8,9,10,11,12};

//     SECTION("assign1")
//     {
//         t.reshape(2,2,3) = tensor<double>{{3,2,1},{1,2,3}};
//         REQUIRE(t == tensor<double>{3,2,1,1,2,3,3,2,1,1,2,3});
//     }
//     SECTION("assign2")
//     {
//         t.reshape(2,2,3).transpose() = tensor<int>{{2,1},{3,2}};
//         REQUIRE(t == tensor<double>{2,2,2,3,3,3,1,1,1,2,2,2});
//     }
//     SECTION("assign3")
//     {
//         using slice_type = tensor<double>::slice_type;
//         t.reshape(2,2,3)(slice_type{},slice_type{},1) = 0;
//         REQUIRE(t == tensor<double>{1,0,3,4,0,6,7,0,9,10,0,12});
//     }
//     SECTION("assign4")
//     {
//         t(t>3&&t<10) = 0;
//         REQUIRE(t == tensor<double>{1,2,3,0,0,0,0,0,0,10,11,12});
//     }
//     SECTION("assign5")
//     {
//         t.reshape(2,2,3)(tensor<int>{{1,0},{0,1}},tensor<int>{{0,1},{1,0}}) = tensor<int>{{8},{9}};
//         REQUIRE(t == tensor<double>{1,2,3,8,8,8,9,9,9,10,11,12});
//     }
//     SECTION("assign6")
//     {
//         std::move(t) = 5;
//         REQUIRE(t == tensor<double>{5,5,5,5,5,5,5,5,5,5,5,5});
//     }
// }

// namespace test_tensor_assignment_corner_cases{

// struct assign_exception{};
// struct throw_on_assign{
//     inline static bool is_throw = false;
//     throw_on_assign() = default;
//     throw_on_assign(const throw_on_assign&) = default;
//     throw_on_assign& operator=(const throw_on_assign&){
//         if (is_throw){
//             throw assign_exception{};
//         }
//         return *this;
//     }
// };

// }

// TEST_CASE("test_tensor_assignment_corner_cases","[test_tensor]")
// {
//     using helpers_for_testing::apply_by_element;
//     using test_tensor_assignment_corner_cases::throw_on_assign;
//     throw_on_assign::is_throw = false;
//     REQUIRE_NOTHROW(throw_on_assign{} = throw_on_assign{});
//     throw_on_assign::is_throw = true;
//     REQUIRE_THROWS(throw_on_assign{} = throw_on_assign{});
//     throw_on_assign::is_throw = false;
//     SECTION("copy_assign_to_same_lvalue")
//     {
//         using value_type = throw_on_assign;
//         using tensor_type = gtensor::tensor<value_type>;
//         //assign may be required during construction
//         tensor_type t({10},value_type{});
//         throw_on_assign::is_throw = true;   //enable throwing on assign
//         const auto ptr_to_first_expected = &(*t.begin());
//         REQUIRE_NOTHROW(t.operator=(t));    //self assignment
//         const auto ptr_to_first_result = &(*t.begin());
//         REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
//     }
//     SECTION("copy_assign_to_same_rvalue")
//     {
//         using value_type = throw_on_assign;
//         using tensor_type = gtensor::tensor<value_type>;
//         //assign may be required during construction
//         tensor_type t({10},value_type{});
//         throw_on_assign::is_throw = true;   //enable throwing on assign
//         REQUIRE_NOTHROW(std::move(t).operator=(t));    //self assignment
//     }
//     SECTION("move_assign_to_same_lvalue")
//     {
//         using value_type = throw_on_assign;
//         using tensor_type = gtensor::tensor<value_type>;
//         //assign may be required during construction
//         tensor_type t({10},value_type{});
//         throw_on_assign::is_throw = true;   //enable throwing on assign
//         const auto ptr_to_first_expected = &(*t.begin());
//         REQUIRE_NOTHROW(t.operator=(std::move(t)));    //self assignment
//         const auto ptr_to_first_result = &(*t.begin());
//         REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
//     }
//     SECTION("move_assign_to_same_rvalue")
//     {
//         using value_type = throw_on_assign;
//         using tensor_type = gtensor::tensor<value_type>;
//         //assign may be required during construction
//         tensor_type t({10},value_type{});
//         throw_on_assign::is_throw = true;   //enable throwing on assign
//         REQUIRE_NOTHROW(std::move(t).operator=(std::move(t)));    //self assignment
//     }
//     SECTION("move_assign_to_lvalue")
//     {
//         using value_type = throw_on_assign;
//         using tensor_type = gtensor::tensor<value_type>;
//         tensor_type lhs({10},value_type{});
//         tensor_type rhs({10},value_type{});
//         throw_on_assign::is_throw = true;   //enable throwing on assign
//         const auto ptr_to_first_expected = &(*rhs.begin());
//         REQUIRE_NOTHROW(lhs.operator=(std::move(rhs)));    //no assignment
//         const auto ptr_to_first_result = &(*lhs.begin());
//         REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //swap impl
//     }
//     SECTION("move_assign_to_rvalue")
//     {
//         using value_type = throw_on_assign;
//         using tensor_type = gtensor::tensor<value_type>;
//         tensor_type lhs({10},value_type{});
//         tensor_type rhs({10},value_type{});
//         throw_on_assign::is_throw = true;   //enable throwing on assign
//         REQUIRE_THROWS_AS(std::move(lhs).operator=(std::move(rhs)),test_tensor_assignment_corner_cases::assign_exception);    //elementwise assignment
//     }
//     SECTION("copy_assign_convert_copy_assign_to_lvalue_same_shape")
//     {
//         using gtensor::tensor;
//         //0lhs,1rhs,2expected_lhs,3expected_rhs
//         auto test_data = std::make_tuple(
//             std::make_tuple(tensor<int>(1),tensor<int>(2),tensor<int>(2),tensor<int>(2)),
//             std::make_tuple(tensor<double>(1),tensor<int>(2),tensor<double>(2),tensor<int>(2)),
//             std::make_tuple(tensor<int>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}}),
//             std::make_tuple(tensor<double>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}}+0,tensor<double>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}})
//         );
//         auto test = [](const auto& t){
//             auto lhs = std::get<0>(t);
//             auto rhs = std::get<1>(t);
//             auto expected_lhs = std::get<2>(t);
//             auto expected_rhs = std::get<3>(t);
//             const auto ptr_to_first_expected = &(*lhs.begin());
//             lhs.operator=(rhs);
//             const auto ptr_to_first_result = &(*lhs.begin());
//             REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
//             REQUIRE(lhs == expected_lhs);
//             REQUIRE(rhs == expected_rhs);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("convert_move_assign_to_lvalue_same_shape")
//     {
//         using gtensor::tensor;
//         //0lhs,1rhs,2expected_lhs,3expected_rhs
//         auto test_data = std::make_tuple(
//             std::make_tuple(tensor<double>(1),tensor<int>(2),tensor<double>(2),tensor<int>(2)),
//             std::make_tuple(tensor<double>{{1,2,3},{4,5,6}},tensor<int>{{7,8,9},{10,11,12}}+0,tensor<double>{{7,8,9},{10,11,12}},tensor<int>{{7,8,9},{10,11,12}})
//         );
//         auto test = [](const auto& t){
//             auto lhs = std::get<0>(t);
//             auto rhs = std::get<1>(t);
//             auto expected_lhs = std::get<2>(t);
//             auto expected_rhs = std::get<3>(t);
//             const auto ptr_to_first_expected = &(*lhs.begin());
//             lhs.operator=(std::move(rhs));
//             const auto ptr_to_first_result = &(*lhs.begin());
//             REQUIRE(ptr_to_first_result == ptr_to_first_expected);  //no reallocation
//             REQUIRE(lhs == expected_lhs);
//             REQUIRE(rhs == expected_rhs);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEST_CASE("test_tensor_elementwise_assignment_exception","[test_tensor]")
// {
//     using tensor_type = gtensor::tensor<int>;
//     using gtensor::value_error;
//     auto lhs = tensor_type{{1,2,3},{4,5,6}}(1);
//     tensor_type rhs{1,2};
//     //shapes not broadcast
//     REQUIRE_THROWS_AS((std::move(lhs) = rhs), value_error);
//     REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}(1) = rhs), value_error);
//     REQUIRE_THROWS_AS((tensor_type{{1,2,3},{4,5,6}}.reshape(-1) = tensor_type{1,2}), value_error);
// }

// //test broadcast assign
// TEST_CASE("test_tensor_assign","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using helpers_for_testing::apply_by_element;
//     SECTION("lvalue_rhs")
//     {
//         //0lhs,1rhs,2expected_lhs,3expected_rhs
//         auto test_data = std::make_tuple(
//             //rhs scalar
//             std::make_tuple(tensor_type{},1,tensor_type{},1),
//             std::make_tuple(tensor_type(1),2,tensor_type(2),2),
//             std::make_tuple(tensor_type{{1,2,3},{4,5,6}},7,tensor_type{{7,7,7},{7,7,7}},7),
//             //rhs 0-dim
//             std::make_tuple(tensor_type{},tensor_type(2),tensor_type{},tensor_type(2)),
//             std::make_tuple(tensor_type(1),tensor_type(2),tensor_type(2),tensor_type(2)),
//             std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type(7),tensor_type{{7,7,7},{7,7,7}},tensor_type(7)),
//             //rhs n-dim
//             std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
//             std::make_tuple(tensor_type(1),tensor_type{},tensor_type(1),tensor_type{}),
//             std::make_tuple(tensor_type(1),tensor_type{2},tensor_type(2),tensor_type{2}),
//             std::make_tuple(tensor_type(1),tensor_type{{1,2},{3,4}},tensor_type(4),tensor_type{{1,2},{3,4}}),
//             std::make_tuple(tensor_type{2},tensor_type{},tensor_type{2},tensor_type{}),
//             std::make_tuple(tensor_type{{3}},tensor_type{},tensor_type{{3}},tensor_type{}),
//             std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{9},{10}},tensor_type{{{9,9},{10,10}},{{9,9},{10,10}}},tensor_type{{9},{10}})
//         );
//         auto test = [](const auto& t){
//             auto lhs = std::get<0>(t);
//             auto rhs = std::get<1>(t);
//             auto expected_lhs = std::get<2>(t);
//             auto expected_rhs = std::get<3>(t);
//             auto& result = lhs.assign(rhs);
//             REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
//             REQUIRE(&result == &lhs);
//             REQUIRE(result == expected_lhs);
//             REQUIRE(rhs == expected_rhs);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("rvalue_rhs")
//     {
//         //0lhs,1rhs,2expected_lhs,3expected_rhs
//         auto test_data = std::make_tuple(
//             //rhs 0-dim
//             std::make_tuple(tensor_type{},tensor_type(2),tensor_type{},tensor_type(2)),
//             std::make_tuple(tensor_type(1),tensor_type(2),tensor_type(2),tensor_type(2)),
//             std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type(7),tensor_type{{7,7,7},{7,7,7}},tensor_type(7)),
//             //rhs n-dim
//             std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
//             std::make_tuple(tensor_type(1),tensor_type{},tensor_type(1),tensor_type{}),
//             std::make_tuple(tensor_type(1),tensor_type{2},tensor_type(2),tensor_type{2}),
//             std::make_tuple(tensor_type(1),tensor_type{{1,2},{3,4}},tensor_type(4),tensor_type{{1,2},{3,4}}),
//             std::make_tuple(tensor_type{2},tensor_type{},tensor_type{2},tensor_type{}),
//             std::make_tuple(tensor_type{{3}},tensor_type{},tensor_type{{3}},tensor_type{}),
//             std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{9},{10}},tensor_type{{{9,9},{10,10}},{{9,9},{10,10}}},tensor_type{{9},{10}})
//         );
//         auto test = [](const auto& t){
//             auto lhs = std::get<0>(t);
//             auto rhs = std::get<1>(t);
//             auto expected_lhs = std::get<2>(t);
//             auto expected_rhs = std::get<3>(t);
//             auto& result = lhs.assign(std::move(rhs));
//             REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
//             REQUIRE(&result == &lhs);
//             REQUIRE(result == expected_lhs);
//             REQUIRE(rhs.empty());
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEST_CASE("test_tensor_is_same","[test_tensor]")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using helpers_for_testing::apply_by_element;

//     const auto t = tensor_type{1,2,3};
//     const auto e = tensor_type{1,2,3}+tensor_type{0,0,0};
//     const auto v = tensor_type{1,2,3}.transpose().transpose();
//     SECTION("test_not_same_tensor")
//     {
//         auto test_data = std::make_tuple(
//             std::make_tuple(t,e,v),
//             std::make_tuple(e,t,v),
//             std::make_tuple(v,e,t)
//         );
//         auto test = [](const auto& t){
//             auto first = std::get<0>(t);
//             auto second = std::get<1>(t);
//             auto third = std::get<2>(t);
//             REQUIRE(!first.is_same(second));
//             REQUIRE(!first.is_same(third));
//             REQUIRE(!second.is_same(first));
//             REQUIRE(!second.is_same(third));
//             REQUIRE(!third.is_same(first));
//             REQUIRE(!third.is_same(second));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_not_same_other"){
//         REQUIRE(!t.is_same(1));
//         REQUIRE(!e.is_same(1.0));
//         REQUIRE(!v.is_same(std::string{}));
//     }
//     SECTION("test_same_tensor")
//     {
//         auto test_data = std::make_tuple(t,e,v);
//         auto test = [](const auto& t){
//             REQUIRE(t.is_same(t));
//             auto t_ref_copy = t;
//             REQUIRE(&t != &t_ref_copy);
//             REQUIRE(t.is_same(t_ref_copy));
//             auto t_ref_move = std::move(t_ref_copy);
//             REQUIRE(t.is_same(t_ref_move));
//             REQUIRE(!t.is_same(t_ref_copy));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEST_CASE("test_tensor_resize","[test_tensor]")
// {
//     using value_type = int;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using gtensor::tensor;
//     using tensor_type = tensor<value_type>;
//     using helpers_for_testing::apply_by_element;
//     SECTION("test_tensor_resize_to_not_bigger")
//     {
//         //0tensor,1new_shape,2expected
//         auto test_data = std::make_tuple(
//             //c_order
//             std::make_tuple(tensor<value_type,c_order>{},std::vector<int>{0},tensor_type{}),
//             std::make_tuple(tensor<value_type,c_order>{},std::array<int,3>{0,2,3},tensor_type{}.reshape(0,2,3)),
//             std::make_tuple(tensor<value_type,c_order>(1),std::vector<int>{0},tensor_type{}),
//             std::make_tuple(tensor<value_type,c_order>(2),std::vector<int>{},tensor_type(2)),
//             std::make_tuple(tensor<value_type,c_order>(1),std::vector<int>{1,1},tensor_type{{1}}),
//             std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{0},tensor_type{}),
//             std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{},tensor_type(1)),
//             std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{3},tensor_type{1,2,3}),
//             std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,3},tensor_type{{1,2,3},{4,5,6}}),
//             std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,2,2},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
//             //f_order
//             std::make_tuple(tensor<value_type,f_order>{},std::vector<int>{0},tensor_type{}),
//             std::make_tuple(tensor<value_type,f_order>{},std::array<int,3>{0,2,3},tensor_type{}.reshape(0,2,3)),
//             std::make_tuple(tensor<value_type,f_order>(1),std::vector<int>{0},tensor_type{}),
//             std::make_tuple(tensor<value_type,f_order>(2),std::vector<int>{},tensor_type(2)),
//             std::make_tuple(tensor<value_type,f_order>(1),std::vector<int>{1,1},tensor_type{{1}}),
//             std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{0},tensor_type{}),
//             std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{},tensor_type(1)),
//             std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{3},tensor_type{1,5,3}),
//             std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,3},tensor_type{{1,3,2},{5,7,6}}),
//             std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},std::vector<int>{2,2,2},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}})
//         );
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto new_shape = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto ten_size = ten.size();
//             auto expected_size = expected.size();
//             REQUIRE(ten_size >= expected_size);
//             ten.resize(new_shape);
//             REQUIRE(ten == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_tensor_resize_to_bigger")
//     {
//         const value_type any{std::numeric_limits<value_type>::max()};
//         //0tensor,1size,2new_shape,3expected
//         auto test_data = std::make_tuple(
//             //c_order
//             std::make_tuple(tensor<value_type,c_order>{},std::vector<int>{1},tensor_type{any}),
//             std::make_tuple(tensor<value_type,c_order>{},std::vector<int>{5},tensor_type{any,any,any,any,any}),
//             std::make_tuple(tensor<value_type,c_order>{},std::vector<int>{2,3},tensor_type{{any,any,any},{any,any,any}}),
//             std::make_tuple(tensor<value_type,c_order>(3),std::vector<int>{5},tensor_type{3,any,any,any,any}),
//             std::make_tuple(tensor<value_type,c_order>(3),std::vector<int>{3,2},tensor_type{{3,any},{any,any},{any,any}}),
//             std::make_tuple(tensor<value_type,c_order>{4},std::vector<int>{5},tensor_type{4,any,any,any,any}),
//             std::make_tuple(tensor<value_type,c_order>{{1,2,3},{4,5,6}},std::vector<int>{2,2,2},tensor_type{{{1,2},{3,4}},{{5,6},{any,any}}}),
//             //f_order
//             std::make_tuple(tensor<value_type,f_order>{},std::vector<int>{1},tensor_type{any}),
//             std::make_tuple(tensor<value_type,f_order>{},std::vector<int>{5},tensor_type{any,any,any,any,any}),
//             std::make_tuple(tensor<value_type,f_order>{},std::vector<int>{2,3},tensor_type{{any,any,any},{any,any,any}}),
//             std::make_tuple(tensor<value_type,f_order>(3),std::vector<int>{5},tensor_type{3,any,any,any,any}),
//             std::make_tuple(tensor<value_type,f_order>(3),std::vector<int>{3,2},tensor_type{{3,any},{any,any},{any,any}}),
//             std::make_tuple(tensor<value_type,f_order>{4},std::vector<int>{5},tensor_type{4,any,any,any,any}),
//             std::make_tuple(tensor<value_type,f_order>{{1,2,3},{4,5,6}},std::vector<int>{2,2,2},tensor_type{{{1,3},{2,any}},{{4,6},{5,any}}})
//         );
//         auto test = [&any](const auto& t){
//             auto ten = std::get<0>(t);
//             auto new_shape = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto expected_shape = expected.shape();
//             ten.resize(new_shape);
//             auto result_shape = ten.shape();
//             REQUIRE(result_shape == expected_shape);
//             auto comparator = [&any](auto result_element, auto expected_element){
//                 return expected_element == any ? true : result_element == expected_element;
//             };
//             REQUIRE(std::equal(ten.begin(),ten.end(),expected.begin(),expected.end(),comparator));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_tensor_resize_init_list_interface")
//     {
//         tensor<value_type,c_order> t0{{1,2,3},{4,5,6}};
//         t0.resize({2,2});
//         REQUIRE(t0 == tensor_type{{1,2},{3,4}});
//     }

// }

// TEMPLATE_TEST_CASE("test_tensor_copy","[test_tensor]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using order = TestType;
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using helpers_for_testing::apply_by_element;
//     //0tensor,1expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1} + tensor_type{0,1,2}, tensor_type{{1,3,5},{4,6,8}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}), tensor_type{{5,6},{2,3}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose(), tensor_type{{5,2},{6,3}}),
//         std::make_tuple((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}} + tensor_type{1})({{{},{},-1},{1}}).transpose(), tensor_type{{5,2},{6,3}}),
//         std::make_tuple(tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1}, tensor_type{{5,2},{6,3}}),
//         std::make_tuple(((tensor_type{-1} + tensor_type{{1,2,3},{4,5,6}}({{{},{},-1},{1}}).transpose() + tensor_type{1})).reshape(4),tensor_type{5,2,6,3})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         auto result = ten.copy(order{});
//         using result_order = typename decltype(result)::order;
//         REQUIRE(std::is_same_v<result_order,order>);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_tensor_equal","[test_tensor]"){
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using bool_tensor_type = gtensor::tensor<bool>;
//     using helpers_for_testing::apply_by_element;
//     //0operand1,1operand2,2expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{},tensor_type{},bool_tensor_type{}),
//         std::make_tuple(tensor_type{},tensor_type(1),bool_tensor_type{}),
//         std::make_tuple(tensor_type(1),tensor_type{},bool_tensor_type{}),
//         std::make_tuple(tensor_type(1),tensor_type(1),bool_tensor_type(true)),
//         std::make_tuple(tensor_type(1),tensor_type(2),bool_tensor_type(false)),
//         std::make_tuple(tensor_type(1),tensor_type{{1,2},{3,1}},bool_tensor_type{{true,false},{false,true}}),
//         std::make_tuple(tensor_type{{1,2},{3,1}},tensor_type(1),bool_tensor_type{{true,false},{false,true}}),
//         std::make_tuple(tensor_type{{1,2},{3,1}},tensor_type{2},bool_tensor_type{{false,true},{false,false}}),
//         std::make_tuple(tensor_type{2},tensor_type{{1,2},{3,1}},bool_tensor_type{{false,true},{false,false}}),
//         std::make_tuple(tensor_type{{1},{5}},tensor_type{{{1,2},{3,1}},{{4,5},{5,6}}},bool_tensor_type{{{true,false},{false,false}},{{false,false},{true,false}}})
//     );
//     auto test = [](const auto& t){
//         auto operand1 = std::get<0>(t);
//         auto operand2 = std::get<1>(t);
//         auto expected = std::get<2>(t);
//         auto result = operand1.equal(operand2);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_tensor_swap","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using helpers_for_testing::apply_by_element;
//     //0tensor0,1tensor1,2expected_tensor0,3expected_tensor1
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3},tensor_type{{4},{5}},tensor_type{{4},{5}},tensor_type{1,2,3}),
//         std::make_tuple(tensor_type{1,2,3}+1,tensor_type{{4},{5}}+1,tensor_type{{5},{6}},tensor_type{2,3,4}),
//         std::make_tuple(tensor_type{{1,2,3}}.transpose(),tensor_type{{4},{5}}.transpose(),tensor_type{{4,5}},tensor_type{{1},{2},{3}})
//     );
//     auto test = [](const auto& t){
//         auto ten1 = std::get<0>(t);
//         auto ten2 = std::get<1>(t);
//         auto expected_ten1 = std::get<2>(t);
//         auto expected_ten2 = std::get<3>(t);
//         auto ten1_copy{ten1};
//         auto ten2_copy{ten2};
//         REQUIRE(ten1_copy.is_same(ten1));
//         REQUIRE(ten2_copy.is_same(ten2));
//         swap(ten1,ten2);
//         REQUIRE(ten1.is_same(ten2_copy));
//         REQUIRE(ten2.is_same(ten1_copy));
//         REQUIRE(ten1 == expected_ten1);
//         REQUIRE(ten2 == expected_ten2);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_meta_data_interface","[test_tensor]")
// {
//     using value_type = double;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using gtensor::tensor;
//     using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
//     using dim_type = typename config_type::dim_type;
//     using index_type = typename config_type::index_type;
//     using shape_type = typename config_type::shape_type;
//     using helpers_for_testing::apply_by_element;
//     //0tensor,1expected_dim,2expected_size,3expected_shape,4expected_strides
//     auto test_data = std::make_tuple(
//         //c_order layout
//         std::make_tuple(tensor<value_type,c_order>(1),dim_type{0},index_type{1},shape_type{},shape_type{}),
//         std::make_tuple(tensor<value_type,c_order>{},dim_type{1},index_type{0},shape_type{0},shape_type{1}),
//         std::make_tuple(tensor<value_type,c_order>{1},dim_type{1},index_type{1},shape_type{1},shape_type{1}),
//         std::make_tuple(tensor<value_type,c_order>{1,2,3},dim_type{1},index_type{3},shape_type{3},shape_type{1}),
//         std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},dim_type{3},index_type{8},shape_type{2,2,2},shape_type{4,2,1}),
//         //f_order layout
//         std::make_tuple(tensor<value_type,f_order>(1),dim_type{0},index_type{1},shape_type{},shape_type{}),
//         std::make_tuple(tensor<value_type,f_order>{},dim_type{1},index_type{0},shape_type{0},shape_type{1}),
//         std::make_tuple(tensor<value_type,f_order>{1},dim_type{1},index_type{1},shape_type{1},shape_type{1}),
//         std::make_tuple(tensor<value_type,f_order>{1,2,3},dim_type{1},index_type{3},shape_type{3},shape_type{1}),
//         std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},dim_type{3},index_type{8},shape_type{2,2,2},shape_type{1,2,4})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto expected_dim = std::get<1>(t);
//         auto expected_size = std::get<2>(t);
//         auto expected_shape = std::get<3>(t);
//         auto expected_strides = std::get<4>(t);

//         auto result_dim = ten.dim();
//         auto result_size = ten.size();
//         auto result_shape = ten.shape();
//         auto result_strides = ten.strides();

//         REQUIRE(result_dim == expected_dim);
//         REQUIRE(result_size == expected_size);
//         REQUIRE(result_shape == expected_shape);
//         REQUIRE(result_strides == expected_strides);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_tensor_data_interface","[test_tensor]",
//     (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
//     (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
//     (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
//     (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
//     (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
//     (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
//     (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
//     (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>)
// )
// {
//     using tensor_type = TestType;
//     using config_type = typename tensor_type::config_type;
//     using traverse_order = typename config_type::order;
//     using index_type = typename tensor_type::index_type;
//     using value_type = typename tensor_type::value_type;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using helpers_for_testing::apply_by_element;
//     //0tensor,1elements_c_traverse,2elements_f_traverse
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(1),std::vector<value_type>{1},std::vector<value_type>{1}),
//         std::make_tuple(tensor_type{},std::vector<value_type>{},std::vector<value_type>{}),
//         std::make_tuple(tensor_type{2},std::vector<value_type>{2},std::vector<value_type>{2}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6},std::vector<value_type>{1,2,3,4,5,6},std::vector<value_type>{1,2,3,4,5,6}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12})
//     );
//     SECTION("test_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_c_traverse = std::get<1>(t);
//             auto elements_f_traverse = std::get<2>(t);
//             auto first = ten.begin();
//             auto last = ten.end();
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             if constexpr (std::is_same_v<traverse_order,c_order>){
//                 REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
//             }else{
//                 REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
//             }
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_reverse_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_c_traverse = std::get<1>(t);
//             auto elements_f_traverse = std::get<2>(t);
//             auto first = ten.rbegin();
//             auto last = ten.rend();
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             if constexpr (std::is_same_v<traverse_order,c_order>){
//                 REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
//             }else{
//                 REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
//             }
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_indexer")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_c_traverse = std::get<1>(t);
//             auto elements_f_traverse = std::get<2>(t);
//             auto indexer = ten.create_indexer();
//             std::vector<value_type> traverse_result{};
//             for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
//                 traverse_result.push_back(indexer[i]);
//             }
//             auto first = traverse_result.begin();
//             auto last = traverse_result.end();
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             if constexpr (std::is_same_v<traverse_order,c_order>){
//                 REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
//             }else{
//                 REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
//             }
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_walker_c_order_traverse")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_c_traverse = std::get<1>(t);
//             auto walker = ten.create_walker();
//             using walker_iterator_type = gtensor::walker_iterator<config_type,decltype(walker),c_order>;
//             walker_iterator_type first{walker, ten.shape(), ten.descriptor().strides_div(c_order{}), index_type{0}};
//             walker_iterator_type last{walker, ten.shape(), ten.descriptor().strides_div(c_order{}), ten.size()};
//             REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_walker_f_order_traverse")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_f_traverse = std::get<2>(t);
//             auto walker = ten.create_walker();
//             using walker_iterator_type = gtensor::walker_iterator<config_type,decltype(walker),f_order>;
//             walker_iterator_type first{walker, ten.shape(), ten.descriptor().strides_div(f_order{}), index_type{0}};
//             walker_iterator_type last{walker, ten.shape(), ten.descriptor().strides_div(f_order{}), ten.size()};
//             REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
//         };
//         apply_by_element(test,test_data);
//     }
//     //test traverse adapter
//     SECTION("test_traverse_adapter_c_order_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_c_traverse = std::get<1>(t);
//             auto a = ten.template traverse_order_adapter<c_order>();
//             auto first = a.begin();
//             auto last = a.end();
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_f_order_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_f_traverse = std::get<2>(t);
//             auto a = ten.template traverse_order_adapter<f_order>();
//             auto first = a.begin();
//             auto last = a.end();
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_c_order_reverse_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_c_traverse = std::get<1>(t);
//             auto a = ten.template traverse_order_adapter<c_order>();
//             auto first = a.rbegin();
//             auto last = a.rend();
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_f_order_reverse_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_f_traverse = std::get<2>(t);
//             auto a = ten.template traverse_order_adapter<f_order>();
//             auto first = a.rbegin();
//             auto last = a.rend();
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_c_order_indexer")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_c_traverse = std::get<1>(t);
//             auto a = ten.template traverse_order_adapter<c_order>();
//             auto indexer = a.create_indexer();
//             std::vector<value_type> traverse_result{};
//             for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
//                 traverse_result.push_back(indexer[i]);
//             }
//             auto first = traverse_result.begin();
//             auto last = traverse_result.end();
//             REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_f_order_indexer")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto elements_f_traverse = std::get<2>(t);
//             auto a = ten.template traverse_order_adapter<f_order>();
//             auto indexer = a.create_indexer();
//             std::vector<value_type> traverse_result{};
//             for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
//                 traverse_result.push_back(indexer[i]);
//             }
//             auto first = traverse_result.begin();
//             auto last = traverse_result.end();
//             REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_tensor_broadcast_iterator","[test_tensor]",
//     (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
//     (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
//     (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
//     (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
//     (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
//     (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
//     (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
//     (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>)
// )
// {
//     using tensor_type = TestType;
//     using config_type = typename tensor_type::config_type;
//     using traverse_order = typename config_type::order;
//     using shape_type = typename tensor_type::shape_type;
//     using value_type = typename tensor_type::value_type;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using helpers_for_testing::apply_by_element;
//     //0tensor,1broadcast_shape,2elements_c_traverse,3elements_f_traverse
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(2),shape_type{1},std::vector<value_type>{2},std::vector<value_type>{2}),
//         std::make_tuple(tensor_type(2),shape_type{1,1},std::vector<value_type>{2},std::vector<value_type>{2}),
//         std::make_tuple(tensor_type(2),shape_type{5},std::vector<value_type>{2,2,2,2,2},std::vector<value_type>{2,2,2,2,2}),
//         std::make_tuple(tensor_type(2),shape_type{2,3},std::vector<value_type>{2,2,2,2,2,2},std::vector<value_type>{2,2,2,2,2,2}),
//         std::make_tuple(tensor_type{1},shape_type{1},std::vector<value_type>{1},std::vector<value_type>{1}),
//         std::make_tuple(tensor_type{1},shape_type{1,1},std::vector<value_type>{1},std::vector<value_type>{1}),
//         std::make_tuple(tensor_type{1},shape_type{5},std::vector<value_type>{1,1,1,1,1},std::vector<value_type>{1,1,1,1,1}),
//         std::make_tuple(tensor_type{1},shape_type{2,3},std::vector<value_type>{1,1,1,1,1,1},std::vector<value_type>{1,1,1,1,1,1}),
//         std::make_tuple(tensor_type{1,2,3,4,5},shape_type{5},std::vector<value_type>{1,2,3,4,5},std::vector<value_type>{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},shape_type{2,5},std::vector<value_type>{1,2,3,4,5,1,2,3,4,5},std::vector<value_type>{1,1,2,2,3,3,4,4,5,5}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},shape_type{2,3},std::vector<value_type>{1,2,3,4,5,6},std::vector<value_type>{1,4,2,5,3,6}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},shape_type{1,2,2,3},std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6},std::vector<value_type>{1,1,4,4,2,2,5,5,3,3,6,6})
//     );

//     SECTION("test_broadcast_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto broadcast_shape = std::get<1>(t);
//             auto elements_c_traverse = std::get<2>(t);
//             auto elements_f_traverse = std::get<3>(t);
//             auto first = ten.begin(broadcast_shape);
//             auto last = ten.end(broadcast_shape);
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             if constexpr (std::is_same_v<traverse_order,c_order>){
//                 REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
//             }else{
//                 REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
//             }
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_reverse_broadcast_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto broadcast_shape = std::get<1>(t);
//             auto elements_c_traverse = std::get<2>(t);
//             auto elements_f_traverse = std::get<3>(t);
//             auto first = ten.rbegin(broadcast_shape);
//             auto last = ten.rend(broadcast_shape);
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             if constexpr (std::is_same_v<traverse_order,c_order>){
//                 REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
//             }else{
//                 REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
//             }
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_c_order_broadcast_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto broadcast_shape = std::get<1>(t);
//             auto elements_c_traverse = std::get<2>(t);
//             auto a = ten.template traverse_order_adapter<c_order>();
//             auto first = a.begin(broadcast_shape);
//             auto last = a.end(broadcast_shape);
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_f_order_broadcast_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto broadcast_shape = std::get<1>(t);
//             auto elements_f_traverse = std::get<3>(t);
//             auto a = ten.template traverse_order_adapter<f_order>();
//             auto first = a.begin(broadcast_shape);
//             auto last = a.end(broadcast_shape);
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_c_order_reverse_broadcast_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto broadcast_shape = std::get<1>(t);
//             auto elements_c_traverse = std::get<2>(t);
//             auto a = ten.template traverse_order_adapter<c_order>();
//             auto first = a.rbegin(broadcast_shape);
//             auto last = a.rend(broadcast_shape);
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_traverse_adapter_f_order_reverse_broadcast_iterator")
//     {
//         auto test = [](const auto& t){
//             tensor_type ten = std::get<0>(t);
//             auto broadcast_shape = std::get<1>(t);
//             auto elements_f_traverse = std::get<3>(t);
//             auto a = ten.template traverse_order_adapter<f_order>();
//             auto first = a.rbegin(broadcast_shape);
//             auto last = a.rend(broadcast_shape);
//             REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
//             REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEST_CASE("test_tensor_data_interface_result_type","test_tensor")
// {
//     using value_type = int;
//     using gtensor::config::c_order;
//     using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
//     using tensor_type = gtensor::tensor<int,c_order,config_type>;
//     using dim_type = tensor_type::dim_type;
//     using index_type = tensor_type::index_type;
//     using shape_type = tensor_type::shape_type;

//     //non const instance
//     //tensor
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().begin()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().end()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().rbegin()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().rend()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().begin(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().end(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().rbegin(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().rend(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<tensor_type>().create_indexer()[std::declval<index_type>()]),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().create_walker(std::declval<dim_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().create_walker()),value_type&>);
//     //adapter
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().template traverse_order_adapter<c_order>().begin()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().template traverse_order_adapter<c_order>().end()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().template traverse_order_adapter<c_order>().rbegin()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().template traverse_order_adapter<c_order>().rend()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().template traverse_order_adapter<c_order>().begin(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().template traverse_order_adapter<c_order>().end(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().template traverse_order_adapter<c_order>().rbegin(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().template traverse_order_adapter<c_order>().rend(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<tensor_type>().template traverse_order_adapter<c_order>().create_indexer()[std::declval<index_type>()]),value_type&>);
//     //const instance
//     //tensor
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().begin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().end()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().rbegin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().rend()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().begin(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().end(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().rbegin(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().rend(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<const tensor_type>().create_indexer()[std::declval<index_type>()]),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().create_walker(std::declval<dim_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().create_walker()),const value_type&>);
//     //adapter
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().template traverse_order_adapter<c_order>().begin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().template traverse_order_adapter<c_order>().end()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().template traverse_order_adapter<c_order>().rbegin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().template traverse_order_adapter<c_order>().rend()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().template traverse_order_adapter<c_order>().begin(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().template traverse_order_adapter<c_order>().end(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().template traverse_order_adapter<c_order>().rbegin(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().template traverse_order_adapter<c_order>().rend(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<const tensor_type>().template traverse_order_adapter<c_order>().create_indexer()[std::declval<index_type>()]),const value_type&>);
// }

// TEST_CASE("test_tensor_view_interface","[test_tensor]")
// {

//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = tensor<value_type>;
//     using shape_type = typename tensor_type::shape_type;
//     using slice_type = typename tensor_type::slice_type;
//     using nop_type = typename slice_type::nop_type;
//     using helpers_for_testing::apply_by_element;
//     const nop_type nop;
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //slice view
//         //init-list subs
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}({{nop,nop,-1},{0,-1}}), tensor_type{{4,5},{1,2}}),
//         //variadic slices subs
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(slice_type{},slice_type{1}), tensor_type{{2,3},{5,6}}),
//         //variadic mixed subs
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(slice_type{},1), tensor_type{2,5}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(1,slice_type{}), tensor_type{4,5,6}),
//         //variadic index subs
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(0), tensor_type{1,2,3}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(0,1), tensor_type(2)),
//         //slice container subs
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(std::vector<slice_type>{slice_type{},slice_type{1,-1}}), tensor_type{{2},{5}}),
//         //transpose view
//         //variadic, no subs
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(), tensor_type{{{1,5},{3,7}},{{2,6},{4,8}}}),
//         //variadic, index subs
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(1,0,2), tensor_type{{{1,2},{5,6}},{{3,4},{7,8}}}),
//         //container subs
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(std::vector<int>{2,0,1}), tensor_type{{{1,3},{5,7}},{{2,4},{6,8}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose({2,0,1}), tensor_type{{{1,3},{5,7}},{{2,4},{6,8}}}),
//         //reshape view
//         //variadic, no subs
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(), tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
//         //variadic, index subs
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(-1,1), tensor_type{{1},{2},{3},{4},{5},{6},{7},{8}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(-1,4), tensor_type{{1,2,3,4},{5,6,7,8}}),
//         //container subs
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(std::vector<int>{2,-1}), tensor_type{{1,2,3,4},{5,6,7,8}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(shape_type{2,2,2}, f_order{}), tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape(shape_type{-1,4}, f_order{}), tensor_type{{1,3,2,4},{5,7,6,8}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape({-1,2}, f_order{}), tensor_type{{1,2},{5,6},{3,4},{7,8}}),
//         //index mapping view
//         //variadic subs
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(tensor<int>{1,0},tensor<int>{0,1}), tensor_type{{5,6},{3,4}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(tensor<int>(1),tensor<int>{{0,1},{1,0}}), tensor_type{{{5,6},{7,8}},{{7,8},{5,6}}}),
//         //container subs
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(std::vector<tensor<int>>{tensor<int>{1,0},tensor<int>{0,1}}), tensor_type{{5,6},{3,4}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(std::vector<tensor<int>>{tensor<int>(1),tensor<int>{{0,1},{1,0}}}), tensor_type{{{5,6},{7,8}},{{7,8},{5,6}}}),
//         //bool mapping view
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(tensor<bool>{{{true,false},{false,true}}}), tensor_type{1,4})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);

//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_reduce","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using helpers_for_testing::apply_by_element;
//     auto sum = [](auto first, auto last){
//         const auto& init = *first;
//         return std::accumulate(++first,last,init,std::plus{});
//     };

//     //0tensor,1axes,2operation,3expected
//     auto test_data = std::make_tuple(
//         //single axis
//         std::make_tuple(tensor_type{1,2,3,4,5},dim_type{0},sum,tensor_type(15)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{0},sum,tensor_type{5,7,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},dim_type{1},sum,tensor_type{6,15}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),dim_type{0},sum,tensor_type{6,15}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),dim_type{1},sum,tensor_type{5,7,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),dim_type{0},sum,tensor_type{11,15,19}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),dim_type{1},sum,tensor_type{18,27}),
//         //axes container
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<dim_type>{0},sum,tensor_type(15)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{0},sum,tensor_type{5,7,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{1},sum,tensor_type{6,15}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{0,1},sum,tensor_type(21)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{0},sum,tensor_type{6,15}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{1},sum,tensor_type{5,7,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(),std::vector<dim_type>{1,0},sum,tensor_type(21)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{0},sum,tensor_type{11,15,19}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{1},sum,tensor_type{18,27}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{0,1,2}+tensor_type(3),std::vector<dim_type>{0,1},sum,tensor_type(45)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<dim_type>{},sum,tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<dim_type>{},sum,tensor_type{{1,2,3},{4,5,6}})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto operation = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         auto result = ten.reduce(axes,operation);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_reduce_overload","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using helpers_for_testing::apply_by_element;
//     auto sum = [](auto first, auto last){
//         const auto& init = *first;
//         return std::accumulate(++first,last,init,std::plus{});
//     };
//     //axes initializer_list
//     REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce({0},sum,true) == tensor_type{{{6,8},{10,12}}});
//     REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce({0},sum,false) == tensor_type{{6,8},{10,12}});
//     REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce({0},sum) == tensor_type{{6,8},{10,12}});
//     REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce({0,1},sum) == tensor_type{16,20});
//     REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce({0,2},sum) == tensor_type{14,22});
//     //like over flatten
//     REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce(sum,false) == tensor_type(36));
//     REQUIRE(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reduce(sum) == tensor_type(36));
// }

// TEST_CASE("test_tensor_slide","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using index_type = typename tensor_type::index_type;
//     using helpers_for_testing::apply_by_element;

//     auto cumsum = [](auto first, auto, auto dfirst, auto dlast){
//         auto cumsum_ = *first;
//         *dfirst = cumsum_;
//         for(++dfirst,++first;dfirst!=dlast;++dfirst,++first){
//             cumsum_+=*first;
//             *dfirst = cumsum_;
//         }
//     };

//     auto diff_1 = [](auto first, auto, auto dfirst, auto dlast){
//         for(;dfirst!=dlast;++dfirst){
//             auto prev = *first;
//             *dfirst = *++first - prev;
//         }
//     };
//     //0tensor,1axis,2operation,3window_size,4window_step,5expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{1,2,3,4,5},dim_type{0},cumsum,index_type{1},index_type{1},tensor_type{1,3,6,10,15}),
//         std::make_tuple(tensor_type{1,2,0,4,3,2,5},dim_type{0},diff_1,index_type{2},index_type{1},tensor_type{1,-2,4,-1,-1,3}),
//         std::make_tuple(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}},dim_type{0},diff_1,index_type{2},index_type{1},tensor_type{{1,-1,0,-4},{1,0,2,5},{-1,0,0,-4}}),
//         std::make_tuple(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}},dim_type{1},diff_1,index_type{2},index_type{1},tensor_type{{1,-2,4},{-1,-1,0},{-2,1,3},{-1,1,-1}})

//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto operation = std::get<2>(t);
//         auto window_size = std::get<3>(t);
//         auto window_step = std::get<4>(t);
//         auto expected = std::get<5>(t);
//         auto result = ten.slide(axis,operation,window_size,window_step);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_slide_overload","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using helpers_for_testing::apply_by_element;

//     auto diff_1 = [](auto first, auto, auto dfirst, auto dlast){
//         for(;dfirst!=dlast;++dfirst){
//             auto prev = *first;
//             *dfirst = *++first - prev;
//         }
//     };
//     //like over flatten
//     REQUIRE(tensor_type{2,3,1,0,2,4,2,1,6,3}.slide(diff_1,2,1) == tensor_type{1,-2,-1,2,2,-2,-1,5,-3});
//     REQUIRE(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}}.slide(diff_1,2,1) == tensor_type{1,-2,4,-2,-1,-1,0,3,-2,1,3,-3,-1,1,-1});
// }

// TEST_CASE("test_tensor_transform","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using helpers_for_testing::apply_by_element;

//     auto sort = [](auto first, auto last){
//         std::sort(first,last);
//     };

//     //0tensor,1axis,2operation,3expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{1,2,0,4,3,2,5},0,sort,tensor_type{0,1,2,2,3,4,5}),
//         std::make_tuple(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}},0,sort,tensor_type{{1,1,0,0},{2,1,0,1},{2,1,2,4},{3,2,2,5}}),
//         std::make_tuple(tensor_type{{1,2,0,4},{2,1,0,0},{3,1,2,5},{2,1,2,1}},1,sort,tensor_type{{0,1,2,4},{0,0,1,2},{1,2,3,5},{1,1,2,2}})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto operation = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         ten.transform(axis,operation);
//         REQUIRE(ten == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_ravel","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1order,2expected
//     auto test_data = std::make_tuple(
//         //c_order
//         std::make_tuple(tensor_type(2),c_order{},tensor_type{2}),
//         std::make_tuple(tensor_type{},c_order{},tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),c_order{},tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3,4,5},c_order{},tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},c_order{},tensor_type{1,2,3,4,5,6,7,8,9,10,11,12}),
//         //f_order
//         std::make_tuple(tensor_type(2),f_order{},tensor_type{2}),
//         std::make_tuple(tensor_type{},f_order{},tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),f_order{},tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3,4,5},f_order{},tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},f_order{},tensor_type{1,7,4,10,2,8,5,11,3,9,6,12})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto order = std::get<1>(t);
//         auto expected = std::get<2>(t);
//         auto result = ten.ravel(order);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_tensor_math_methods","[test_tensor]")
// {
//     using value_type = int;
//     using gtensor::tensor;
//     using tensor_type = gtensor::tensor<value_type>;

//     //all
//     REQUIRE(tensor_type(2).all() == tensor<bool>(true));
//     REQUIRE(tensor_type(0).all(true) == tensor<bool>(false));
//     REQUIRE(tensor_type{1,2,3,4,5}.all() == tensor<bool>(true));
//     REQUIRE(tensor_type{1,0,3,0,5}.all() == tensor<bool>(false));
//     REQUIRE(tensor_type{{{3,1,2,3},{1,2,3,2}},{{1,2,3,4},{1,2,3,4}}}.all() == tensor<bool>(true));
//     REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(true) == tensor<bool>{{{false}}});
//     REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(2) == tensor<bool>{{false,false},{true,true}});
//     REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(1,true) == tensor<bool>{{{false,true,true,false}},{{true,true,true,true}}});
//     REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all({0,1}) == tensor<bool>{false,true,true,false});
//     REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all({1,-1},true) == tensor<bool>{{{false}},{{true}}});
//     REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(std::vector<int>{0,1}) == tensor<bool>{false,true,true,false});
//     REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(std::vector<int>{1,-1},true) == tensor<bool>{{{false}},{{true}}});

//     //any
//     REQUIRE(tensor_type(2).any() == tensor<bool>(true));
//     REQUIRE(tensor_type(0).any(true) == tensor<bool>(false));
//     REQUIRE(tensor_type{1,2,3,4,5}.any() == tensor<bool>(true));
//     REQUIRE(tensor_type{0,0,0,0,0}.any() == tensor<bool>(false));
//     REQUIRE(tensor_type{{{3,1,2,3},{1,2,3,2}},{{1,2,3,4},{1,2,3,4}}}.any() == tensor<bool>(true));
//     REQUIRE(tensor_type{{{0,0,0,0},{0,0,0,0}},{{0,0,0,0},{0,0,0,0}}}.any(true) == tensor<bool>{{{false}}});
//     REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(2) == tensor<bool>{{true,false},{true,true}});
//     REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(1,true) == tensor<bool>{{{false,true,true,false}},{{true,true,true,false}}});
//     REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any({0,1}) == tensor<bool>{true,true,true,false});
//     REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any({1,-1},true) == tensor<bool>{{{true}},{{true}}});
//     REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(std::vector<int>{0,1}) == tensor<bool>{true,true,true,false});
//     REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(std::vector<int>{1,-1},true) == tensor<bool>{{{true}},{{true}}});

//     //max
//     REQUIRE(tensor_type(2).max() == tensor_type(2));
//     REQUIRE(tensor_type(2).max(true) == tensor_type(2));
//     REQUIRE(tensor_type(2).max(true,10) == tensor_type(10));
//     REQUIRE(tensor_type(2).max(true,-10) == tensor_type(2));
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max() == tensor_type(4));
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(true) == tensor_type{{{4}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(true,10) == tensor_type{{{10}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(true,-10) == tensor_type{{{4}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(2) == tensor_type{{3,4},{3,3}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(0,true) == tensor_type{{{-2,3,3,3},{4,3,3,2}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(0,true,1) == tensor_type{{{1,3,3,3},{4,3,3,2}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max({1,-1}) == tensor_type{4,3});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max({0,1},true) == tensor_type{{{4,3,3,3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max({0,1},true,0) == tensor_type{{{4,3,3,3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(std::vector<int>{0,1},true) == tensor_type{{{4,3,3,3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(std::vector<int>{0,1},true,0) == tensor_type{{{4,3,3,3}}});

//     //min
//     REQUIRE(tensor_type(2).min() == tensor_type(2));
//     REQUIRE(tensor_type(2).min(true) == tensor_type(2));
//     REQUIRE(tensor_type(2).min(true,10) == tensor_type(2));
//     REQUIRE(tensor_type(2).min(true,-10) == tensor_type(-10));
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min() == tensor_type(-3));
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true) == tensor_type{{{-3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true,10) == tensor_type{{{-3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true,-10) == tensor_type{{{-10}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(2) == tensor_type{{-3,0},{-2,-2}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(0,true) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(0,true,1) == tensor_type{{{-3,1,1,-2},{0,1,-2,0}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({1,-1}) == tensor_type{-3,-2});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({0,1},true) == tensor_type{{{-3,3,-2,-2}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({0,1},true,0) == tensor_type{{{-3,0,-2,-2}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(std::vector<int>{0,1},true) == tensor_type{{{-3,3,-2,-2}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(std::vector<int>{0,1},true,0) == tensor_type{{{-3,0,-2,-2}}});

//     //sum
//     REQUIRE(tensor_type(2).sum() == tensor_type(2));
//     REQUIRE(tensor_type(2).sum(true) == tensor_type(2));
//     REQUIRE(tensor_type(2).sum(true,10) == tensor_type(12));
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum() == tensor_type(19));
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(true) == tensor_type{{{19}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(true,10) == tensor_type{{{29}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(2) == tensor_type{{1,10},{5,3}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(0,true) == tensor_type{{{-5,6,4,1},{4,6,1,2}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(0,true,1) == tensor_type{{{-4,7,5,2},{5,7,2,3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum({1,-1}) == tensor_type{11,8});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum({0,1},true) == tensor_type{{{-1,12,5,3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum({0,1},true,0) == tensor_type{{{-1,12,5,3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(std::vector<int>{0,1},true) == tensor_type{{{-1,12,5,3}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(std::vector<int>{0,1},true,0) == tensor_type{{{-1,12,5,3}}});

//     //prod
//     REQUIRE(tensor_type(2).prod() == tensor_type(2));
//     REQUIRE(tensor_type(2).prod(true) == tensor_type(2));
//     REQUIRE(tensor_type(2).prod(true,10) == tensor_type(20));
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod() == tensor_type(7776));
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(true) == tensor_type{{{7776}}});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(true,10) == tensor_type{{{77760}}});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(2) == tensor_type{{12,12},{3,18}});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(0,true) == tensor_type{{{2,6,1,3},{6,3,3,4}}});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(0,true,1) == tensor_type{{{2,6,1,3},{6,3,3,4}}});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod({1,-1}) == tensor_type{144,54});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod({0,1},true) == tensor_type{{{12,18,3,12}}});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod({0,1},true,0) == tensor_type{{{0,0,0,0}}});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(std::vector<int>{0,1},true) == tensor_type{{{12,18,3,12}}});
//     REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(std::vector<int>{0,1},true,0) == tensor_type{{{0,0,0,0}}});

//     //cumsum
//     REQUIRE(tensor_type(2).cumsum() == tensor_type{2});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum() == tensor_type{-3,0,3,1,5,8,11,11,9,12,13,16,16,19,17,19});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(0) == tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-5,6,4,1},{4,6,1,2}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(1) == tensor_type{{{-3,3,3,-2},{1,6,6,-2}},{{-2,3,1,3},{-2,6,-1,5}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(2) == tensor_type{{{-3,0,3,1},{4,7,10,10}},{{-2,1,2,5},{0,3,1,3}}});

//     //cumprod
//     REQUIRE(tensor_type(2).cumprod() == tensor_type{2});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod() == tensor_type{-3,-9,-27,54,216,648,1944,0,0,0,0,0,0,0,0,0});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(0) == tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{6,9,3,-6},{0,9,-6,0}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(1) == tensor_type{{{-3,3,3,-2},{-12,9,9,0}},{{-2,3,1,3},{0,9,-2,6}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(2) == tensor_type{{{-3,-9,-27,54},{4,12,36,0}},{{-2,-6,-6,-18},{0,0,0,0}}});
// }

// TEST_CASE("test_tensor_statistic_methods","[test_tensor]")
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::tensor_close;
//     using tensor_type = gtensor::tensor<value_type>;

//     //ptp
//     REQUIRE(tensor_type(2).ptp() == tensor_type(0));
//     REQUIRE(tensor_type(0).ptp(true) == tensor_type(0));
//     REQUIRE(tensor_type{1,2,3,4,5}.ptp(true) == tensor_type{4});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp() == tensor_type(7));
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(true) == tensor_type{{{7}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(2) == tensor_type{{6,4},{5,5}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(1,true) == tensor_type{{{7,0,0,2}},{{2,0,3,1}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp({0,1}) == tensor_type{7,0,5,5});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp({1,-1},true) == tensor_type{{{7}},{{5}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(std::vector<int>{0,1}) == tensor_type{7,0,5,5});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(std::vector<int>{1,-1},true) == tensor_type{{{7}},{{5}}});

//     //mean
//     REQUIRE(tensor_type(2).mean() == tensor_type(2));
//     REQUIRE(tensor_type(0).mean(true) == tensor_type(0));
//     REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.mean(true), tensor_type{3},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean() ,tensor_type(1.187),1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(true) ,tensor_type{{{1.187}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(2) ,tensor_type{{0.25,2.5},{1.25,0.75}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(1,true) ,tensor_type{{{0.5,3.0,3.0,-1.0}},{{-1.0,3.0,-0.5,2.5}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean({0,1}) ,tensor_type{-0.25,3.0,1.25,0.75},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean({1,-1},true) ,tensor_type{{{1.375}},{{1.0}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(std::vector<int>{0,1}) ,tensor_type{-0.25,3.0,1.25,0.75},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(std::vector<int>{1,-1},true) ,tensor_type{{{1.375}},{{1.0}}},1E-2,1E-2));

//     //median
//     REQUIRE(tensor_type(2).median() == tensor_type(2));
//     REQUIRE(tensor_type(0).median(true) == tensor_type(0));
//     REQUIRE(tensor_type{1,2,3,4,5}.median(true) == tensor_type{3});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median() == tensor_type(2.5));
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(true) == tensor_type{{{2.5}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(2) == tensor_type{{0.5,3.0},{2.0,1.0}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(1,true) == tensor_type{{{0.5,3.0,3.0,-1.0}},{{-1.0,3.0,-0.5,2.5}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median({0,1}) == tensor_type{-1,3,2,1});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median({1,-1},true) == tensor_type{{{3.0}},{{1.5}}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(std::vector<int>{0,1}) == tensor_type{-1,3,2,1});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(std::vector<int>{1,-1},true) == tensor_type{{{3.0}},{{1.5}}});

//     //var
//     REQUIRE(tensor_type(2).var() == tensor_type(0));
//     REQUIRE(tensor_type(0).var(true) == tensor_type(0));
//     REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.var(true), tensor_type{2},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var() ,tensor_type(5.152),1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(true) ,tensor_type{{{5.152}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(2) ,tensor_type{{7.6875,2.25},{4.1875,3.6875}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(1,true) ,tensor_type{{{12.25,0.0,0.0,1.0}},{{1.0,0.0,2.25,0.25}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var({0,1}) ,tensor_type{7.1875,0.0,4.1875,3.6875},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var({1,-1},true) ,tensor_type{{{6.234}},{{4.0}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(std::vector<int>{0,1}) ,tensor_type{7.1875,0.0,4.1875,3.6875},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(std::vector<int>{1,-1},true) ,tensor_type{{{6.234}},{{4.0}}},1E-2,1E-2));

//     //std
//     REQUIRE(tensor_type(2).std() == tensor_type(0));
//     REQUIRE(tensor_type(0).std(true) == tensor_type(0));
//     REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.std(true), tensor_type{1.414},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.std() ,tensor_type(2.279),1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.std(true) ,tensor_type{{{2.270}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.std(2) ,tensor_type{{2.772,1.5},{2.046,1.920}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.std(1,true) ,tensor_type{{{3.5,0.0,0.0,1.0}},{{1.0,0.0,1.5,0.5}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.std({0,1}) ,tensor_type{2.680,0.0,2.046,1.920},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.std({1,-1},true) ,tensor_type{{{2.496}},{{2.0}}},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.std(std::vector<int>{0,1}) ,tensor_type{2.680,0.0,2.046,1.920},1E-2,1E-2));
//     REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.std(std::vector<int>{1,-1},true) ,tensor_type{{{2.496}},{{2.0}}},1E-2,1E-2));
// }

// TEST_CASE("test_tensor_indexing_methods","[test_tensor]")
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using gtensor::tensor_close;
//     using tensor_type = gtensor::tensor<value_type>;

//     //take
//     REQUIRE(tensor_type(2).take(tensor<int>{}) == tensor_type{});
//     REQUIRE(tensor_type(3).take(tensor<int>{},0) == tensor_type{});
//     REQUIRE(tensor_type(2).take(tensor<int>{0}) == tensor_type{2});
//     REQUIRE(tensor_type(2).take(tensor<int>{{0,0},{0,0}}) == tensor_type{{2,2},{2,2}});
//     REQUIRE(tensor_type{1,2,3,4,5}.take(tensor<int>{}) == tensor_type{});
//     REQUIRE(tensor_type{1,2,3,4,5}.take(tensor<int>{1,0,2,0,3,3,1}) == tensor_type{2,1,3,1,4,4,2});
//     REQUIRE(tensor_type{1,2,3,4,5}.take(tensor<int>{{1,0,2},{0,3,3}},0) == tensor_type{{2,1,3},{1,4,4}});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{}) == tensor_type{});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{1,2,2,0,3}) == tensor_type{3,3,3,-3,-2});
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{1,0,0,1},0) ==
//         tensor_type{{{-2,3,1,3},{0,3,-2,2}},{{-3,3,3,-2},{4,3,3,0}},{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}
//     );
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{1,0,1,0},1) ==
//         tensor_type{{{4,3,3,0},{-3,3,3,-2},{4,3,3,0},{-3,3,3,-2}},{{0,3,-2,2},{-2,3,1,3},{0,3,-2,2},{-2,3,1,3}}}
//     );
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{1,0,3,2,2},2) ==
//         tensor_type{{{3,-3,-2,3,3},{3,4,0,3,3}},{{3,-2,3,1,1},{3,0,2,-2,-2}}}
//     );
//     REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{{1,0,1},{0,1,0}},1) ==
//         tensor_type{{{{4,3,3,0},{-3,3,3,-2},{4,3,3,0}},{{-3,3,3,-2},{4,3,3,0},{-3,3,3,-2}}},{{{0,3,-2,2},{-2,3,1,3},{0,3,-2,2}},{{-2,3,1,3},{0,3,-2,2},{-2,3,1,3}}}}
//     );
// }

// TEST_CASE("test_tensor_sort_search_methods_sort","[test_tensor]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;

//     //sort
//     tensor_type t{4,6,-6,-6,-5,6,-1,-8,-5,-9,-5,-1,5,-8,0,-6,-7,6,-7,-8,-5,7,7,-8,3,0,-2,-10,-7,-7,9,-9,-3,-4,-7,-9,-3,8,-4,-5,-1,-2,-1,-6,-2,8,-6,2,-7,9};
//     tensor_type x{{{7,-3,2,-2,4},{4,6,6,4,5},{7,6,4,4,1},{5,-6,-10,-3,-9}},{{-8,0,8,-3,-3},{-7,-4,-9,-2,7},{3,0,5,-10,-7},{7,9,-3,-3,9}},{{4,9,-5,-9,-5},{-1,7,-7,3,-9},{-4,7,6,6,2},{9,6,-4,-8,-3}}};
//     SECTION("sort")
//     {
//         t.sort();
//         REQUIRE(t == tensor_type{-10,-9,-9,-9,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-3,-3,-2,-2,-2,-1,-1,-1,-1,0,0,2,3,4,5,6,6,6,7,7,8,8,9,9});
//         x.sort();
//         REQUIRE(x == tensor_type{{{-3,-2,2,4,7},{4,4,5,6,6},{1,4,4,6,7},{-10,-9,-6,-3,5}},{{-8,-3,-3,0,8},{-9,-7,-4,-2,7},{-10,-7,0,3,5},{-3,-3,7,9,9}},{{-9,-5,-5,4,9},{-9,-7,-1,3,7},{-4,2,6,6,7},{-8,-4,-3,6,9}}});
//     }
//     SECTION("sort_axis")
//     {
//         t.sort(0);
//         REQUIRE(t == tensor_type{-10,-9,-9,-9,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-3,-3,-2,-2,-2,-1,-1,-1,-1,0,0,2,3,4,5,6,6,6,7,7,8,8,9,9});
//         SECTION("sort_0")
//         {
//             x.sort(0);
//             REQUIRE(x == tensor_type{{{-8,-3,-5,-9,-5},{-7,-4,-9,-2,-9},{-4,0,4,-10,-7},{5,-6,-10,-8,-9}},{{4,0,2,-3,-3},{-1,6,-7,3,5},{3,6,5,4,1},{7,6,-4,-3,-3}},{{7,9,8,-2,4},{4,7,6,4,7},{7,7,6,6,2},{9,9,-3,-3,9}}});
//         }
//         SECTION("sort_0")
//         {
//             x.sort(1);
//             REQUIRE(x == tensor_type{{{4,-6,-10,-3,-9},{5,-3,2,-2,1},{7,6,4,4,4},{7,6,6,4,5}},{{-8,-4,-9,-10,-7},{-7,0,-3,-3,-3},{3,0,5,-3,7},{7,9,8,-2,9}},{{-4,6,-7,-9,-9},{-1,7,-5,-8,-5},{4,7,-4,3,-3},{9,9,6,6,2}}});
//         }
//     }
//     SECTION("sort_axis_comparator")
//     {
//         t.sort(0,std::greater<void>{});
//         REQUIRE(t == tensor_type{9,9,8,8,7,7,6,6,6,5,4,3,2,0,0,-1,-1,-1,-1,-2,-2,-2,-3,-3,-4,-4,-5,-5,-5,-5,-5,-6,-6,-6,-6,-6,-7,-7,-7,-7,-7,-7,-8,-8,-8,-8,-9,-9,-9,-10});
//         x.sort(1,std::greater<void>{});
//         REQUIRE(x == tensor_type{{{7,6,6,4,5},{7,6,4,4,4},{5,-3,2,-2,1},{4,-6,-10,-3,-9}},{{7,9,8,-2,9},{3,0,5,-3,7},{-7,0,-3,-3,-3},{-8,-4,-9,-10,-7}},{{9,9,6,6,2},{4,7,-4,3,-3},{-1,7,-5,-8,-5},{-4,6,-7,-9,-9}}});
//     }
// }

// TEST_CASE("test_tensor_sort_search_methods","[test_tensor]")
// {
//     using value_type = double;
//     using gtensor::tensor;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::take_along_axis;

//     const tensor_type t{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}};
//     //argsort
//     REQUIRE(take_along_axis(t,t.argsort(),-1) == tensor_type{{{-3,-2,3,3},{0,3,3,4}},{{-2,1,3,3},{-2,0,2,3}}});
//     REQUIRE(take_along_axis(t,t.argsort(0),0) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}},{{-2,3,3,3},{4,3,3,2}}});
//     REQUIRE(take_along_axis(t,t.argsort(1,std::greater<void>{}),1) == tensor_type{{{4,3,3,0},{-3,3,3,-2}},{{0,3,1,3},{-2,3,-2,2}}});
//     //argmax
//     REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax() == tensor<int>(20));
//     REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(0) == tensor<int>{{2,0,2,1,2},{2,1,1,1,1}});
//     REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(1,true) == tensor<int>{{{1,0,1,1,0}},{{1,0,1,0,0}},{{0,0,0,1,0}}});
//     REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(2) == tensor<int>{{1,0},{3,2},{0,0}});
//     REQUIRE(take_along_axis(t,t.argmax(0,true),0) == tensor_type{{{-2,3,3,3},{4,3,3,2}}});
//     REQUIRE(take_along_axis(t,t.argmax(1,true),1) == tensor_type{{{4,3,3,0}},{{0,3,1,3}}});
//     REQUIRE(take_along_axis(t,t.argmax(2,true),2) == tensor_type{{{3},{4}},{{3},{3}}});
//     //argmin
//     REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin() == tensor<int>(29));
//     REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(0) == tensor<int>{{0,1,0,0,0},{0,0,2,0,2}});
//     REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(1,true) == tensor<int>{{{0,1,0,0,1}},{{0,1,0,1,1}},{{1,1,1,0,1}}});
//     REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(2) == tensor<int>{{2,1},{0,4},{3,4}});
//     REQUIRE(take_along_axis(t,t.argmin(0,true),0) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}}});
//     REQUIRE(take_along_axis(t,t.argmin(1,true),1) == tensor_type{{{-3,3,3,-2}},{{-2,3,-2,2}}});
//     REQUIRE(take_along_axis(t,t.argmin(2,true),2) == tensor_type{{{-3},{0}},{{-2},{-2}}});
//     //nonzero
//     REQUIRE(t(t.nonzero()) == tensor_type{-3,3,3,-2,4,3,3,-2,3,1,3,3,-2,2});
// }

TEMPLATE_TEST_CASE("test_tensor_view_chain","[test_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type=double;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using slice_type = typename tensor_type::slice_type;
    using helpers_for_testing::apply_by_element;

    const auto input = tensor_type{
        {{{{7,5,8,5},{0,5,5,1},{3,8,0,8}}},{{{0,0,2,5},{1,2,3,0},{6,7,3,7}}}},
        {{{{4,8,0,7},{0,0,2,4},{1,5,8,5}}},{{{6,8,4,8},{4,1,3,2},{7,0,6,2}}}},
        {{{{7,3,6,4},{2,6,4,7},{0,3,3,1}}},{{{2,1,3,0},{4,7,4,4},{7,6,3,3}}}}
    };  //(3,2,1,3,4)

    //0tensor,1view_maker,2expected
    auto test_data = std::make_tuple(
        //transpose
        std::make_tuple(input,[](const auto& t_){return t_.transpose().transpose(1,0,2,4,3);},
            tensor_type{{{{{7,0},{4,6},{7,2}}},{{{5,0},{8,8},{3,1}}},{{{8,2},{0,4},{6,3}}},{{{5,5},{7,8},{4,0}}}},{{{{0,1},{0,4},{2,4}}},{{{5,2},{0,1},{6,7}}},{{{5,3},{2,3},{4,4}}},{{{1,0},{4,2},{7,4}}}},{{{{3,6},{1,7},{0,7}}},{{{8,7},{5,0},{3,6}}},{{{0,3},{8,6},{3,3}}},{{{8,7},{5,2},{1,3}}}}}
        ),
        //slice
        std::make_tuple(input,[](const auto& t_){return t_(1,slice_type{},0,slice_type{{},{},-1},slice_type{{},-1})(slice_type{},1,slice_type{{},{},2});},
            tensor_type{{0,2},{4,3}}
        ),
        //reshape
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,6,4},c_order{}).reshape({3,24},c_order{});},
            tensor_type{{7,5,8,5,0,5,5,1,3,8,0,8,0,0,2,5,1,2,3,0,6,7,3,7},{4,8,0,7,0,0,2,4,1,5,8,5,6,8,4,8,4,1,3,2,7,0,6,2},{7,3,6,4,2,6,4,7,0,3,3,1,2,1,3,0,4,7,4,4,7,6,3,3}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,6,4},f_order{}).reshape({3,24},f_order{});},
            tensor_type{{7,0,0,1,3,6,5,0,5,2,8,7,8,2,5,3,0,3,5,5,1,0,8,7},{4,6,0,4,1,7,8,8,0,1,5,0,0,4,2,3,8,6,7,8,4,2,5,2},{7,2,2,4,0,7,3,1,6,7,3,6,6,3,4,4,3,3,4,0,7,4,1,3}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,6,4},c_order{}).reshape({3,24},f_order{});},
            tensor_type{{7,0,3,0,1,6,5,5,8,0,2,7,8,5,0,2,3,3,5,1,8,5,0,7},{4,0,1,6,4,7,8,0,5,8,1,0,0,2,8,4,3,6,7,4,5,8,2,2},{7,2,0,2,4,7,3,6,3,1,7,6,6,4,3,3,4,3,4,7,1,0,4,3}}
        ),
        //transpose reshape
        std::make_tuple(input,[](const auto& t_){return t_.transpose(2,0,1,3,4).reshape({6,12},c_order{});},
            tensor_type{{7,5,8,5,0,5,5,1,3,8,0,8},{0,0,2,5,1,2,3,0,6,7,3,7},{4,8,0,7,0,0,2,4,1,5,8,5},{6,8,4,8,4,1,3,2,7,0,6,2},{7,3,6,4,2,6,4,7,0,3,3,1},{2,1,3,0,4,7,4,4,7,6,3,3}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.transpose(2,0,1,3,4).reshape({6,12},f_order{});},
            tensor_type{{7,0,3,5,5,8,8,5,0,5,1,8},{4,0,1,8,0,5,0,2,8,7,4,5},{7,2,0,3,6,3,6,4,3,4,7,1},{0,1,6,0,2,7,2,3,3,5,0,7},{6,4,7,8,1,0,4,3,6,8,2,2},{2,4,7,1,7,6,3,4,3,0,4,3}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,1,4,6},c_order{}).transpose(2,0,1,3);},
            tensor_type{{{{7,5,8,5,0,5}},{{4,8,0,7,0,0}},{{7,3,6,4,2,6}}},{{{5,1,3,8,0,8}},{{2,4,1,5,8,5}},{{4,7,0,3,3,1}}},{{{0,0,2,5,1,2}},{{6,8,4,8,4,1}},{{2,1,3,0,4,7}}},{{{3,0,6,7,3,7}},{{3,2,7,0,6,2}},{{4,4,7,6,3,3}}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({3,1,4,6},f_order{}).transpose(2,0,1,3);},
            tensor_type{{{{7,3,5,8,0,1}},{{4,1,0,0,8,4}},{{7,0,6,6,3,7}}},{{{0,6,2,2,3,0}},{{6,7,1,4,6,2}},{{2,7,7,3,3,4}}},{{{0,5,8,5,5,8}},{{0,8,5,2,7,5}},{{2,3,3,4,4,1}}},{{{1,0,7,3,5,7}},{{4,8,0,3,8,2}},{{4,1,6,4,0,3}}}}
        ),
        //transpose slice
        std::make_tuple(input,[](const auto& t_){return t_({{},{1},{},{{},{},-1},{1,-1}}).transpose(3,2,1,0,4)({{{},{},2},{},{},{{},{},2},{}});},
            tensor_type{{{{{7,3},{6,3}}}},{{{{0,2},{1,3}}}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.transpose(3,2,1,4,0)(slice_type{},0,slice_type{},2,slice_type{{},{},2});},
            tensor_type{{{8,6},{2,3}},{{5,4},{3,4}},{{0,3},{3,3}}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(2,1,0).transpose();},
            tensor_type{{2,4,7},{1,7,6},{3,4,3},{0,4,3}}
        ),
        //slice reshape
        std::make_tuple(input,[](const auto& t_){return t_.reshape({6,3,4},c_order{})(slice_type{},1,slice_type{1,{},2});},
            tensor_type{{5,1},{2,0},{0,4},{1,2},{6,7},{7,4}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_.reshape({6,3,4},f_order{})(slice_type{},1,slice_type{1,{},2});},
            tensor_type{{5,1},{0,4},{6,7},{2,0},{1,2},{7,4}}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(1,slice_type{},0,slice_type{{},{},-1}).reshape({-1},c_order{});},
            tensor_type{1,5,8,5,0,0,2,4,4,8,0,7,7,0,6,2,4,1,3,2,6,8,4,8}
        ),
        std::make_tuple(input,[](const auto& t_){return t_(1,slice_type{},0,slice_type{{},{},-1}).reshape({-1},f_order{});},
            tensor_type{1,7,0,4,4,6,5,0,0,1,8,8,8,6,2,3,0,4,5,2,4,2,7,8}
        ),
        //expression input
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_.reshape({-1},c_order{});},
            tensor_type{21,19,22,19,17,20,19,17,20,22,20,23,20,18,21,20,18,21,17,15,18,15,13,16,15,13,16}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_.reshape({-1},f_order{});},
            tensor_type{21,22,17,19,20,15,19,20,15,19,20,15,17,18,13,17,18,13,22,23,18,20,21,16,20,21,16}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_.transpose();},
            tensor_type{{{21,22,17},{19,20,15},{19,20,15}},{{19,20,15},{17,18,13},{17,18,13}},{{22,23,18},{20,21,16},{20,21,16}}}
        ),
        std::make_tuple(
            input.reshape(-1)({{0,27}}).reshape(3,3,3)({{0,1},{0,1},{}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{0,1},{},{0,1}}) + input.reshape(-1)({{0,27,1}}).reshape(3,3,3)({{},{0,1},{0,1}}),
            [](const auto& t_){return t_(slice_type{1},0,slice_type{1,{},-1});},
            tensor_type{{20,22},{15,17}}
        )
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto view_maker = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = view_maker(ten);
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}