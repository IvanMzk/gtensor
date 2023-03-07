#include <tuple>
#include "catch.hpp"
#include "tensor_init_list.hpp"

TEST_CASE("test_list_depth","[test_tensor_init_list]"){
    gtensor::detail::nested_initializer_list_type<int,1>::type l1 = {1,2,3};
    gtensor::detail::nested_initializer_list_type<int,2>::type l2 = {{1,2},{3,4},{5,6}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3 = {{{1},{2}},{{3},{4}},{{5},{6}}};
    gtensor::detail::nested_initializer_list_type<bool,2>::type l4 = {{true},{false}};
    REQUIRE(gtensor::detail::nested_initialiser_list_depth<decltype(l1)>::value == 1);
    REQUIRE(gtensor::detail::nested_initialiser_list_depth<decltype(l2)>::value == 2);
    REQUIRE(gtensor::detail::nested_initialiser_list_depth<decltype(l3)>::value == 3);
    REQUIRE(gtensor::detail::nested_initialiser_list_depth<decltype(l4)>::value == 2);
}

TEST_CASE("test_list_parser_empty","[test_tensor_init_list]"){
    gtensor::detail::nested_initializer_list_type<int,1>::type l1 = {};
    gtensor::detail::nested_initializer_list_type<int,2>::type l2 = {{2},{2},{}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3 = {{{},{2}},{{},{}},{{},{}}};
    REQUIRE_THROWS_AS(gtensor::detail::list_parse(l1), gtensor::tensor_init_list_exception);
    REQUIRE_THROWS_AS(gtensor::detail::list_parse(l2), gtensor::tensor_init_list_exception);
    REQUIRE_THROWS_AS(gtensor::detail::list_parse(l3), gtensor::tensor_init_list_exception);

}

TEMPLATE_TEST_CASE("test_list_parser","[test_tensor_init_list]",std::vector<std::size_t>, std::vector<std::int64_t>)
{
    using container_type = TestType;
    using value_type = typename container_type::value_type;
    using gtensor::detail::nested_initializer_list_type;
    using gtensor::detail::list_parse;
    nested_initializer_list_type<int,1>::type l1 = {1,2,3};
    nested_initializer_list_type<int,2>::type l2 = {{1,2},{3,4},{5,6}};
    nested_initializer_list_type<int,3>::type l3 = {{{1},{2}},{{3},{4}},{{5},{6}}};
    REQUIRE(list_parse<value_type, container_type>(l1) == container_type{3});
    REQUIRE(list_parse<value_type, container_type>(l2) == container_type{3,2});
    REQUIRE(list_parse<value_type, container_type>(l3) == container_type{3,2,1});
}

TEMPLATE_TEST_CASE("test_list_parser_exception","[test_tensor_init_list]",std::size_t, std::uint64_t){
    gtensor::detail::nested_initializer_list_type<int,2>::type l2 = {{1,2,3},{3,4},{5,6}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3 = {{{1},{2}},{{3},{4},{0}},{{5},{6}}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3_ = {{{1,2},{2}},{{3},{4}},{{5},{6}}};
    REQUIRE_THROWS_AS(gtensor::detail::list_parse<TestType>(l2), gtensor::tensor_init_list_exception);
    REQUIRE_THROWS_AS(gtensor::detail::list_parse<TestType>(l3), gtensor::tensor_init_list_exception);
    REQUIRE_THROWS_AS(gtensor::detail::list_parse<TestType>(l3_), gtensor::tensor_init_list_exception);
}

TEST_CASE("test_fill_from_list","[test_tensor_init_list]"){
    gtensor::detail::nested_initializer_list_type<int,2>::type l2 = {{1,2,3},{3,4},{5,6}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3 = {{{1},{2}},{{3},{4},{0}},{{5},{6}}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3_ = {{{1,2},{}},{{3},{4}},{{5},{6}}};
    std::vector<int> v(7);
    REQUIRE(gtensor::detail::fill_from_list(l2,v.begin()) == 7);
    REQUIRE(v == std::vector<int>{1,2,3,3,4,5,6});
    std::vector<int> v1(7);
    REQUIRE(gtensor::detail::fill_from_list(l3,v1.begin()) == 7);
    REQUIRE(v1 == std::vector<int>{1,2,3,4,0,5,6});
    std::vector<int> v2(6);
    REQUIRE(gtensor::detail::fill_from_list(l3_,v2.begin()) == 6);
    REQUIRE(v2 == std::vector<int>{1,2,3,4,5,6});
}

TEST_CASE("test_fill_from_list_empty","[test_tensor_init_list]"){
    std::initializer_list<int> l = {};
    std::vector<int> v;
    REQUIRE(gtensor::detail::fill_from_list(l,v.begin()) == 0);
    REQUIRE(v.size() == 0);
}

TEST_CASE("test_list_size","[test_tensor_init_list]"){
    using gtensor::detail::list_size;

    gtensor::detail::nested_initializer_list_type<int,1>::type l1 = {1,2,3};
    gtensor::detail::nested_initializer_list_type<int,1>::type l1_empty = {};
    gtensor::detail::nested_initializer_list_type<int,2>::type l2 = {{1,2},{3,4},{5,6}};
    gtensor::detail::nested_initializer_list_type<int,2>::type l2_empty = {{},{},{}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3 = {{{1},{2}},{{3},{4}},{{5},{6}}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3_empty = {{{},{}},{{},{}},{{},{}}};
    gtensor::detail::nested_initializer_list_type<int,3>::type l3_bad_shape = {{{1},{2,2,2}},{{},{4,4}},{{5,5,5},{6}}};

    REQUIRE(list_size(l1) == 3);
    REQUIRE(list_size(l1_empty) == 0);
    REQUIRE(list_size(l2) == 6);
    REQUIRE(list_size(l2_empty) == 0);
    REQUIRE(list_size(l3_empty) == 0);
    REQUIRE(list_size(l3) == 6);
    REQUIRE(list_size(l3_bad_shape) == 10);
}
