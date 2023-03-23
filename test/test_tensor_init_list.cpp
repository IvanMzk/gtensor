#include <tuple>
#include "catch.hpp"
#include "tensor_init_list.hpp"
#include "helpers_for_testing.hpp"

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

TEST_CASE("test_nested_initialiser_list_value_type","[test_tensor_init_list]"){
    using gtensor::detail::nested_initialiser_list_value_type;
    REQUIRE(std::is_same_v<nested_initialiser_list_value_type<std::initializer_list<int>>::type, int>);
    REQUIRE(std::is_same_v<nested_initialiser_list_value_type<std::initializer_list<std::initializer_list<std::tuple<int,float>>>>::type, std::tuple<int,float>>);
    REQUIRE(std::is_same_v<nested_initialiser_list_value_type<std::initializer_list<std::initializer_list<std::initializer_list<std::tuple<int,float>>>>>::type, std::tuple<int,float>>);
}


TEMPLATE_TEST_CASE("test_list_parser","[test_tensor_init_list]",std::vector<std::size_t>, std::vector<std::int64_t>)
{
    using container_type = TestType;
    using value_type = typename container_type::value_type;
    using gtensor::detail::nested_init_list1_type;
    using gtensor::detail::nested_init_list2_type;
    using gtensor::detail::nested_init_list3_type;
    using gtensor::detail::list_parse;

    REQUIRE(list_parse<value_type, container_type>(nested_init_list1_type<int>{}) == container_type{0});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list1_type<int>{1,2,3}) == container_type{3});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list2_type<int>{{},{},{}}) == container_type{3,0});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list2_type<int>{{1,2},{3,4},{5,6}}) == container_type{3,2});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list3_type<int>{{{},{}},{{},{}},{{},{}}}) == container_type{3,2,0});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list3_type<int>{{{1},{2}},{{3},{4}},{{5},{6}}}) == container_type{3,2,1});
}

TEMPLATE_TEST_CASE("test_list_parser_exception","[test_tensor_init_list]",std::vector<std::size_t>, std::vector<std::int64_t>)
{
    using container_type = TestType;
    using value_type = typename container_type::value_type;
    using gtensor::detail::nested_init_list1_type;
    using gtensor::detail::nested_init_list2_type;
    using gtensor::detail::nested_init_list3_type;
    using gtensor::tensor_init_list_exception;
    using gtensor::detail::list_parse;

    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list2_type<int>{{1},{},{}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list2_type<int>{{},{1},{}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list2_type<int>{{},{},{1}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list2_type<int>{{1,2,3},{3,4},{5,6}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list3_type<int>{{{1},{2}},{{3},{4}},{{},{}}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list3_type<int>{{{1},{2}},{{3},{4},{0}},{{5},{6}}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list3_type<int>{{{1,2},{2}},{{3},{4}},{{5},{6}}})), tensor_init_list_exception);
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
