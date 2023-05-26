#include <tuple>
#include "catch.hpp"
#include "tensor_init_list.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_list_depth","[test_tensor_init_list]")
{
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::detail::nested_initialiser_list_depth;
    REQUIRE(nested_initialiser_list_depth<nested_init_list1<int>>::value == 1);
    REQUIRE(nested_initialiser_list_depth<nested_init_list2<int>>::value == 2);
    REQUIRE(nested_initialiser_list_depth<nested_init_list3<int>>::value == 3);
}

TEST_CASE("test_nested_initialiser_list_value_type","[test_tensor_init_list]")
{
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::detail::nested_initialiser_list_value_type;
    REQUIRE(std::is_same_v<nested_initialiser_list_value_type<nested_init_list1<int>>::type, int>);
    REQUIRE(std::is_same_v<nested_initialiser_list_value_type<nested_init_list2<std::tuple<int,float>>>::type, std::tuple<int,float>>);
    REQUIRE(std::is_same_v<nested_initialiser_list_value_type<nested_init_list3<std::tuple<int,float>>>::type, std::tuple<int,float>>);
}

TEMPLATE_TEST_CASE("test_list_parser","[test_tensor_init_list]",std::vector<std::size_t>, std::vector<std::int64_t>)
{
    using container_type = TestType;
    using value_type = typename container_type::value_type;
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::detail::list_parse;

    REQUIRE(list_parse<value_type, container_type>(nested_init_list1<int>{}) == container_type{0});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list1<int>{1,2,3}) == container_type{3});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list2<int>{{}}) == container_type{1,0});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list2<int>{{},{},{}}) == container_type{3,0});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list2<int>{{1,2},{3,4},{5,6}}) == container_type{3,2});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list3<int>{{{}}}) == container_type{1,1,0});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list3<int>{{{},{}},{{},{}},{{},{}}}) == container_type{3,2,0});
    REQUIRE(list_parse<value_type, container_type>(nested_init_list3<int>{{{1},{2}},{{3},{4}},{{5},{6}}}) == container_type{3,2,1});
}

TEMPLATE_TEST_CASE("test_list_parser_exception","[test_tensor_init_list]",std::vector<std::size_t>, std::vector<std::int64_t>)
{
    using container_type = TestType;
    using value_type = typename container_type::value_type;
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::tensor_init_list_exception;
    using gtensor::detail::list_parse;

    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list2<int>{{1},{},{}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list2<int>{{},{1},{}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list2<int>{{},{},{1}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list2<int>{{1,2,3},{3,4},{5,6}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list3<int>{{{1},{2}},{{3},{4}},{{},{}}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list3<int>{{{1},{2}},{{3},{4},{0}},{{5},{6}}})), tensor_init_list_exception);
    REQUIRE_THROWS_AS((list_parse<value_type, container_type>(nested_init_list3<int>{{{1,2},{2}},{{3},{4}},{{5},{6}}})), tensor_init_list_exception);
}

TEST_CASE("test_copy_from_list","[test_tensor_init_list]")
{
    using container_type = std::vector<int>;
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using gtensor::detail::copy_from_list;

    auto make_copy = [](const auto& n, auto init_list){
        container_type container(n);
        copy_from_list(init_list, container.begin());
        return container;
    };
    auto make_mapped_copy = [](const auto& n, auto init_list, auto mapper){
        container_type container(n);
        copy_from_list(init_list, container.begin(), mapper);
        return container;
    };
    //0result,1expected
    using test_type = std::tuple<container_type, container_type>;
    auto test_data = GENERATE_COPY(
        test_type{make_copy(0,nested_init_list1<int>{}), container_type{}},
        test_type{make_copy(5,nested_init_list1<int>{1,2,3,4,5}), container_type{1,2,3,4,5}},
        test_type{make_copy(0,nested_init_list2<int>{{},{},{}}), container_type{}},
        test_type{make_copy(6,nested_init_list2<int>{{1,2},{3,4},{5,6}}), container_type{1,2,3,4,5,6}},
        test_type{make_copy(0,nested_init_list3<int>{{{},{}},{{},{}},{{},{}}}), container_type{}},
        test_type{make_copy(18,nested_init_list3<int>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}},{{13,14,15},{16,17,18}}}), container_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}},
        test_type{
            make_mapped_copy(
                6,
                nested_init_list2<int>{{1,2},{3,4},{5,6}},
                [](auto i){int map[]={5,4,3,2,1,0}; return map[i];}
            ),
            container_type{6,5,4,3,2,1}
        },
        test_type{
            make_mapped_copy(
                6,
                nested_init_list2<int>{{1,2},{3,4},{5,6}},
                [](auto i){int map[]={1,4,5,0,3,2}; return map[i];}
            ),
            container_type{4,1,6,5,2,3}
        }
    );
    auto result = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    REQUIRE(result == expected);
}

TEST_CASE("test_list_size","[test_tensor_init_list]")
{
    using gtensor::detail::nested_init_list1;
    using gtensor::detail::nested_init_list2;
    using gtensor::detail::nested_init_list3;
    using size_type = std::int64_t;
    using gtensor::detail::list_size;
    //0result,1expected
    using test_type = std::tuple<size_type, size_type>;
    auto test_data = GENERATE(
        test_type{list_size<size_type>(nested_init_list1<int>{}),0},
        test_type{list_size<size_type>(nested_init_list1<int>{1,2,3,4,5}),5},
        test_type{list_size<size_type>(nested_init_list2<int>{{},{},{}}),0},
        test_type{list_size<size_type>(nested_init_list2<int>{{1,2},{3,4},{5,6}}),6},
        test_type{list_size<size_type>(nested_init_list3<int>{{{},{}},{{},{}},{{},{}}}),0},
        test_type{list_size<size_type>(nested_init_list3<int>{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}}),12}
    );
    auto result = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    REQUIRE(result == expected);
}
