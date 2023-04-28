#include <tuple>
#include <iostream>
#include <sstream>
#include "catch.hpp"
#include "helpers_for_testing.hpp"

namespace test_helpers_for_testing{
struct A{};
struct B{};
struct C{};
struct D{};
struct E{};
struct F{};

template<typename...Ts>
struct type_list{
    using type = type_list<Ts...>;
    static constexpr std::size_t size = sizeof...(Ts);
};

template<typename F, typename S> struct type_pair{
    using first_type = F;
    using second_type = S;
};

}   //end of namespace test_helpers_for_testing

TEST_CASE("test_list_concat","[test_helpers_for_testing]")
{
    using helpers_for_testing::list_concat;
    using test_helpers_for_testing::type_list;
    using test_helpers_for_testing::A;
    using test_helpers_for_testing::B;
    using test_helpers_for_testing::C;
    using test_helpers_for_testing::D;
    using test_helpers_for_testing::E;
    using test_helpers_for_testing::F;

    using l1 = type_list<A>;
    using l2 = type_list<B,C>;
    using l3 = type_list<D,E,F>;
    using l4 = type_list<>;

    REQUIRE(std::is_same_v<list_concat<l1,l2>::type, type_list<A,B,C>>);
    REQUIRE(std::is_same_v<list_concat<l2,l1>::type, type_list<B,C,A>>);
    REQUIRE(std::is_same_v<list_concat<l4,l1>::type, type_list<A>>);
    REQUIRE(std::is_same_v<list_concat<l3,l4,l1,l2>::type, type_list<D,E,F,A,B,C>>);
}

TEST_CASE("test_types_cross_product_with_type_list","[test_helpers_for_testing]")
{
    using helpers_for_testing::cross_product;
    using test_helpers_for_testing::type_list;
    using test_helpers_for_testing::type_pair;
    using test_helpers_for_testing::A;
    using test_helpers_for_testing::B;
    using test_helpers_for_testing::C;
    using test_helpers_for_testing::D;
    using test_helpers_for_testing::E;
    using test_helpers_for_testing::F;

    REQUIRE(std::is_same_v<
        cross_product<type_list, type_list<A,B,C>, type_list<D,E,F>>::type ,
        type_list<type_list<A,D>,type_list<A,E>,type_list<A,F>,type_list<B,D>,type_list<B,E>,type_list<B,F>,type_list<C,D>,type_list<C,E>,type_list<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<type_pair, type_list<A,B,C>, type_list<D,E,F>>::type ,
        type_list<type_pair<A,D>,type_pair<A,E>,type_pair<A,F>,type_pair<B,D>,type_pair<B,E>,type_pair<B,F>,type_pair<C,D>,type_pair<C,E>,type_pair<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<type_list, type_list<A,A,C>, type_list<D,E,E>>::type ,
        type_list<type_list<A,D>,type_list<A,E>,type_list<A,E>,type_list<A,D>,type_list<A,E>,type_list<A,E>,type_list<C,D>,type_list<C,E>,type_list<C,E>> >
    );
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<A>, type_list<B>>::type , type_list<type_list<A,B>>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<>, type_list<B>>::type , type_list<>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<>, type_list<>>::type , type_list<>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<A,B,C>, type_list<>>::type , type_list<>>);
}

TEST_CASE("test_types_cross_product_with_tuple","[test_helpers_for_testing]")
{
    using helpers_for_testing::cross_product;
    using test_helpers_for_testing::A;
    using test_helpers_for_testing::B;
    using test_helpers_for_testing::C;
    using test_helpers_for_testing::D;
    using test_helpers_for_testing::E;
    using test_helpers_for_testing::F;

    REQUIRE(std::is_same_v<
        cross_product<std::tuple, std::tuple<A,B,C>, std::tuple<D,E,F>>::type ,
        std::tuple<std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,F>,std::tuple<B,D>,std::tuple<B,E>,std::tuple<B,F>,std::tuple<C,D>,std::tuple<C,E>,std::tuple<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<std::pair, std::tuple<A,B,C>, std::tuple<D,E,F>>::type ,
        std::tuple<std::pair<A,D>,std::pair<A,E>,std::pair<A,F>,std::pair<B,D>,std::pair<B,E>,std::pair<B,F>,std::pair<C,D>,std::pair<C,E>,std::pair<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<std::tuple, std::tuple<A,A,C>, std::tuple<D,E,E>>::type ,
        std::tuple<std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,E>,std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,E>,std::tuple<C,D>,std::tuple<C,E>,std::tuple<C,E>> >
    );
    REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<A>, std::tuple<B>>::type , std::tuple<std::tuple<A,B>>>);
    REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<>, std::tuple<B>>::type , std::tuple<>>);
    REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<>, std::tuple<>>::type , std::tuple<>>);
    REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<A,B,C>, std::tuple<>>::type , std::tuple<>>);
}

TEST_CASE("test_apply_by_element","[test_helpers_for_testing]")
{
    using helpers_for_testing::apply_by_element;

    SECTION("std_tuple")
    {
        auto tests = std::make_tuple(
            [](const auto& t){REQUIRE(t == t);},
            [](const auto& t){REQUIRE((t+0) == t);},
            [](const auto& t){REQUIRE(t+1 > t);},
            [](const auto& t){REQUIRE(t-1 < t);}
        );

        auto test_data = std::make_tuple(
            int{0},
            int{1},
            float{0},
            float{1},
            double{0},
            double{1}
        );

        auto apply_tests = [&test_data](auto& test){
            apply_by_element(test, test_data);
        };
        apply_by_element(apply_tests, tests);
    }
    SECTION("tuple_for_testing")
    {
        using helpers_for_testing::create_tuple;
        using helpers_for_testing::get;
        auto tests = create_tuple(
            [](const auto& t){REQUIRE(get<0>(t)+get<1>(t) == get<2>(t));},
            [](const auto& t){REQUIRE(get<0>(t)-get<1>(t) == get<3>(t));}
        );

        auto test_data = create_tuple(
            create_tuple(5,5,10,0),
            create_tuple(1,1,2,0),
            create_tuple(2,18,20,-16),
            create_tuple(14,16,30,-2)
        );

        auto apply_tests = [&test_data](auto& test){
            apply_by_element(test, test_data);
        };
        apply_by_element(apply_tests, tests);
    }
}

TEST_CASE("test_apply_by_element_return","[test_helpers_for_testing]")
{
    using helpers_for_testing::apply_by_element;
    auto inc = [](const auto& t){return t+1;};

    SECTION("std_tuple")
    {
        auto t = std::make_tuple(1,2,3);
        REQUIRE(apply_by_element(inc,t) == std::make_tuple(2,3,4));
    }
    SECTION("tuple_for_testing")
    {
        using helpers_for_testing::create_tuple;
        auto t = create_tuple(1,2,3);
        REQUIRE(apply_by_element(inc,t) == create_tuple(2,3,4));
    }
}

TEST_CASE("test_light_tuple_for_testing_get","[test_helpers_for_testing]")
{
    using ltp::ltuple;
    using ltp::get;

    ltuple<int,double,std::vector<int>> t_{1,2.0,{1,2,3}};
    SECTION("not_const_arg")
    {
        auto& t = t_;
        REQUIRE(get<0>(t) == 1);
        REQUIRE(get<1>(t) == 2);
        REQUIRE(get<2>(t) == std::vector<int>{1,2,3});
        REQUIRE(std::is_same_v<decltype(get<0>(t)),int&>);
        REQUIRE(std::is_same_v<decltype(get<1>(t)),double&>);
        REQUIRE(std::is_same_v<decltype(get<2>(t)),std::vector<int>&>);
    }
    SECTION("const_arg")
    {
        const auto& t = t_;
        REQUIRE(get<0>(t) == 1);
        REQUIRE(get<1>(t) == 2);
        REQUIRE(get<2>(t) == std::vector<int>{1,2,3});
        REQUIRE(std::is_same_v<decltype(get<0>(t)),const int&>);
        REQUIRE(std::is_same_v<decltype(get<1>(t)),const double&>);
        REQUIRE(std::is_same_v<decltype(get<2>(t)),const std::vector<int>&>);
    }
    SECTION("rvalue_arg")
    {
        auto& t = t_;
        REQUIRE(get<0>(std::move(t)) == 1);
        REQUIRE(get<1>(std::move(t)) == 2);
        REQUIRE(get<2>(std::move(t)) == std::vector<int>{1,2,3});
        REQUIRE(std::is_same_v<decltype(get<0>(std::move(t))),int&&>);
        REQUIRE(std::is_same_v<decltype(get<1>(std::move(t))),double&&>);
        REQUIRE(std::is_same_v<decltype(get<2>(std::move(t))),std::vector<int>&&>);
    }
}

TEST_CASE("test_light_tuple_for_testing_operator==,!=","[test_helpers_for_testing]")
{
    using ltp::ltuple;
    using ltp::create_ltuple;

    ltuple<int> t0{0};
    ltuple<int> t1{1};
    REQUIRE(t0 == t0);
    REQUIRE(t1 != t0);
    REQUIRE(ltuple<int>{1} == ltuple<int>{1});
    REQUIRE(ltuple<int>{1} == ltuple<double>{1});
    REQUIRE(ltuple<int>{1} != ltuple<int>{0});
    REQUIRE(ltuple<int,double>{1,2.0} == ltuple<int,double>{1,2.0});
    REQUIRE(ltuple<int,double>{1,2.0} != ltuple<int,double>{1,3.0});
    REQUIRE(ltuple<int,double,std::string>{1,2.0,"abc"} == ltuple<int,double,std::string>{1,2.0,"abc"});
    REQUIRE(ltuple<int,double,std::string>{1,2.0,"abc"} != ltuple<int,double,std::string>{1,2.0,"def"});
}