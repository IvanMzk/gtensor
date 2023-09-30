/*
* GTensor - matrix computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

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
}

TEST_CASE("test_generate_lehmer","[test_helpers_for_testing]")
{
    using helpers_for_testing::generate_lehmer;

    const std::size_t n = 50;
    std::vector<std::size_t> res(n);
    SECTION("init_1")
    {
        generate_lehmer(res.begin(),res.end(),1);
        REQUIRE(res == std::vector<std::size_t>{279470273,1196210100,1795977874,3523022591,1091671578,3704055081,1929315071,1612890431,1665815703,3458738122,2060731663,2646341249,1864281884,374573440,1808675010,3676207556,260926693,2378977307,4109152188,3412486940,216934916,3861127178,4272950010,767728855,3867421730,3145820344,1503004262,2090242676,3910709429,2424267053,2030314576,3315377645,2904869620,4249708872,3900740825,181298020,134379285,1860617027,2638162249,388652375,311585419,2308523930,20784244,155920847,2486579790,589399664,1402887382,3570972666,1571825228,2961475146});
    }
    SECTION("init_123")
    {
        generate_lehmer(res.begin(),res.end(),123);
        REQUIRE(res == std::vector<std::size_t>{15105251,1104954406,1861946661,3835049593,1131618073,332242117,1082552728,817027627,3031868792,223027197,66924380,3377426802,1673405309,3122860210,3423694389,1201963833,2029212202,556432973,2914546077,3124066393,913190922,2472240884,1586841728,4236336054,3246470780,388846122,185930713,3696778979,4275890466,1832104440,620589970,4064524981,816678107,3023149045,3049752174,824820005,3643750182,1222627898,2371409802,559601924,3965268209,480602184,2556462012,1998395017,906636509,3776682016,756456346,1142974236,60974949,3484190514});
    }
}