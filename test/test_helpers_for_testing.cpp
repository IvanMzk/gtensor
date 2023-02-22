#include <tuple>
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

TEST_CASE("test_list_concat","[test_helpers_for_testing]"){
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

TEST_CASE("test_types_cross_product_with_type_list","[test_helpers_for_testing]"){
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

TEST_CASE("test_types_cross_product_with_tuple","[test_helpers_for_testing]"){
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

TEST_CASE("test_apply_by_element","[test_helpers_for_testing]"){
    using helpers_for_testing::apply_by_element;

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