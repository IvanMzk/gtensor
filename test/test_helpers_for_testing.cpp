#include <tuple>
#include "catch.hpp"
#include "helpers_for_testing.hpp"

namespace test_cross_product{
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

}   //end of namespace test_cross_product

TEST_CASE("test_types_cross_product_with_type_list","[test_helpers_for_testing]"){
    using helpers_for_testing::cross_product;
    using test_cross_product::type_list;
    using test_cross_product::type_pair;
    using test_cross_product::A;
    using test_cross_product::B;
    using test_cross_product::C;
    using test_cross_product::D;
    using test_cross_product::E;
    using test_cross_product::F;

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
    using test_cross_product::A;
    using test_cross_product::B;
    using test_cross_product::C;
    using test_cross_product::D;
    using test_cross_product::E;
    using test_cross_product::F;

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