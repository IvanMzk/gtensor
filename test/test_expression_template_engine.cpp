#include "catch.hpp"
#include "expression_template_engine.hpp"

namespace test_cross_product{
struct A{};
struct B{};
struct C{};
struct D{};
struct E{};
struct F{};
template<typename F, typename S> struct type_pair{};

}   //end of namespace test_cross_product



TEST_CASE("test_cross_product","[test_expression_template_engine]"){
    using gtensor::detail::cross_product;
    using gtensor::detail::type_list;
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

TEST_CASE("test","[test_expression_template_engine]"){
    
}