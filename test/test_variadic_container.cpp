#include <utility>
#include <string>
#include "catch.hpp"
#include "variadic_container.hpp"

namespace test_variadic_container{

struct A{
    int i;
    A():
        i{1}
    {}
    A(int i_):
        i{i_}
    {}
    friend bool operator==(const A& lhs, const A& rhs){
        return lhs.i == rhs.i;
    }
};

}



TEST_CASE("test_variadic_container","[test_variadic_container]"){
    using gtensor::detail::variadic_container;
    using gtensor::detail::get;
    using test_variadic_container::A;

    variadic_container<int> s1{1};
    REQUIRE(get<0>{}(s1) == 1);
    REQUIRE(!std::is_const_v<decltype(get<0>{}(s1))>);
    const auto& cr1 = s1;
    REQUIRE(get<0>{}(cr1) == 1);
    REQUIRE(std::is_const_v<std::remove_reference_t<decltype(get<0>{}(cr1))>>);


    // variadic_container<int,float> s2{1,2};
    // REQUIRE(get<0>{}(s2) == 1);
    // REQUIRE(get<1>{}(s2) == 2);
    // REQUIRE(!std::is_const_v<decltype(get<0>{}(s2))>);
    // REQUIRE(!std::is_const_v<decltype(get<1>{}(s2))>);


}