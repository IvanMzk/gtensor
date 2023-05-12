#include <iostream>
#include "catch.hpp"
//#include "tensor.hpp"

namespace test_tmp{

template<typename T>
struct ttt{

    template<typename...Subs> static constexpr bool enable_v = (std::is_integral_v<Subs>&&...);

    template<class...Subs, std::enable_if_t<enable_v<Subs...>,int> =0>
    auto operator()(const Subs&...subs){}


    template<class...Subs, std::enable_if_t<std::conjunction_v<std::is_integral<Subs>...>,int> =0>
    auto f(const Subs&...subs){}
};

}

TEST_CASE("test_tmp","[test_tmp]"
)
{
    using test_tmp::ttt;
    ttt<int> t{};
    t(1,2,3);
    t.f(1,2,3);
}