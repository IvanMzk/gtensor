#include "catch.hpp"
#include "libdivide.h"



TEST_CASE("test_libdivide_operator==","[test_libdivide]"){
    using index_type = std::int64_t;
    using divider_type = libdivide::divider<index_type>;

    divider_type d1{7};
    divider_type d2{7};
    divider_type d3{5};

    REQUIRE(d1 == d2);
    REQUIRE(d1 != d3);


}