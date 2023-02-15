#include <tuple>
#include <vector>
#include "catch.hpp"
#include "broadcast.hpp"

TEMPLATE_TEST_CASE("test_variadic_broadcast_shape","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using result_shape_type = std::vector<std::int64_t>;
    using test_type = std::tuple<result_shape_type, result_shape_type>;
    using gtensor::detail::broadcast_shape;
    //0result broadcast shape,1expected broadcast shape
    auto test_data = GENERATE(
        test_type(broadcast_shape<result_shape_type>(shape_type{1}), result_shape_type{1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,2,3}), result_shape_type{1,2,3}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1}, shape_type{1}), result_shape_type{1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1}, shape_type{1}, shape_type{1}), result_shape_type{1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{5}, shape_type{5}), result_shape_type{5}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,1}, shape_type{1}), result_shape_type{1,1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,1}, shape_type{1}, shape_type{1,1,1}, shape_type{1,1}), result_shape_type{1,1,1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,1}, shape_type{1,1}, shape_type{1,1}), result_shape_type{1,1}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,5}, shape_type{5,1}), result_shape_type{5,5}),
        test_type(broadcast_shape<result_shape_type>(shape_type{1,5}, shape_type{5,1}, shape_type{1,5}, shape_type{1,1}), result_shape_type{5,5}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,3,4}, shape_type{3,4}), result_shape_type{2,3,4}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,3,4}, shape_type{3,4}, shape_type{1,1,1,1}, shape_type{5,1,1,1}), result_shape_type{5,2,3,4}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,1,4}, shape_type{3,1}, shape_type{3,4}), result_shape_type{2,3,4}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4}),
        test_type(broadcast_shape<result_shape_type>(shape_type{2,1}, shape_type{2,4}, shape_type{3,1,4}), result_shape_type{3,2,4})
    );

    auto result_broadcast_shape = std::get<0>(test_data);
    auto expected_broadcast_shape = std::get<1>(test_data);
    REQUIRE(result_broadcast_shape.size() == expected_broadcast_shape.size());
    REQUIRE(std::equal(expected_broadcast_shape.begin(),expected_broadcast_shape.end(),result_broadcast_shape.begin()));
}

TEMPLATE_TEST_CASE("test_variadic_broadcast_shape_exception","[test_descriptor]", std::vector<std::int64_t>)
{
    using shape_type = TestType;
    using gtensor::detail::broadcast_shape;
    using gtensor::broadcast_exception;

    SECTION("1-shape"){
        //shape1,shape2
        REQUIRE_THROWS_AS(broadcast_shape<shape_type>(shape_type{}), broadcast_exception);
    }
    SECTION("2-shapes"){
        //shape1,shape2
        using test_type = std::tuple<shape_type, shape_type>;
        auto test_data = GENERATE(
            test_type(shape_type{}, shape_type{}),
            test_type(shape_type{1}, shape_type{}),
            test_type(shape_type{}, shape_type{1}),
            test_type(shape_type{3}, shape_type{2}),
            test_type(shape_type{2}, shape_type{3}),
            test_type(shape_type{1,2}, shape_type{3}),
            test_type(shape_type{1,2}, shape_type{4,3}),
            test_type(shape_type{3,2}, shape_type{4,2}),
            test_type(shape_type{5,1,2}, shape_type{4,4,2})
        );
        auto shape1 = std::get<0>(test_data);
        auto shape2 = std::get<1>(test_data);
        REQUIRE_THROWS_AS(broadcast_shape<shape_type>(shape1, shape2), broadcast_exception);
    }
    SECTION("3-shapes"){
        //shape1,shape2,shape3
        using test_type = std::tuple<shape_type, shape_type, shape_type>;
        auto test_data = GENERATE(
            test_type(shape_type{}, shape_type{1}, shape_type{}),
            test_type(shape_type{3}, shape_type{3}, shape_type{2}),
            test_type(shape_type{1,2}, shape_type{3}, shape_type{1}),
            test_type(shape_type{1,2}, shape_type{1,1}, shape_type{4,4}),
            test_type(shape_type{5,1,2}, shape_type{2,2}, shape_type{4,4,2})
        );
        auto shape1 = std::get<0>(test_data);
        auto shape2 = std::get<1>(test_data);
        auto shape3 = std::get<2>(test_data);
        REQUIRE_THROWS_AS(broadcast_shape<shape_type>(shape1, shape2, shape3), broadcast_exception);
    }
}

