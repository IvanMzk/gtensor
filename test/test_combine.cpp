#include "catch.hpp"
#include "gtensor.hpp"
#include "combine.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_make_stack_shape","[test_combine]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::detail::make_stack_shape;
    using test_type = std::tuple<shape_type,size_type,index_type,shape_type>;
    //0shape,1direction,2tensors_number,3expected
    auto test_data = GENERATE(
        test_type{shape_type{5},size_type{0},index_type{1},shape_type{1,5}},
        test_type{shape_type{5},size_type{1},index_type{1},shape_type{5,1}},
        test_type{shape_type{3,4},size_type{0},index_type{7},shape_type{7,3,4}},
        test_type{shape_type{3,4},size_type{1},index_type{7},shape_type{3,7,4}},
        test_type{shape_type{3,4},size_type{2},index_type{7},shape_type{3,4,7}},
        test_type{shape_type{3,4,5},size_type{0},index_type{7},shape_type{7,3,4,5}},
        test_type{shape_type{3,4,5},size_type{1},index_type{7},shape_type{3,7,4,5}},
        test_type{shape_type{3,4,5},size_type{2},index_type{7},shape_type{3,4,7,5}},
        test_type{shape_type{3,4,5},size_type{3},index_type{7},shape_type{3,4,5,7}}
    );

    auto shape = std::get<0>(test_data);
    auto direction = std::get<1>(test_data);
    auto tensors_number = std::get<2>(test_data);
    auto expected = std::get<3>(test_data);
    auto result = make_stack_shape(shape,direction,tensors_number);
    REQUIRE(result == expected);
}