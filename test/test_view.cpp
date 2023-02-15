#include <tuple>
#include "catch.hpp"
#include "test_config.hpp"
#include "gtensor.hpp"
#include "helpers_for_testing.hpp"

TEMPLATE_TEST_CASE("test_view","[test_view]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = float;
    using config_type = TestType;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using tensor_type = gtensor::tensor<value_type,config_type>;
    using bool_tensor_type =  gtensor::tensor<bool,config_type>;
    auto nop = gtensor::slice_traits<config_type>::nop_type{};
    using helpers_for_testing::apply_by_element;

    //0result,1expected
    auto test_data = std::make_tuple(
        //slice view
        std::make_tuple(tensor_type{1}({{}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{0}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{0,1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{{},1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{{},{},1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{nop,1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{nop,nop,1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{}})({{}}),tensor_type{1}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{}}),tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{}}),tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{1},{},{}}),tensor_type{{{7,8},{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{-1},{}}),tensor_type{{{5,6}},{{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{0,1},{0,-1},{}}),tensor_type{{{1,2},{3,4}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{{},{},-1}}),tensor_type{{{2,1},{4,3},{6,5}},{{8,7},{10,9},{12,11}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{{},{},-1}})({{0,1},{0,-1}}),tensor_type{{{2,1},{4,3}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{{},{},2},{{},{},-1}}),tensor_type{{{2,1},{6,5}},{{8,7},{12,11}}}),
        //transpose view
        std::make_tuple(tensor_type{1}.transpose(),tensor_type{1}),
        std::make_tuple(tensor_type{1}.transpose(0),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}.transpose(),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5}.transpose(0),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}.transpose(),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}.transpose(1,0),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}.transpose(0,1),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose().transpose(),tensor_type{{{1,2},{3,4},{5,6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(2,1,0),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(2,0,1),tensor_type{{{1,3,5}},{{2,4,6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(1,0,2),tensor_type{{{1,2}},{{3,4}},{{5,6}}}),
        //subdim view
        std::make_tuple(tensor_type{1}(),tensor_type{1}),
        std::make_tuple(tensor_type{1}()(),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}(),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}}}(0),tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}}}(0,0),tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}}}(0,1),tensor_type{4,5,6}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(1),tensor_type{{5,6},{7,8}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(2,0),tensor_type{9,10}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(2)(0),tensor_type{9,10}),
        //reshape view
        std::make_tuple(tensor_type{1}.reshape(),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}.reshape(1,5),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}.reshape(5,1),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}.reshape(6,2), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}),
        //mapping view index tensor
        std::make_tuple(tensor_type{1}(tensor_type{0}), tensor_type{1}),
        std::make_tuple(tensor_type{1}(tensor_type{0,0,0}), tensor_type{1,1,1}),
        std::make_tuple(tensor_type{1}(tensor_type{0,0,0})(tensor_type{0,0,0,0}), tensor_type{1,1,1,1}),
        std::make_tuple(tensor_type{1,2,3,4,5}(tensor_type{1,1,0,0}), tensor_type{2,2,1,1}),
        std::make_tuple(tensor_type{1,2,3,4,5}(tensor_type{1,1,0,0})(tensor_type{0,3,3,0}), tensor_type{2,1,1,2}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(tensor_type{1,3}),tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(tensor_type{1,3}, tensor_type{0,1}),tensor_type{{5,6},{15,16}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(tensor_type{1,3}, tensor_type{1}),tensor_type{{7,8},{15,16}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}(tensor_type{{0,0},{3,3}}, tensor_type{{0,2},{0,2}}),tensor_type{{1,3},{10,12}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}(tensor_type{3,2,1,0})(tensor_type{{0,0},{3,3}}, tensor_type{{0,2},{0,2}}),tensor_type{{10,12},{1,3}}),
        //mapping view bool tensor
        std::make_tuple(tensor_type{1}(bool_tensor_type{false}), tensor_type{}),
        std::make_tuple(tensor_type{1}(bool_tensor_type{true}), tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}(bool_tensor_type{false,true,false,true,false}), tensor_type{2,4}),
        std::make_tuple(tensor_type{1,2,3,4,5}(bool_tensor_type{true,true}), tensor_type{1,2}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{true}),tensor_type{{{1,2},{3,4}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{true,true}),tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{false,true,false,true}),tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{{false,true},{true,false}}),tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{{{false,true}},{{true,false}}}),tensor_type{2,3}),
        std::make_tuple([](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x>tensor_type{10});}(),tensor_type{11,12,13,14,15,16}),
        std::make_tuple([](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x<tensor_type{5});}(),tensor_type{1,2,3,4}),
        std::make_tuple([](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x>tensor_type{5} && x<tensor_type{10});}(),tensor_type{6,7,8,9}),
        //view composition
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose().reshape(2,3), tensor_type{{1,4,2},{5,3,6}})

    );

    auto test = [](auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result.equals(expected));
    };
    apply_by_element(test,test_data);
}