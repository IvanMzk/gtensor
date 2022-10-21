#include <tuple>
#include "catch.hpp"
#include "test_config.hpp"
#include "gtensor.hpp"

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
    using test_type = std::tuple<tensor_type, tensor_type>;
    auto nop = config_type::nop_type{};
    //result tensor, expected tensor
    auto test_data = GENERATE_COPY(
        //slice view
        test_type{tensor_type{1}({{}}).copy(),tensor_type{1}},
        test_type{tensor_type{1}({{0}}).copy(),tensor_type{1}},
        test_type{tensor_type{1}({{0,1}}).copy(),tensor_type{1}},
        test_type{tensor_type{1}({{{},1}}).copy(),tensor_type{1}},
        test_type{tensor_type{1}({{{},{},1}}).copy(),tensor_type{1}},
        test_type{tensor_type{1}({{nop,1}}).copy(),tensor_type{1}},
        test_type{tensor_type{1}({{nop,nop,1}}).copy(),tensor_type{1}},
        test_type{tensor_type{1}({{}})({{}}).copy(),tensor_type{1}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{}}).copy(),tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{}}).copy(),tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{1},{},{}}).copy(),tensor_type{{{7,8},{9,10},{11,12}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{-1},{}}).copy(),tensor_type{{{5,6}},{{11,12}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{0,1},{0,-1},{}}).copy(),tensor_type{{{1,2},{3,4}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{{},{},-1}}).copy(),tensor_type{{{2,1},{4,3},{6,5}},{{8,7},{10,9},{12,11}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{{},{},-1}})({{0,1},{0,-1}}).copy(),tensor_type{{{2,1},{4,3}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{{},{},2},{{},{},-1}}).copy(),tensor_type{{{2,1},{6,5}},{{8,7},{12,11}}}},
        //transpose view
        test_type{tensor_type{1}.transpose().copy(),tensor_type{1}},
        test_type{tensor_type{1}.transpose(0).copy(),tensor_type{1}},
        test_type{tensor_type{1,2,3,4,5}.transpose().copy(),tensor_type{1,2,3,4,5}},
        test_type{tensor_type{1,2,3,4,5}.transpose(0).copy(),tensor_type{1,2,3,4,5}},
        test_type{tensor_type{{1,2,3,4,5}}.transpose().copy(),tensor_type{{1},{2},{3},{4},{5}}},
        test_type{tensor_type{{1,2,3,4,5}}.transpose(1,0).copy(),tensor_type{{1},{2},{3},{4},{5}}},
        test_type{tensor_type{{1,2,3,4,5}}.transpose(0,1).copy(),tensor_type{{1,2,3,4,5}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}}}.transpose().transpose().copy(),tensor_type{{{1,2},{3,4},{5,6}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}}}.transpose().copy(),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}}}.transpose(2,1,0).copy(),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}}}.transpose(2,0,1).copy(),tensor_type{{{1,3,5}},{{2,4,6}}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}}}.transpose(1,0,2).copy(),tensor_type{{{1,2}},{{3,4}},{{5,6}}}},
        //subdim view
        test_type{tensor_type{1}().copy(),tensor_type{1}},
        test_type{tensor_type{1}()().copy(),tensor_type{1}},
        test_type{tensor_type{1,2,3,4,5}().copy(),tensor_type{1,2,3,4,5}},
        test_type{tensor_type{{{1,2,3},{4,5,6}}}(0).copy(),tensor_type{{1,2,3},{4,5,6}}},
        test_type{tensor_type{{{1,2,3},{4,5,6}}}(0,0).copy(),tensor_type{1,2,3}},
        test_type{tensor_type{{{1,2,3},{4,5,6}}}(0,1).copy(),tensor_type{4,5,6}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(1).copy(),tensor_type{{5,6},{7,8}}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(2,0).copy(),tensor_type{9,10}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(2)(0).copy(),tensor_type{9,10}},
        //reshape view
        test_type{tensor_type{1}.reshape().copy(),tensor_type{1}},
        test_type{tensor_type{1,2,3,4,5}.reshape(1,5).copy(),tensor_type{{1,2,3,4,5}}},
        test_type{tensor_type{1,2,3,4,5}.reshape(5,1).copy(),tensor_type{{1},{2},{3},{4},{5}}},
        test_type{tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}.reshape(6,2).copy(), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}},
        //mapping view index tensor
        test_type{tensor_type{1}(tensor_type{0}).copy(), tensor_type{1}},
        test_type{tensor_type{1}(tensor_type{0,0,0}).copy(), tensor_type{1,1,1}},
        test_type{tensor_type{1}(tensor_type{0,0,0})(tensor_type{0,0,0,0}).copy(), tensor_type{1,1,1,1}},
        test_type{tensor_type{1,2,3,4,5}(tensor_type{1,1,0,0}).copy(), tensor_type{2,2,1,1}},
        test_type{tensor_type{1,2,3,4,5}(tensor_type{1,1,0,0})(tensor_type{0,3,3,0}).copy(), tensor_type{2,1,1,2}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(tensor_type{1,3}).copy(),tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(tensor_type{1,3}, tensor_type{0,1}).copy(),tensor_type{{5,6},{15,16}}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(tensor_type{1,3}, tensor_type{1}).copy(),tensor_type{{7,8},{15,16}}},
        test_type{tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}(tensor_type{{0,0},{3,3}}, tensor_type{{0,2},{0,2}}).copy(),tensor_type{{1,3},{10,12}}},
        test_type{tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}(tensor_type{3,2,1,0})(tensor_type{{0,0},{3,3}}, tensor_type{{0,2},{0,2}}).copy(),tensor_type{{10,12},{1,3}}},
        //mapping view bool tensor
        //test_type{tensor_type{1}(bool_tensor_type{false}).copy(), tensor_type{}}
        test_type{tensor_type{1}(bool_tensor_type{true}).copy(), tensor_type{1}},
        test_type{tensor_type{1,2,3,4,5}(bool_tensor_type{false,true,false,true,false}).copy(), tensor_type{2,4}},
        test_type{tensor_type{1,2,3,4,5}(bool_tensor_type{true,true}).copy(), tensor_type{1,2}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{true}).copy(),tensor_type{{{1,2},{3,4}}}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{true,true}).copy(),tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{false,true,false,true}).copy(),tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{{false,true},{true,false}}).copy(),tensor_type{{3,4},{5,6}}},
        test_type{tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{{{false,true}},{{true,false}}}).copy(),tensor_type{2,3}},
        test_type{[](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x>tensor_type{10});}(),tensor_type{11,12,13,14,15,16}},
        test_type{[](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x<tensor_type{5});}(),tensor_type{1,2,3,4}},
        test_type{[](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x>tensor_type{5} && x<tensor_type{10});}(),tensor_type{6,7,8,9}},
        //view composition
        test_type{tensor_type{{1,2,3},{4,5,6}}.transpose().reshape(2,3).copy(), tensor_type{{1,4,2},{5,3,6}}}
    );

    auto result_tensor = std::get<0>(test_data);
    auto expected_tensor = std::get<1>(test_data);
    REQUIRE(result_tensor.equals(expected_tensor));
}