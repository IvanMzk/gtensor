#include <algorithm>
#include "catch.hpp"
#include "builder.hpp"
#include "reduce.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

namespace test_reduce_binary_{
    struct max{
        template<typename T, typename U>
        auto operator()(const T& t, const U& u){
            return t>u ? t:u;
        }
    };
    struct min{
        template<typename T, typename U>
        auto operator()(const T& t, const U& u){
            return t<u ? t:u;
        }
    };
}

TEMPLATE_TEST_CASE("test_reduce_binary","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::reduce_binary;
    using gtensor::detail::no_value;
    using test_reduce_binary_::max;
    using test_reduce_binary_::min;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{
        {{{{7,4,6,5,7},{3,1,3,3,8},{3,5,6,7,6},{5,7,1,1,6}},{{6,4,0,3,8},{5,3,3,8,7},{0,1,7,2,3},{5,5,0,2,5}},{{8,7,7,4,5},{1,8,6,8,4},{2,7,1,6,2},{6,5,6,0,3}}},
        {{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}}},
        {{{{1,7,2,7,8},{1,7,1,8,1},{2,3,4,2,0},{7,5,8,5,0}},{{5,5,1,3,8},{0,8,0,0,2},{5,1,2,3,0},{6,7,3,7,4}},{{8,0,7,0,0},{2,4,1,5,8},{5,6,8,4,8},{4,1,3,2,7}}},
        {{{0,6,2,7,3},{6,4,2,6,4},{7,0,3,3,1},{2,1,3,0,4}},{{7,4,4,7,6},{3,3,6,7,4},{1,7,4,0,1},{2,3,0,6,8}},{{2,4,1,6,0},{3,5,2,6,7},{5,7,5,4,4},{7,8,0,2,2}}}},
        {{{{0,7,1,1,0},{2,7,5,3,3},{6,5,4,8,6},{4,8,0,6,4}},{{0,0,5,8,0},{8,1,6,4,7},{2,5,4,6,3},{0,4,0,2,7}},{{6,0,3,6,4},{1,5,3,8,0},{8,7,2,4,0},{8,3,2,3,6}}},
        {{{2,8,5,4,4},{0,0,3,8,5},{4,1,4,2,1},{4,1,8,1,1}},{{7,2,8,8,3},{3,4,3,3,6},{1,6,2,7,7},{0,5,4,6,1}},{{1,4,0,7,6},{8,7,6,8,2},{6,4,0,5,8},{6,4,2,4,0}}}},
        {{{{0,5,0,8,6},{5,5,3,8,1},{8,3,7,8,5},{1,4,3,4,4}},{{4,0,0,6,8},{4,8,0,1,7},{6,2,6,4,2},{4,7,5,8,1}},{{3,3,1,5,5},{2,4,6,0,5},{3,1,7,6,5},{6,2,8,1,2}}},
        {{{4,7,5,2,1},{6,5,3,1,5},{8,8,5,5,4},{3,3,4,1,5}},{{7,8,2,8,1},{6,0,2,4,5},{8,4,5,0,3},{7,2,5,0,0}},{{2,2,2,7,8},{1,0,7,5,8},{0,2,5,4,4},{1,3,5,8,4}}}}
    };  //(4,2,3,4,5)

    //0tensor,1axes,2functor,3keep_dims,4initial,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(test_ten,std::vector<int>{4},std::plus<void>{},false,no_value{},tensor_type{{{{29,18,27,20},{21,26,13,17},{31,27,18,20}},{{21,11,16,27},{19,17,5,28},{24,16,27,30}}},{{{25,18,11,25},{22,10,11,27},{15,20,31,17}},{{18,22,14,10},{28,23,13,19},{13,23,25,19}}},{{{9,20,29,22},{13,26,20,13},{19,17,21,22}},{{23,16,12,15},{28,19,23,16},{18,31,23,16}}},{{{19,22,31,16},{18,20,20,25},{17,17,22,19}},{{19,20,30,16},{26,17,20,14},{21,21,15,21}}}}),
        std::make_tuple(test_ten,std::vector<int>{3,4},std::plus<void>{},false,no_value{},tensor_type{{{94,77,96},{75,69,97}},{{79,70,83},{64,83,80}},{{80,72,79},{66,86,88}},{{88,83,75},{85,77,78}}}),
        std::make_tuple(test_ten,std::vector<int>{2,3,4},std::plus<void>{},false,no_value{},tensor_type{{267,241},{232,227},{231,240},{246,240}}),
        std::make_tuple(test_ten,std::vector<int>{0,4},std::plus<void>{},false,no_value{},tensor_type{{{82,78,98,83},{74,82,64,82},{82,81,92,78}},{{81,69,72,68},{101,76,61,77},{76,91,90,86}}}),
        std::make_tuple(test_ten,std::vector<int>{2,4},std::plus<void>{},false,no_value{},tensor_type{{{81,71,58,57},{64,44,48,85}},{{62,48,53,69},{59,68,52,48}},{{41,63,70,57},{69,66,58,47}},{{54,59,73,60},{66,58,65,51}}}),
        std::make_tuple(test_ten,std::vector<int>{0,2,4},std::plus<void>{},false,no_value{},tensor_type{{238,241,254,243},{258,236,223,231}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,4},std::plus<void>{},false,no_value{},tensor_type{{163,147,170,151},{175,158,125,159},{158,172,182,164}}),
        std::make_tuple(test_ten,std::vector<int>{0,3,4},std::plus<void>{},false,no_value{},tensor_type{{341,302,333},{290,315,343}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,3,4},std::plus<void>{},false,no_value{},tensor_type{631,617,676}),
        std::make_tuple(test_ten,std::vector<int>{0},std::plus<void>{},false,no_value{},tensor_type{{{{8,23,9,21,21},{11,20,12,22,13},{19,16,21,25,17},{17,24,12,16,14}},{{15,9,6,20,24},{17,20,9,13,23},{13,9,19,15,8},{15,23,8,19,17}},{{25,10,18,15,14},{6,21,16,21,17},{18,21,18,20,15},{24,11,19,6,18}}},{{{8,23,19,18,13},{12,9,11,22,15},{27,11,17,10,7},{9,12,22,7,18}},{{22,19,20,30,10},{18,11,12,18,17},{12,18,11,8,12},{15,16,12,18,16}},{{12,16,4,23,21},{14,15,23,19,20},{14,21,16,16,23},{19,23,11,22,11}}}}),
        std::make_tuple(test_ten,std::vector<int>{2},std::plus<void>{},false,no_value{},tensor_type{{{{21,15,13,12,20},{9,12,12,19,19},{5,13,14,15,11},{16,17,7,3,14}},{{10,13,14,15,12},{8,7,12,11,6},{13,11,11,4,9},{11,21,14,19,20}}},{{{14,12,10,10,16},{3,19,2,13,11},{12,10,14,9,8},{17,13,14,14,11}},{{9,14,7,20,9},{12,12,10,19,15},{13,14,12,7,6},{11,12,3,8,14}}},{{{6,7,9,15,4},{11,13,14,15,10},{16,17,10,18,9},{12,15,2,11,17}},{{10,14,13,19,13},{11,11,12,19,13},{11,11,6,14,16},{10,10,14,11,2}}},{{{7,8,1,19,19},{11,17,9,9,13},{17,6,20,18,12},{11,13,16,13,7}},{{13,17,9,17,10},{13,5,12,10,18},{16,14,15,9,11},{11,8,14,9,9}}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1},std::plus<void>{},false,no_value{},tensor_type{{{16,46,28,39,34},{23,29,23,44,28},{46,27,38,35,24},{26,36,34,23,32}},{{37,28,26,50,34},{35,31,21,31,40},{25,27,30,23,20},{30,39,20,37,33}},{{37,26,22,38,35},{20,36,39,40,37},{32,42,34,36,38},{43,34,30,28,29}}}),
        std::make_tuple(test_ten,std::vector<int>{1,2},std::plus<void>{},false,no_value{},tensor_type{{{31,28,27,27,32},{17,19,24,30,25},{18,24,25,19,20},{27,38,21,22,34}},{{23,26,17,30,25},{15,31,12,32,26},{25,24,26,16,14},{28,25,17,22,25}},{{16,21,22,34,17},{22,24,26,34,23},{27,28,16,32,25},{22,25,16,22,19}},{{20,25,10,36,29},{24,22,21,19,31},{33,20,35,27,23},{22,21,30,22,16}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,2},std::plus<void>{},false,no_value{},tensor_type{{90,100,76,127,103},{78,96,83,115,105},{103,96,102,94,82},{99,109,84,88,94}}),
        std::make_tuple(test_ten,std::vector<int>{1,3},std::plus<void>{},false,no_value{},tensor_type{{{28,28,38,33,42},{31,29,20,33,33},{34,52,39,32,36}},{{26,33,25,38,21},{29,38,20,33,33},{36,35,27,29,36}},{{22,37,30,33,24},{21,27,32,44,34},{44,34,18,45,26}},{{35,40,30,37,31},{46,31,25,31,27},{18,17,41,36,41}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,3},std::plus<void>{},false,no_value{},tensor_type{{111,138,123,141,118},{127,125,97,141,127},{132,138,125,142,139}}),
        std::make_tuple(test_ten,std::vector<int>{1,2,3},std::plus<void>{},false,no_value{},tensor_type{{93,109,97,98,111},{91,106,72,100,90},{87,98,80,122,84},{99,88,96,104,99}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},std::plus<void>{},false,no_value{},tensor_type(1924)),
        //initial
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},std::plus<void>{},false,value_type{-24},tensor_type(1900)),
        std::make_tuple(test_ten,std::vector<int>{0},std::plus<void>{},false,value_type{-5},tensor_type{{{{3,18,4,16,16},{6,15,7,17,8},{14,11,16,20,12},{12,19,7,11,9}},{{10,4,1,15,19},{12,15,4,8,18},{8,4,14,10,3},{10,18,3,14,12}},{{20,5,13,10,9},{1,16,11,16,12},{13,16,13,15,10},{19,6,14,1,13}}},{{{3,18,14,13,8},{7,4,6,17,10},{22,6,12,5,2},{4,7,17,2,13}},{{17,14,15,25,5},{13,6,7,13,12},{7,13,6,3,7},{10,11,7,13,11}},{{7,11,-1,18,16},{9,10,18,14,15},{9,16,11,11,18},{14,18,6,17,6}}}}),
        std::make_tuple(test_ten,std::vector<int>{4},std::plus<void>{},false,value_type{5},tensor_type{{{{34,23,32,25},{26,31,18,22},{36,32,23,25}},{{26,16,21,32},{24,22,10,33},{29,21,32,35}}},{{{30,23,16,30},{27,15,16,32},{20,25,36,22}},{{23,27,19,15},{33,28,18,24},{18,28,30,24}}},{{{14,25,34,27},{18,31,25,18},{24,22,26,27}},{{28,21,17,20},{33,24,28,21},{23,36,28,21}}},{{{24,27,36,21},{23,25,25,30},{22,22,27,24}},{{24,25,35,21},{31,22,25,19},{26,26,20,26}}}}),
        std::make_tuple(test_ten,std::vector<int>{1,2,4},std::plus<void>{},false,value_type{-5},tensor_type{{140,110,101,137},{116,111,100,112},{105,124,123,99},{115,112,133,106}}),
        //keep_dims
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},std::plus<void>{},true,no_value{},tensor_type{{{{{1924}}}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},std::plus<void>{},true,value_type{-24},tensor_type{{{{{1900}}}}}),
        std::make_tuple(test_ten,std::vector<int>{0,1,3},std::plus<void>{},true,value_type{-5},tensor_type{{{{{106,133,118,136,113}},{{122,120,92,136,122}},{{127,133,120,137,134}}}}}),
        std::make_tuple(test_ten,std::vector<int>{1,3,4},std::plus<void>{},true,value_type{-5},tensor_type{{{{{164}},{{141}},{{188}}}},{{{{138}},{{148}},{{158}}}},{{{{141}},{{153}},{{162}}}},{{{{168}},{{155}},{{148}}}}}),
        std::make_tuple(test_ten,0,std::plus<void>{},true,value_type{-5},tensor_type{{{{{3,18,4,16,16},{6,15,7,17,8},{14,11,16,20,12},{12,19,7,11,9}},{{10,4,1,15,19},{12,15,4,8,18},{8,4,14,10,3},{10,18,3,14,12}},{{20,5,13,10,9},{1,16,11,16,12},{13,16,13,15,10},{19,6,14,1,13}}},{{{3,18,14,13,8},{7,4,6,17,10},{22,6,12,5,2},{4,7,17,2,13}},{{17,14,15,25,5},{13,6,7,13,12},{7,13,6,3,7},{10,11,7,13,11}},{{7,11,-1,18,16},{9,10,18,14,15},{9,16,11,11,18},{14,18,6,17,6}}}}}),
        std::make_tuple(test_ten,std::vector<int>{4},std::plus<void>{},true,value_type{-5},tensor_type{{{{{24},{13},{22},{15}},{{16},{21},{8},{12}},{{26},{22},{13},{15}}},{{{16},{6},{11},{22}},{{14},{12},{0},{23}},{{19},{11},{22},{25}}}},{{{{20},{13},{6},{20}},{{17},{5},{6},{22}},{{10},{15},{26},{12}}},{{{13},{17},{9},{5}},{{23},{18},{8},{14}},{{8},{18},{20},{14}}}},{{{{4},{15},{24},{17}},{{8},{21},{15},{8}},{{14},{12},{16},{17}}},{{{18},{11},{7},{10}},{{23},{14},{18},{11}},{{13},{26},{18},{11}}}},{{{{14},{17},{26},{11}},{{13},{15},{15},{20}},{{12},{12},{17},{14}}},{{{14},{15},{25},{11}},{{21},{12},{15},{9}},{{16},{16},{10},{16}}}}}),
        //functor
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},max{},false,no_value{},tensor_type(8)),
        std::make_tuple(test_ten,std::vector<int>{0,1,2,3,4},min{},false,no_value{},tensor_type(0)),
        std::make_tuple(test_ten,std::vector<int>{1,3},max{},false,no_value{},tensor_type{{{8,7,7,7,8},{6,6,7,8,8},{8,8,8,8,7}},{{7,7,8,8,8},{7,8,6,7,8},{8,8,8,6,8}},{{6,8,8,8,6},{8,6,8,8,7},{8,7,6,8,8}},{{8,8,7,8,6},{8,8,6,8,8},{6,4,8,8,8}}}),
        std::make_tuple(test_ten,std::vector<int>{1,3},max{},false,value_type{7},tensor_type{{{8,7,7,7,8},{7,7,7,8,8},{8,8,8,8,7}},{{7,7,8,8,8},{7,8,7,7,8},{8,8,8,7,8}},{{7,8,8,8,7},{8,7,8,8,7},{8,7,7,8,8}},{{8,8,7,8,7},{8,8,7,8,8},{7,7,8,8,8}}}),
        std::make_tuple(test_ten,std::vector<int>{0,2},min{},false,no_value{},tensor_type{{{0,0,0,0,0},{0,1,0,0,0},{0,1,1,2,0},{0,1,0,0,0}},{{0,2,0,2,0},{0,0,1,0,1},{0,0,0,0,1},{0,1,0,0,0}}}),
        std::make_tuple(test_ten,std::vector<int>{0,2},min{},false,value_type{1},tensor_type{{{0,0,0,0,0},{0,1,0,0,0},{0,1,1,1,0},{0,1,0,0,0}},{{0,1,0,1,0},{0,0,1,0,1},{0,0,0,0,1},{0,1,0,0,0}}}),
        //unsorted, negative axes
        std::make_tuple(test_ten,std::vector<int>{3,0,1},std::plus<void>{},true,value_type{-5},tensor_type{{{{{106,133,118,136,113}},{{122,120,92,136,122}},{{127,133,120,137,134}}}}}),
        std::make_tuple(test_ten,std::vector<int>{1,2,0,3,-1},std::plus<void>{},true,value_type{-24},tensor_type{{{{{1900}}}}}),
        std::make_tuple(test_ten,std::vector<int>{2,-1,-4,},std::plus<void>{},false,value_type{-5},tensor_type{{140,110,101,137},{116,111,100,112},{105,124,123,99},{115,112,133,106}}),
        //input 1d
        std::make_tuple(tensor_type{1,2,3,4,5},0,std::plus<void>{},false,no_value{},tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},std::plus<void>{},false,no_value{},tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,std::plus<void>{},false,value_type{4},tensor_type(19)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},std::plus<void>{},false,value_type{4},tensor_type(19)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,std::plus<void>{},true,no_value{},tensor_type{15}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},std::plus<void>{},true,no_value{},tensor_type{15}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,std::plus<void>{},true,value_type{-4},tensor_type{11}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},std::plus<void>{},true,value_type{-4},tensor_type{11}),
        //input unit dims
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},0,std::plus<void>{},false,value_type{0},tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{1},std::plus<void>{},false,value_type{0},tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},2,std::plus<void>{},true,value_type{0},tensor_type{{{15}}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{2},std::plus<void>{},false,value_type{0},tensor_type{{15}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{0,1},std::plus<void>{},false,value_type{0},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{1,2},std::plus<void>{},true,value_type{0},tensor_type{{{15}}}),
        std::make_tuple(tensor_type{{{1,2,3,4,5}}},std::vector<int>{0,1,2},std::plus<void>{},false,value_type{0},tensor_type(15)),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},0,std::plus<void>{},false,value_type{0},tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},1,std::plus<void>{},false,value_type{0},tensor_type{{15}}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},2,std::plus<void>{},true,value_type{0},tensor_type{{{1},{2},{3},{4},{5}}}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},std::vector<int>{0,1},std::plus<void>{},false,value_type{0},tensor_type{15}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},std::vector<int>{0,2},std::plus<void>{},false,value_type{0},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1},{2},{3},{4},{5}}},std::vector<int>{0,2},std::plus<void>{},true,value_type{0},tensor_type{{{1},{2},{3},{4},{5}}}),
        //empty axes container
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},std::plus<void>{},true,no_value{},tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},std::plus<void>{},true,value_type{2},tensor_type{{3,4,5},{6,7,8}}),
        //reduce zero size axes
        std::make_tuple(tensor_type{},std::vector<int>{0},std::plus<void>{},false,value_type{0},tensor_type(0)),
        std::make_tuple(tensor_type{},std::vector<int>{0},std::plus<void>{},false,value_type{3},tensor_type(3)),
        std::make_tuple(tensor_type{},std::vector<int>{0},std::plus<void>{},true,value_type{2},tensor_type{2}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0},std::plus<void>{},false,value_type{4},tensor_type{{4,4,4},{4,4,4}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0},std::plus<void>{},true,value_type{4},tensor_type{{{4,4,4},{4,4,4}}}),
        std::make_tuple(tensor_type{}.reshape(0,2,0,3),std::vector<int>{0,2},std::plus<void>{},false,value_type{0},tensor_type{{0,0,0},{0,0,0}}),
        //empty result
        std::make_tuple(tensor_type{}.reshape(2,0,3),std::vector<int>{0},std::plus<void>{},false,no_value{},tensor_type{}.reshape(0,3)),
        std::make_tuple(tensor_type{}.reshape(2,0,3),std::vector<int>{0},std::plus<void>{},true,value_type{0},tensor_type{}.reshape(1,0,3)),
        std::make_tuple(tensor_type{}.reshape(0,2,0,3),std::vector<int>{0,1},std::plus<void>{},false,value_type{0},tensor_type{}.reshape(0,3)),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten,std::vector<int>{0,1,2},std::plus<void>{},false,value_type{0},tensor_type{{360,400,304,508,412},{312,384,332,460,420},{412,384,408,376,328},{396,436,336,352,376}}),
        std::make_tuple((test_ten-1)*(test_ten+1),std::vector<int>{1,2,3},std::plus<void>{},false,value_type{0},tensor_type{{491,613,533,540,639},{465,590,322,550,530},{497,532,368,728,438},{537,446,490,628,513}}),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,1)+test_ten(2,0,2),std::vector<int>{0,1,2},std::plus<void>{},false,value_type{0},tensor_type{{430,324,312,539,399},{266,388,331,579,325},{471,472,346,322,210},{487,429,240,312,486}}),
        std::make_tuple((test_ten+test_ten(0))*(test_ten(1,1)-test_ten(2,0,2)),std::vector<int>{1,2,3},std::plus<void>{},false,value_type{0},tensor_type{{-422,194,-6,-98,176},{-388,206,51,-79,284},{-329,169,26,-155,179},{-383,129,6,-87,220}})
    );
    auto test_reduce_binary = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto initial = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto result = reduce_binary(policy...,ten,axes,functor,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };

    SECTION("default_policy")
    {
        test_reduce_binary();
    }
    SECTION("exec_pol<4>")
    {
        test_reduce_binary(multithreading::exec_pol<4>{});
    }
}

TEMPLATE_TEST_CASE("test_reduce_binary_exception","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::reduce_binary;
    using gtensor::detail::no_value;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axes,2functor,3keep_dims,4initial
    auto test_data = std::make_tuple(
        //reduce zero size axes without initial
        std::make_tuple(tensor_type{},0,std::plus<void>{},false,no_value{}),
        std::make_tuple(tensor_type{},std::vector<int>{0},std::plus<void>{},true,no_value{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0},std::plus<void>{},false,no_value{}),
        std::make_tuple(tensor_type{}.reshape(0,2,0,3),std::vector<int>{0,2},std::plus<void>{},false,no_value{}),
        //axes out of range
        std::make_tuple(tensor_type{},1,std::plus<void>{},false,value_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5},1,std::plus<void>{},false,value_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{1},std::plus<void>{},false,value_type{0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},2,std::plus<void>{},false,value_type{0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{0,1,2},std::plus<void>{},false,value_type{0}),
        //repeating axes
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{0,0},std::plus<void>{},false,value_type{0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{2,-1},std::plus<void>{},false,value_type{0})
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto keep_dims = std::get<3>(t);
        auto initial = std::get<4>(t);
        REQUIRE_THROWS_AS(reduce_binary(multithreading::exec_pol<1>{},ten,axes,functor,keep_dims,initial),value_error);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_reduce_binary_flatten","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using test_reduce_binary_::max;
    using test_reduce_binary_::min;
    using gtensor::detail::no_value;
    using gtensor::reduce_binary;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{
        {{{{7,4,6,5,7},{3,1,3,3,8},{3,5,6,7,6},{5,7,1,1,6}},{{6,4,0,3,8},{5,3,3,8,7},{0,1,7,2,3},{5,5,0,2,5}},{{8,7,7,4,5},{1,8,6,8,4},{2,7,1,6,2},{6,5,6,0,3}}},
        {{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}}},
        {{{{1,7,2,7,8},{1,7,1,8,1},{2,3,4,2,0},{7,5,8,5,0}},{{5,5,1,3,8},{0,8,0,0,2},{5,1,2,3,0},{6,7,3,7,4}},{{8,0,7,0,0},{2,4,1,5,8},{5,6,8,4,8},{4,1,3,2,7}}},
        {{{0,6,2,7,3},{6,4,2,6,4},{7,0,3,3,1},{2,1,3,0,4}},{{7,4,4,7,6},{3,3,6,7,4},{1,7,4,0,1},{2,3,0,6,8}},{{2,4,1,6,0},{3,5,2,6,7},{5,7,5,4,4},{7,8,0,2,2}}}},
        {{{{0,7,1,1,0},{2,7,5,3,3},{6,5,4,8,6},{4,8,0,6,4}},{{0,0,5,8,0},{8,1,6,4,7},{2,5,4,6,3},{0,4,0,2,7}},{{6,0,3,6,4},{1,5,3,8,0},{8,7,2,4,0},{8,3,2,3,6}}},
        {{{2,8,5,4,4},{0,0,3,8,5},{4,1,4,2,1},{4,1,8,1,1}},{{7,2,8,8,3},{3,4,3,3,6},{1,6,2,7,7},{0,5,4,6,1}},{{1,4,0,7,6},{8,7,6,8,2},{6,4,0,5,8},{6,4,2,4,0}}}},
        {{{{0,5,0,8,6},{5,5,3,8,1},{8,3,7,8,5},{1,4,3,4,4}},{{4,0,0,6,8},{4,8,0,1,7},{6,2,6,4,2},{4,7,5,8,1}},{{3,3,1,5,5},{2,4,6,0,5},{3,1,7,6,5},{6,2,8,1,2}}},
        {{{4,7,5,2,1},{6,5,3,1,5},{8,8,5,5,4},{3,3,4,1,5}},{{7,8,2,8,1},{6,0,2,4,5},{8,4,5,0,3},{7,2,5,0,0}},{{2,2,2,7,8},{1,0,7,5,8},{0,2,5,4,4},{1,3,5,8,4}}}}
    };  //(4,2,3,4,5)

    //0tensor,1functor,2keep_dims,3initial,4expected
    auto test_data = std::make_tuple(
        //keep_dims is false
        std::make_tuple(tensor_type{}, std::plus<void>{}, false, value_type{0}, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, std::multiplies<void>{}, false, value_type{1}, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::plus<void>{}, false, value_type{0}, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::multiplies<void>{}, false, value_type{1}, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::plus<void>{}, false, no_value{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::plus<void>{}, false, no_value{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::plus<void>{}, false, no_value{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::plus<void>{}, false, no_value{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::multiplies<void>{}, false, no_value{}, tensor_type(720)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, false, no_value{}, tensor_type(9)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, false, no_value{}, tensor_type(0)),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::plus<void>{}, false, no_value{}, tensor_type(28)),
        //keep_dims is true
        std::make_tuple(tensor_type{}, std::plus<void>{}, true, value_type{0}, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}, std::multiplies<void>{}, true, value_type{1}, tensor_type{value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(2,1,0), std::plus<void>{}, true, value_type{0}, tensor_type{{{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), std::multiplies<void>{}, true, value_type{1}, tensor_type{{{value_type{1}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::plus<void>{}, true, no_value{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::plus<void>{}, true, no_value{}, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::plus<void>{}, true, no_value{}, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::plus<void>{}, true, no_value{}, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::multiplies<void>{}, true, no_value{}, tensor_type{{720}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, true, no_value{}, tensor_type{{9}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, true, no_value{}, tensor_type{{0}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::plus<void>{}, true, no_value{}, tensor_type{{{28}}}),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten,std::plus<void>{},false,value_type{0},tensor_type(7696)),
        std::make_tuple((test_ten-1)*(test_ten+1),std::plus<void>{},false,value_type{0},tensor_type(10450)),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,1)+test_ten(2,0,2),std::plus<void>{},false,value_type{0},tensor_type(7668)),
        std::make_tuple((test_ten+test_ten(0))*(test_ten(1,1)-test_ten(2,0,2)),std::plus<void>{},false,value_type{0},tensor_type(-307))
    );

    auto test_reduce_binary_flatten = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto functor = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = reduce_binary(policy...,ten,gtensor::detail::no_value{},functor,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };

    SECTION("default_policy")
    {
        test_reduce_binary_flatten();
    }
    SECTION("exec_pol<4>")
    {
        test_reduce_binary_flatten(multithreading::exec_pol<4>{});
    }
}

TEMPLATE_TEST_CASE("test_reduce_binary_big","[test_reduce]",
    (multithreading::exec_pol<1>),
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = std::size_t;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = tensor_type::shape_type;
    using helpers_for_testing::generate_lehmer;
    using helpers_for_testing::apply_by_element;

    tensor_type t(shape_type{32,16,8,64,4,16}); //1<<24
    generate_lehmer(t.begin(),t.end(),[](const auto& e){return e%2;},123);

    //0ten,1axes,2binary_f,3keep_dims,4initial,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::cref(t),std::vector<int>{0,2,3,5},std::plus<void>{},false,value_type{0},tensor_type{{130985,131241,131122,131418},{131063,130985,130771,131109},{130880,130776,131173,130602},{130953,131504,130845,130713},{130533,131118,131072,130999},{131537,131601,131137,130747},{130771,131109,131092,131087},{131009,131288,131239,131240},{131045,130487,131331,131042},{130738,130992,131102,131046},{131303,130886,131084,131374},{130716,131235,131133,130959},{130922,131557,131289,131151},{130930,130964,131054,130756},{131444,131149,131506,130919},{131779,130963,131140,130513}}),
        std::make_tuple(std::cref(t),std::vector<int>{0,1,3,5},std::plus<void>{},false,value_type{0},tensor_type{{262550,262341,261387,262795},{262126,262338,263037,261910},{262039,261752,262161,261542},{261610,261842,262138,262175},{262190,262597,262274,262151},{262391,262182,262827,261800},{261970,262571,261781,261997},{261732,262232,262485,261305}}),
        std::make_tuple(std::cref(t),std::vector<int>{0,1,2,3,5},std::plus<void>{},false,value_type{0},tensor_type{2096608,2097855,2098090,2095675}),
        std::make_tuple(std::cref(t),std::vector<int>{1,2,3,4,5},std::plus<void>{},false,value_type{0},tensor_type{262294,262306,261408,262194,261907,262785,262000,262093,262364,261515,261966,262240,262489,262095,262097,262023,262345,261632,262444,262166,262217,262027,262144,262242,262540,261711,262199,261935,262167,262702,261858,262123}),
        std::make_tuple(std::cref(t),std::vector<int>{0,1,2,3,4,5},std::plus<void>{},false,value_type{0}, tensor_type(8388228)),
        std::make_tuple(std::cref(t),gtensor::detail::no_value{},std::plus<void>{},false,value_type{0},tensor_type(8388228))
    );
    auto test = [](const auto& t){
        auto& ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto binary_f = std::get<2>(t);
        auto keep_dims = std::get<3>(t);
        auto initial = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = gtensor::reduce_binary(policy{},ten,axes,binary_f,keep_dims,initial);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

