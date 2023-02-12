#include <tuple>
#include "catch.hpp"
#include "gtensor.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

namespace test_expression_template_engine{

//test configs
using test_config_div_native = typename test_config::config_host_engine_div_selector<gtensor::config::engine_expression_template,gtensor::config::mode_div_native>::config_type;
using test_config_div_libdivide = typename test_config::config_host_engine_div_selector<gtensor::config::engine_expression_template,gtensor::config::mode_div_libdivide>::config_type;

//wrapper to get access to protected tensor engine
template<typename T>
struct test_tensor : public T
{
    using base_type = T;
    test_tensor(const T& base):
        base_type{base}
    {}
    using base_type::engine;
};

template<typename T>
auto make_test_tensor(T&& t){return test_tensor<std::decay_t<T>>{std::forward<T>(t)};}

//test tensors maker
//make tuple of tensors of same shape and value using chain of operators and views
//result tensor is evaluated to {{{1,2,3},{4,5,6}}}, shape is (1,2,3)
template<typename ValT, typename CfgT>
struct test_data{
    using tensor_type = gtensor::tensor<ValT, CfgT>;
    typename CfgT::nop_type nop{};
    auto operator()()const{
        return std::make_tuple(
            //storage
            tensor_type{{{1,2,3},{4,5,6}}},
            //slice of storage
            tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}}({{1,2},{},{nop,nop,2}}),
            //transpose of storage
            tensor_type{{{1},{4}},{{2},{5}},{{3},{6}}}.transpose(),
            //subdimension of storage
            tensor_type{{{{0,0,0},{0,0,0}}},{{{1,2,3},{4,5,6}}}}(1),
            //reshape of storage
            tensor_type{{{1},{2},{3}},{{4},{5},{6}}}.reshape(1,2,3),
            //view chain of storage
            tensor_type{{{1},{3},{5}},{{2},{4},{6}}}.transpose().reshape(1,2,3),
            //broadcast
            tensor_type{{{0},{3}}} + tensor_type{1,2,3},
            tensor_type{2} * tensor_type{-1,-1,-1} + (tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}}) + tensor_type{5,5,5} - tensor_type{3},
            //trivial broadcast
            tensor_type{{{-1,-1,-1},{-1,-1,-1}}} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{1,1,1},{4,4,4}}},
            //slice of broadcast
            (tensor_type{2} * tensor_type{{1,1,1,1,1,1}} + tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}} - tensor_type{{{3,3,3,3,3,3}}} + tensor_type{1})({{1,2},{},{nop,nop,2}}),
            //slice chain of broadcast
            (tensor_type{2} * tensor_type{{1,1,1,1,1,1}} + tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{0,3,0,2,0,1},{0,6,0,5,0,4}}} - tensor_type{{{3,3,3,3,3,3}}} + tensor_type{1})({{},{},{nop,nop,-1}})({{1,2},{},{nop,nop,2}})
        );
    }
};

}   //end of namespace test_expression_template_engine

TEST_CASE("test_is_trivial","[test_expression_template_engine]")
{
    using value_type = float;
    using tensor_type = gtensor::tensor<value_type>;
    using test_expression_template_engine::test_tensor;
    using test_type = std::tuple<bool, bool>;
    using helpers_for_testing::apply_by_element;
    //0tensor,1expected_is_trivial
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}+tensor_type{1},true),
        std::make_tuple(tensor_type{1,2,3,4,5}+tensor_type{1,2,3,4,5},true),
        std::make_tuple((tensor_type{1,2,3}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5}),true),
        std::make_tuple(tensor_type{1,2,3,4,5}+tensor_type{1},false),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5}}+tensor_type{{1,2,3,4,5}},false)
    );

    auto test = [](auto& test_data){
        auto t = test_tensor{std::get<0>(test_data)};
        auto expected_is_trivial = std::get<1>(test_data);
        auto result_is_trivial = t.engine().is_trivial();
        REQUIRE(result_is_trivial == expected_is_trivial);
    };

    apply_by_element(test, test_data);
}

TEST_CASE("test_walker","[test_expression_template_engine]")
{
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_expression_template_engine::test_tensor;
    using test_expression_template_engine::make_test_tensor;
    using helpers_for_testing::apply_by_element;

    //{{{1,2,3},{4,5,6}}}, shape is (1,2,3), direction indexes are (2,1,0)
    auto test_data = test_expression_template_engine::test_data<value_type,config_type>();

    //tests for broadcast walker
    auto tests = std::make_tuple(
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); REQUIRE(*w == value_type{1});},
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(0,1); REQUIRE(*w == value_type{2});},
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(1,1); REQUIRE(*w == value_type{4});},
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(0,1); w.walk(1,1); REQUIRE(*w == value_type{5});},
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(1,1); w.walk(0,2); REQUIRE(*w == value_type{6});},
        //walk in direction 2 has no effect, its dimension 1
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(2,1); REQUIRE(*w == value_type{1});},
        //walk in direction 3 has no effect, no such dimension
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(3,1); REQUIRE(*w == value_type{1});},
        //call reset when on the end of direction
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(0,2); w.reset(0); REQUIRE(*w == value_type{1});},
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(1,1); w.walk(0,2); w.reset(1); REQUIRE(*w == value_type{3});},
        //reset on direction with dimension 1 has no effect
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(1,1); w.walk(0,2); w.reset(2); REQUIRE(*w == value_type{6});},
        //reset on direction 3 has no effect, no such direction
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(1,1); w.walk(0,1); w.reset(3); REQUIRE(*w == value_type{5});},
        //reset to first from any position
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(1,1); w.walk(0,2); w.reset(); REQUIRE(*w == value_type{1});},
        [](auto& t_){auto t = make_test_tensor(t_); auto w = t.engine().create_walker(); w.walk(0,2); w.reset(); REQUIRE(*w == value_type{1});}
    );

    auto apply_tests = [&test_data](auto& test){
        //apply test to each tensor in test_data
        apply_by_element(test, test_data());
    };

    //apply to each test in tests
    apply_by_element(apply_tests, tests);

}

TEST_CASE("test_trivial_walker","[test_expression_template_engine]")
{
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_expression_template_engine::test_tensor;
    using test_expression_template_engine::make_test_tensor;
    using helpers_for_testing::apply_by_element;

    //{{{1,2,3},{4,5,6}}}, shape is (1,2,3), direction indexes are (2,1,0)
    auto test_data = test_expression_template_engine::test_data<value_type,config_type>();

    auto test = [](auto& t_){
        auto t = make_test_tensor(t_);
        auto w = t.engine().create_trivial_walker();
    };

}

// TEMPLATE_LIST_TEST_CASE("test_indexer","[test_expression_template_engine]",
//     typename test_expression_template_helpers::makers_type_list<test_expression_template_helpers::test_tensor>::type
// )
// {
//     using value_type = typename TestType::value_type;
//     using test_type = std::tuple<value_type,value_type>;

//     //0result,1expected
//     auto test_data = GENERATE(
//         test_type{TestType{}().create_indexer()[0],value_type{1}},
//         test_type{TestType{}().create_indexer()[5],value_type{6}},
//         test_type{TestType{}().create_indexer()[1],value_type{2}},
//         test_type{TestType{}().create_indexer()[2],value_type{3}},
//         test_type{TestType{}().create_indexer()[4],value_type{5}},
//         test_type{TestType{}().create_indexer()[3],value_type{4}}
//     );
//     auto result = std::get<0>(test_data);
//     auto expected = std::get<1>(test_data);
//     REQUIRE(result == expected);
// }

// TEST_CASE("test_result_type","[test_expression_template_engine]"){
//     using value_type = float;
//     using reference_type = value_type&;
//     using const_reference_type = const value_type&;
//     using test_expression_template_helpers::test_tensor;
//     using test_expression_template_helpers::make_test_tensor;
//     using test_config_type = typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type;
//     using index_type = typename test_config_type::index_type;
//     using tensor_type = gtensor::tensor<value_type, test_config_type>;

//     auto t = make_test_tensor<test_tensor>(tensor_type{1,2,3});
//     SECTION("storage"){
//         auto& re = t.engine();
//         const auto& cre = re;
//         REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),reference_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),reference_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.begin()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.end()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.begin()),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.end()),const_reference_type>);
//     }
//     SECTION("view_of_storage"){
//         auto vt = make_test_tensor<test_tensor>(t({{}}));
//         auto& re = vt.engine();
//         const auto& cre = re;
//         REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),reference_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),reference_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.begin()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.end()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.begin()),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.end()),const_reference_type>);
//     }
//     SECTION("view_view_of_storage"){
//         auto vvt = make_test_tensor<test_tensor>(t({{}}).transpose().reshape());
//         auto& re = vvt.engine();
//         const auto& cre = re;
//         REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),reference_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),reference_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.begin()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*re.end()),reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.begin()),const_reference_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.end()),const_reference_type>);
//     }
//     SECTION("evaluating"){
//         auto e = make_test_tensor<test_tensor>(t+t+t);
//         auto& re = e.engine();
//         const auto& cre = re;
//         REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.begin()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.end()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.begin()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.end()),value_type>);
//     }
//     SECTION("view_of_evaluating"){
//         auto ve = make_test_tensor<test_tensor>((t+t+t).transpose());
//         auto& re = ve.engine();
//         const auto& cre = re;
//         REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.begin()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.end()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.begin()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.end()),value_type>);
//     }
//     SECTION("view_view_of_evaluating"){
//         auto vve = make_test_tensor<test_tensor>((t+t+t).transpose().reshape()({{}})());
//         auto& re = vve.engine();
//         const auto& cre = re;
//         REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.begin()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*re.end()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.begin()),value_type>);
//         REQUIRE(std::is_same_v<decltype(*cre.end()),value_type>);
//     }
// }

// namespace test_iterator_helpers{
// using test_iterator_makers = typename helpers_for_testing::list_concat<
//     typename test_expression_template_helpers::makers_type_list<test_expression_template_helpers::test_broadcast_iterator_tensor>::type,
//     typename test_expression_template_helpers::makers_trivial_type_list<test_expression_template_helpers::test_trivial_iterator_tensor>::type
//     >::type;
// }

// TEMPLATE_LIST_TEST_CASE("test_iterator","[test_expression_template_engine]",test_iterator_helpers::test_iterator_makers)
// {
//     SECTION("test_iter_deref"){
//         using value_type = typename TestType::value_type;
//         using test_type = std::tuple<value_type, value_type>;
//         //0deref,1expected_deref
//         auto test_data = GENERATE(
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); return *it;}(), value_type{1}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; return *it;}(), value_type{2}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; return *it;}(), value_type{3}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; return *it;}(), value_type{4}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; return *it;}(), value_type{5}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; return *it;}(), value_type{6}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; return *it;}(), value_type{5}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; --it; return *it;}(), value_type{4}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; return *it;}(), value_type{3}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; --it; return *it;}(), value_type{2}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; --it; --it; return *it;}(), value_type{1}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; return *it;}(), value_type{6}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; return *it;}(), value_type{5}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; return *it;}(), value_type{4}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; return *it;}(), value_type{3}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; return *it;}(), value_type{2}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; return *it;}(), value_type{1}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; return *it;}(), value_type{2}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; ++it; return *it;}(), value_type{3}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; return *it;}(), value_type{4}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; ++it; return *it;}(), value_type{5}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; ++it; ++it; return *it;}(), value_type{6}},
//             test_type{TestType{}().begin()[0], value_type{1}},
//             test_type{TestType{}().begin()[1], value_type{2}},
//             test_type{TestType{}().begin()[2], value_type{3}},
//             test_type{TestType{}().begin()[3], value_type{4}},
//             test_type{TestType{}().begin()[4], value_type{5}},
//             test_type{TestType{}().begin()[5], value_type{6}},
//             test_type{TestType{}().end()[-1], value_type{6}},
//             test_type{TestType{}().end()[-2], value_type{5}},
//             test_type{TestType{}().end()[-3], value_type{4}},
//             test_type{TestType{}().end()[-4], value_type{3}},
//             test_type{TestType{}().end()[-5], value_type{2}},
//             test_type{TestType{}().end()[-6], value_type{1}},
//             test_type{*(TestType{}().begin()+0), value_type{1}},
//             test_type{*(TestType{}().begin()+1), value_type{2}},
//             test_type{*(TestType{}().begin()+2), value_type{3}},
//             test_type{*(TestType{}().begin()+3), value_type{4}},
//             test_type{*(TestType{}().begin()+4), value_type{5}},
//             test_type{*(TestType{}().begin()+5), value_type{6}},
//             test_type{*(TestType{}().end()-1), value_type{6}},
//             test_type{*(TestType{}().end()-2), value_type{5}},
//             test_type{*(TestType{}().end()-3), value_type{4}},
//             test_type{*(TestType{}().end()-4), value_type{3}},
//             test_type{*(TestType{}().end()-5), value_type{2}},
//             test_type{*(TestType{}().end()-6), value_type{1}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); it+=0; return *it;}(), value_type{1}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); it+=1; return *it;}(), value_type{2}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); it+=2; return *it;}(), value_type{3}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); it+=3; return *it;}(), value_type{4}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); it+=4; return *it;}(), value_type{5}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); it+=5; return *it;}(), value_type{6}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); it-=1; return *it;}(), value_type{6}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); it-=2; return *it;}(), value_type{5}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); it-=3; return *it;}(), value_type{4}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); it-=4; return *it;}(), value_type{3}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); it-=5; return *it;}(), value_type{2}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); it-=6; return *it;}(), value_type{1}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it;  return *(it-2);}(), value_type{1}},
//             test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it;  return *(it+2);}(), value_type{5}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it;  return *(it+1);}(), value_type{6}},
//             test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it;  return *(it-2);}(), value_type{3}}
//         );
//         auto deref = std::get<0>(test_data);
//         auto expected_deref = std::get<1>(test_data);
//         REQUIRE(deref == expected_deref);
//     }
//     SECTION("test_iter_cmp"){
//         using test_type = std::tuple<bool,bool>;
//         //0cmp_result,1expected_cmp_result
//         auto test_data = GENERATE(
//             test_type{[](){auto t = TestType{}(); return t.begin() > t.end();}(), false},
//             test_type{[](){auto t = TestType{}(); return t.end() > t.begin();}(), true},
//             test_type{[](){auto t = TestType{}(); return t.begin() < t.end();}(), true},
//             test_type{[](){auto t = TestType{}(); return t.end() < t.begin();}(), false},
//             test_type{[](){auto t = TestType{}(); return t.end() > t.end();}(), false},
//             test_type{[](){auto t = TestType{}(); return t.begin() > t.begin();}(), false},
//             test_type{[](){auto t = TestType{}(); return t.end() < t.end();}(), false},
//             test_type{[](){auto t = TestType{}(); return t.begin() < t.begin();}(), false},
//             test_type{[](){auto t = TestType{}(); return t.end() >= t.end();}(), true},
//             test_type{[](){auto t = TestType{}(); return t.begin() >= t.begin();}(), true},
//             test_type{[](){auto t = TestType{}(); return t.end() <= t.end();}(), true},
//             test_type{[](){auto t = TestType{}(); return t.begin() <= t.begin();}(), true}
//         );
//         auto cmp_result = std::get<0>(test_data);
//         auto expected_cmp_result = std::get<1>(test_data);
//         REQUIRE(cmp_result == expected_cmp_result);
//     }
// }

TEMPLATE_TEST_CASE("test_broadcast_iterator","[test_expression_template_engine]",
    test_expression_template_engine::test_config_div_native,
    test_expression_template_engine::test_config_div_libdivide
)
{
    using value_type = float;
    using config_type = TestType;
    using shape_type = typename config_type::shape_type;
    using tensor_type = gtensor::tensor<value_type,config_type>;
    using test_expression_template_engine::test_tensor;
    using helpers_for_testing::apply_by_element;

    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3}, shape_type{3}, std::vector<value_type>{1,2,3}),
        std::make_tuple(tensor_type{1,2,3}, shape_type{1,1,3}, std::vector<value_type>{1,2,3}),
        std::make_tuple(tensor_type{1,2,3}, shape_type{2,3}, std::vector<value_type>{1,2,3,1,2,3}),
        std::make_tuple(tensor_type{1,2,3}, shape_type{2,2,3}, std::vector<value_type>{1,2,3,1,2,3,1,2,3,1,2,3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, shape_type{1,2,3}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, shape_type{2,2,3}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1},{2},{3}}, shape_type{3,3}, std::vector<value_type>{1,1,1,2,2,2,3,3,3}),
        std::make_tuple(tensor_type{-1} + tensor_type{1,2,3} + tensor_type{{1},{2},{3}} + tensor_type{1}, shape_type{1,3,3}, std::vector<value_type>{2,3,4,3,4,5,4,5,6}),
        std::make_tuple((tensor_type{-1} + tensor_type{1,2,3} + tensor_type{{1},{2},{3}} + tensor_type{1})({{{},{},2}}), shape_type{1,2,3}, std::vector<value_type>{2,3,4,4,5,6})
    );
    auto test = [](const auto& test_data_){
        auto test_ten = test_tensor{std::get<0>(test_data_)};
        auto broadcast_shape = std::get<1>(test_data_);
        auto expected = std::get<2>(test_data_);
        auto begin = test_ten.engine().begin_broadcast(broadcast_shape);
        auto end = test_ten.engine().end_broadcast(broadcast_shape);
        REQUIRE(std::distance(begin,end) == expected.size());
        REQUIRE(std::equal(expected.begin(),expected.end(), begin));
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_broadcast_assignment","[test_expression_template_engine]"){
    using value_type = float;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::broadcast_exception;

    //tensor assignment
    auto t_lhs = tensor_type();
    t_lhs = tensor_type{1,2,3};
    REQUIRE(std::equal(t_lhs.begin(), t_lhs.end(), std::vector<float>{1,2,3}.begin()));

    //broadcast assignment
    auto lhs = tensor_type{1,2,3,4,5};
    lhs({{{},{},2}}) = tensor_type{0};

    auto lhs1 = tensor_type{{1,2,3},{4,5,6}};
    lhs1() = tensor_type{0,1,2};
    REQUIRE(std::equal(lhs1.begin(), lhs1.end(), std::vector<float>{0,1,2,0,1,2}.begin()));

    auto lhs2 = tensor_type{{1,2,3},{4,5,6}};
    lhs2(1) = tensor_type{0,1,2};
    REQUIRE(std::equal(lhs2.begin(), lhs2.end(), std::vector<float>{1,2,3,0,1,2}.begin()));

    auto lhs4 = tensor_type{1,2,3,4,5};
    lhs4({{{},{},-1}})({{{},{},2}}) = tensor_type{0,1,2};
    REQUIRE(std::equal(lhs4.begin(), lhs4.end(), std::vector<float>{2,2,1,4,0}.begin()));

    //broadcast assignment exception, every element of lhs must be assigned only once
    auto lhs3 = tensor_type{0};
    REQUIRE_THROWS_AS((lhs3() = tensor_type{0,1,2}), broadcast_exception);

    //not compile, broadcast assignment to evaluating tensor
    //lhs+lhs = tensor_type{1};

    //not compile, broadcast assignment to view of evaluating tensor
    // auto e = lhs+lhs;
    // e() = tensor_type{1};
}