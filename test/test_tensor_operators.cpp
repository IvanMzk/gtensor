#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"
#include "tensor.hpp"

namespace test_tensor_operators_{

struct unary_ident_ref{
    template<typename T>
    T& operator()(T& t)const{
        return t;
    }
};
struct unary_ident_const_ref{
    template<typename T>
    const T& operator()(const T& t)const{
        return t;
    }
};
struct unary_square{
    template<typename T>
    auto operator()(const T& t)const{
        return t*t;
    }
};
struct binary_mul{
    template<typename T1, typename T2>
    auto operator()(const T1& t1, const T2& t2)const{
        return t1*t2;
    }
};
struct binary_sub{
    template<typename T1, typename T2>
    auto operator()(const T1& t1, const T2& t2)const{
        return t1-t2;
    }
};
struct ternary_add_mul{
    template<typename T1, typename T2, typename T3>
    auto operator()(const T1& t1, const T2& t2, const T3& t3)const{
        return (t1+t2)*t3;
    }
};

struct assign{
    template<typename T1, typename T2>
    void operator()(T1& t1, const T2& t2)const{
        t1 = t2;
    }
};
struct assign_add{
    template<typename T1, typename T2>
    void operator()(T1& t1, const T2& t2)const{
        t1 += t2;
    }
};

}

//test generalized operators
TEST_CASE("test_n_operator","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using test_tensor_operators_::unary_square;
    using test_tensor_operators_::binary_mul;
    using test_tensor_operators_::binary_sub;
    using test_tensor_operators_::ternary_add_mul;
    using gtensor::n_operator;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(unary_square{},std::make_tuple(tensor_type{}),tensor_type{}),
        std::make_tuple(unary_square{},std::make_tuple(tensor_type(2)),tensor_type(4)),
        std::make_tuple(unary_square{},std::make_tuple(tensor_type{1,2,3,4,5}),tensor_type{1,4,9,16,25}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type(5),2),tensor_type(10)),
        std::make_tuple(binary_mul{},std::make_tuple(3,tensor_type(5)),tensor_type(15)),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type(5),tensor_type(4)),tensor_type(20)),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{1,2,3,4,5},2),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(3,tensor_type{1,2,3,4,5}),tensor_type{3,6,9,12,15}),
        std::make_tuple(binary_sub{},std::make_tuple(3,tensor_type{1,2,3,4,5}),tensor_type{2,1,0,-1,-2}),
        std::make_tuple(binary_sub{},std::make_tuple(tensor_type{1,2,3,4,5},3),tensor_type{-2,-1,0,1,2}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{1,2,3,4,5},tensor_type(2)),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{2}),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{5,4,3,2,1}),tensor_type{5,8,9,8,5}),
        std::make_tuple(binary_mul{},std::make_tuple(4,tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),tensor_type{{{4,8},{12,16}},{{20,24},{28,32}}}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type(4), tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),tensor_type{{{4,8},{12,16}},{{20,24},{28,32}}}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{-1,2}),tensor_type{{{-1,4},{-3,8}},{{-5,12},{-7,16}}}),
        std::make_tuple(binary_mul{},std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{-1},{2}}),tensor_type{{{-1,-2},{6,8}},{{-5,-6},{14,16}}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(tensor_type(1),tensor_type(2),tensor_type(3)),tensor_type(9)),
        std::make_tuple(ternary_add_mul{},std::make_tuple(tensor_type(1),tensor_type{1,2,3,4,5},tensor_type(3)),tensor_type{6,9,12,15,18}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(tensor_type(1),tensor_type{1,2,3},tensor_type{{1},{2},{3}}),tensor_type{{2,3,4},{4,6,8},{6,9,12}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(-1,tensor_type{1,2,3},tensor_type{{1},{2},{3}}),tensor_type{{0,1,2},{0,2,4},{0,3,6}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(tensor_type{{4,5,6},{7,8,9}},tensor_type{1,2,3},2),tensor_type{{10,14,18},{16,20,24}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(-1,tensor_type{{1,2,3},{4,5,6}},2),tensor_type{{0,2,4},{6,8,10}})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        auto operands = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto apply_n_operator = [f](auto&&...operands){
            return n_operator(f,operands...);
        };
        auto result = std::apply(apply_n_operator, operands);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_a_operator","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using test_tensor_operators_::assign;
    using test_tensor_operators_::assign_add;
    using gtensor::a_operator;
    using helpers_for_testing::apply_by_element;
    //0operation,1lhs,2rhs,3expected
    auto test_data = std::make_tuple(
        //rhs scalar
        std::make_tuple(assign{},tensor_type{},2,tensor_type{}),
        std::make_tuple(assign{},tensor_type(2),1,tensor_type(1)),
        std::make_tuple(assign{},tensor_type{1,2,3,4,5},3,tensor_type{3,3,3,3,3}),
        std::make_tuple(assign{},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},4,tensor_type{{{4,4},{4,4}},{{4,4},{4,4}}}),
        std::make_tuple(assign_add{},tensor_type{},2,tensor_type{}),
        std::make_tuple(assign_add{},tensor_type(2),1,tensor_type(3)),
        std::make_tuple(assign_add{},tensor_type{1,2,3,4,5},3,tensor_type{4,5,6,7,8}),
        std::make_tuple(assign_add{},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},4,tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}),
        //rhs tensor
        std::make_tuple(assign{},tensor_type{},tensor_type{},tensor_type{}),
        std::make_tuple(assign{},tensor_type{},tensor_type(1),tensor_type{}),
        std::make_tuple(assign{},tensor_type{},tensor_type{1},tensor_type{}),
        std::make_tuple(assign{},tensor_type(1),tensor_type{},tensor_type(1)),
        std::make_tuple(assign{},tensor_type(1),tensor_type(2),tensor_type(2)),
        std::make_tuple(assign{},tensor_type(2),tensor_type{3},tensor_type(3)),
        std::make_tuple(assign{},tensor_type{1,2,3,4,5},tensor_type{6},tensor_type{6,6,6,6,6}),
        std::make_tuple(assign{},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{-1},{1}},tensor_type{{{-1,-1},{1,1}},{{-1,-1},{1,1}}}),
        std::make_tuple(assign_add{},tensor_type{},tensor_type{},tensor_type{}),
        std::make_tuple(assign_add{},tensor_type{},tensor_type(1),tensor_type{}),
        std::make_tuple(assign_add{},tensor_type{},tensor_type{1},tensor_type{}),
        std::make_tuple(assign_add{},tensor_type(1),tensor_type{},tensor_type(1)),
        std::make_tuple(assign_add{},tensor_type(1),tensor_type(2),tensor_type(3)),
        std::make_tuple(assign_add{},tensor_type(2),tensor_type{3},tensor_type(5)),
        std::make_tuple(assign_add{},tensor_type{1,2,3,4,5},tensor_type{6},tensor_type{7,8,9,10,11}),
        std::make_tuple(assign_add{},tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},tensor_type{{-1},{1}},tensor_type{{{0,1},{4,5}},{{4,5},{8,9}}}),
        //assign multiple times to lhs
        std::make_tuple(assign{},tensor_type(3),tensor_type{1,2,3,4,5},tensor_type(5)),
        std::make_tuple(assign{},tensor_type{0},tensor_type{1,2,3,4,5},tensor_type{5}),
        std::make_tuple(assign{},tensor_type{0,0},tensor_type{{1,2},{3,4},{5,6}},tensor_type{5,6}),
        std::make_tuple(assign_add{},tensor_type(3),tensor_type{1,2,3,4,5},tensor_type(18)),
        std::make_tuple(assign_add{},tensor_type{0},tensor_type{1,2,3,4,5},tensor_type{15}),
        std::make_tuple(assign_add{},tensor_type{-1,1},tensor_type{{1,2},{3,4},{5,6}},tensor_type{8,13})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        auto lhs = std::get<1>(t);
        auto rhs = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto& result = a_operator(f,lhs,rhs);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

//test operators
TEST_CASE("test_gtensor_unary_operator","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //0operand,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type(1),tensor_type(-1)),
        std::make_tuple(tensor_type(-1),tensor_type(1)),
        std::make_tuple(tensor_type{{1,-2},{-3,4}},tensor_type{{-1,2},{3,-4}})
    );
    auto test = [](const auto& t){
        auto operand = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = -operand;
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_binary_operator","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //0operand1,1operand2,2expected
    auto test_data = std::make_tuple(
        //other operand
        std::make_tuple(1,tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{},2,tensor_type{}),
        std::make_tuple(1,tensor_type(2),tensor_type(3)),
        std::make_tuple(tensor_type(1),2,tensor_type(3)),
        std::make_tuple(1,tensor_type{{1,2},{3,4}},tensor_type{{2,3},{4,5}}),
        std::make_tuple(tensor_type{{1,2},{3,4}},2,tensor_type{{3,4},{5,6}}),
        //0-dim operand
        std::make_tuple(tensor_type(1), tensor_type(2),tensor_type(3)),
        std::make_tuple(tensor_type(1), tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type(2),tensor_type{}),
        std::make_tuple(tensor_type(1), tensor_type{{1,2},{3,4}},tensor_type{{2,3},{4,5}}),
        std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type(2),tensor_type{{3,4},{5,6}}),
        //n-dim,n-dim
        std::make_tuple(tensor_type{},tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1},tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type{2},tensor_type{}),
        std::make_tuple(tensor_type{1},tensor_type{{1,2},{3,4}},tensor_type{{2,3},{4,5}}),
        std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{2},tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{1,2},tensor_type{{3},{4}},tensor_type{{4,5},{5,6}}),
        std::make_tuple(tensor_type{{3},{4}},tensor_type{1,2},tensor_type{{4,5},{5,6}}),
        std::make_tuple(tensor_type{{3},{4}}+1,tensor_type{1,2}*2,tensor_type{{6,8},{7,9}}),
        std::make_tuple(tensor_type{{3},{4}}+tensor_type(1),tensor_type{1,2}*tensor_type{1,2},tensor_type{{5,8},{6,9}})
    );
    auto test = [](const auto& t){
        auto operand1 = std::get<0>(t);
        auto operand2 = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = operand1 + operand2;
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_operator_rvalue_operand","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    tensor_type t1{1,2,3};
    tensor_type t2{4,5,6};
    SECTION("unary_operator")
    {
        auto result = -std::move(t1);
        REQUIRE(result == tensor_type{-1,-2,-3});
        REQUIRE(t1.empty());
    }
    SECTION("binary_operator_left_rvalue")
    {
        auto result = std::move(t1)+t2;
        REQUIRE(result == tensor_type{5,7,9});
        REQUIRE(t1.empty());
    }
    SECTION("binary_operator_right_rvalue")
    {
        auto result = t1+std::move(t2);
        REQUIRE(result == tensor_type{5,7,9});
        REQUIRE(t2.empty());
    }
    SECTION("binary_operator_both_rvalue")
    {
        auto result = std::move(t1)+std::move(t2);
        REQUIRE(result == tensor_type{5,7,9});
        REQUIRE(t1.empty());
        REQUIRE(t2.empty());
    }
}

//str
TEMPLATE_TEST_CASE("test_tensor_operators_str","[test_tensor_operators]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::str;
    REQUIRE(str(tensor_type(2))==std::string("[(){2}]"));
    REQUIRE(str(tensor_type(1.234567),2)==std::string("[(){1.2}]"));
    REQUIRE(str(tensor_type(1.23451),5)==std::string("[(){1.2345}]"));
    REQUIRE(str(tensor_type{})==std::string("[(0){}]"));
    REQUIRE(str(tensor_type{}.reshape(0,2,0,3,0))==std::string("[(0,2,0,3,0){}]"));
    REQUIRE(str(tensor_type{1})==std::string("[(1){1}]"));
    REQUIRE(str(tensor_type{1,2,3,4,5})==std::string("[(5){1,2,3,4,5}]"));
    REQUIRE(str(tensor_type{{1,2,3},{4,5,6}})==std::string("[(2,3){{1,2,3},{4,5,6}}]"));
    REQUIRE(str(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}})==std::string("[(2,2,3){{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}]"));
    //unprintable value_type
    REQUIRE(str(tensor<std::pair<int,int>>{std::make_pair(0,0),std::make_pair(1,1),std::make_pair(2,1)})==std::string("[(3){...}]"));
    REQUIRE(str(tensor<std::pair<int,int>>{{std::make_pair(0,0),std::make_pair(1,1),std::make_pair(2,1)}})==std::string("[(1,3){{...}}]"));
    REQUIRE(str(tensor<std::pair<int,int>>{{std::make_pair(0,0)},{std::make_pair(1,1)},{std::make_pair(2,1)}})==std::string("[(3,1){{...},{...},{...}}]"));
}

//test assign operators
TEST_CASE("test_gtensor_assign_operator","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::assign;
    using helpers_for_testing::apply_by_element;
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        //other operand
        std::make_tuple(tensor_type{},2,tensor_type{},2),
        std::make_tuple(tensor_type(1),2,tensor_type(2),2),
        std::make_tuple(tensor_type{{1,2},{3,4}},2,tensor_type{{2,2},{2,2}},2),
        //0-dim operand
        std::make_tuple(tensor_type(1), tensor_type(2),tensor_type(2),tensor_type(2)),
        std::make_tuple(tensor_type(1), tensor_type{},tensor_type(1),tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type(2),tensor_type{},tensor_type(2)),
        std::make_tuple(tensor_type(1), tensor_type{{1,2},{3,4}},tensor_type(4),tensor_type{{1,2},{3,4}}),
        std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type(2),tensor_type{{2,2},{2,2}},tensor_type(2)),
        //n-dim,n-dim
        std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1},tensor_type{},tensor_type{1},tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type{2},tensor_type{},tensor_type{2}),
        std::make_tuple(tensor_type{1},tensor_type{{1,2},{3,4}},tensor_type{4},tensor_type{{1,2},{3,4}}),
        std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{2},tensor_type{{2,2},{2,2}},tensor_type{2}),
        std::make_tuple(tensor_type{1,2},tensor_type{{3},{4}},tensor_type{4,4},tensor_type{{3},{4}}),
        std::make_tuple(tensor_type{{3},{4}},tensor_type{1,2},tensor_type{{2},{2}},tensor_type{1,2})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto& result = assign(lhs,rhs);
        REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_compound_assign_operator","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        //other operand
        std::make_tuple(tensor_type{},2,tensor_type{},2),
        std::make_tuple(tensor_type(1),2,tensor_type(3),2),
        std::make_tuple(tensor_type{{1,2},{3,4}},2,tensor_type{{3,4},{5,6}},2),
        //0-dim operand
        std::make_tuple(tensor_type(1), tensor_type(2),tensor_type(3),tensor_type(2)),
        std::make_tuple(tensor_type(1), tensor_type{},tensor_type(1),tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type(2),tensor_type{},tensor_type(2)),
        std::make_tuple(tensor_type(1), tensor_type{{1,2},{3,4}},tensor_type(11),tensor_type{{1,2},{3,4}}),
        std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type(2),tensor_type{{3,4},{5,6}},tensor_type(2)),
        //n-dim,n-dim
        std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1},tensor_type{},tensor_type{1},tensor_type{}),
        std::make_tuple(tensor_type{},tensor_type{2},tensor_type{},tensor_type{2}),
        std::make_tuple(tensor_type{1},tensor_type{{1,2},{3,4}},tensor_type{11},tensor_type{{1,2},{3,4}}),
        std::make_tuple(tensor_type{{1,2},{3,4}},tensor_type{2},tensor_type{{3,4},{5,6}},tensor_type{2}),
        std::make_tuple(tensor_type{1,2},tensor_type{{3},{4}},tensor_type{8,9},tensor_type{{3},{4}}),
        std::make_tuple(tensor_type{{3},{4}},tensor_type{1,2},tensor_type{{6},{7}},tensor_type{1,2})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto& result = lhs+=rhs;
        REQUIRE(std::is_same_v<decltype(lhs),std::remove_reference_t<decltype(result)>>);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_assign_operator_compound_assign_operator_rvalue_operand","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::assign;
    tensor_type lhs{1,2,3};
    tensor_type rhs{4,5,6};
    SECTION("assign_operator_rvalue_operand")
    {
        assign(lhs,std::move(rhs));
        REQUIRE(lhs == tensor_type{4,5,6});
        REQUIRE(rhs.empty());
    }
    SECTION("compound_assign_operator_rvalue_operand")
    {
        lhs+=std::move(rhs);
        REQUIRE(lhs == tensor_type{5,7,9});
        REQUIRE(rhs.empty());
    }
}

TEST_CASE("test_gtensor_assign_operator_lhs_is_view","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::assign;
    using helpers_for_testing::apply_by_element;
    //0parent,1lhs_view_maker,2rhs,3expected_parent,4expected_lhs,5expected_rhs
    auto test_data = std::make_tuple(
        //rhs scalar
        std::make_tuple(tensor_type{},[](const auto& t){return t();},1,tensor_type{},tensor_type{},1),
        std::make_tuple(tensor_type(1),[](const auto& t){return t.transpose();},2,tensor_type(2),tensor_type(2),2),
        std::make_tuple(tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},7,tensor_type{1,7,7,7,7,6},tensor_type{7,7,7,7},7),
        //rhs 0-dim
        std::make_tuple(tensor_type{},[](const auto& t){return t();},tensor_type(1),tensor_type{},tensor_type{},tensor_type(1)),
        std::make_tuple(tensor_type(1),[](const auto& t){return t.transpose();},tensor_type(2),tensor_type(2),tensor_type(2),tensor_type(2)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},tensor_type(7),tensor_type{1,7,7,7,7,6},tensor_type{7,7,7,7},tensor_type(7)),
        //rhs n-dim
        std::make_tuple(tensor_type{},[](const auto& t){return t();},tensor_type{1},tensor_type{},tensor_type{},tensor_type{1}),
        std::make_tuple(tensor_type(1),[](const auto& t){return t.transpose();},tensor_type{2},tensor_type(2),tensor_type(2),tensor_type{2}),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
            [](const auto& t){return t({{},{1}});},
            tensor_type{-1,1},
            tensor_type{{{1,2},{-1,1}},{{5,6},{-1,1}}},
            tensor_type{{{-1,1}},{{-1,1}}},
            tensor_type{-1,1}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            tensor_type{-1,1},
            tensor_type{{{1,5},{-1,1}},{{2,6},{-1,1}}},
            tensor_type{{{-1,1}},{{-1,1}}},
            tensor_type{-1,1}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            tensor_type{-1,1} + tensor_type{{1},{2}},
            tensor_type{{{1,5},{1,3}},{{2,6},{1,3}}},
            tensor_type{{{1,3}},{{1,3}}},
            tensor_type{{0,2},{1,3}}
        )
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto lhs_view_maker = std::get<1>(t);
        auto rhs = std::get<2>(t);
        auto expected_parent = std::get<3>(t);
        auto expected_lhs = std::get<4>(t);
        auto expected_rhs = std::get<5>(t);
        auto lhs = lhs_view_maker(parent);
        auto& result = assign(lhs,rhs);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(parent == expected_parent);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_compound_assign_operator_lhs_is_view","[test_tensor_operators]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using helpers_for_testing::apply_by_element;
    //0parent,1lhs_view_maker,2rhs,3expected_parent,4expected_lhs,5expected_rhs
    auto test_data = std::make_tuple(
        //rhs scalar
        std::make_tuple(tensor_type{},[](const auto& t){return t();},1,tensor_type{},tensor_type{},1),
        std::make_tuple(tensor_type(1),[](const auto& t){return t.transpose();},2,tensor_type(3),tensor_type(3),2),
        std::make_tuple(tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},7,tensor_type{1,9,10,11,12,6},tensor_type{9,10,11,12},7),
        std::make_tuple(tensor_type{1,2,3,4,5,6},[](const auto& t){return t(tensor<int>{1,1,3,0,0});},7,tensor_type{15,16,3,11,5,6},tensor_type{16,16,11,15,15},7),
        //rhs 0-dim
        std::make_tuple(tensor_type{},[](const auto& t){return t();},tensor_type(1),tensor_type{},tensor_type{},tensor_type(1)),
        std::make_tuple(tensor_type(1),[](const auto& t){return t.transpose();},tensor_type(2),tensor_type(3),tensor_type(3),tensor_type(2)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},[](const auto& t){return t({{1,-1}});},tensor_type(7),tensor_type{1,9,10,11,12,6},tensor_type{9,10,11,12},tensor_type(7)),
        //rhs n-dim
        std::make_tuple(tensor_type{},[](const auto& t){return t();},tensor_type{1},tensor_type{},tensor_type{},tensor_type{1}),
        std::make_tuple(tensor_type(1),[](const auto& t){return t.transpose();},tensor_type{2},tensor_type(3),tensor_type(3),tensor_type{2}),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},
            [](const auto& t){return t({{},{1}});},
            tensor_type{-1,1},
            tensor_type{{{1,2},{2,5}},{{5,6},{6,9}}},
            tensor_type{{{2,5}},{{6,9}}},
            tensor_type{-1,1}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            tensor_type{-1,1},
            tensor_type{{{1,5},{2,8}},{{2,6},{3,9}}},
            tensor_type{{{2,8}},{{3,9}}},
            tensor_type{-1,1}
        ),
        std::make_tuple(
            tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(),
            [](const auto& t){return t({{},{1}});},
            tensor_type{-1,1} + tensor_type{{1},{2}},
            tensor_type{{{1,5},{4,12}},{{2,6},{5,13}}},
            tensor_type{{{4,12}},{{5,13}}},
            tensor_type{{0,2},{1,3}}
        )
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto lhs_view_maker = std::get<1>(t);
        auto rhs = std::get<2>(t);
        auto expected_parent = std::get<3>(t);
        auto expected_lhs = std::get<4>(t);
        auto expected_rhs = std::get<5>(t);
        auto lhs = lhs_view_maker(parent);
        auto& result = lhs+=rhs;
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(parent == expected_parent);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_assignment_self_assignment","[test_tensor_operators]")
{
    struct assign_exception{};
    struct throw_on_assign{
        throw_on_assign() = default;
        throw_on_assign(const throw_on_assign&) = default;
        throw_on_assign& operator=(const throw_on_assign&){
            throw assign_exception{};
        }
    };
    REQUIRE_THROWS(throw_on_assign{} = throw_on_assign{});
    using value_type = throw_on_assign;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::assign;
    tensor_type t({10},value_type{});
    REQUIRE_NOTHROW(assign(t,t));
    REQUIRE_THROWS_AS(assign(t,tensor_type(value_type{})),assign_exception);
    auto v = t.transpose();
    REQUIRE_NOTHROW(assign(v,v));
    REQUIRE_THROWS_AS(assign(v,t),assign_exception);
}

TEST_CASE("test_gtensor_assignment_self_assignment_expression","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::assign;
    using helpers_for_testing::apply_by_element;
    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},tensor_type{},tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type(1),tensor_type{},tensor_type(1),tensor_type{}),
        std::make_tuple(tensor_type(1),tensor_type(2),tensor_type(3),tensor_type(2)),
        std::make_tuple(tensor_type(1),tensor_type{{1,2},{3,4}},tensor_type(11),tensor_type{{1,2},{3,4}}),
        std::make_tuple(tensor_type{1},tensor_type{{1,2},{3,4}},tensor_type(11),tensor_type{{1,2},{3,4}}),
        std::make_tuple(tensor_type{5,6},tensor_type{{1,2},{3,4}},tensor_type{9,12},tensor_type{{1,2},{3,4}})
    );
    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        assign(lhs,lhs+rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_assign_operator_compound_assign_operator_exception","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::assign;
    using gtensor::value_error;
    tensor_type lhs{1,2,3};
    tensor_type rhs{4,5};
    SECTION("assign_operator_exception")
    {
        REQUIRE_THROWS_AS(assign(lhs,rhs), value_error);
    }
    SECTION("compound_assign_operator_rvalue_operand")
    {
        REQUIRE_THROWS_AS((lhs+=rhs), value_error);
    }
}

//test tensor equality
TEST_CASE("test_tensor_equal_routine","test_tensor_operators")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type inf = std::numeric_limits<value_type>::infinity();
    //0ten_0,1ten_1,2equal_nan,3expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(1),tensor_type(1),false,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),false,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),false,true),
        std::make_tuple(tensor_type{},tensor_type{},false,true),
        std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type{1,2,3,4,5,6},false,true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}},false,true),
        std::make_tuple(tensor_type(1),tensor_type(2),false,false),
        std::make_tuple(tensor_type(1),tensor_type(-1),false,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),false,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),false,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),false,false),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),false,false),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),false,false),
        std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type{0,2,3,4,5,6},false,false),
        std::make_tuple(tensor_type{nan,2.0,3.0,4.0,5.0,6.0},tensor_type{nan,2.0,3.0,4.0,5.0,6.0},false,false),
        std::make_tuple(tensor_type{nan,2.0,3.0,4.0,5.0,6.0},tensor_type{1.0,2.0,3.0,4.0,5.0,6.0},false,false),
        std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type{{1,2,3},{4,5,6}},false,false),
        //nan_equal true
        std::make_tuple(tensor_type(1),tensor_type(1),true,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),true,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),true,true),
        std::make_tuple(tensor_type{},tensor_type{},true,true),
        std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type{1,2,3,4,5,6},true,true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}},true,true),
        std::make_tuple(tensor_type(1),tensor_type(2),true,false),
        std::make_tuple(tensor_type(1),tensor_type(-1),true,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),false,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),false,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),true,true),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),false,false),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),true,false),
        std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type{0,2,3,4,5,6},true,false),
        std::make_tuple(tensor_type{nan,2.0,3.0,4.0,5.0,6.0},tensor_type{nan,2.0,3.0,4.0,5.0,6.0},true,true),
        std::make_tuple(tensor_type{nan,2.0,3.0,4.0,5.0,6.0},tensor_type{1.0,2.0,3.0,4.0,5.0,6.0},false,false),
        std::make_tuple(tensor_type{1,2,3,4,5,6},tensor_type{{1,2,3},{4,5,6}},true,false),
        //not floating point type
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},false,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},true,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},true,false),
        //mixed types
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-15,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-15,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},true,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},false,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},true,false),
        std::make_tuple(tensor<double>{1.0,2.0,3.0,4.0,5.0},tensor<float>{1.0,2.0,3.0,4.0,5.0},false,true),
        std::make_tuple(tensor<double>{1.0,2.0,3.0,4.0,5.0},tensor<float>{1.0,2.0+1E-6,3.0,4.0,5.0},false,false),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},false,false),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},true,true)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto equal_nan = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = tensor_equal(ten_0,ten_1,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_equal_routine_different_config_traverse_order","[test_tensor_operators]")
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using gtensor::tensor_equal;
    using c_config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<c_order>,value_type>;
    using f_config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<f_order>,value_type>;

    REQUIRE(tensor_equal(tensor<value_type,c_order,c_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor<value_type,c_order,f_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}));
    REQUIRE(tensor_equal(tensor<value_type,c_order,f_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor<value_type,c_order,c_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}));
}

//test tensor fuzzy equality
TEST_CASE("test_tensor_close","test_tensor_operators")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type inf = std::numeric_limits<value_type>::infinity();
    //0ten_0,1ten_1,2relative_tolerance,3absolute_tolerance,4equal_nan,5expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{},tensor_type{},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2+1E-11,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{{1.1},{2.2},{3.3},{inf}},tensor_type{{1.1},{2.2},{3.3},{inf}},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-9),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2-1E-9,3.3},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2,3.3}},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,nan,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,false,false),
        //nan_equal true
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{},tensor_type{},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2+1E-11,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{{1.1},{2.2},{3.3},{inf}},tensor_type{{1.1},{2.2},{3.3},{inf}},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-9),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2-1E-9,3.3},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2,3.3}},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,nan,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,true,true),
        //vary tolerance
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-12,1E-12,false,false),
        //near zero: absolute tolerance plays
        std::make_tuple(tensor_type(1E-15),tensor_type(1E-15+1E-30),1E-16,1E-10,false,true),
        std::make_tuple(tensor_type(1E-15),tensor_type(1E-15+1E-30),1E-16,1E-32,false,false),
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-14,1E-10,false,true),
        //big: relative tolerance plays
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-15,1E-10,false,true),
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-16,1E-10,false,false),
        //not floating point type
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},1E-10,1E-10,false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},1E-10,1E-10,true,false),
        //mixed types
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},1E-10,1E-10,true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,false),
        std::make_tuple(tensor<double>{1.0,2.0,3.0,4.0,5.0},tensor<float>{1.0,2.0+1E-11,3.0,4.0,5.0},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},1E-10,1E-10,false,false),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},1E-10,1E-10,true,true)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto relative_tolerance = std::get<2>(t);
        auto absolute_tolerance = std::get<3>(t);
        auto equal_nan = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto result = tensor_close(ten_0,ten_1,relative_tolerance,absolute_tolerance,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_close_default_precision","test_tensor_operators")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type e = std::numeric_limits<value_type>::epsilon();
    //0ten_0,1ten_1,2equal_nan,3expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),false,true),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),false,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},false,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},false,false),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{-1.0-e,0.0,0.0,1.0+e,nan},false,false),
        std::make_tuple(tensor_type{{-1.0,0.0,0.0+e},{1.0+e,2.1,3.3}},tensor_type{{-1.0-e,0.0},{0.0,1.0+e},{2.1,3.3}},false,false),
        //nan_equal true
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),true,true),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),true,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},true,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},true,false),
        std::make_tuple(tensor_type{nan,-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{nan,-1.0-e,0.0,0.0,1.0+e,nan},true,true),
        std::make_tuple(tensor_type{{nan,-1.0,0.0},{0.0+e,1.0+e,nan}},tensor_type{{nan,-1.0-e},{0.0,0.0},{1.0+e,nan}},true,false)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto equal_nan = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = tensor_close(ten_0,ten_1,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_close_different_config_traverse_order","[test_tensor_operators]")
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using gtensor::tensor_close;
    using c_config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<c_order>,value_type>;
    using f_config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<f_order>,value_type>;

    REQUIRE(tensor_close(tensor<value_type,c_order,c_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor<value_type,c_order,f_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}));
    REQUIRE(tensor_close(tensor<value_type,c_order,f_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor<value_type,c_order,c_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}));
}

TEST_CASE("test_allclose","test_tensor_operators")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::allclose;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type inf = std::numeric_limits<value_type>::infinity();
    //0ten_0,1ten_1,2relative_tolerance,3absolute_tolerance,4equal_nan,5expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{},tensor_type{},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2+1E-11,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{{1.1},{2.2},{3.3},{inf}},tensor_type{{1.1},{2.2},{3.3},{inf}},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-9),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2-1E-9,3.3},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,nan,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,false,false),
        //nan_equal true
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{},tensor_type{},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2+1E-11,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{{1.1},{2.2},{3.3},{inf}},tensor_type{{1.1},{2.2},{3.3},{inf}},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-9),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2-1E-9,3.3},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,nan,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,true,true),
        //vary tolerance
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-12,1E-12,false,false),
        //near zero: absolute tolerance plays
        std::make_tuple(tensor_type(1E-15),tensor_type(1E-15+1E-30),1E-16,1E-10,false,true),
        std::make_tuple(tensor_type(1E-15),tensor_type(1E-15+1E-30),1E-16,1E-32,false,false),
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-14,1E-10,false,true),
        //big: relative tolerance plays
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-15,1E-10,false,true),
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-16,1E-10,false,false),
        //not floating point type
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},1E-10,1E-10,false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},1E-10,1E-10,true,false),
        //mixed types
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},1E-10,1E-10,true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,false),
        std::make_tuple(tensor<double>{1.0,2.0,3.0,4.0,5.0},tensor<float>{1.0,2.0+1E-11,3.0,4.0,5.0},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},1E-10,1E-10,false,false),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},1E-10,1E-10,true,true),
        //broadcast shape
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2+1E-11,3.3}},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2,3.3}},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2-1E-11,3.3},{1.1,2.2+1E-11,3.3}},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,nan},tensor_type{{1.1,2.2-1E-11,nan},{1.1,2.2+1E-11,nan}},1E-10,1E-10,true,true)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto relative_tolerance = std::get<2>(t);
        auto absolute_tolerance = std::get<3>(t);
        auto equal_nan = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto result = allclose(ten_0,ten_1,relative_tolerance,absolute_tolerance,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_allclose_default_precision","test_tensor_operators")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::allclose;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type e = std::numeric_limits<value_type>::epsilon();
    //0ten_0,1ten_1,2equal_nan,3expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),false,true),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),false,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},false,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},false,false),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{-1.0-e,0.0,0.0,1.0+e,nan},false,false),
        std::make_tuple(tensor_type{{-1.0,0.0,0.0+e,1.0+e,2.1,3.3},{-1.0,0.0,0.0+e,1.0+e,2.1,3.3}},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1,3.3},false,true),
        //nan_equal true
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),true,true),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),true,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},true,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},true,false),
        std::make_tuple(tensor_type{nan,-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{nan,-1.0-e,0.0,0.0,1.0+e,nan},true,true),
        std::make_tuple(tensor_type{{0.0+e,1.0+e,nan},{0.0+e,1.0+e,nan}},tensor_type{{0.0+e,1.0+e,nan}},true,true)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto equal_nan = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = allclose(ten_0,ten_1,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//test tensor fuzzy elementwise equality
TEST_CASE("test_isclose","test_tensor_operators")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::isclose;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type inf = std::numeric_limits<value_type>::infinity();
    //0ten_0,1ten_1,2relative_tolerance,3absolute_tolerance,4equal_nan,5expected
    auto test_data = std::make_tuple(
        //nan_equal false
        //near zero
        std::make_tuple(tensor_type(0.0), tensor_type(0.0), 1E-10, 1E-10, std::false_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-11), 1E-10, 1E-10, std::false_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-21), 1E-10, 1E-10, std::false_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-9), 1E-10, 1E-10, std::false_type{}, bool_tensor_type(false)),
        std::make_tuple(tensor_type(0.0), tensor_type{0.0+1E-11,0.0-1E-11,0.0+1E-8}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false}),
        //near one
        std::make_tuple(tensor_type(1.1), tensor_type{1.1+1E-11,1.1-1E-11,1.1+1E-8}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false}),
        //near big
        std::make_tuple(tensor_type(4E15), tensor_type{4E15+1.0,4E15-1.0,4E15+10.0}, 1E-15, 1E-16, std::false_type{}, bool_tensor_type{true,true,false}),
        //nans
        std::make_tuple(tensor_type{inf,2.2,nan,3.3}, tensor_type{inf,2.2,nan,3.3}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor_type{inf,2.2,nan,3.3}, tensor_type{inf,2.2,1.1,3.3}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor_type{inf,2.2,1.1,3.3}, tensor_type{inf,2.2,nan,3.3}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false,true}),
        //nan_equal true
        //near zero
        std::make_tuple(tensor_type(0.0), tensor_type(0.0), 1E-10, 1E-10, std::true_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-11), 1E-10, 1E-10, std::true_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-21), 1E-10, 1E-10, std::true_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-9), 1E-10, 1E-10, std::true_type{}, bool_tensor_type(false)),
        std::make_tuple(tensor_type(0.0), tensor_type{0.0+1E-11,0.0-1E-11,0.0+1E-8}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,false}),
        //near one
        std::make_tuple(tensor_type(1.1), tensor_type{1.1+1E-11,1.1-1E-11,1.1+1E-8}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,false}),
        //near big
        std::make_tuple(tensor_type(4E15), tensor_type{4E15+1.0,4E15-1.0,4E15+10.0}, 1E-15, 1E-16, std::true_type{}, bool_tensor_type{true,true,false}),
        //nans
        std::make_tuple(tensor_type{inf,2.2,nan,3.3}, tensor_type{inf,2.2,nan,3.3}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor_type{inf,2.2,nan,3.3}, tensor_type{inf,2.2,1.1,3.3}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor_type{inf,2.2,1.1,3.3}, tensor_type{inf,2.2,nan,3.3}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,false,true}),
        //not floating point type
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<int>{1,2,3,4}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<int>{1,2,3,3}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,true,false}),
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<int>{1,2,3,4}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<int>{1,2,3,3}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,true,false}),
        //mixed types
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<double>{1.0,2.0,3.0,4.0}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0}, tensor<int>{1,2,3,4}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0},tensor<int>{1,2,3,4},1E-10,1E-10, std::false_type{}, bool_tensor_type{true,false,true,true}),
        std::make_tuple(tensor<int>{1,2,3,4},tensor<double>{1.0,2.0,3.0,4.0},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0},tensor<int>{1,2,3,4},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0},tensor<int>{1,2,3,4},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,false,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0},tensor<int>{1,2,3,4},1E-10,1E-10, std::false_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0},tensor<int>{1,2,3,4},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor<double>{1.0,2.0,3.0,4.0},tensor<float>{1.0,2.0+1E-11,3.0,4.0},1E-10,1E-10, std::false_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0},tensor<float>{1.0,nan,3.0,4.0},1E-10,1E-10, std::false_type{}, bool_tensor_type{true,false,true,true}),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0},tensor<float>{1.0,nan,3.0,4.0},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,true,true,true})
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto relative_tolerance = std::get<2>(t);
        auto absolute_tolerance = std::get<3>(t);
        auto equal_nan = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto result = isclose(ten_0,ten_1,relative_tolerance,absolute_tolerance,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_isclose_default_precision","test_tensor_operators")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::isclose;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type e = std::numeric_limits<value_type>::epsilon();
    //0ten_0,1ten_1,2equal_nan,3expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),std::false_type{},bool_tensor_type(true)),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),std::false_type{},bool_tensor_type(true)),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),std::false_type{},bool_tensor_type(true)),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},std::false_type{},bool_tensor_type{true,true,true,true,true}),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},std::false_type{},bool_tensor_type{true,true,true,true,false}),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{-1.0-e,0.0,0.0,1.0+e,nan},std::false_type{},bool_tensor_type{true,true,true,true,false}),
        std::make_tuple(
            tensor_type{{-1.0,0.0,0.0+e,1.0+e,2.1,3.3},{-1.0,0.0,0.0+e,1.0+e,2.1,3.3}},
            tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1,3.3},
            std::false_type{},
            bool_tensor_type{{true,true,true,true,true,true},{true,true,true,true,true,true}}
        ),
        //nan_equal true
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),std::true_type{},bool_tensor_type(true)),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),std::true_type{},bool_tensor_type(true)),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),std::true_type{},bool_tensor_type(true)),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},std::true_type{},bool_tensor_type{true,true,true,true,true}),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},std::true_type{},bool_tensor_type{true,true,true,true,false}),
        std::make_tuple(tensor_type{nan,-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{nan,-1.0-e,0.0,0.0,1.0+e,nan},std::true_type{},bool_tensor_type{true,true,true,true,true,true}),
        std::make_tuple(tensor_type{{0.0+e,1.0-e,nan},{0.0+e,1.0+e,nan}},tensor_type{{0.0,1.0,nan}},std::true_type{},bool_tensor_type{{true,true,true},{true,true,true}}),
        std::make_tuple(tensor_type{{0.0+e,1.0+e,nan},{0.0+e,1.0+e,nan}},tensor_type{{nan,1.0+e,nan}},std::true_type{},bool_tensor_type{{false,true,true},{false,true,true}}),
        std::make_tuple(tensor_type{{nan,1.0+e,nan},{0.0+e,1.0+e,nan}},tensor_type{{0.0+e,1.0+e,nan}},std::true_type{},bool_tensor_type{{false,true,true},{true,true,true}})
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto equal_nan = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = isclose(ten_0,ten_1,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_allclose_different_config_traverse_order","[test_tensor_operators]")
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using gtensor::allclose;
    using c_config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<c_order>,value_type>;
    using f_config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<f_order>,value_type>;

    REQUIRE(allclose(tensor<value_type,c_order,c_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor<value_type,c_order,f_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}));
    REQUIRE(allclose(tensor<value_type,c_order,f_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor<value_type,c_order,c_config_type>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}));
    REQUIRE(allclose(tensor<value_type,c_order,c_config_type>{{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}},tensor<value_type,c_order,f_config_type>{{1,2,3},{4,5,6}}));
    REQUIRE(allclose(tensor<value_type,c_order,f_config_type>{{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}}},tensor<value_type,c_order,c_config_type>{{1,2,3},{4,5,6}}));
}

//test cast
TEST_CASE("test_cast","test_tensor_operators")
{
    using gtensor::tensor;
    using gtensor::cast;
    using helpers_for_testing::apply_by_element;
    //0ten,1cast_type_value,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor<int>(1),double{},tensor<double>(1)),
        std::make_tuple(tensor<int>{1,2,3,4,5},double{},tensor<double>{1,2,3,4,5}),
        std::make_tuple(tensor<double>{1.1,2.2,3.3,4.4,5.5},std::size_t{},tensor<std::size_t>{1,2,3,4,5}),
        std::make_tuple(tensor<double>{-1.1,-2.2,3.3,4.4,5.5},std::int64_t{},tensor<std::int64_t>{-1,-2,3,4,5})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto cast_type_value = std::get<1>(t);
        using cast_type = decltype(cast_type_value);
        auto expected = std::get<2>(t);
        auto result = cast<cast_type>(ten);
        using result_value_type = typename decltype(result)::value_type;
        using expected_value_type = typename decltype(expected)::value_type;
        REQUIRE(std::is_same_v<result_value_type,expected_value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//test where
TEST_CASE("test_where","test_tensor_operators")
{
    using gtensor::tensor;
    using gtensor::where;
    using helpers_for_testing::apply_by_element;
    //0condition,1if_true_tensor,2if_false_tensor,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor<bool>(true), tensor<double>{1,2,3,4,5} , tensor<double>(-1), tensor<double>{1,2,3,4,5}),
        std::make_tuple(tensor<bool>(false), tensor<double>{1,2,3,4,5} , tensor<double>(-1), tensor<double>{-1,-1,-1,-1,-1}),
        std::make_tuple(tensor<bool>{false,false,true,true,false}, tensor<int>{1,2,3,4,5} , std::size_t{0}, tensor<std::size_t>{0,0,3,4,0}),
        std::make_tuple(tensor<bool>{false,false,true,true,false}, 1, -1, tensor<int>{-1,-1,1,1,-1})
    );
    auto test = [](const auto& t){
        auto condition = std::get<0>(t);
        auto if_true_tensor = std::get<1>(t);
        auto if_false_tensor = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = where(condition,if_true_tensor,if_false_tensor);
        using result_value_type = typename decltype(result)::value_type;
        using expected_value_type = typename decltype(expected)::value_type;
        REQUIRE(std::is_same_v<result_value_type,expected_value_type>);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//test operators semantic
TEST_CASE("test_tensor_arithmetic_operators_semantic","[test_tensor_operators]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    SECTION("unary+")
    {
        REQUIRE(+tensor_type{1,2,3} == tensor_type{1,2,3});
    }
    SECTION("unary-")
    {
        REQUIRE(-tensor_type{1,2,3} == tensor_type{-1,-2,-3});
    }
    SECTION("operator+")
    {
        REQUIRE(tensor_type{1,2,3}+tensor_type{4,5,6} == tensor_type{5,7,9});
    }
    SECTION("operator-")
    {
        REQUIRE(tensor_type{1,2,3}-tensor_type{4,5,6} == tensor_type{-3,-3,-3});
    }
    SECTION("operator*")
    {
        REQUIRE(tensor_type{1,2,3}*tensor_type{4,5,6} == tensor_type{4,10,18});
    }
    SECTION("operator/")
    {
        REQUIRE(tensor_type{4,5,6,7}/tensor_type{1,2,3,4} == tensor_type{4,2,2,1});
    }
    SECTION("operator%")
    {
        REQUIRE(tensor_type{4,5,6,7}%tensor_type{1,2,3,4} == tensor_type{0,1,0,3});
    }
}

TEST_CASE("test_tensor_bitwise_operators_semantic","[test_tensor_operators]")
{
    using value_type = std::uint32_t;
    using tensor_type = gtensor::tensor<value_type>;
    SECTION("operator~")
    {
        REQUIRE(~tensor_type{1,2,4} == tensor_type{4294967294,4294967293,4294967291});
    }
    SECTION("operator&")
    {
        REQUIRE((tensor_type{1,2,3}&tensor_type{4,5,6}) == tensor_type{0,0,2});
    }
    SECTION("operator|")
    {
        REQUIRE((tensor_type{1,2,3}|tensor_type{4,5,6}) == tensor_type{5,7,7});
    }
    SECTION("operator^")
    {
        REQUIRE((tensor_type{1,2,3}^tensor_type{4,5,6}) == tensor_type{5,7,5});
    }
    SECTION("operator<<")
    {
        REQUIRE((tensor_type{1,2,3}<<tensor_type{4,5,6}) == tensor_type{16,64,192});
    }
    SECTION("operator>>")
    {
        REQUIRE((tensor_type{4,5,6}>>tensor_type{1,2,3}) == tensor_type{2,1,0});
    }
}

TEST_CASE("test_tensor_strict_comparison_operators_semantic","[test_tensor_operators]")
{
    using value_type = int;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    SECTION("equal")
    {
        REQUIRE(gtensor::equal(tensor_type{1,2,3,4},tensor_type{4,2,3,1}) == tensor<bool>{false,true,true,false});
    }
    SECTION("not_equal")
    {
        REQUIRE(gtensor::not_equal(tensor_type{1,2,3,4},tensor_type{4,2,3,1}) == tensor<bool>{true,false,false,true});
    }
    SECTION("operator>")
    {
        REQUIRE((tensor_type{1,2,3,4}>tensor_type{4,2,3,1}) == tensor<bool>{false,false,false,true});
    }
    SECTION("operator>=")
    {
        REQUIRE((tensor_type{1,2,3,4}>=tensor_type{4,2,3,1}) == tensor<bool>{false,true,true,true});
    }
    SECTION("operator<")
    {
        REQUIRE((tensor_type{1,2,3,4}<tensor_type{4,2,3,1}) == tensor<bool>{true,false,false,false});
    }
    SECTION("operator<=")
    {
        REQUIRE((tensor_type{1,2,3,4}<=tensor_type{4,2,3,1}) == tensor<bool>{true,true,true,false});
    }
}

TEST_CASE("test_tensor_logical_operators_semantic","[test_tensor_operators]")
{
    using value_type = bool;
    using tensor_type = gtensor::tensor<value_type>;
    SECTION("operator!")
    {
        REQUIRE((!tensor_type{true,false,true,false}) == tensor_type{false,true,false,true});
    }
    SECTION("operator&&")
    {
        REQUIRE((tensor_type{true,false,true,false}&&tensor_type{true,false,false,false}) == tensor_type{true,false,false,false});
    }
    SECTION("operator||")
    {
        REQUIRE((tensor_type{true,false,true,false}||tensor_type{true,false,false,false}) == tensor_type{true,false,true,false});
    }
}

TEST_CASE("test_tensor_assign_operators_semantic","[test_tensor_operators]")
{
    using value_type = std::uint32_t;
    using tensor_type = gtensor::tensor<value_type>;
    tensor_type lhs{4,5,6};
    SECTION("assign")
    {
        REQUIRE(gtensor::assign(lhs,tensor_type{1,2,3}) == tensor_type{1,2,3});
    }
    SECTION("operator+=")
    {
        REQUIRE((lhs+=tensor_type{1,2,3}) == tensor_type{5,7,9});
    }
    SECTION("operator-=")
    {
        REQUIRE((lhs-=tensor_type{1,2,3}) == tensor_type{3,3,3});
    }
    SECTION("operator*=")
    {
        REQUIRE((lhs*=tensor_type{1,2,3}) == tensor_type{4,10,18});
    }
    SECTION("operator/=")
    {
        REQUIRE((lhs/=tensor_type{1,2,3}) == tensor_type{4,2,2});
    }
    SECTION("operator%=")
    {
        REQUIRE((lhs%=tensor_type{1,2,3}) == tensor_type{0,1,0});
    }
    SECTION("operator&=")
    {
        REQUIRE((lhs&=tensor_type{1,2,3}) == tensor_type{0,0,2});
    }
    SECTION("operator|=")
    {
        REQUIRE((lhs|=tensor_type{1,2,3}) == tensor_type{5,7,7});
    }
    SECTION("operator^=")
    {
        REQUIRE((lhs^=tensor_type{1,2,3}) == tensor_type{5,7,5});
    }
    SECTION("operator<<=")
    {
        REQUIRE((lhs<<=tensor_type{1,2,3}) == tensor_type{8,20,48});
    }
    SECTION("operator>>=")
    {
        REQUIRE((lhs>>=tensor_type{1,2,3}) == tensor_type{2,1,0});
    }
}
