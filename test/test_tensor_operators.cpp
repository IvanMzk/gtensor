#include "catch.hpp"
#include "helpers_for_testing.hpp"
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
        std::make_tuple(ternary_add_mul{},std::make_tuple(tensor_type(1),tensor_type{1,2,3},tensor_type{{1},{2},{3}}),tensor_type{{2,3,4},{4,6,8},{6,9,12}})
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
        std::make_tuple(tensor_type{{3},{4}},tensor_type{1,2},tensor_type{{4,5},{5,6}})
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
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_compaund_assign_operator","[test_tensor_operators]")
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
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_gtensor_assign_operator_from_rvalue","[test_tensor_operators]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::assign;
    tensor_type lhs{1,2,3};
    tensor_type rhs{4,5,6};
    REQUIRE(!rhs.empty());
    assign(lhs,std::move(rhs));
    REQUIRE(rhs.empty());
    REQUIRE(lhs == tensor_type{4,5,6});
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

TEST_CASE("test_tensor_comparison_operators_semantic","[test_tensor_operators]")
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
        REQUIRE(!tensor_type{true,false,true,false} == tensor_type{false,true,false,true});
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