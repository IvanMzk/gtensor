/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <tuple>
#include <iostream>
#include <complex>
#include "catch.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"
#include "statistic.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_tensor_of_tensor_construction","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;

    //shape
    REQUIRE(tensor_type_1(std::vector<int>{2,3}) == tensor_type_1{{tensor_type_0{},tensor_type_0{},tensor_type_0{}},{tensor_type_0{},tensor_type_0{},tensor_type_0{}}});
    REQUIRE(tensor_type_1{std::vector<int>{2,3}} == tensor_type_1{{tensor_type_0{},tensor_type_0{},tensor_type_0{}},{tensor_type_0{},tensor_type_0{},tensor_type_0{}}});
    REQUIRE(tensor_type_2(std::vector<int>{2,3}) == tensor_type_2{{tensor_type_1{},tensor_type_1{},tensor_type_1{}},{tensor_type_1{},tensor_type_1{},tensor_type_1{}}});
    REQUIRE(tensor_type_2{std::vector<int>{2,3}} == tensor_type_2{{tensor_type_1{},tensor_type_1{},tensor_type_1{}},{tensor_type_1{},tensor_type_1{},tensor_type_1{}}});

    //default
    REQUIRE(tensor_type_1() == tensor_type_1(std::vector<int>{0}));
    REQUIRE(tensor_type_1{} == tensor_type_1(std::vector<int>{0}));
    REQUIRE(tensor_type_2() == tensor_type_2(std::vector<int>{0}));
    REQUIRE(tensor_type_2{} == tensor_type_2(std::vector<int>{0}));

    //0d
    REQUIRE(tensor_type_1(2) == tensor_type_1(tensor_type_0(2)));
    REQUIRE(tensor_type_1{2} == tensor_type_1(tensor_type_0(2)));
    REQUIRE(tensor_type_2(3) == tensor_type_2(tensor_type_1(tensor_type_0(3))));
    REQUIRE(tensor_type_2{3} == tensor_type_2(tensor_type_1(tensor_type_0(3))));

    //shape and value
    REQUIRE(tensor_type_1(std::vector<int>{2,3},1.1) == tensor_type_1{{tensor_type_0(1.1),tensor_type_0(1.1),tensor_type_0(1.1)},{tensor_type_0(1.1),tensor_type_0(1.1),tensor_type_0(1.1)}});
    REQUIRE(tensor_type_1{std::vector<int>{2,3},2} == tensor_type_1{{tensor_type_0(2),tensor_type_0(2),tensor_type_0(2)},{tensor_type_0(2),tensor_type_0(2),tensor_type_0(2)}});
    REQUIRE(tensor_type_2(std::vector<int>{2,3},3) == tensor_type_2{{tensor_type_1(3),tensor_type_1(3),tensor_type_1(3)},{tensor_type_1(3),tensor_type_1(3),tensor_type_1(3)}});
    REQUIRE(tensor_type_2{std::vector<int>{2,3},3} == tensor_type_2{{tensor_type_1(3),tensor_type_1(3),tensor_type_1(3)},{tensor_type_1(3),tensor_type_1(3),tensor_type_1(3)}});
}

TEST_CASE("test_tensor_of_tensor_strict_equality","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{3,2,1};
    const auto t2 = tensor_type_0{0,2,1};
    const auto t3 = tensor_type_0{1,2,0};
    const auto t4 = tensor_type_0{2,2,3};
    const auto t5 = tensor_type_0{1,3,0};

    const auto a = tensor_type_1{{t0,t1},{t2,t3},{t4,t5}};
    const auto b = tensor_type_1{{t0,t1},{t1,t2},{t3,t4}};
    const auto c = tensor_type_1{t3,t4};
    const auto d = tensor_type_1{t4,t3};

    const auto x = tensor_type_2{{c,d},{c,d}};
    const auto z = tensor_type_2{d,c};
    const auto y = tensor_type_2{{d,c},{d,c}};
    const auto u = tensor_type_2{c,d};

    REQUIRE(a==a);
    REQUIRE(a==a.copy());
    REQUIRE(a!=b);
    REQUIRE(a!=c);
    REQUIRE(b!=c);
    REQUIRE(a+b+c==c+b+a);
    REQUIRE(a+b+c!=a+b+d);

    REQUIRE(x==x.copy());
    REQUIRE(x!=y);
    REQUIRE(y!=z);
    REQUIRE(x!=z);
    REQUIRE(x+z==y+u);

    REQUIRE(tensor_equal(x+z,y+u));
}

TEST_CASE("test_tensor_of_tensor_strict_equality_nan_equal","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;
    static constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    const auto t0 = tensor_type_0{1.1,2.2,nan,3.3};
    const auto t1 = tensor_type_0{1.1,2.2,nan,3.3};
    const auto t2 = tensor_type_0{1.1,2.2,3.3,nan};

    const auto a = tensor_type_1{{t0,t0},{t1,t1}};
    const auto b = tensor_type_1{{t0,t1},{t1,t1}};
    const auto c = tensor_type_1{{t0,t2},{t1,t1}};

    const auto x = tensor_type_2{{a,a},{b,b}};
    const auto y = tensor_type_2{{a,b},{b,b}};
    const auto z = tensor_type_2{{a,c},{b,b}};

    REQUIRE(!tensor_equal(a,b));
    REQUIRE(tensor_equal(a,b,true));
    REQUIRE(!tensor_equal(a,c));
    REQUIRE(!tensor_equal(a,c,true));

    REQUIRE(!tensor_equal(x,y));
    REQUIRE(tensor_equal(x,y,true));
    REQUIRE(!tensor_equal(x,z));
    REQUIRE(!tensor_equal(x,z,true));
}

TEST_CASE("test_tensor_of_tensor_close_equality","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;

    const auto t0 = tensor_type_0{1.12345,2.12345,3.12345};
    const auto t1 = tensor_type_0{1.12545,2.12345,3.12345};

    const auto a = tensor_type_1{{t0,t0},{t1,t1}};
    const auto b = tensor_type_1{{t0,t1},{t1,t1}};

    const auto x = tensor_type_2{{a,a},{b,b}};
    const auto y = tensor_type_2{{a,b},{b,b}};

    REQUIRE(!tensor_close(a,b));
    REQUIRE(tensor_close(a,b,1E-3,1E-3));
    REQUIRE(!tensor_close(x,y));
    REQUIRE(tensor_close(x,y,1E-3,1E-3));
}

TEST_CASE("test_tensor_of_tensor_close_equality_nan_equal","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;
    static constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    const auto t0 = tensor_type_0{1.12345,2.12345,nan,3.12345};
    const auto t1 = tensor_type_0{1.12545,2.12345,nan,3.12345};
    const auto t2 = tensor_type_0{1.12545,2.12345,3.12345,nan};

    const auto a = tensor_type_1{{t0,t0},{t1,t1}};
    const auto b = tensor_type_1{{t0,t1},{t1,t1}};
    const auto c = tensor_type_1{{t0,t2},{t1,t1}};

    const auto x = tensor_type_2{{a,a},{b,b}};
    const auto y = tensor_type_2{{a,b},{b,b}};
    const auto z = tensor_type_2{{a,c},{b,b}};

    REQUIRE(!tensor_close(a,b));
    REQUIRE(!tensor_close(a,b,1E-3,1E-3));
    REQUIRE(!tensor_close(a,b,1E-6,1E-6,true));
    REQUIRE(tensor_close(a,b,1E-3,1E-3,true));
    REQUIRE(!tensor_close(a,c,1E-3,1E-3,true));

    REQUIRE(!tensor_close(x,y));
    REQUIRE(!tensor_close(x,y,1E-3,1E-3));
    REQUIRE(!tensor_close(x,y,1E-6,1E-6,true));
    REQUIRE(tensor_close(x,y,1E-3,1E-3,true));
    REQUIRE(!tensor_close(x,z,1E-3,1E-3,true));
}

TEST_CASE("test_tensor_of_tensor_expressions","[test_tensor_of_tensor]")
{

    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,1};


    auto a = tensor_type_1{{t0,t1},{t1,t2}};
    auto b = tensor_type_1{{t2,t1},{t1,t0}};
    auto c = tensor_type_1{t2,t1};
    auto d = tensor_type_1{t0,t2};

    auto x = tensor_type_2{{a,b},{b,a}};
    auto y = tensor_type_2{c,d};

    REQUIRE(2.2+a+c-1 == tensor_type_1{{2.2+t0+t2-1,2.2+t1+t1-1},{2.2+t1+t2-1,2.2+t2+t1-1}});
    REQUIRE((a+b)*(c-1) == tensor_type_1{{(t0+t2)*(t2-1),(t1+t1)*(t1-1)},{(t1+t1)*(t2-1),(t2+t0)*(t1-1)}});

    REQUIRE(x+y == tensor_type_2{{a+c,b+d},{b+c,a+d}});
}

TEST_CASE("test_tensor_of_tensor_assign_1","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,1};

    const auto a = tensor_type_1{{t0,t1},{t1,t2}};
    const auto b = tensor_type_1{t2,t1};

    SECTION("value_assign_tensor")
    {
        tensor_type_1 c{};
        const auto a_copy = a.copy();
        c = a;
        REQUIRE(c==a);
        c+=1;
        REQUIRE(c!=a);
        REQUIRE(a==a_copy);
    }
    SECTION("value_assign_scalar")
    {
        auto c=a.copy();
        c = 1.12;
        REQUIRE(c==tensor_type_1{1.12});
    }
    SECTION("value_assign_scalar_to_unit_size")
    {
        auto c=tensor_type_1{0.1};
        c = 1.12;
        REQUIRE(c==tensor_type_1{1.12});
    }
    SECTION("broadcast_assign_tensor")
    {
        auto c = a.copy();
        c.assign(b);
        REQUIRE(c == tensor_type_1{{t2,t1},{t2,t1}});
    }
    SECTION("broadcast_assign_scalar")
    {
        auto c = a.copy();
        c.assign(1.12);
        REQUIRE(c == tensor_type_1{{tensor_type_0(3,1.12),tensor_type_0(3,1.12)},{tensor_type_0(3,1.12),tensor_type_0(3,1.12)}});
    }
    SECTION("broadcast_assign_plus_tensor")
    {
        auto c = a.copy();
        c+=b;
        REQUIRE(c == tensor_type_1{{t0+t2,t1+t1},{t1+t2,t2+t1}});
    }
    SECTION("broadcast_assign_plus_scalar")
    {
        auto c = a.copy();
        c+=1.1;
        REQUIRE(c == tensor_type_1{{t0+1.1,t1+1.1},{t1+1.1,t2+1.1}});
    }
    SECTION("broadcast_assign_view_tensor")
    {
        auto c = a.copy();
        c.transpose()=b;
        REQUIRE(c == tensor_type_1{{t2,t2},{t1,t1}});
    }
    SECTION("broadcast_assign_view_scalar")
    {
        auto c = a.copy();
        c({{},{1}}) = 1.12;
        REQUIRE(c == tensor_type_1{{t0,tensor_type_0(3,1.12)},{t1,tensor_type_0(3,1.12)}});
    }
    SECTION("broadcast_assign_plus_view_tensor")
    {
        auto c = a.copy();
        c(tensor<bool>{{true,false},{true,true}}) += tensor_type_1{t0,t1,t2};
        REQUIRE(c == tensor_type_1{{(t0+t0).copy(),t1},{(t1+t1).copy(),(t2+t2).copy()}});
    }
}

TEST_CASE("test_tensor_of_tensor_assign_2","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,1};

    const auto a = tensor_type_1{{t0,t1},{t1,t2}};
    const auto b = tensor_type_1{{t2,t1},{t0,t1}};

    const auto x = tensor_type_2{{a,b},{b,a}};
    const auto y = tensor_type_2{b,a};

    SECTION("value_assign_tensor")
    {
        tensor_type_2 z{};
        const auto x_copy = x.copy();
        z = x;
        REQUIRE(z==x);
        z+=1;
        REQUIRE(z!=x);
        REQUIRE(x==x_copy);
    }
    SECTION("value_assign_scalar")
    {
        auto z=x.copy();
        z = 1.12;
        REQUIRE(z==tensor_type_2{1.12});
    }
    SECTION("value_assign_scalar_to_unit_size")
    {
        auto z=tensor_type_2{0.1};
        z = 1.12;
        REQUIRE(z==tensor_type_2{1.12});
    }
    SECTION("broadcast_assign_tensor")
    {
        auto z = x.copy();
        z.assign(y);
        REQUIRE(z == tensor_type_2{{b,a},{b,a}});
    }
    SECTION("broadcast_assign_scalar")
    {
        auto z = x.copy();
        z.assign(1.12);
        const auto t = tensor_type_0(3,1.12);
        const auto d = tensor_type_1{{t,t},{t,t}};
        REQUIRE(z == tensor_type_2{{d,d},{d,d}});
    }
    SECTION("broadcast_assign_plus_tensor")
    {
        auto z = x.copy();
        z+=y;
        REQUIRE(z == tensor_type_2{{a+b,b+a},{b+b,a+a}});
    }
    SECTION("broadcast_assign_plus_scalar")
    {
        auto z = x.copy();
        z+=1.1;
        REQUIRE(z == tensor_type_2{{a+1.1,b+1.1},{b+1.1,a+1.1}});
    }
    SECTION("broadcast_assign_view_tensor")
    {
        auto z = x.copy();
        z.transpose()=y;
        REQUIRE(z == tensor_type_2{{b,b},{a,a}});
    }
    SECTION("broadcast_assign_view_scalar")
    {
        auto z = x.copy();
        z({{},{1}}) = 1.12;
        const auto t = tensor_type_0(3,1.12);
        const auto d = tensor_type_1{{t,t},{t,t}};
        REQUIRE(z == tensor_type_2{{a,d},{b,d}});
    }
    SECTION("broadcast_assign_plus_view_tensor")
    {
        auto z = x.copy();
        z(tensor<bool>{{true,false},{true,true}}) += tensor_type_2{a,b,(a+b).copy()};
        REQUIRE(z == tensor_type_2{{(a+a).copy(),b},{(b+b).copy(),(a+a+b).copy()}});
    }
}

TEST_CASE("test_tensor_of_tensor_broadcast_routines","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;

    const auto t0 = tensor_type_0{1.1,2.2,3.3};
    const auto t1 = tensor_type_0{2.2,0.1,1.1};
    const auto t2 = tensor_type_0{2.1,1.1,1.1};

    const auto a = tensor_type_1{{t0,t1},{t1,t2}};
    const auto b = tensor_type_1{{t2,t1},{t1,t0}};
    const auto c = tensor_type_1{t2,t1};

    const auto x = tensor_type_2{{a,b},{b,a}};
    const auto y = tensor_type_2{b,a};

    REQUIRE(pow(a,b) == tensor_type_1{{pow(t0,t2),pow(t1,t1)},{pow(t1,t1),pow(t2,t0)}});
    REQUIRE(pow(a,c) == tensor_type_1{{pow(t0,t2),pow(t1,t1)},{pow(t1,t2),pow(t2,t1)}});
    REQUIRE(sin(a) == tensor_type_1{{sin(t0),sin(t1)},{sin(t1),sin(t2)}});
    REQUIRE(sqrt(b+c) == tensor_type_1{{sqrt(t2+t2),sqrt(t1+t1)},{sqrt(t1+t2),sqrt(t0+t1)}});

    REQUIRE(pow(x,y) == tensor_type_2{{pow(a,b),pow(b,a)},{pow(b,b),pow(a,a)}});
    REQUIRE(sin(x+y) == tensor_type_2{{sin(a+b),sin(b+a)},{sin(b+b),sin(a+a)}});

    auto ai = gtensor::cast<int>(a);
    REQUIRE(std::is_same_v<typename decltype(ai)::element_type,int>);
    REQUIRE(ai == tensor<tensor<int>>{{tensor<int>{1,2,3},tensor<int>{2,0,1}},{tensor<int>{2,0,1},tensor<int>{2,1,1}}});

    auto xi = gtensor::cast<int>(x);
    REQUIRE(std::is_same_v<typename decltype(xi)::element_type,int>);
    REQUIRE(xi == tensor<tensor<tensor<int>>>{{gtensor::cast<int>(a),gtensor::cast<int>(b)},{gtensor::cast<int>(b),gtensor::cast<int>(a)}});
}

TEST_CASE("test_tensor_of_tensor_broadcast_routines_std_complex","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<std::complex<double>>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using tensor_type_2 = tensor<tensor_type_1>;
    using namespace std::complex_literals;

    const auto t0 = tensor_type_0{{1.1+2.2i,2.2+1.1i},{3.2+0.1i,0.2+1.3i}};
    const auto t1 = tensor_type_0{{2.1+2.2i,1.2+1.1i},{1.2+0.1i,3.2+1.3i}};
    const auto a = tensor_type_1{{t0,t1},{t1,t0}};
    const auto b = tensor_type_1{{t1,t1},{t0,t1}};
    const auto c = tensor_type_1{t0,t1};
    const auto x = tensor_type_2{{a,b},{b,a}};
    const auto y = tensor_type_2{b,b};

    REQUIRE(conj(a) == tensor_type_1{{conj(t0),conj(t1)},{conj(t1),conj(t0)}});
    REQUIRE(real(a) == tensor_type_1{{real(t0),real(t1)},{real(t1),real(t0)}});
    REQUIRE(imag(a) == tensor_type_1{{imag(t0),imag(t1)},{imag(t1),imag(t0)}});
    REQUIRE(sin(a) == tensor_type_1{{sin(t0),sin(t1)},{sin(t1),sin(t0)}});
    REQUIRE(pow(a,conj(b)) == tensor_type_1{{pow(t0,conj(t1)),pow(t1,conj(t1))},{pow(t1,conj(t0)),pow(t0,conj(t1))}});
    REQUIRE(pow(a,conj(c)) == tensor_type_1{{pow(t0,conj(t0)),pow(t1,conj(t1))},{pow(t1,conj(t0)),pow(t0,conj(t1))}});
    REQUIRE(conj(x) == tensor_type_2{{conj(a),conj(b)},{conj(b),conj(a)}});
    REQUIRE(pow(x,conj(y)) == tensor_type_2{{pow(a,conj(b)),pow(b,conj(b))},{pow(b,conj(b)),pow(a,conj(b))}});
}

TEST_CASE("test_tensor_of_tensor_reduce_binary","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using gtensor::detail::no_value;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,2};
    const auto t3 = tensor_type_0{0,3,1};

    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};
    const auto b = tensor_type_1{{{t1,t3,t1},{t2,t1,t0}},{{t3,t2,t0},{t0,t2,t1}}};
    const auto c = tensor_type_1{t2,t1,t3};

    //ten,axes,intial,expected
    auto test_data = std::make_tuple(
        //input tensor
        //no initial
        std::make_tuple(a,no_value{},no_value{},tensor_type_1(tensor_type_0{14,20,23})),
        std::make_tuple(a,0,no_value{},tensor_type_1{{tensor_type_0{1,5,4},tensor_type_0{1,5,4},tensor_type_0{4,1,3}},{tensor_type_0{4,1,3},tensor_type_0{3,3,5},tensor_type_0{1,5,4}}}),
        std::make_tuple(a,1,no_value{},tensor_type_1{{tensor_type_0{3,2,4},tensor_type_0{2,4,3},tensor_type_0{3,3,5}},{tensor_type_0{2,4,3},tensor_type_0{2,4,6},tensor_type_0{2,3,2}}}),
        std::make_tuple(a,2,no_value{},tensor_type_1{{tensor_type_0{3,6,6},tensor_type_0{5,3,6}},{tensor_type_0{3,5,5},tensor_type_0{3,6,6}}}),
        std::make_tuple(a,std::vector<int>{0,1},no_value{},tensor_type_1{tensor_type_0{5,6,7},tensor_type_0{4,8,9},tensor_type_0{5,6,7}}),
        std::make_tuple(a,std::vector<int>{0,2},no_value{},tensor_type_1{tensor_type_0{6,11,11},tensor_type_0{8,9,12}}),
        std::make_tuple(a,std::vector<int>{1,2},no_value{},tensor_type_1{tensor_type_0{8,9,12},tensor_type_0{6,11,11}}),
        std::make_tuple(a,std::vector<int>{0,1,2},no_value{},tensor_type_1(tensor_type_0{14,20,23})),
        //initial
        std::make_tuple(a,no_value{},tensor_type_0{1,0,-1},tensor_type_1(tensor_type_0{15,20,22})),
        std::make_tuple(a,0,tensor_type_0{1,0,-1},tensor_type_1{{tensor_type_0{2,5,3},tensor_type_0{2,5,3},tensor_type_0{5,1,2}},{tensor_type_0{5,1,2},tensor_type_0{4,3,4},tensor_type_0{2,5,3}}}),
        std::make_tuple(a,1,tensor_type_0{1,0,-1},tensor_type_1{{tensor_type_0{4,2,3},tensor_type_0{3,4,2},tensor_type_0{4,3,4}},{tensor_type_0{3,4,2},tensor_type_0{3,4,5},tensor_type_0{3,3,1}}}),
        std::make_tuple(a,2,tensor_type_0{1,0,-1},tensor_type_1{{tensor_type_0{4,6,5},tensor_type_0{6,3,5}},{tensor_type_0{4,5,4},tensor_type_0{4,6,5}}}),
        std::make_tuple(a,std::vector<int>{0,1},tensor_type_0{1,0,-1},tensor_type_1{tensor_type_0{6,6,6},tensor_type_0{5,8,8},tensor_type_0{6,6,6}}),
        std::make_tuple(a,std::vector<int>{0,2},tensor_type_0{1,0,-1},tensor_type_1{tensor_type_0{7,11,10},tensor_type_0{9,9,11}}),
        std::make_tuple(a,std::vector<int>{1,2},tensor_type_0{1,0,-1},tensor_type_1{tensor_type_0{9,9,11},tensor_type_0{7,11,10}}),
        std::make_tuple(a,std::vector<int>{0,1,2},tensor_type_0{1,0,-1},tensor_type_1(tensor_type_0{15,20,22})),
        //input expression
        //no initial
        std::make_tuple(a+b+c,no_value{},no_value{},tensor_type_1(tensor_type_0{47,51,60})),
        std::make_tuple(a+b+c,0,no_value{},tensor_type_1{{tensor_type_0{7,10,10},tensor_type_0{7,9,9},tensor_type_0{7,9,9}},{tensor_type_0{11,6,12},tensor_type_0{11,4,10},tensor_type_0{4,13,10}}}),
        std::make_tuple(a+b+c,1,no_value{},tensor_type_1{{tensor_type_0{11,5,11},tensor_type_0{8,7,7},tensor_type_0{6,11,11}},{tensor_type_0{7,11,11},tensor_type_0{10,6,12},tensor_type_0{5,11,8}}}),
        std::make_tuple(a+b+c,2,no_value{},tensor_type_1{{tensor_type_0{11,13,13},tensor_type_0{14,10,16}},{tensor_type_0{10,15,15},tensor_type_0{12,13,16}}}),
        std::make_tuple(a+b+c,std::vector<int>{0,1},no_value{},tensor_type_1{tensor_type_0{18,16,22},tensor_type_0{18,13,19},tensor_type_0{11,22,19}}),
        std::make_tuple(a+b+c,std::vector<int>{0,2},no_value{},tensor_type_1{tensor_type_0{21,28,28},tensor_type_0{26,23,32}}),
        std::make_tuple(a+b+c,std::vector<int>{1,2},no_value{},tensor_type_1{tensor_type_0{25,23,29},tensor_type_0{22,28,31}}),
        std::make_tuple(a+b+c,std::vector<int>{0,1,2},no_value{},tensor_type_1(tensor_type_0{47,51,60})),
        //initial
        std::make_tuple(a+b+c,no_value{},tensor_type_0{1,0,-1},tensor_type_1(tensor_type_0{48,51,59})),
        std::make_tuple(a+b+c,0,tensor_type_0{1,0,-1},tensor_type_1{{tensor_type_0{8,10,9},tensor_type_0{8,9,8},tensor_type_0{8,9,8}},{tensor_type_0{12,6,11},tensor_type_0{12,4,9},tensor_type_0{5,13,9}}}),
        std::make_tuple(a+b+c,1,tensor_type_0{1,0,-1},tensor_type_1{{tensor_type_0{12,5,10},tensor_type_0{9,7,6},tensor_type_0{7,11,10}},{tensor_type_0{8,11,10},tensor_type_0{11,6,11},tensor_type_0{6,11,7}}}),
        std::make_tuple(a+b+c,2,tensor_type_0{1,0,-1},tensor_type_1{{tensor_type_0{12,13,12},tensor_type_0{15,10,15}},{tensor_type_0{11,15,14},tensor_type_0{13,13,15}}}),
        std::make_tuple(a+b+c,std::vector<int>{0,1},tensor_type_0{1,0,-1},tensor_type_1{tensor_type_0{19,16,21},tensor_type_0{19,13,18},tensor_type_0{12,22,18}}),
        std::make_tuple(a+b+c,std::vector<int>{0,2},tensor_type_0{1,0,-1},tensor_type_1{tensor_type_0{22,28,27},tensor_type_0{27,23,31}}),
        std::make_tuple(a+b+c,std::vector<int>{1,2},tensor_type_0{1,0,-1},tensor_type_1{tensor_type_0{26,23,28},tensor_type_0{23,28,30}}),
        std::make_tuple(a+b+c,std::vector<int>{0,1,2},tensor_type_0{1,0,-1},tensor_type_1(tensor_type_0{48,51,59}))
    );

    auto test_reduce_binary = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto initial = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto ten_copy = ten.copy();
            auto result = reduce_binary(policy...,ten,axes,std::plus<void>{},false,initial);
            REQUIRE(ten == ten_copy);
            REQUIRE(result == expected);
        };
        helpers_for_testing::apply_by_element(test,test_data);
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

TEST_CASE("test_tensor_of_tensor_reduce_range","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using gtensor::detail::no_value;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,2};
    const auto t3 = tensor_type_0{0,3,1};

    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};
    const auto b = tensor_type_1{{{t1,t3,t1},{t2,t1,t0}},{{t3,t2,t0},{t0,t2,t1}}};
    const auto c = tensor_type_1{t2,t1,t3};

    //ten,axes,expected
    auto test_data = std::make_tuple(
        //input tensor
        std::make_tuple(a,no_value{},tensor_type_1(tensor_type_0{14,20,23})),
        std::make_tuple(a,0,tensor_type_1{{tensor_type_0{1,5,4},tensor_type_0{1,5,4},tensor_type_0{4,1,3}},{tensor_type_0{4,1,3},tensor_type_0{3,3,5},tensor_type_0{1,5,4}}}),
        std::make_tuple(a,1,tensor_type_1{{tensor_type_0{3,2,4},tensor_type_0{2,4,3},tensor_type_0{3,3,5}},{tensor_type_0{2,4,3},tensor_type_0{2,4,6},tensor_type_0{2,3,2}}}),
        std::make_tuple(a,2,tensor_type_1{{tensor_type_0{3,6,6},tensor_type_0{5,3,6}},{tensor_type_0{3,5,5},tensor_type_0{3,6,6}}}),
        std::make_tuple(a,std::vector<int>{0,1},tensor_type_1{tensor_type_0{5,6,7},tensor_type_0{4,8,9},tensor_type_0{5,6,7}}),
        std::make_tuple(a,std::vector<int>{0,2},tensor_type_1{tensor_type_0{6,11,11},tensor_type_0{8,9,12}}),
        std::make_tuple(a,std::vector<int>{1,2},tensor_type_1{tensor_type_0{8,9,12},tensor_type_0{6,11,11}}),
        std::make_tuple(a,std::vector<int>{0,1,2},tensor_type_1(tensor_type_0{14,20,23})),
        //input expression
        std::make_tuple(a+b+c,no_value{},tensor_type_1(tensor_type_0{47,51,60})),
        std::make_tuple(a+b+c,0,tensor_type_1{{tensor_type_0{7,10,10},tensor_type_0{7,9,9},tensor_type_0{7,9,9}},{tensor_type_0{11,6,12},tensor_type_0{11,4,10},tensor_type_0{4,13,10}}}),
        std::make_tuple(a+b+c,1,tensor_type_1{{tensor_type_0{11,5,11},tensor_type_0{8,7,7},tensor_type_0{6,11,11}},{tensor_type_0{7,11,11},tensor_type_0{10,6,12},tensor_type_0{5,11,8}}}),
        std::make_tuple(a+b+c,2,tensor_type_1{{tensor_type_0{11,13,13},tensor_type_0{14,10,16}},{tensor_type_0{10,15,15},tensor_type_0{12,13,16}}}),
        std::make_tuple(a+b+c,std::vector<int>{0,1},tensor_type_1{tensor_type_0{18,16,22},tensor_type_0{18,13,19},tensor_type_0{11,22,19}}),
        std::make_tuple(a+b+c,std::vector<int>{0,2},tensor_type_1{tensor_type_0{21,28,28},tensor_type_0{26,23,32}}),
        std::make_tuple(a+b+c,std::vector<int>{1,2},tensor_type_1{tensor_type_0{25,23,29},tensor_type_0{22,28,31}}),
        std::make_tuple(a+b+c,std::vector<int>{0,1,2},tensor_type_1(tensor_type_0{47,51,60}))
    );

    auto test_reduce_range = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto ten_copy = ten.copy();
            auto sum = [](auto first, auto last){
                auto init = (*first).copy();
                return std::accumulate(++first,last,init,std::plus<void>{});
            };
            auto result = reduce_range(policy...,ten,axes,sum,false,true);
            REQUIRE(ten == ten_copy);
            REQUIRE(result == expected);
        };
        helpers_for_testing::apply_by_element(test,test_data);
    };

    SECTION("default_policy")
    {
        test_reduce_range();
    }
    SECTION("exec_pol<4>")
    {
        test_reduce_range(multithreading::exec_pol<4>{});
    }
}

TEST_CASE("test_tensor_of_tensor_slide","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using gtensor::detail::no_value;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,2};
    const auto t3 = tensor_type_0{0,3,1};

    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};
    const auto b = tensor_type_1{{{t1,t3,t1},{t2,t1,t0}},{{t3,t2,t0},{t0,t2,t1}}};
    const auto c = tensor_type_1{t2,t1,t3};

    //0ten,1axis,2expected
    auto test_data = std::make_tuple(
        //tensor
        std::make_tuple(a,no_value{},
            tensor_type_1{
                tensor_type_0{1,2,3},tensor_type_0{1,5,4},tensor_type_0{3,6,6},tensor_type_0{5,6,7},tensor_type_0{7,7,9},tensor_type_0{8,9,12},
                tensor_type_0{8,12,13},tensor_type_0{9,14,16},tensor_type_0{11,14,17},tensor_type_0{13,15,19},tensor_type_0{14,17,22},tensor_type_0{14,20,23}
            }
        ),
        std::make_tuple(a,0,
            tensor_type_1{
                {{tensor_type_0{1,2,3},tensor_type_0{0,3,1},tensor_type_0{2,1,2}},{tensor_type_0{2,0,1},tensor_type_0{2,1,2},tensor_type_0{1,2,3}}},
                {{tensor_type_0{1,5,4},tensor_type_0{1,5,4},tensor_type_0{4,1,3}},{tensor_type_0{4,1,3},tensor_type_0{3,3,5},tensor_type_0{1,5,4}}}
            }
        ),
        std::make_tuple(a,1,
            tensor_type_1{
                {{tensor_type_0{1,2,3},tensor_type_0{0,3,1},tensor_type_0{2,1,2}},{tensor_type_0{3,2,4},tensor_type_0{2,4,3},tensor_type_0{3,3,5}}},
                {{tensor_type_0{0,3,1},tensor_type_0{1,2,3},tensor_type_0{2,0,1}},{tensor_type_0{2,4,3},tensor_type_0{2,4,6},tensor_type_0{2,3,2}}}
            }
        ),
        std::make_tuple(a,2,
            tensor_type_1{
                {{tensor_type_0{1,2,3},tensor_type_0{1,5,4},tensor_type_0{3,6,6}},{tensor_type_0{2,0,1},tensor_type_0{4,1,3},tensor_type_0{5,3,6}}},
                {{tensor_type_0{0,3,1},tensor_type_0{1,5,4},tensor_type_0{3,5,5}},{tensor_type_0{2,1,2},tensor_type_0{3,3,5},tensor_type_0{3,6,6}}}
            }
        ),
        std::make_tuple(a.reshape(2,6),1,
            tensor_type_1{
                {tensor_type_0{1,2,3},tensor_type_0{1,5,4},tensor_type_0{3,6,6},tensor_type_0{5,6,7},tensor_type_0{7,7,9},tensor_type_0{8,9,12}},
                {tensor_type_0{0,3,1},tensor_type_0{1,5,4},tensor_type_0{3,5,5},tensor_type_0{5,6,7},tensor_type_0{6,8,10},tensor_type_0{6,11,11}}
            }
        ),
        //expression
        std::make_tuple(a+b+c,no_value{},
            tensor_type_1{
                tensor_type_0{5,3,6},tensor_type_0{7,9,9},tensor_type_0{11,13,13},tensor_type_0{17,15,18},tensor_type_0{23,16,22},tensor_type_0{25,23,29},
                tensor_type_0{27,30,33},tensor_type_0{32,33,39},tensor_type_0{35,38,44},tensor_type_0{40,42,51},tensor_type_0{45,45,57},tensor_type_0{47,51,60}
            }
        ),
        std::make_tuple(a+b+c,2,
            tensor_type_1{
                {{tensor_type_0{5,3,6},tensor_type_0{7,9,9},tensor_type_0{11,13,13}},{tensor_type_0{6,2,5},tensor_type_0{12,3,9},tensor_type_0{14,10,16}}},
                {{tensor_type_0{2,7,4},tensor_type_0{7,10,10},tensor_type_0{10,15,15}},{tensor_type_0{5,4,7},tensor_type_0{10,7,13},tensor_type_0{12,13,16}}}
            }
        )
    );

    auto test_slide = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto expected = std::get<2>(t);
            auto ten_copy = ten.copy();
            auto cumsum = [](auto first, auto, auto dfirst, auto dlast){
                auto cumsum_ = gtensor::detail::make_copy(*first);
                *dfirst = cumsum_;
                for(++dfirst,++first;dfirst!=dlast;++dfirst,++first){
                    cumsum_+=*first;
                    *dfirst = cumsum_;
                }
            };
            auto result = gtensor::slide<gtensor::detail::copy_type_t<tensor_type_1>::value_type>(policy...,ten,axes,cumsum,1,1);
            REQUIRE(ten == ten_copy);
            REQUIRE(result == expected);
        };
        helpers_for_testing::apply_by_element(test,test_data);
    };

    SECTION("default_policy")
    {
        test_slide();
    }
    SECTION("exec_pol<4>")
    {
        test_slide(multithreading::exec_pol<4>{});
    }
}

TEMPLATE_TEST_CASE("test_tensor_of_tensor_routines","[test_tensor_of_tensor]",
    //inner tensor layout
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using gtensor::tensor;
    using gtensor::detail::no_value;
    using tensor_type_0 = tensor<double,TestType>;
    using tensor_type_1 = tensor<tensor_type_0>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,2};
    const auto t3 = tensor_type_0{0,3,1};

    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};

    //sum
    REQUIRE(a.sum() == tensor_type_1(tensor_type_0{14,20,23}));
    REQUIRE(a.sum(-1) == tensor_type_1{{tensor_type_0{3,6,6},tensor_type_0{5,3,6}},{tensor_type_0{3,5,5},tensor_type_0{3,6,6}}});
    REQUIRE(a.sum({0,-1}) == tensor_type_1{tensor_type_0{6,11,11},tensor_type_0{8,9,12}});
    //cumsum
    REQUIRE(a.cumsum() == tensor_type_1{tensor_type_0{1,2,3},tensor_type_0{1,5,4},tensor_type_0{3,6,6},tensor_type_0{5,6,7},tensor_type_0{7,7,9},tensor_type_0{8,9,12},
        tensor_type_0{8,12,13},tensor_type_0{9,14,16},tensor_type_0{11,14,17},tensor_type_0{13,15,19},tensor_type_0{14,17,22},tensor_type_0{14,20,23}}
    );
    REQUIRE(a.cumsum(-1) == tensor_type_1{{{tensor_type_0{1,2,3},tensor_type_0{1,5,4},tensor_type_0{3,6,6}},{tensor_type_0{2,0,1},tensor_type_0{4,1,3},tensor_type_0{5,3,6}}},
        {{tensor_type_0{0,3,1},tensor_type_0{1,5,4},tensor_type_0{3,5,5}},{tensor_type_0{2,1,2},tensor_type_0{3,3,5},tensor_type_0{3,6,6}}}}
    );
    //prod
    REQUIRE(a.prod() == tensor_type_1(tensor_type_0{0,0,648}));
    REQUIRE(a.prod(-1) == tensor_type_1{{tensor_type_0{0,6,6},tensor_type_0{4,0,6}},{tensor_type_0{0,0,3},tensor_type_0{0,6,6}}});
    REQUIRE(a.prod({0,-1}) == tensor_type_1{tensor_type_0{0,0,18},tensor_type_0{0,0,36}});
    //cumprod
    REQUIRE(a.cumprod() == tensor_type_1{tensor_type_0{1,2,3},tensor_type_0{0,6,3},tensor_type_0{0,6,6},tensor_type_0{0,0,6},tensor_type_0{0,0,12},tensor_type_0{0,0,36},
        tensor_type_0{0,0,36},tensor_type_0{0,0,108},tensor_type_0{0,0,108},tensor_type_0{0,0,216},tensor_type_0{0,0,648},tensor_type_0{0,0,648}}
    );
    REQUIRE(a.cumprod(-1) == tensor_type_1{{{tensor_type_0{1,2,3},tensor_type_0{0,6,3},tensor_type_0{0,6,6}},{tensor_type_0{2,0,1},tensor_type_0{4,0,2},tensor_type_0{4,0,6}}},
        {{tensor_type_0{0,3,1},tensor_type_0{0,6,3},tensor_type_0{0,0,3}},{tensor_type_0{2,1,2},tensor_type_0{2,2,6},tensor_type_0{0,6,6}}}}
    );
    //diff
    REQUIRE(diff(a) == tensor_type_1{{{tensor_type_0{-1,1,-2},tensor_type_0{2,-2,1}},{tensor_type_0{0,1,1},tensor_type_0{-1,1,1}}},
        {{tensor_type_0{1,-1,2},tensor_type_0{1,-2,-2}},{tensor_type_0{-1,1,1},tensor_type_0{-1,1,-2}}}}
    );
    REQUIRE(diff(a.reshape(-1)) == tensor_type_1{tensor_type_0{-1,1,-2},tensor_type_0{2,-2,1},tensor_type_0{0,-1,-1},tensor_type_0{0,1,1},tensor_type_0{-1,1,1},tensor_type_0{-1,1,-2},
        tensor_type_0{1,-1,2},tensor_type_0{1,-2,-2},tensor_type_0{0,1,1},tensor_type_0{-1,1,1},tensor_type_0{-1,1,-2}}
    );
    REQUIRE(diff(a.reshape(-1),2) == tensor_type_1{tensor_type_0{3,-3,3},tensor_type_0{-2,1,-2},tensor_type_0{0,2,2},tensor_type_0{-1,0,0},tensor_type_0{0,0,-3},tensor_type_0{2,-2,4},
        tensor_type_0{0,-1,-4},tensor_type_0{-1,3,3},tensor_type_0{-1,0,0},tensor_type_0{0,0,-3}}
    );
    REQUIRE(diff2(a.reshape(-1)) == tensor_type_1{tensor_type_0{3,-3,3},tensor_type_0{-2,1,-2},tensor_type_0{0,2,2},tensor_type_0{-1,0,0},tensor_type_0{0,0,-3},tensor_type_0{2,-2,4},
        tensor_type_0{0,-1,-4},tensor_type_0{-1,3,3},tensor_type_0{-1,0,0},tensor_type_0{0,0,-3}}
    );
    //matmul
    REQUIRE(matmul(a(0),a(1).transpose()) == tensor_type_1{{tensor_type_0{4,12,8},tensor_type_0{2,11,11}},{tensor_type_0{4,2,10},tensor_type_0{6,8,11}}});

    //mean
    REQUIRE(tensor_close(a.mean(),tensor_type_1(tensor_type_0{1.16666667,1.66666667,1.91666667}),1E-6,1E-6));
    REQUIRE(tensor_close(a.mean(-1),tensor_type_1{{tensor_type_0{1.0,2.0,2.0},tensor_type_0{1.66666667,1.0,2.0}},{tensor_type_0{1.0,1.66666667,1.66666667},tensor_type_0{1.0,2.0,2.0}}},1E-6,1E-6));
    REQUIRE(tensor_close(a.mean({0,-1}),tensor_type_1{tensor_type_0{1.0,1.83333333,1.83333333},tensor_type_0{1.33333333,1.5,2.0}},1E-6,1E-6));
    //var
    REQUIRE(tensor_close(a.var(),tensor_type_1(tensor_type_0{0.63888889,1.05555556,0.74305556}),1E-6,1E-6));
    REQUIRE(tensor_close(a.var(-1),tensor_type_1{{tensor_type_0{0.66666667,0.66666667,0.66666667},tensor_type_0{0.22222222,0.66666667,0.66666667}},
        {tensor_type_0{0.66666667,1.55555556,0.88888889},tensor_type_0{0.66666667,0.66666667,0.66666667}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.var({0,-1}),tensor_type_1{tensor_type_0{0.66666667,1.13888889,0.80555556},tensor_type_0{0.55555556,0.91666667,0.66666667}},1E-6,1E-6));
    //stdev
    REQUIRE(tensor_close(a.stdev(),tensor_type_1(tensor_type_0{0.79930525,1.02740233,0.8620067}),1E-6,1E-6));
    REQUIRE(tensor_close(a.stdev(-1),tensor_type_1{{tensor_type_0{0.81649658,0.81649658,0.81649658},tensor_type_0{0.47140452,0.81649658,0.81649658}},
        {tensor_type_0{0.81649658,1.24721913,0.94280904},tensor_type_0{0.81649658,0.81649658,0.81649658}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.stdev({0,-1}),tensor_type_1{tensor_type_0{0.81649658,1.06718737,0.89752747},tensor_type_0{0.74535599,0.95742711,0.81649658}},1E-6,1E-6));
}

TEMPLATE_TEST_CASE("test_tensor_of_tensor_routines_policy","[test_tensor_of_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using policy = multithreading::exec_pol<4>;
    using gtensor::tensor;
    using gtensor::detail::no_value;
    using tensor_type_0 = tensor<double,TestType>;
    using tensor_type_1 = tensor<tensor_type_0>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,2};
    const auto t3 = tensor_type_0{0,3,1};

    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};

    //sum
    REQUIRE(a.sum(policy{}) == tensor_type_1(tensor_type_0{14,20,23}));
    REQUIRE(a.sum(policy{},-1) == tensor_type_1{{tensor_type_0{3,6,6},tensor_type_0{5,3,6}},{tensor_type_0{3,5,5},tensor_type_0{3,6,6}}});
    REQUIRE(a.sum(policy{},{0,-1}) == tensor_type_1{tensor_type_0{6,11,11},tensor_type_0{8,9,12}});
    //cumsum
    REQUIRE(a.cumsum(policy{}) == tensor_type_1{tensor_type_0{1,2,3},tensor_type_0{1,5,4},tensor_type_0{3,6,6},tensor_type_0{5,6,7},tensor_type_0{7,7,9},tensor_type_0{8,9,12},
        tensor_type_0{8,12,13},tensor_type_0{9,14,16},tensor_type_0{11,14,17},tensor_type_0{13,15,19},tensor_type_0{14,17,22},tensor_type_0{14,20,23}}
    );
    REQUIRE(a.cumsum(policy{},-1) == tensor_type_1{{{tensor_type_0{1,2,3},tensor_type_0{1,5,4},tensor_type_0{3,6,6}},{tensor_type_0{2,0,1},tensor_type_0{4,1,3},tensor_type_0{5,3,6}}},
        {{tensor_type_0{0,3,1},tensor_type_0{1,5,4},tensor_type_0{3,5,5}},{tensor_type_0{2,1,2},tensor_type_0{3,3,5},tensor_type_0{3,6,6}}}}
    );
    //prod
    REQUIRE(a.prod(policy{}) == tensor_type_1(tensor_type_0{0,0,648}));
    REQUIRE(a.prod(policy{},-1) == tensor_type_1{{tensor_type_0{0,6,6},tensor_type_0{4,0,6}},{tensor_type_0{0,0,3},tensor_type_0{0,6,6}}});
    REQUIRE(a.prod(policy{},{0,-1}) == tensor_type_1{tensor_type_0{0,0,18},tensor_type_0{0,0,36}});
    //cumprod
    REQUIRE(a.cumprod(policy{}) == tensor_type_1{tensor_type_0{1,2,3},tensor_type_0{0,6,3},tensor_type_0{0,6,6},tensor_type_0{0,0,6},tensor_type_0{0,0,12},tensor_type_0{0,0,36},
        tensor_type_0{0,0,36},tensor_type_0{0,0,108},tensor_type_0{0,0,108},tensor_type_0{0,0,216},tensor_type_0{0,0,648},tensor_type_0{0,0,648}}
    );
    REQUIRE(a.cumprod(policy{},-1) == tensor_type_1{{{tensor_type_0{1,2,3},tensor_type_0{0,6,3},tensor_type_0{0,6,6}},{tensor_type_0{2,0,1},tensor_type_0{4,0,2},tensor_type_0{4,0,6}}},
        {{tensor_type_0{0,3,1},tensor_type_0{0,6,3},tensor_type_0{0,0,3}},{tensor_type_0{2,1,2},tensor_type_0{2,2,6},tensor_type_0{0,6,6}}}}
    );
    //diff
    REQUIRE(diff(policy{},a) == tensor_type_1{{{tensor_type_0{-1,1,-2},tensor_type_0{2,-2,1}},{tensor_type_0{0,1,1},tensor_type_0{-1,1,1}}},
        {{tensor_type_0{1,-1,2},tensor_type_0{1,-2,-2}},{tensor_type_0{-1,1,1},tensor_type_0{-1,1,-2}}}}
    );
    REQUIRE(diff(policy{},a.reshape(-1)) == tensor_type_1{tensor_type_0{-1,1,-2},tensor_type_0{2,-2,1},tensor_type_0{0,-1,-1},tensor_type_0{0,1,1},tensor_type_0{-1,1,1},tensor_type_0{-1,1,-2},
        tensor_type_0{1,-1,2},tensor_type_0{1,-2,-2},tensor_type_0{0,1,1},tensor_type_0{-1,1,1},tensor_type_0{-1,1,-2}}
    );
    REQUIRE(diff(policy{},a.reshape(-1),2) == tensor_type_1{tensor_type_0{3,-3,3},tensor_type_0{-2,1,-2},tensor_type_0{0,2,2},tensor_type_0{-1,0,0},tensor_type_0{0,0,-3},tensor_type_0{2,-2,4},
        tensor_type_0{0,-1,-4},tensor_type_0{-1,3,3},tensor_type_0{-1,0,0},tensor_type_0{0,0,-3}}
    );
    REQUIRE(diff2(policy{},a.reshape(-1)) == tensor_type_1{tensor_type_0{3,-3,3},tensor_type_0{-2,1,-2},tensor_type_0{0,2,2},tensor_type_0{-1,0,0},tensor_type_0{0,0,-3},tensor_type_0{2,-2,4},
        tensor_type_0{0,-1,-4},tensor_type_0{-1,3,3},tensor_type_0{-1,0,0},tensor_type_0{0,0,-3}}
    );
    //matmul
    REQUIRE(matmul(a(0),a(1).transpose()) == tensor_type_1{{tensor_type_0{4,12,8},tensor_type_0{2,11,11}},{tensor_type_0{4,2,10},tensor_type_0{6,8,11}}});

    //mean
    REQUIRE(tensor_close(a.mean(policy{}),tensor_type_1(tensor_type_0{1.16666667,1.66666667,1.91666667}),1E-6,1E-6));
    REQUIRE(tensor_close(a.mean(policy{},-1),tensor_type_1{{tensor_type_0{1.0,2.0,2.0},tensor_type_0{1.66666667,1.0,2.0}},
        {tensor_type_0{1.0,1.66666667,1.66666667},tensor_type_0{1.0,2.0,2.0}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.mean(policy{},{0,-1}),tensor_type_1{tensor_type_0{1.0,1.83333333,1.83333333},tensor_type_0{1.33333333,1.5,2.0}},1E-6,1E-6));
    //var
    REQUIRE(tensor_close(a.var(policy{}),tensor_type_1(tensor_type_0{0.63888889,1.05555556,0.74305556}),1E-6,1E-6));
    REQUIRE(tensor_close(a.var(policy{},-1),tensor_type_1{{tensor_type_0{0.66666667,0.66666667,0.66666667},tensor_type_0{0.22222222,0.66666667,0.66666667}},
        {tensor_type_0{0.66666667,1.55555556,0.88888889},tensor_type_0{0.66666667,0.66666667,0.66666667}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.var(policy{},{0,-1}),tensor_type_1{tensor_type_0{0.66666667,1.13888889,0.80555556},tensor_type_0{0.55555556,0.91666667,0.66666667}},1E-6,1E-6));
    //stdev
    REQUIRE(tensor_close(a.stdev(policy{}),tensor_type_1(tensor_type_0{0.79930525,1.02740233,0.8620067}),1E-6,1E-6));
    REQUIRE(tensor_close(a.stdev(policy{},-1),tensor_type_1{{tensor_type_0{0.81649658,0.81649658,0.81649658},tensor_type_0{0.47140452,0.81649658,0.81649658}},
        {tensor_type_0{0.81649658,1.24721913,0.94280904},tensor_type_0{0.81649658,0.81649658,0.81649658}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.stdev(policy{},{0,-1}),tensor_type_1{tensor_type_0{0.81649658,1.06718737,0.89752747},tensor_type_0{0.74535599,0.95742711,0.81649658}},1E-6,1E-6));
}

TEST_CASE("test_tensor_of_tensor_expression_routines","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using gtensor::detail::no_value;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,2};
    const auto t3 = tensor_type_0{0,3,1};

    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};
    const auto b = tensor_type_1{{{t1,t3,t1},{t2,t1,t0}},{{t3,t2,t0},{t0,t2,t1}}};
    const auto c = tensor_type_1{t2,t1,t3};

    //sum
    REQUIRE((a+b+c).sum() == tensor_type_1(tensor_type_0{47,51,60}));
    REQUIRE((a+b+c).sum(-1) == tensor_type_1{{tensor_type_0{11,13,13},tensor_type_0{14,10,16}},{tensor_type_0{10,15,15},tensor_type_0{12,13,16}}});
    REQUIRE((a+b+c).sum({0,-1}) == tensor_type_1{tensor_type_0{21,28,28},tensor_type_0{26,23,32}});
    //cumsum
    REQUIRE((a+b+c).cumsum() == tensor_type_1{tensor_type_0{5,3,6},tensor_type_0{7,9,9},tensor_type_0{11,13,13},tensor_type_0{17,15,18},tensor_type_0{23,16,22},
        tensor_type_0{25,23,29},tensor_type_0{27,30,33},tensor_type_0{32,33,39},tensor_type_0{35,38,44},tensor_type_0{40,42,51},tensor_type_0{45,45,57},tensor_type_0{47,51,60}}
    );
    REQUIRE((a+b+c).cumsum(-1) == tensor_type_1{{{tensor_type_0{5,3,6},tensor_type_0{7,9,9},tensor_type_0{11,13,13}},{tensor_type_0{6,2,5},tensor_type_0{12,3,9},tensor_type_0{14,10,16}}},
        {{tensor_type_0{2,7,4},tensor_type_0{7,10,10},tensor_type_0{10,15,15}},{tensor_type_0{5,4,7},tensor_type_0{10,7,13},tensor_type_0{12,13,16}}}}
    );
    //prod
    REQUIRE((a+b+c).prod() == tensor_type_1(tensor_type_0{4320000,7620480,152409600}));
    REQUIRE((a+b+c).prod(-1) == tensor_type_1{{tensor_type_0{40,72,72},tensor_type_0{72,14,140}},{tensor_type_0{30,105,120},tensor_type_0{50,72,126}}});
    REQUIRE((a+b+c).prod({0,-1}) == tensor_type_1{tensor_type_0{1200,7560,8640},tensor_type_0{3600,1008,17640}});
    //cumprod
    REQUIRE((a+b+c).cumprod() == tensor_type_1{tensor_type_0{5,3,6},tensor_type_0{10,18,18},tensor_type_0{40,72,72},tensor_type_0{240,144,360},tensor_type_0{1440,144,1440},
        tensor_type_0{2880,1008,10080},tensor_type_0{5760,7056,40320},tensor_type_0{28800,21168,241920},tensor_type_0{86400,105840,1209600},tensor_type_0{432000,423360,8467200},
        tensor_type_0{2160000,1270080,50803200},tensor_type_0{4320000,7620480,152409600}}
    );
    REQUIRE((a+b+c).cumprod(-1) == tensor_type_1{{{tensor_type_0{5,3,6},tensor_type_0{10,18,18},tensor_type_0{40,72,72}},{tensor_type_0{6,2,5},tensor_type_0{36,2,20},
        tensor_type_0{72,14,140}}},{{tensor_type_0{2,7,4},tensor_type_0{10,21,24},tensor_type_0{30,105,120}},{tensor_type_0{5,4,7},tensor_type_0{25,12,42},tensor_type_0{50,72,126}}}}
    );
    //diff
    REQUIRE(diff(a+b+c) == tensor_type_1{{{tensor_type_0{-3,3,-3},tensor_type_0{2,-2,1}},{tensor_type_0{0,-1,-1},tensor_type_0{-4,6,3}}},{{tensor_type_0{3,-4,2},
        tensor_type_0{-2,2,-1}},{tensor_type_0{0,-1,-1},tensor_type_0{-3,3,-3}}}}
    );
    REQUIRE(diff((a+b+c).reshape(-1)) == tensor_type_1{tensor_type_0{-3,3,-3},tensor_type_0{2,-2,1},tensor_type_0{2,-2,1},tensor_type_0{0,-1,-1},tensor_type_0{-4,6,3},
        tensor_type_0{0,0,-3},tensor_type_0{3,-4,2},tensor_type_0{-2,2,-1},tensor_type_0{2,-1,2},tensor_type_0{0,-1,-1},tensor_type_0{-3,3,-3}}
    );
    REQUIRE(diff((a+b+c).reshape(-1),2) == tensor_type_1{tensor_type_0{5,-5,4},tensor_type_0{0,0,0},tensor_type_0{-2,1,-2},tensor_type_0{-4,7,4},
        tensor_type_0{4,-6,-6},tensor_type_0{3,-4,5},tensor_type_0{-5,6,-3},tensor_type_0{4,-3,3},tensor_type_0{-2,0,-3},tensor_type_0{-3,4,-2}}
    );
    REQUIRE(diff2((a+b+c).reshape(-1)) == tensor_type_1{tensor_type_0{5,-5,4},tensor_type_0{0,0,0},tensor_type_0{-2,1,-2},tensor_type_0{-4,7,4},
        tensor_type_0{4,-6,-6},tensor_type_0{3,-4,5},tensor_type_0{-5,6,-3},tensor_type_0{4,-3,3},tensor_type_0{-2,0,-3},tensor_type_0{-3,4,-2}}
    );
    //matmul
    REQUIRE(matmul((a+b+c)(0),(a+b+c)(1).transpose()) == tensor_type_1{{tensor_type_0{32,59,62},tensor_type_0{43,54,72}},{tensor_type_0{48,52,79},tensor_type_0{64,53,80}}});

    //mean
    REQUIRE(tensor_close((a+b+c).mean(),tensor_type_1(tensor_type_0{3.91666667,4.25,5.0}),1E-6,1E-6));
    REQUIRE(tensor_close((a+b+c).mean(-1),tensor_type_1{{tensor_type_0{3.66666667,4.33333333,4.33333333},tensor_type_0{4.66666667,3.33333333,5.33333333}},
        {tensor_type_0{3.33333333,5.0,5.0},tensor_type_0{4.0,4.33333333,5.33333333}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close((a+b+c).mean({0,-1}),tensor_type_1{tensor_type_0{3.5,4.66666667,4.66666667},tensor_type_0{4.33333333,3.83333333,5.33333333}},1E-6,1E-6));
    //var
    REQUIRE(tensor_close((a+b+c).var(),tensor_type_1(tensor_type_0{2.40972222,3.52083333,1.83333333}),1E-6,1E-6));
    REQUIRE(tensor_close((a+b+c).var(-1),tensor_type_1{{tensor_type_0{1.55555556,1.55555556,1.55555556},tensor_type_0{3.55555556,6.88888889,1.55555556}},
        {tensor_type_0{1.55555556,2.66666667,0.66666667},tensor_type_0{2.0,1.55555556,2.88888889}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close((a+b+c).var({0,-1}),tensor_type_1{tensor_type_0{1.58333333,2.22222222,1.22222222},tensor_type_0{2.88888889,4.47222222,2.22222222}},1E-6,1E-6));
    //stdev
    REQUIRE(tensor_close((a+b+c).stdev(),tensor_type_1(tensor_type_0{1.552328,1.87638837,1.3540064}),1E-6,1E-6));
    REQUIRE(tensor_close((a+b+c).stdev(-1),tensor_type_1{{tensor_type_0{1.24721913,1.24721913,1.24721913},tensor_type_0{1.88561808,2.62466929,1.24721913}},
        {tensor_type_0{1.24721913,1.63299316,0.81649658},tensor_type_0{1.41421356,1.24721913,1.69967317}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close((a+b+c).stdev({0,-1}),tensor_type_1{tensor_type_0{1.25830574,1.49071198,1.1055416},tensor_type_0{1.69967317,2.11476292,1.49071198}},1E-6,1E-6));
}

TEMPLATE_TEST_CASE("test_tensor_of_tensor_expression_routines_policy","[test_tensor_of_tensor]",
    multithreading::exec_pol<4>
)
{
    using policy = TestType;
    using gtensor::tensor;
    using gtensor::detail::no_value;
    using tensor_type_0 = tensor<double>;
    using tensor_type_1 = tensor<tensor_type_0>;

    const auto t0 = tensor_type_0{1,2,3};
    const auto t1 = tensor_type_0{2,0,1};
    const auto t2 = tensor_type_0{2,1,2};
    const auto t3 = tensor_type_0{0,3,1};

    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};
    const auto b = tensor_type_1{{{t1,t3,t1},{t2,t1,t0}},{{t3,t2,t0},{t0,t2,t1}}};
    const auto c = tensor_type_1{t2,t1,t3};

    //sum
    REQUIRE((a+b+c).sum(policy{}) == tensor_type_1(tensor_type_0{47,51,60}));
    REQUIRE((a+b+c).sum(policy{},-1) == tensor_type_1{{tensor_type_0{11,13,13},tensor_type_0{14,10,16}},{tensor_type_0{10,15,15},tensor_type_0{12,13,16}}});
    REQUIRE((a+b+c).sum(policy{},{0,-1}) == tensor_type_1{tensor_type_0{21,28,28},tensor_type_0{26,23,32}});
    //cumsum
    REQUIRE((a+b+c).cumsum(policy{}) == tensor_type_1{tensor_type_0{5,3,6},tensor_type_0{7,9,9},tensor_type_0{11,13,13},tensor_type_0{17,15,18},tensor_type_0{23,16,22},
        tensor_type_0{25,23,29},tensor_type_0{27,30,33},tensor_type_0{32,33,39},tensor_type_0{35,38,44},tensor_type_0{40,42,51},tensor_type_0{45,45,57},tensor_type_0{47,51,60}}
    );
    REQUIRE((a+b+c).cumsum(policy{},-1) == tensor_type_1{{{tensor_type_0{5,3,6},tensor_type_0{7,9,9},tensor_type_0{11,13,13}},{tensor_type_0{6,2,5},tensor_type_0{12,3,9},tensor_type_0{14,10,16}}},
        {{tensor_type_0{2,7,4},tensor_type_0{7,10,10},tensor_type_0{10,15,15}},{tensor_type_0{5,4,7},tensor_type_0{10,7,13},tensor_type_0{12,13,16}}}}
    );
    //prod
    REQUIRE((a+b+c).prod(policy{}) == tensor_type_1(tensor_type_0{4320000,7620480,152409600}));
    REQUIRE((a+b+c).prod(policy{},-1) == tensor_type_1{{tensor_type_0{40,72,72},tensor_type_0{72,14,140}},{tensor_type_0{30,105,120},tensor_type_0{50,72,126}}});
    REQUIRE((a+b+c).prod(policy{},{0,-1}) == tensor_type_1{tensor_type_0{1200,7560,8640},tensor_type_0{3600,1008,17640}});
    //cumprod
    REQUIRE((a+b+c).cumprod(policy{}) == tensor_type_1{tensor_type_0{5,3,6},tensor_type_0{10,18,18},tensor_type_0{40,72,72},tensor_type_0{240,144,360},tensor_type_0{1440,144,1440},
        tensor_type_0{2880,1008,10080},tensor_type_0{5760,7056,40320},tensor_type_0{28800,21168,241920},tensor_type_0{86400,105840,1209600},tensor_type_0{432000,423360,8467200},
        tensor_type_0{2160000,1270080,50803200},tensor_type_0{4320000,7620480,152409600}}
    );
    REQUIRE((a+b+c).cumprod(policy{},-1) == tensor_type_1{{{tensor_type_0{5,3,6},tensor_type_0{10,18,18},tensor_type_0{40,72,72}},{tensor_type_0{6,2,5},tensor_type_0{36,2,20},
        tensor_type_0{72,14,140}}},{{tensor_type_0{2,7,4},tensor_type_0{10,21,24},tensor_type_0{30,105,120}},{tensor_type_0{5,4,7},tensor_type_0{25,12,42},tensor_type_0{50,72,126}}}}
    );
    //diff
    REQUIRE(diff(policy{},a+b+c) == tensor_type_1{{{tensor_type_0{-3,3,-3},tensor_type_0{2,-2,1}},{tensor_type_0{0,-1,-1},tensor_type_0{-4,6,3}}},{{tensor_type_0{3,-4,2},
        tensor_type_0{-2,2,-1}},{tensor_type_0{0,-1,-1},tensor_type_0{-3,3,-3}}}}
    );
    REQUIRE(diff(policy{},(a+b+c).reshape(-1)) == tensor_type_1{tensor_type_0{-3,3,-3},tensor_type_0{2,-2,1},tensor_type_0{2,-2,1},tensor_type_0{0,-1,-1},tensor_type_0{-4,6,3},
        tensor_type_0{0,0,-3},tensor_type_0{3,-4,2},tensor_type_0{-2,2,-1},tensor_type_0{2,-1,2},tensor_type_0{0,-1,-1},tensor_type_0{-3,3,-3}}
    );
    REQUIRE(diff(policy{},(a+b+c).reshape(-1),2) == tensor_type_1{tensor_type_0{5,-5,4},tensor_type_0{0,0,0},tensor_type_0{-2,1,-2},tensor_type_0{-4,7,4},
        tensor_type_0{4,-6,-6},tensor_type_0{3,-4,5},tensor_type_0{-5,6,-3},tensor_type_0{4,-3,3},tensor_type_0{-2,0,-3},tensor_type_0{-3,4,-2}}
    );
    REQUIRE(diff2(policy{},(a+b+c).reshape(-1)) == tensor_type_1{tensor_type_0{5,-5,4},tensor_type_0{0,0,0},tensor_type_0{-2,1,-2},tensor_type_0{-4,7,4},
        tensor_type_0{4,-6,-6},tensor_type_0{3,-4,5},tensor_type_0{-5,6,-3},tensor_type_0{4,-3,3},tensor_type_0{-2,0,-3},tensor_type_0{-3,4,-2}}
    );
    //matmul
    REQUIRE(matmul((a+b+c)(0),(a+b+c)(1).transpose()) == tensor_type_1{{tensor_type_0{32,59,62},tensor_type_0{43,54,72}},{tensor_type_0{48,52,79},tensor_type_0{64,53,80}}});

    //mean
    REQUIRE(tensor_close((a+b+c).mean(policy{}),tensor_type_1(tensor_type_0{3.91666667,4.25,5.0}),1E-6,1E-6));
    REQUIRE(tensor_close((a+b+c).mean(policy{},-1),tensor_type_1{{tensor_type_0{3.66666667,4.33333333,4.33333333},tensor_type_0{4.66666667,3.33333333,5.33333333}},
        {tensor_type_0{3.33333333,5.0,5.0},tensor_type_0{4.0,4.33333333,5.33333333}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close((a+b+c).mean(policy{},{0,-1}),tensor_type_1{tensor_type_0{3.5,4.66666667,4.66666667},tensor_type_0{4.33333333,3.83333333,5.33333333}},1E-6,1E-6));
    //var
    REQUIRE(tensor_close((a+b+c).var(policy{}),tensor_type_1(tensor_type_0{2.40972222,3.52083333,1.83333333}),1E-6,1E-6));
    REQUIRE(tensor_close((a+b+c).var(policy{},-1),tensor_type_1{{tensor_type_0{1.55555556,1.55555556,1.55555556},tensor_type_0{3.55555556,6.88888889,1.55555556}},
        {tensor_type_0{1.55555556,2.66666667,0.66666667},tensor_type_0{2.0,1.55555556,2.88888889}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close((a+b+c).var(policy{},{0,-1}),tensor_type_1{tensor_type_0{1.58333333,2.22222222,1.22222222},tensor_type_0{2.88888889,4.47222222,2.22222222}},1E-6,1E-6));
    //stdev
    REQUIRE(tensor_close((a+b+c).stdev(policy{}),tensor_type_1(tensor_type_0{1.552328,1.87638837,1.3540064}),1E-6,1E-6));
    REQUIRE(tensor_close((a+b+c).stdev(policy{},-1),tensor_type_1{{tensor_type_0{1.24721913,1.24721913,1.24721913},tensor_type_0{1.88561808,2.62466929,1.24721913}},
        {tensor_type_0{1.24721913,1.63299316,0.81649658},tensor_type_0{1.41421356,1.24721913,1.69967317}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close((a+b+c).stdev(policy{},{0,-1}),tensor_type_1{tensor_type_0{1.25830574,1.49071198,1.1055416},tensor_type_0{1.69967317,2.11476292,1.49071198}},1E-6,1E-6));
}

TEST_CASE("test_tensor_of_tensor_routines_std_complex","[test_tensor_of_tensor]")
{
    using gtensor::tensor;
    using tensor_type_0 = tensor<std::complex<double>>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using namespace std::complex_literals;

    const auto t0 = tensor_type_0{1.1+2.2i,2.2+1.1i,3.2+0.1i};
    const auto t1 = tensor_type_0{2.1+2.2i,1.2+1.1i,1.2+0.1i};
    const auto t2 = tensor_type_0{0.1+1.2i,1.1+2.2i,1.2+2.1i};
    const auto t3 = tensor_type_0{0.2+2.2i,2.1+2.3i,0.2+2.3i};
    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};
    const auto b = tensor_type_1{{{t1,t3,t1},{t2,t1,t0}},{{t3,t2,t0},{t0,t2,t1}}};
    const auto c = tensor_type_1{t2,t1,t3};

    //sum
    REQUIRE(tensor_close(a.sum(), tensor_type_1(tensor_type_0{9.5+23.4i, 20.8+20.1i, 19.4+13.8i})));
    REQUIRE(tensor_close(a.sum(-1), tensor_type_1{{tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i},tensor_type_0{3.3+5.6i,4.5+4.4i,5.6+2.3i}},
        {tensor_type_0{3.4+6.6i,5.5+4.5i,4.6+2.5i},tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i}}}));
    REQUIRE(tensor_close(a.sum({0,-1}),tensor_type_1{tensor_type_0{4.8+12.2i,10.9+10.1i,9.2+7.0i},tensor_type_0{4.7+11.2i,9.9+10.0i,10.2+6.8i}}));
    //cumsum
    REQUIRE(tensor_close(a.cumsum(),tensor_type_1{tensor_type_0{1.1+2.2i,2.2+1.1i,3.2+0.1i},tensor_type_0{1.3+4.4i,4.3+3.4i,3.4+2.4i},tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i},
        tensor_type_0{3.5+7.8i,6.6+6.7i,5.8+4.6i},tensor_type_0{3.6+9.0i,7.7+8.9i,7.0+6.7i},tensor_type_0{4.7+11.2i,9.9+10.0i,10.2+6.8i},tensor_type_0{4.9+13.4i,12.0+12.3i,10.4+9.1i},
        tensor_type_0{6.0+15.6i,14.2+13.4i,13.6+9.2i},tensor_type_0{8.1+17.8i,15.4+14.5i,14.8+9.3i},tensor_type_0{8.2+19.0i,16.5+16.7i,16.0+11.4i},
        tensor_type_0{9.3+21.2i,18.7+17.8i,19.2+11.5i},tensor_type_0{9.5+23.4i,20.8+20.1i,19.4+13.8i}}
    ));
    REQUIRE(tensor_close(a.cumsum(-1),tensor_type_1{{{tensor_type_0{1.1+2.2i,2.2+1.1i,3.2+0.1i},tensor_type_0{1.3+4.4i,4.3+3.4i,3.4+2.4i},tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i}},
        {tensor_type_0{2.1+2.2i,1.2+1.1i,1.2+0.1i},tensor_type_0{2.2+3.4i,2.3+3.3i,2.4+2.2i},tensor_type_0{3.3+5.6i,4.5+4.4i,5.6+2.3i}}},{{tensor_type_0{0.2+2.2i,2.1+2.3i,0.2+2.3i},
        tensor_type_0{1.3+4.4i,4.3+3.4i,3.4+2.4i},tensor_type_0{3.4+6.6i,5.5+4.5i,4.6+2.5i}},{tensor_type_0{0.1+1.2i,1.1+2.2i,1.2+2.1i},
        tensor_type_0{1.2+3.4i,3.3+3.3i,4.4+2.2i},tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i}}}}
    ));
    //prod
    REQUIRE(tensor_close(a.prod(),tensor_type_1(tensor_type_0{-4625.00567341+4384.36281148i,-41998.46350343+11731.21360869i,-1176.87861768+26497.95898164i}),1E-6,1E-6));
    REQUIRE(tensor_close(a.prod(-1),tensor_type_1{{tensor_type_0{-3.894-5.258i,-13.915+12.705i,-15.006+9.717i},tensor_type_0{-8.701-2.332i,-6.655+7.26i,3.672+8.571i}},
        {tensor_type_0{-15.994-4.158i,-5.599+11.143i,-0.246+8.897i},tensor_type_0{-3.894-5.258i,-13.915+12.705i,-15.006+9.717i}}},1E-6,1E-6));
    REQUIRE(tensor_close(a.prod({0,-1}),tensor_type_1{tensor_type_0{40.417872+100.287704i,-63.66173-226.19014i,-82.760673-135.898764i},
        tensor_type_0{21.620038+54.830666i,0.366025-185.574675i,-138.386439-92.935602i}},1E-6,1E-6
    ));
    //cumprod
    REQUIRE(tensor_close(a.cumprod(),tensor_type_1{tensor_type_0{1.10000000e+00+2.20000000e+00i,2.20000000e+00+1.10000000e+00i,3.20000000e+00+1.00000000e-01i},
        tensor_type_0{-4.62000000e+00+2.86000000e+00i,2.09000000e+00+7.37000000e+00i,4.10000000e-01+7.38000000e+00i},
        tensor_type_0{-3.89400000e+00-5.25800000e+00i,-1.39150000e+01+1.27050000e+01i,-1.50060000e+01+9.71700000e+00i},
        tensor_type_0{3.39020000e+00-1.96086000e+01i,-3.06735000e+01-6.05000000e-02i,-1.89789000e+01+1.01598000e+01i},
        tensor_type_0{2.38693400e+01+2.10738000e+00i,-3.36077500e+01-6.75482500e+01i,-4.41102600e+01-2.76639300e+01i},
        tensor_type_0{2.16200380e+01+5.48306660e+01i,3.66025000e-01-1.85574675e+02i,-1.38386439e+02-9.29356020e+01i},
        tensor_type_0{-1.16303458e+02+5.85302168e+01i,4.27590405e+02-3.88864960e+02i,1.86074597e+02-3.36875930e+02i},
        tensor_type_0{-2.56700280e+02-1.91484368e+02i,1.36845035e+03-3.85153467e+02i,6.29126303e+02-1.05939552e+03i},
        tensor_type_0{-1.17804979e+02-9.66857790e+02i,2.06580923e+03+1.04311122e+03i,8.60891115e+02-1.20836199e+03i},
        tensor_type_0{1.14844885e+03-2.38051753e+02i,-2.24545357e+01+5.69220265e+03i,3.57062952e+03+3.57836954e+02i},
        tensor_type_0{1.78700759e+03+2.26473054e+03i,-6.31082289e+03+1.24981458e+04i,1.13902308e+04+1.50214120e+03i},
        tensor_type_0{-4.62500567e+03+4.38436281e+03i,-4.19984635e+04+1.17312136e+04i,-1.17687862e+03+2.64979590e+04i}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(a.cumprod(-1),tensor_type_1{{{tensor_type_0{1.1+2.2i,2.2+1.1i,3.2+0.1i},tensor_type_0{-4.62+2.86i,2.09+7.37i,0.41+7.38i},
        tensor_type_0{-3.894-5.258i,-13.915+12.705i,-15.006+9.717i}},{tensor_type_0{2.1+2.2i,1.2+1.1i,1.2+0.1i},tensor_type_0{-2.43+2.74i,-1.1+3.85i,1.23+2.64i},
        tensor_type_0{-8.701-2.332i,-6.655+7.26i,3.672+8.571i}}},{{tensor_type_0{0.2+2.2i,2.1+2.3i,0.2+2.3i},tensor_type_0{-4.62+2.86i,2.09+7.37i,0.41+7.38i},
        tensor_type_0{-15.994-4.158i,-5.599+11.143i,-0.246+8.897i}},{tensor_type_0{0.1+1.2i,1.1+2.2i,1.2+2.1i},tensor_type_0{-2.53+1.54i,0.0+6.05i,3.63+6.84i},
        tensor_type_0{-3.894-5.258i,-13.915+12.705i,-15.006+9.717i}}}},1E-6,1E-6
    ));
    //diff
    REQUIRE(tensor_close(diff(a),tensor_type_1{{{tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i},tensor_type_0{-0.1-1.0i,-1.0-0.1i,1.0-0.2i}},
    {tensor_type_0{-2.0-1.0i,-0.1+1.1i,0.0+2.0i},tensor_type_0{1.0+1.0i,1.1-1.1i,2.0-2.0i}}},{{tensor_type_0{0.9+0.0i,0.1-1.2i,3.0-2.2i},
    tensor_type_0{1.0+0.0i,-1.0+0.0i,-2.0+0.0i}},{tensor_type_0{1.0+1.0i,1.1-1.1i,2.0-2.0i},tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i}}}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(diff(a.reshape(-1)),tensor_type_1{tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i},tensor_type_0{-0.1-1.0i,-1.0-0.1i,1.0-0.2i},
        tensor_type_0{2.0+1.0i,0.1-1.1i,0.0-2.0i},tensor_type_0{-2.0-1.0i,-0.1+1.1i,0.0+2.0i},tensor_type_0{1.0+1.0i,1.1-1.1i,2.0-2.0i},tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i},
        tensor_type_0{0.9+0.0i,0.1-1.2i,3.0-2.2i},tensor_type_0{1.0+0.0i,-1.0+0.0i,-2.0+0.0i},tensor_type_0{-2.0-1.0i,-0.1+1.1i,0.0+2.0i},tensor_type_0{1.0+1.0i,1.1-1.1i,2.0-2.0i},
        tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(diff(a.reshape(-1),2),tensor_type_1{tensor_type_0{0.8-1.0i,-0.9-1.3i,4.0-2.4i},tensor_type_0{2.1+2.0i,1.1-1.0i,-1.0-1.8i},
        tensor_type_0{-4.0-2.0i,-0.2+2.2i,0.0+4.0i},tensor_type_0{3.0+2.0i,1.2-2.2i,2.0-4.0i},tensor_type_0{-1.9-1.0i,-1.2+2.3i,-5.0+4.2i},tensor_type_0{1.8+0.0i,0.2-2.4i,6.0-4.4i},
        tensor_type_0{0.1+0.0i,-1.1+1.2i,-5.0+2.2i},tensor_type_0{-3.0-1.0i,0.9+1.1i,2.0+2.0i},tensor_type_0{3.0+2.0i,1.2-2.2i,2.0-4.0i},tensor_type_0{-1.9-1.0i,-1.2+2.3i,-5.0+4.2i}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(diff2(a.reshape(-1)),tensor_type_1{tensor_type_0{0.8-1.0i,-0.9-1.3i,4.0-2.4i},tensor_type_0{2.1+2.0i,1.1-1.0i,-1.0-1.8i},
        tensor_type_0{-4.0-2.0i,-0.2+2.2i,0.0+4.0i},tensor_type_0{3.0+2.0i,1.2-2.2i,2.0-4.0i},tensor_type_0{-1.9-1.0i,-1.2+2.3i,-5.0+4.2i},tensor_type_0{1.8+0.0i,0.2-2.4i,6.0-4.4i},
        tensor_type_0{0.1+0.0i,-1.1+1.2i,-5.0+2.2i},tensor_type_0{-3.0-1.0i,0.9+1.1i,2.0+2.0i},tensor_type_0{3.0+2.0i,1.2-2.2i,2.0-4.0i},tensor_type_0{-1.9-1.0i,-1.2+2.3i,-5.0+4.2i}},1E-6,1E-6
    ));

    //matmul
    REQUIRE(tensor_close(matmul(a(0),a(1).transpose()),tensor_type_1{{tensor_type_0{-11.67 +8.46i,3.08+18.59i,2.05+17.4i},tensor_type_0{-9.77+4.86i,-0.66+20.57i,-0.55+17.4i}},
        {tensor_type_0{-9.48+13.64i,1.42+14.86i,7.47+10.06i},tensor_type_0{-9.58 +7.14i,0.99+17.27i,5.27+16.86i}}},1E-6,1E-6
    ));

    //mean
    REQUIRE(tensor_close(a.mean(),tensor_type_1(tensor_type_0{0.79166667+1.95i,1.73333333+1.675i,1.61666667+1.15i}),1E-6,1E-6));
    REQUIRE(tensor_close(a.mean(-1),tensor_type_1{{tensor_type_0{0.46666667+1.86666667i,1.8+1.86666667i,1.53333333+1.5i},tensor_type_0{1.1+1.86666667i,1.5+1.46666667i,1.86666667+0.76666667i}},
        {tensor_type_0{1.13333333+2.2i,1.83333333+1.5i,1.53333333+0.83333333i},tensor_type_0{0.46666667+1.86666667i,1.8+1.86666667i,1.53333333+1.5i}}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(a.mean({0,-1}),tensor_type_1{tensor_type_0{0.8+2.03333333i,1.81666667+1.68333333i,1.53333333+1.16666667i},tensor_type_0{0.78333333+1.86666667i,1.65+1.66666667i,1.7+1.13333333i}},1E-6,1E-6));
    //var
    REQUIRE(tensor_close(a.var(),tensor<tensor<double>>(tensor<double>{0.71159722,0.58576389,2.51722222}),1E-6,1E-6));
    REQUIRE(tensor_close(a.var(-1),tensor<tensor<double>>{{tensor<double>{0.42444444,0.54222222,2.54222222},tensor<double>{0.88888889,0.51555556,1.77777778}},
        {tensor<double>{0.60222222,0.52222222,2.63111111},tensor<double>{0.42444444,0.54222222,2.54222222}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.var({0,-1}),tensor<tensor<double>>{tensor<double>{0.65222222,0.56611111,2.69777778},tensor<double>{0.75694444,0.59138889,2.32222222}},1E-6,1E-6));
    //stdev
    REQUIRE(tensor_close(a.stdev(),tensor<tensor<double>>(tensor<double>{0.84356222,0.76535213,1.58657563}),1E-6,1E-6));
    REQUIRE(tensor_close(a.stdev(-1),tensor<tensor<double>>{{tensor<double>{0.65149401,0.7363574,1.59443477},tensor<double>{0.94280904,0.71802197,1.33333333}},
        {tensor<double>{0.77602978,0.72264945,1.62207001},tensor<double>{0.65149401,0.7363574,1.59443477}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.stdev({0,-1}),tensor<tensor<double>>{tensor<double>{0.80760276,0.75240356,1.64249133},tensor<double>{0.87002554,0.76901813,1.52388393}},1E-6,1E-6));
}

TEMPLATE_TEST_CASE("test_tensor_of_tensor_routines_std_complex_policy","[test_tensor_of_tensor]",
    multithreading::exec_pol<4>
)
{
    using policy = TestType;
    using gtensor::tensor;
    using tensor_type_0 = tensor<std::complex<double>>;
    using tensor_type_1 = tensor<tensor_type_0>;
    using namespace std::complex_literals;

    const auto t0 = tensor_type_0{1.1+2.2i,2.2+1.1i,3.2+0.1i};
    const auto t1 = tensor_type_0{2.1+2.2i,1.2+1.1i,1.2+0.1i};
    const auto t2 = tensor_type_0{0.1+1.2i,1.1+2.2i,1.2+2.1i};
    const auto t3 = tensor_type_0{0.2+2.2i,2.1+2.3i,0.2+2.3i};
    const auto a = tensor_type_1{{{t0,t3,t2},{t1,t2,t0}},{{t3,t0,t1},{t2,t0,t3}}};
    const auto b = tensor_type_1{{{t1,t3,t1},{t2,t1,t0}},{{t3,t2,t0},{t0,t2,t1}}};
    const auto c = tensor_type_1{t2,t1,t3};

    //sum
    REQUIRE(tensor_close(a.sum(policy{}), tensor_type_1(tensor_type_0{9.5+23.4i, 20.8+20.1i, 19.4+13.8i})));
    REQUIRE(tensor_close(a.sum(policy{},-1), tensor_type_1{{tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i},tensor_type_0{3.3+5.6i,4.5+4.4i,5.6+2.3i}},
        {tensor_type_0{3.4+6.6i,5.5+4.5i,4.6+2.5i},tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i}}}));
    REQUIRE(tensor_close(a.sum(policy{},{0,-1}),tensor_type_1{tensor_type_0{4.8+12.2i,10.9+10.1i,9.2+7.0i},tensor_type_0{4.7+11.2i,9.9+10.0i,10.2+6.8i}}));
    //cumsum
    REQUIRE(tensor_close(a.cumsum(policy{}),tensor_type_1{tensor_type_0{1.1+2.2i,2.2+1.1i,3.2+0.1i},tensor_type_0{1.3+4.4i,4.3+3.4i,3.4+2.4i},tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i},
        tensor_type_0{3.5+7.8i,6.6+6.7i,5.8+4.6i},tensor_type_0{3.6+9.0i,7.7+8.9i,7.0+6.7i},tensor_type_0{4.7+11.2i,9.9+10.0i,10.2+6.8i},tensor_type_0{4.9+13.4i,12.0+12.3i,10.4+9.1i},
        tensor_type_0{6.0+15.6i,14.2+13.4i,13.6+9.2i},tensor_type_0{8.1+17.8i,15.4+14.5i,14.8+9.3i},tensor_type_0{8.2+19.0i,16.5+16.7i,16.0+11.4i},
        tensor_type_0{9.3+21.2i,18.7+17.8i,19.2+11.5i},tensor_type_0{9.5+23.4i,20.8+20.1i,19.4+13.8i}}
    ));
    REQUIRE(tensor_close(a.cumsum(policy{},-1),tensor_type_1{{{tensor_type_0{1.1+2.2i,2.2+1.1i,3.2+0.1i},tensor_type_0{1.3+4.4i,4.3+3.4i,3.4+2.4i},tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i}},
        {tensor_type_0{2.1+2.2i,1.2+1.1i,1.2+0.1i},tensor_type_0{2.2+3.4i,2.3+3.3i,2.4+2.2i},tensor_type_0{3.3+5.6i,4.5+4.4i,5.6+2.3i}}},{{tensor_type_0{0.2+2.2i,2.1+2.3i,0.2+2.3i},
        tensor_type_0{1.3+4.4i,4.3+3.4i,3.4+2.4i},tensor_type_0{3.4+6.6i,5.5+4.5i,4.6+2.5i}},{tensor_type_0{0.1+1.2i,1.1+2.2i,1.2+2.1i},
        tensor_type_0{1.2+3.4i,3.3+3.3i,4.4+2.2i},tensor_type_0{1.4+5.6i,5.4+5.6i,4.6+4.5i}}}}
    ));
    //prod
    REQUIRE(tensor_close(a.prod(policy{}),tensor_type_1(tensor_type_0{-4625.00567341+4384.36281148i,-41998.46350343+11731.21360869i,-1176.87861768+26497.95898164i}),1E-6,1E-6));
    REQUIRE(tensor_close(a.prod(policy{},-1),tensor_type_1{{tensor_type_0{-3.894-5.258i,-13.915+12.705i,-15.006+9.717i},tensor_type_0{-8.701-2.332i,-6.655+7.26i,3.672+8.571i}},
        {tensor_type_0{-15.994-4.158i,-5.599+11.143i,-0.246+8.897i},tensor_type_0{-3.894-5.258i,-13.915+12.705i,-15.006+9.717i}}},1E-6,1E-6));
    REQUIRE(tensor_close(a.prod(policy{},{0,-1}),tensor_type_1{tensor_type_0{40.417872+100.287704i,-63.66173-226.19014i,-82.760673-135.898764i},
        tensor_type_0{21.620038+54.830666i,0.366025-185.574675i,-138.386439-92.935602i}},1E-6,1E-6
    ));
    //cumprod
    REQUIRE(tensor_close(a.cumprod(policy{}),tensor_type_1{tensor_type_0{1.10000000e+00+2.20000000e+00i,2.20000000e+00+1.10000000e+00i,3.20000000e+00+1.00000000e-01i},
        tensor_type_0{-4.62000000e+00+2.86000000e+00i,2.09000000e+00+7.37000000e+00i,4.10000000e-01+7.38000000e+00i},
        tensor_type_0{-3.89400000e+00-5.25800000e+00i,-1.39150000e+01+1.27050000e+01i,-1.50060000e+01+9.71700000e+00i},
        tensor_type_0{3.39020000e+00-1.96086000e+01i,-3.06735000e+01-6.05000000e-02i,-1.89789000e+01+1.01598000e+01i},
        tensor_type_0{2.38693400e+01+2.10738000e+00i,-3.36077500e+01-6.75482500e+01i,-4.41102600e+01-2.76639300e+01i},
        tensor_type_0{2.16200380e+01+5.48306660e+01i,3.66025000e-01-1.85574675e+02i,-1.38386439e+02-9.29356020e+01i},
        tensor_type_0{-1.16303458e+02+5.85302168e+01i,4.27590405e+02-3.88864960e+02i,1.86074597e+02-3.36875930e+02i},
        tensor_type_0{-2.56700280e+02-1.91484368e+02i,1.36845035e+03-3.85153467e+02i,6.29126303e+02-1.05939552e+03i},
        tensor_type_0{-1.17804979e+02-9.66857790e+02i,2.06580923e+03+1.04311122e+03i,8.60891115e+02-1.20836199e+03i},
        tensor_type_0{1.14844885e+03-2.38051753e+02i,-2.24545357e+01+5.69220265e+03i,3.57062952e+03+3.57836954e+02i},
        tensor_type_0{1.78700759e+03+2.26473054e+03i,-6.31082289e+03+1.24981458e+04i,1.13902308e+04+1.50214120e+03i},
        tensor_type_0{-4.62500567e+03+4.38436281e+03i,-4.19984635e+04+1.17312136e+04i,-1.17687862e+03+2.64979590e+04i}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(a.cumprod(policy{},-1),tensor_type_1{{{tensor_type_0{1.1+2.2i,2.2+1.1i,3.2+0.1i},tensor_type_0{-4.62+2.86i,2.09+7.37i,0.41+7.38i},
        tensor_type_0{-3.894-5.258i,-13.915+12.705i,-15.006+9.717i}},{tensor_type_0{2.1+2.2i,1.2+1.1i,1.2+0.1i},tensor_type_0{-2.43+2.74i,-1.1+3.85i,1.23+2.64i},
        tensor_type_0{-8.701-2.332i,-6.655+7.26i,3.672+8.571i}}},{{tensor_type_0{0.2+2.2i,2.1+2.3i,0.2+2.3i},tensor_type_0{-4.62+2.86i,2.09+7.37i,0.41+7.38i},
        tensor_type_0{-15.994-4.158i,-5.599+11.143i,-0.246+8.897i}},{tensor_type_0{0.1+1.2i,1.1+2.2i,1.2+2.1i},tensor_type_0{-2.53+1.54i,0.0+6.05i,3.63+6.84i},
        tensor_type_0{-3.894-5.258i,-13.915+12.705i,-15.006+9.717i}}}},1E-6,1E-6
    ));
    //diff
    REQUIRE(tensor_close(diff(policy{},a),tensor_type_1{{{tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i},tensor_type_0{-0.1-1.0i,-1.0-0.1i,1.0-0.2i}},
    {tensor_type_0{-2.0-1.0i,-0.1+1.1i,0.0+2.0i},tensor_type_0{1.0+1.0i,1.1-1.1i,2.0-2.0i}}},{{tensor_type_0{0.9+0.0i,0.1-1.2i,3.0-2.2i},
    tensor_type_0{1.0+0.0i,-1.0+0.0i,-2.0+0.0i}},{tensor_type_0{1.0+1.0i,1.1-1.1i,2.0-2.0i},tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i}}}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(diff(policy{},a.reshape(-1)),tensor_type_1{tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i},tensor_type_0{-0.1-1.0i,-1.0-0.1i,1.0-0.2i},
        tensor_type_0{2.0+1.0i,0.1-1.1i,0.0-2.0i},tensor_type_0{-2.0-1.0i,-0.1+1.1i,0.0+2.0i},tensor_type_0{1.0+1.0i,1.1-1.1i,2.0-2.0i},tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i},
        tensor_type_0{0.9+0.0i,0.1-1.2i,3.0-2.2i},tensor_type_0{1.0+0.0i,-1.0+0.0i,-2.0+0.0i},tensor_type_0{-2.0-1.0i,-0.1+1.1i,0.0+2.0i},tensor_type_0{1.0+1.0i,1.1-1.1i,2.0-2.0i},
        tensor_type_0{-0.9+0.0i,-0.1+1.2i,-3.0+2.2i}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(diff(policy{},a.reshape(-1),2),tensor_type_1{tensor_type_0{0.8-1.0i,-0.9-1.3i,4.0-2.4i},tensor_type_0{2.1+2.0i,1.1-1.0i,-1.0-1.8i},
        tensor_type_0{-4.0-2.0i,-0.2+2.2i,0.0+4.0i},tensor_type_0{3.0+2.0i,1.2-2.2i,2.0-4.0i},tensor_type_0{-1.9-1.0i,-1.2+2.3i,-5.0+4.2i},tensor_type_0{1.8+0.0i,0.2-2.4i,6.0-4.4i},
        tensor_type_0{0.1+0.0i,-1.1+1.2i,-5.0+2.2i},tensor_type_0{-3.0-1.0i,0.9+1.1i,2.0+2.0i},tensor_type_0{3.0+2.0i,1.2-2.2i,2.0-4.0i},tensor_type_0{-1.9-1.0i,-1.2+2.3i,-5.0+4.2i}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(diff2(policy{},a.reshape(-1)),tensor_type_1{tensor_type_0{0.8-1.0i,-0.9-1.3i,4.0-2.4i},tensor_type_0{2.1+2.0i,1.1-1.0i,-1.0-1.8i},
        tensor_type_0{-4.0-2.0i,-0.2+2.2i,0.0+4.0i},tensor_type_0{3.0+2.0i,1.2-2.2i,2.0-4.0i},tensor_type_0{-1.9-1.0i,-1.2+2.3i,-5.0+4.2i},tensor_type_0{1.8+0.0i,0.2-2.4i,6.0-4.4i},
        tensor_type_0{0.1+0.0i,-1.1+1.2i,-5.0+2.2i},tensor_type_0{-3.0-1.0i,0.9+1.1i,2.0+2.0i},tensor_type_0{3.0+2.0i,1.2-2.2i,2.0-4.0i},tensor_type_0{-1.9-1.0i,-1.2+2.3i,-5.0+4.2i}},1E-6,1E-6
    ));

    //matmul
    REQUIRE(tensor_close(matmul(a(0),a(1).transpose()),tensor_type_1{{tensor_type_0{-11.67 +8.46i,3.08+18.59i,2.05+17.4i},tensor_type_0{-9.77+4.86i,-0.66+20.57i,-0.55+17.4i}},
        {tensor_type_0{-9.48+13.64i,1.42+14.86i,7.47+10.06i},tensor_type_0{-9.58 +7.14i,0.99+17.27i,5.27+16.86i}}},1E-6,1E-6
    ));

    //mean
    REQUIRE(tensor_close(a.mean(policy{}),tensor_type_1(tensor_type_0{0.79166667+1.95i,1.73333333+1.675i,1.61666667+1.15i}),1E-6,1E-6));
    REQUIRE(tensor_close(a.mean(policy{},-1),tensor_type_1{{tensor_type_0{0.46666667+1.86666667i,1.8+1.86666667i,1.53333333+1.5i},tensor_type_0{1.1+1.86666667i,1.5+1.46666667i,1.86666667+0.76666667i}},
        {tensor_type_0{1.13333333+2.2i,1.83333333+1.5i,1.53333333+0.83333333i},tensor_type_0{0.46666667+1.86666667i,1.8+1.86666667i,1.53333333+1.5i}}},1E-6,1E-6
    ));
    REQUIRE(tensor_close(a.mean(policy{},{0,-1}),tensor_type_1{tensor_type_0{0.8+2.03333333i,1.81666667+1.68333333i,1.53333333+1.16666667i},tensor_type_0{0.78333333+1.86666667i,1.65+1.66666667i,1.7+1.13333333i}},1E-6,1E-6));
    //var
    REQUIRE(tensor_close(a.var(policy{}),tensor<tensor<double>>(tensor<double>{0.71159722,0.58576389,2.51722222}),1E-6,1E-6));
    REQUIRE(tensor_close(a.var(policy{},-1),tensor<tensor<double>>{{tensor<double>{0.42444444,0.54222222,2.54222222},tensor<double>{0.88888889,0.51555556,1.77777778}},
        {tensor<double>{0.60222222,0.52222222,2.63111111},tensor<double>{0.42444444,0.54222222,2.54222222}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.var(policy{},{0,-1}),tensor<tensor<double>>{tensor<double>{0.65222222,0.56611111,2.69777778},tensor<double>{0.75694444,0.59138889,2.32222222}},1E-6,1E-6));
    //stdev
    REQUIRE(tensor_close(a.stdev(policy{}),tensor<tensor<double>>(tensor<double>{0.84356222,0.76535213,1.58657563}),1E-6,1E-6));
    REQUIRE(tensor_close(a.stdev(policy{},-1),tensor<tensor<double>>{{tensor<double>{0.65149401,0.7363574,1.59443477},tensor<double>{0.94280904,0.71802197,1.33333333}},
        {tensor<double>{0.77602978,0.72264945,1.62207001},tensor<double>{0.65149401,0.7363574,1.59443477}}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(a.stdev(policy{},{0,-1}),tensor<tensor<double>>{tensor<double>{0.80760276,0.75240356,1.64249133},tensor<double>{0.87002554,0.76901813,1.52388393}},1E-6,1E-6));
}

