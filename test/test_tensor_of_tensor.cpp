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

TEST_CASE("test_tensor_of_tensor_broadcast_complex_routines","[test_tensor_of_tensor]")
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

// TEST_CASE("test_tensor_of_tensor_routines","[test_tensor_of_tensor]")
// {
//     using gtensor::tensor;
//     using value_type = std::complex<double>;
//     using tensor_type = gtensor::tensor<value_type>;
//     using namespace std::complex_literals;

//     const tensor_type a{{{1.1+2.2i,2.2+1.1i,1.5+0.3i},{3.3+4.4i,4.4+3.3i,0.2+0.7i}},{{1.6+2.3i,2.1+1.2i,1.9+0.3i},{3.5+4.1i,4.4+3.0i,0.2+1.7i}}};

//     //sum
//     REQUIRE(tensor_close(a.sum(),tensor_type(26.400000000000002+24.6i)));
//     REQUIRE(tensor_close(a.sum(-1),tensor_type{{4.8+3.6i,7.9+8.4i},{5.6+3.8i,8.1+8.8i}}));
//     REQUIRE(tensor_close(a.sum({0,-1}),tensor_type{10.4+7.4i,16.0+17.2i}));
//     //cumsum
//     REQUIRE(tensor_close(a.cumsum(),tensor_type{1.1+2.2i,3.3+3.3i,4.8+3.6i,8.1+8.0i,12.5+11.3i,12.7+12.0i,14.3+14.3i,16.4+15.5i,18.3+15.8i,21.8+19.9i,26.2+22.9i,26.4+24.6i}));
//     REQUIRE(tensor_close(a.cumsum(-1),tensor_type{{{1.1+2.2i,3.3+3.3i,4.8+3.6i},{3.3+4.4i,7.7+7.7i,7.9+8.4i}},{{1.6+2.3i,3.7+3.5i,5.6+3.8i},{3.5+4.1i,7.9+7.1i,8.1+8.8i}}}));
//     //prod
//     REQUIRE(tensor_close(a.prod(),tensor_type(-126861.3528684975+30811.523198692543i)));
//     REQUIRE(tensor_close(a.prod(-1),tensor_type{{-1.815+9.075i,-21.175+6.05i},{-0.885+13.005i,-47.898+10.978i}}));
//     REQUIRE(tensor_close(a.prod({0,-1}),tensor_type{-116.4141-31.63545i,947.82325-522.24205i}));
//     //cumprod
//     REQUIRE(tensor_close(
//         a.cumprod(),
//         tensor_type{1.10000000e+00+2.20000000e+00i,0.00000000e+00+6.05000000e+00i,-1.81500000e+00+9.07500000e+00i,-4.59195000e+01+2.19615000e+01i,-2.74518750e+02-5.49037500e+01i,-1.64711250e+01-2.03143875e+02i,4.40877113e+02-3.62913788e+02i,1.36133848e+03-2.33066419e+02i,2.65646304e+03-3.44246513e+01i,9.43876171e+03+1.07710122e+04i,9.21751497e+03+7.57087387e+04i,-1.26861353e+05+3.08115232e+04i},
//         1E-6,
//         1E-6
//     ));
//     REQUIRE(tensor_close(a.cumprod(-1),tensor_type{{{1.1+2.2i,0.0+6.05i,-1.815+9.075i},{3.3+4.4i,0.0+30.25i,-21.175+6.05i}},{{1.6+2.3i,0.6+6.75i,-0.885+13.005i},{3.5+4.1i,3.1+28.54i,-47.898+10.978i}}}));
//     //diff
//     REQUIRE(tensor_close(diff(a),tensor_type{{{1.1-1.1i,-0.7-0.8i},{1.1-1.1i,-4.2-2.6i}},{{0.5-1.1i,-0.2-0.9i},{0.9-1.1i,-4.2-1.3i}}}));
//     //gradient
//     REQUIRE(tensor_close(gradient(a,-1),tensor_type{{{1.1-1.1i,0.2-0.95i,-0.7-0.8i},{1.1-1.1i,-1.55-1.85i,-4.2-2.6i}},{{0.5-1.1i,0.15-1.0i,-0.2-0.9i},{0.9-1.1i,-1.65-1.2i,-4.2-1.3i}}}));
//     //matmul
//     REQUIRE(tensor_close(matmul(a,tensor_type{{1.1+2.2i,2.2+1.1i},{3.3+4.4i,4.4+3.3i},{2.1+2.2i,2.1+1.1i}}),
//         tensor_type{{{1.28+22.08i,8.87+20.43i},{-7.17+44.26i,10.54+44.04i}},{{1.68+24.06i,9.93+21.75i},{-7.17+45.48i,11.2+44.38i}}}
//     ));

//     //mean
//     REQUIRE(tensor_close(a.mean(),tensor_type(2.2+2.05i)));
//     REQUIRE(tensor_close(a.mean(-1),tensor_type{{1.6+1.2i,2.63333333+2.8i},{1.86666667+1.26666667i,2.7+2.93333333i}},1E-6,1E-6));
//     REQUIRE(tensor_close(a.mean({0,-1}),tensor_type{1.73333333+1.23333333i,2.66666667+2.86666667i},1E-6,1E-6));
//     //var
//     REQUIRE(tensor_close(a.var(),tensor<double>(3.7258333333333336),1E-6,1E-6));
//     REQUIRE(tensor_close(a.var(-1),tensor<double>{{0.81333333,5.56888889},{0.71111111,4.22222222}},1E-6,1E-6));
//     REQUIRE(tensor_close(a.var({0,-1}),tensor<double>{0.78111111,4.90111111},1E-6,1E-6));
//     //stdev
//     REQUIRE(tensor_close(a.stdev(),tensor<double>(1.9302417810557655),1E-6,1E-6));
//     REQUIRE(tensor_close(a.stdev(-1),tensor<double>{{0.90184995,2.35984934},{0.84327404,2.05480467}},1E-6,1E-6));
//     REQUIRE(tensor_close(a.stdev({0,-1}),tensor<double>{0.88380491,2.21384532},1E-6,1E-6));
// }

// TEMPLATE_TEST_CASE("test_tensor_of_tensor_routines_policy","[test_tensor_of_tensor]",
//     multithreading::exec_pol<4>
// )
// {
//     using policy = TestType;
//     using gtensor::tensor;
//     using value_type = std::complex<double>;
//     using tensor_type = gtensor::tensor<value_type>;
//     using namespace std::complex_literals;

//     const tensor_type a{{{1.1+2.2i,2.2+1.1i,1.5+0.3i},{3.3+4.4i,4.4+3.3i,0.2+0.7i}},{{1.6+2.3i,2.1+1.2i,1.9+0.3i},{3.5+4.1i,4.4+3.0i,0.2+1.7i}}};

//     //sum
//     REQUIRE(tensor_close(a.sum(policy{}),tensor_type(26.400000000000002+24.6i)));
//     REQUIRE(tensor_close(a.sum(policy{},-1),tensor_type{{4.8+3.6i,7.9+8.4i},{5.6+3.8i,8.1+8.8i}}));
//     REQUIRE(tensor_close(a.sum(policy{},{0,-1}),tensor_type{10.4+7.4i,16.0+17.2i}));
//     //cumsum
//     REQUIRE(tensor_close(a.cumsum(policy{}),tensor_type{1.1+2.2i,3.3+3.3i,4.8+3.6i,8.1+8.0i,12.5+11.3i,12.7+12.0i,14.3+14.3i,16.4+15.5i,18.3+15.8i,21.8+19.9i,26.2+22.9i,26.4+24.6i}));
//     REQUIRE(tensor_close(a.cumsum(policy{},-1),tensor_type{{{1.1+2.2i,3.3+3.3i,4.8+3.6i},{3.3+4.4i,7.7+7.7i,7.9+8.4i}},{{1.6+2.3i,3.7+3.5i,5.6+3.8i},{3.5+4.1i,7.9+7.1i,8.1+8.8i}}}));
//     //prod
//     REQUIRE(tensor_close(a.prod(policy{}),tensor_type(-126861.3528684975+30811.523198692543i)));
//     REQUIRE(tensor_close(a.prod(policy{},-1),tensor_type{{-1.815+9.075i,-21.175+6.05i},{-0.885+13.005i,-47.898+10.978i}}));
//     REQUIRE(tensor_close(a.prod(policy{},{0,-1}),tensor_type{-116.4141-31.63545i,947.82325-522.24205i}));
//     //cumprod
//     REQUIRE(tensor_close(
//         a.cumprod(policy{}),
//         tensor_type{1.10000000e+00+2.20000000e+00i,0.00000000e+00+6.05000000e+00i,-1.81500000e+00+9.07500000e+00i,-4.59195000e+01+2.19615000e+01i,-2.74518750e+02-5.49037500e+01i,-1.64711250e+01-2.03143875e+02i,4.40877113e+02-3.62913788e+02i,1.36133848e+03-2.33066419e+02i,2.65646304e+03-3.44246513e+01i,9.43876171e+03+1.07710122e+04i,9.21751497e+03+7.57087387e+04i,-1.26861353e+05+3.08115232e+04i},
//         1E-6,
//         1E-6
//     ));
//     REQUIRE(tensor_close(a.cumprod(policy{},-1),tensor_type{{{1.1+2.2i,0.0+6.05i,-1.815+9.075i},{3.3+4.4i,0.0+30.25i,-21.175+6.05i}},{{1.6+2.3i,0.6+6.75i,-0.885+13.005i},{3.5+4.1i,3.1+28.54i,-47.898+10.978i}}}));
//     //diff
//     REQUIRE(tensor_close(diff(policy{},a),tensor_type{{{1.1-1.1i,-0.7-0.8i},{1.1-1.1i,-4.2-2.6i}},{{0.5-1.1i,-0.2-0.9i},{0.9-1.1i,-4.2-1.3i}}}));
//     //gradient
//     REQUIRE(tensor_close(gradient(policy{},a,-1),tensor_type{{{1.1-1.1i,0.2-0.95i,-0.7-0.8i},{1.1-1.1i,-1.55-1.85i,-4.2-2.6i}},{{0.5-1.1i,0.15-1.0i,-0.2-0.9i},{0.9-1.1i,-1.65-1.2i,-4.2-1.3i}}}));

//     //mean
//     REQUIRE(tensor_close(a.mean(policy{}),tensor_type(2.2+2.05i)));
//     REQUIRE(tensor_close(a.mean(policy{},-1),tensor_type{{1.6+1.2i,2.63333333+2.8i},{1.86666667+1.26666667i,2.7+2.93333333i}},1E-6,1E-6));
//     REQUIRE(tensor_close(a.mean(policy{},{0,-1}),tensor_type{1.73333333+1.23333333i,2.66666667+2.86666667i},1E-6,1E-6));
//     //var
//     REQUIRE(tensor_close(a.var(policy{}),tensor<double>(3.7258333333333336),1E-6,1E-6));
//     REQUIRE(tensor_close(a.var(policy{},-1),tensor<double>{{0.81333333,5.56888889},{0.71111111,4.22222222}},1E-6,1E-6));
//     REQUIRE(tensor_close(a.var(policy{},{0,-1}),tensor<double>{0.78111111,4.90111111},1E-6,1E-6));
//     //stdev
//     REQUIRE(tensor_close(a.stdev(policy{}),tensor<double>(1.9302417810557655),1E-6,1E-6));
//     REQUIRE(tensor_close(a.stdev(policy{},-1),tensor<double>{{0.90184995,2.35984934},{0.84327404,2.05480467}},1E-6,1E-6));
//     REQUIRE(tensor_close(a.stdev(policy{},{0,-1}),tensor<double>{0.88380491,2.21384532},1E-6,1E-6));
// }

