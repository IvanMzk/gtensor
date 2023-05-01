#include <iostream>
#include "catch.hpp"
#include "common.hpp"
#include "tensor.hpp"


TEST_CASE("test_is_iterator","[test_common]")
{
    using gtensor::detail::is_iterator_v;
    REQUIRE(!is_iterator_v<int>);
    REQUIRE(!is_iterator_v<std::vector<int>>);
    REQUIRE(is_iterator_v<std::vector<int>::iterator>);
}

TEST_CASE("test_is_tensor","[test_common]")
{
    using gtensor::tensor;
    using gtensor::integral;
    using gtensor::detail::is_tensor_of_type_v;
    using gtensor::detail::is_bool_tensor_v;

    REQUIRE(is_tensor_of_type_v<tensor<int>,int>);
    REQUIRE(is_tensor_of_type_v<tensor<integral<std::int64_t>>,integral<std::int64_t>>);
    REQUIRE(is_tensor_of_type_v<tensor<int>,integral<std::int64_t>>);
    REQUIRE(is_tensor_of_type_v<tensor<std::size_t>,std::int64_t>);
    REQUIRE(is_tensor_of_type_v<tensor<bool>,std::int64_t>);
    REQUIRE(is_tensor_of_type_v<tensor<bool>,int>);

    REQUIRE(!is_tensor_of_type_v<tensor<float>,integral<std::int64_t>>);

    REQUIRE(!is_tensor_of_type_v<tensor<integral<std::int64_t>>,std::int64_t>);
    REQUIRE(!is_tensor_of_type_v<std::vector<int>,int>);
    REQUIRE(!is_tensor_of_type_v<std::string,int>);
    REQUIRE(!is_tensor_of_type_v<std::vector<bool>,int>);

    REQUIRE(is_bool_tensor_v<tensor<bool>>);
    REQUIRE(!is_bool_tensor_v<tensor<int>>);
    REQUIRE(!is_bool_tensor_v<tensor<float>>);
    REQUIRE(!is_bool_tensor_v<std::vector<int>>);
    REQUIRE(!is_bool_tensor_v<std::string>);
    REQUIRE(!is_bool_tensor_v<std::vector<bool>>);
}

namespace test_has_member_function{

    struct test_type{
        void f(){std::cout<<std::endl<<"f";};
        int g()const;
        double g();
        int h(double) const;
        double h(int);

        int* begin();

        const int* rbegin()const;

        using size_type = std::size_t;
        int operator[](size_type);
    };

    struct public_derived_type : public test_type{};
    struct private_derived_type : private test_type{
        using test_type::f;
        using test_type::g;
    };


    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(f,void(T::*)(),has_f);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(f,void(T::*)()const,has_f_const);

    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(g,decltype(std::declval<const T&>().g())(T::*)()const,has_g_const);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(g,decltype(std::declval<T>().g())(T::*)(),has_g);

    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(h,decltype(std::declval<const T&>().h(std::declval<double>()))(T::*)(double)const,has_h_double_const);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(h,decltype(std::declval<const T&>().h(std::declval<double>()))(T::*)(int)const,has_h_int_const);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(h,decltype(std::declval<T>().h(std::declval<int>()))(T::*)(int),has_h_int);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(h,decltype(std::declval<T>().h(std::declval<std::int64_t>()))(T::*)(std::int64_t),has_h_int64);


    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(begin,decltype(std::declval<T&>().begin())(T::*)(),has_begin);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(begin,decltype(std::declval<const T&>().begin())(T::*)()const,has_begin_const);

    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(rbegin,decltype(std::declval<T&>().rbegin())(T::*)(),has_rbegin);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(rbegin,decltype(std::declval<const T&>().rbegin())(T::*)()const,has_rbegin_const);

    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(operator[],decltype(std::declval<T>()[std::declval<typename T::size_type>()])(T::*)(typename T::size_type),has_subscript_operator);
}

TEST_CASE("test_has_member_function","[test_common]")
{
    using test_has_member_function::test_type;
    using test_has_member_function::public_derived_type;
    using test_has_member_function::private_derived_type;

    REQUIRE(test_has_member_function::has_f<test_type>{}());
    REQUIRE(!test_has_member_function::has_f_const<test_type>{}());
    REQUIRE(test_has_member_function::has_f<public_derived_type>{}());
    REQUIRE(!test_has_member_function::has_f_const<public_derived_type>{}());

    REQUIRE(!test_has_member_function::has_f<private_derived_type>{}());
    REQUIRE(!test_has_member_function::has_f_const<private_derived_type>{}());

    REQUIRE(test_has_member_function::has_g_const<test_type>{}());
    REQUIRE(test_has_member_function::has_g<test_type>{}());
    REQUIRE(test_has_member_function::has_g_const<public_derived_type>{}());
    REQUIRE(test_has_member_function::has_g<public_derived_type>{}());
    REQUIRE(!test_has_member_function::has_g_const<private_derived_type>{}());
    REQUIRE(!test_has_member_function::has_g<private_derived_type>{}());

    REQUIRE(test_has_member_function::has_h_double_const<test_type>{}());
    REQUIRE(!test_has_member_function::has_h_int_const<test_type>{}());
    REQUIRE(test_has_member_function::has_h_int<test_type>{}());
    REQUIRE(!test_has_member_function::has_h_int64<test_type>{}());

    REQUIRE(test_has_member_function::has_subscript_operator<test_type>{}());

    REQUIRE(test_has_member_function::has_begin<test_type>{}());
    REQUIRE(!test_has_member_function::has_begin_const<test_type>{}());

    REQUIRE(!test_has_member_function::has_rbegin<test_type>{}());
    REQUIRE(test_has_member_function::has_rbegin_const<test_type>{}());

}

TEST_CASE("test_is_static_castable","[test_common]")
{
    using gtensor::detail::is_static_castable_v;

    struct A{};
    struct B{
        operator A(){return A{};}
    };
    REQUIRE(is_static_castable_v<int,int>);
    REQUIRE(is_static_castable_v<int,const int&>);
    REQUIRE(is_static_castable_v<int,int&&>);
    REQUIRE(is_static_castable_v<int,double>);
    REQUIRE(is_static_castable_v<double,int>);
    REQUIRE(is_static_castable_v<int&,int>);
    REQUIRE(is_static_castable_v<int&,int&>);
    REQUIRE(is_static_castable_v<int&,const int&>);
    REQUIRE(is_static_castable_v<int&,int&&>);
    REQUIRE(is_static_castable_v<int&&,int>);
    REQUIRE(is_static_castable_v<int&&,const int&>);
    REQUIRE(is_static_castable_v<int&&,int&&>);
    REQUIRE(is_static_castable_v<B,A>);

    REQUIRE(!is_static_castable_v<int&&,int&>);
    REQUIRE(!is_static_castable_v<int,int&>);
    REQUIRE(!is_static_castable_v<int,int*>);
    REQUIRE(!is_static_castable_v<int*,int>);
    REQUIRE(!is_static_castable_v<A,B>);
}