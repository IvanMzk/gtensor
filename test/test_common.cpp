#include <iostream>
#include <list>
#include "catch.hpp"
#include "common.hpp"
#include "tensor.hpp"


TEST_CASE("test_is_tensor_of_type","[test_common]")
{
    using gtensor::tensor;
    using gtensor::detail::is_tensor_of_type_v;
    using gtensor::detail::is_bool_tensor_v;

    REQUIRE(is_tensor_of_type_v<tensor<int>,int>);
    REQUIRE(is_tensor_of_type_v<tensor<std::size_t>,std::int64_t>);
    REQUIRE(is_tensor_of_type_v<tensor<bool>,std::int64_t>);
    REQUIRE(is_tensor_of_type_v<tensor<bool>,int>);
    REQUIRE(is_tensor_of_type_v<tensor<double>,std::int64_t>);

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

TEST_CASE("test_is_iterator","[test_common]")
{
    using gtensor::detail::is_iterator_v;
    REQUIRE(!is_iterator_v<int>);
    REQUIRE(!is_iterator_v<std::vector<int>>);
    REQUIRE(is_iterator_v<const double*>);
    REQUIRE(is_iterator_v<std::string*>);
    REQUIRE(is_iterator_v<std::vector<int>::iterator>);
    REQUIRE(is_iterator_v<std::vector<int>::const_iterator>);
    REQUIRE(is_iterator_v<std::list<int>::iterator>);
    REQUIRE(is_iterator_v<std::list<int>::const_iterator>);
}

TEST_CASE("test_is_random_access_iterator","[test_common]")
{
    using gtensor::detail::is_random_access_iterator_v;
    REQUIRE(!is_random_access_iterator_v<int>);
    REQUIRE(!is_random_access_iterator_v<std::vector<int>>);
    REQUIRE(!is_random_access_iterator_v<std::list<int>::iterator>);
    REQUIRE(!is_random_access_iterator_v<std::list<int>::const_iterator>);
    REQUIRE(is_random_access_iterator_v<const double*>);
    REQUIRE(is_random_access_iterator_v<std::string*>);
    REQUIRE(is_random_access_iterator_v<std::vector<int>::iterator>);
    REQUIRE(is_random_access_iterator_v<std::vector<int>::const_iterator>);
}

TEST_CASE("test_has_callable_iterator","[test_tensor_implementation]")
{
    struct const_iterable{
        const int* begin()const{return nullptr;}
        const int* end()const{return nullptr;}
    };
    struct non_const_iterable{
        int* begin(){return nullptr;}
        int* end(){return nullptr;}
    };
    struct iterable{
        const int* begin()const{return nullptr;}
        const int* end()const{return nullptr;}
        int* begin(){return nullptr;}
        int* end(){return nullptr;}
    };

    using gtensor::detail::has_callable_iterator;
    REQUIRE(!has_callable_iterator<void>::value);
    REQUIRE(!has_callable_iterator<void*>::value);
    REQUIRE(!has_callable_iterator<int>::value);
    REQUIRE(!has_callable_iterator<int*>::value);
    REQUIRE(!has_callable_iterator<int[]>::value);
    REQUIRE(!has_callable_iterator<const non_const_iterable>::value);

    REQUIRE(has_callable_iterator<non_const_iterable>::value);
    REQUIRE(has_callable_iterator<const_iterable>::value);
    REQUIRE(has_callable_iterator<const const_iterable>::value);
    REQUIRE(has_callable_iterator<iterable>::value);
    REQUIRE(has_callable_iterator<const iterable>::value);
    REQUIRE(has_callable_iterator<std::list<int>>::value);
    REQUIRE(has_callable_iterator<std::vector<int>>::value);
    REQUIRE(has_callable_iterator<std::string>::value);
}

TEST_CASE("test_has_callable_random_access_iterator","[test_tensor_implementation]")
{
    struct const_iterable{
        const int* begin()const{return nullptr;}
        const int* end()const{return nullptr;}
    };
    struct non_const_iterable{
        int* begin(){return nullptr;}
        int* end(){return nullptr;}
    };
    struct iterable{
        const int* begin()const{return nullptr;}
        const int* end()const{return nullptr;}
        int* begin(){return nullptr;}
        int* end(){return nullptr;}
    };

    using gtensor::detail::has_callable_random_access_iterator;
    REQUIRE(!has_callable_random_access_iterator<void>::value);
    REQUIRE(!has_callable_random_access_iterator<void*>::value);
    REQUIRE(!has_callable_random_access_iterator<int>::value);
    REQUIRE(!has_callable_random_access_iterator<int*>::value);
    REQUIRE(!has_callable_random_access_iterator<int[]>::value);
    REQUIRE(!has_callable_random_access_iterator<const non_const_iterable>::value);
    REQUIRE(!has_callable_random_access_iterator<std::list<int>>::value);

    REQUIRE(has_callable_random_access_iterator<non_const_iterable>::value);
    REQUIRE(has_callable_random_access_iterator<const_iterable>::value);
    REQUIRE(has_callable_random_access_iterator<const const_iterable>::value);
    REQUIRE(has_callable_random_access_iterator<iterable>::value);
    REQUIRE(has_callable_random_access_iterator<const iterable>::value);
    REQUIRE(has_callable_random_access_iterator<std::vector<int>>::value);
    REQUIRE(has_callable_random_access_iterator<std::string>::value);
}

TEST_CASE("test_has_callable_reverse_iterator","[test_tensor_implementation]")
{
    struct const_reverse_iterable{
        const int* rbegin()const{return nullptr;}
        const int* rend()const{return nullptr;}
    };
    struct non_const_reverse_iterable{
        int* rbegin(){return nullptr;}
        int* rend(){return nullptr;}
    };
    struct reverse_iterable{
        const int* rbegin()const{return nullptr;}
        const int* rend()const{return nullptr;}
        int* rbegin(){return nullptr;}
        int* rend(){return nullptr;}
    };

    using gtensor::detail::has_callable_reverse_iterator;
    REQUIRE(!has_callable_reverse_iterator<void>::value);
    REQUIRE(!has_callable_reverse_iterator<void*>::value);
    REQUIRE(!has_callable_reverse_iterator<int>::value);
    REQUIRE(!has_callable_reverse_iterator<int*>::value);
    REQUIRE(!has_callable_reverse_iterator<int[]>::value);
    REQUIRE(!has_callable_reverse_iterator<const non_const_reverse_iterable>::value);

    REQUIRE(has_callable_reverse_iterator<non_const_reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<const_reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<const const_reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<const reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<std::list<int>>::value);
    REQUIRE(has_callable_reverse_iterator<std::vector<int>>::value);
    REQUIRE(has_callable_reverse_iterator<std::string>::value);
}

TEST_CASE("test_has_callable_subscript_operator","[test_tensor_implementation]")
{
    struct const_subscriptable{
        using size_type = std::size_t;
        int operator[](size_type)const;
    };
    struct non_const_subscriptable{
        using difference_type = std::ptrdiff_t;
        int& operator[](difference_type);
    };
    struct subscriptable{
        using size_type = std::size_t;
        const int& operator[](size_type)const;
        int& operator[](size_type);
    };

    using gtensor::detail::has_callable_subscript_operator;
    REQUIRE(!has_callable_subscript_operator<void>());
    REQUIRE(!has_callable_subscript_operator<void*>());
    REQUIRE(!has_callable_subscript_operator<int>());
    REQUIRE(!has_callable_subscript_operator<int*>());
    REQUIRE(!has_callable_subscript_operator<int[]>());
    REQUIRE(!has_callable_subscript_operator<std::list<int>>());
    REQUIRE(!has_callable_subscript_operator<const non_const_subscriptable>());

    REQUIRE(has_callable_subscript_operator<non_const_subscriptable>());
    REQUIRE(has_callable_subscript_operator<const_subscriptable>());
    REQUIRE(has_callable_subscript_operator<const const_subscriptable>());
    REQUIRE(has_callable_subscript_operator<subscriptable>());
    REQUIRE(has_callable_subscript_operator<const subscriptable>());
    REQUIRE(has_callable_subscript_operator<std::vector<int>>());
    REQUIRE(has_callable_subscript_operator<std::string>());
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

    GENERATE_HAS_METHOD_SIGNATURE(f,void(T::*)(),has_f);
    GENERATE_HAS_METHOD_SIGNATURE(f,void(T::*)()const,has_f_const);

    GENERATE_HAS_METHOD_SIGNATURE(g,decltype(std::declval<T>().g())(T::*)(),has_g);
    GENERATE_HAS_METHOD_SIGNATURE(g,decltype(std::declval<const T&>().g())(T::*)()const,has_g_const);

    GENERATE_HAS_METHOD_SIGNATURE(h,decltype(std::declval<const T&>().h(std::declval<double>()))(T::*)(double)const,has_h_double_const);

    GENERATE_HAS_METHOD_SIGNATURE(h,decltype(std::declval<const T&>().h(std::declval<double>()))(T::*)(int)const,has_h_int_const);


    GENERATE_HAS_METHOD_SIGNATURE(h,decltype(std::declval<T>().h(std::declval<int>()))(T::*)(int),has_h_int);
    GENERATE_HAS_METHOD_SIGNATURE(h,decltype(std::declval<T>().h(std::declval<std::int64_t>()))(T::*)(std::int64_t),has_h_int64);


    GENERATE_HAS_METHOD_SIGNATURE(begin,decltype(std::declval<T&>().begin())(T::*)(),has_begin);
    GENERATE_HAS_METHOD_SIGNATURE(begin,decltype(std::declval<const T&>().begin())(T::*)()const,has_begin_const);

    GENERATE_HAS_METHOD_SIGNATURE(rbegin,decltype(std::declval<T&>().rbegin())(T::*)(),has_rbegin);
    GENERATE_HAS_METHOD_SIGNATURE(rbegin,decltype(std::declval<const T&>().rbegin())(T::*)()const,has_rbegin_const);

    GENERATE_HAS_METHOD_SIGNATURE(operator[],decltype(std::declval<T>()[std::declval<typename T::size_type>()])(T::*)(typename T::size_type),has_subscript_operator);
}

TEST_CASE("test_has_member_function","[test_common]")
{
    using test_has_member_function::test_type;

    REQUIRE(test_has_member_function::has_f<test_type>{}());
    REQUIRE(!test_has_member_function::has_f_const<test_type>{}());


    REQUIRE(test_has_member_function::has_g<test_type>{}());
    REQUIRE(test_has_member_function::has_g_const<test_type>{}());

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