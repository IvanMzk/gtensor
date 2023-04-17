#include "catch.hpp"
#include "forward_decl.hpp"
#include "common.hpp"


TEST_CASE("test_is_iterator","[test_tensor]")
{
    using gtensor::detail::is_iterator;
    REQUIRE(!is_iterator<int>);
    REQUIRE(!is_iterator<std::vector<int>>);
    REQUIRE(is_iterator<std::vector<int>::iterator>);
}

// TEST_CASE("test_is_tensor","[test_tensor]")
// {
//     using gtensor::tensor;
//     using gtensor::integral;
//     using gtensor::detail::is_tensor_of_type_v;
//     using gtensor::detail::is_bool_tensor_v;

//     REQUIRE(is_tensor_of_type_v<tensor<int>,int>);
//     REQUIRE(is_tensor_of_type_v<tensor<integral<std::int64_t>>,integral<std::int64_t>>);
//     REQUIRE(is_tensor_of_type_v<tensor<int>,integral<std::int64_t>>);
//     REQUIRE(is_tensor_of_type_v<tensor<std::size_t>,std::int64_t>);
//     REQUIRE(is_tensor_of_type_v<tensor<bool>,std::int64_t>);
//     REQUIRE(is_tensor_of_type_v<tensor<bool>,int>);

//     REQUIRE(!is_tensor_of_type_v<tensor<float>,integral<std::int64_t>>);

//     REQUIRE(!is_tensor_of_type_v<tensor<integral<std::int64_t>>,std::int64_t>);
//     REQUIRE(!is_tensor_of_type_v<std::vector<int>,int>);
//     REQUIRE(!is_tensor_of_type_v<std::string,int>);
//     REQUIRE(!is_tensor_of_type_v<std::vector<bool>,int>);

//     REQUIRE(is_bool_tensor_v<tensor<bool>>);
//     REQUIRE(!is_bool_tensor_v<tensor<int>>);
//     REQUIRE(!is_bool_tensor_v<tensor<float>>);
//     REQUIRE(!is_bool_tensor_v<std::vector<int>>);
//     REQUIRE(!is_bool_tensor_v<std::string>);
//     REQUIRE(!is_bool_tensor_v<std::vector<bool>>);
// }

namespace test_has_member_function{

    struct test_type{
        void f();
        int g()const;
        double g();
        int h(double) const;
        double h(double);
    };

    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(f,void(T::*)(),f);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(f,void(T::*)()const,f_const);

    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(g,decltype(std::declval<const T&>().g())(T::*)()const,g_const);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(g,decltype(std::declval<T>().g())(T::*)(),g);

    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(h,decltype(std::declval<const T&>().h(std::declval<double>()))(T::*)(double)const,h_double_const);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(h,decltype(std::declval<const T&>().h(std::declval<double>()))(T::*)(int)const,h_int_const);
    GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(h,decltype(std::declval<T>().h(std::declval<double>()))(T::*)(double),h_double);
}

TEST_CASE("test_has_member_function","[test_tensor]")
{
    using test_has_member_function::test_type;

    REQUIRE(test_has_member_function::has_member_function_f<test_type>{}());
    REQUIRE(!test_has_member_function::has_member_function_f_const<test_type>{}());

    REQUIRE(test_has_member_function::has_member_function_g_const<test_type>{}());
    REQUIRE(test_has_member_function::has_member_function_g<test_type>{}());

    REQUIRE(test_has_member_function::has_member_function_h_double_const<test_type>{}());
    REQUIRE(test_has_member_function::has_member_function_h_double<test_type>{}());
    REQUIRE(!test_has_member_function::has_member_function_h_int_const<test_type>{}());


    // REQUIRE(!test_has_member_function::has_member_function_h_const<test_type>{}());
    // REQUIRE(!test_has_member_function::has_member_function_h_int_const<test_type>{}());
    // REQUIRE(!test_has_member_function::has_member_function_h_crdouble_const<test_type>{}());


}