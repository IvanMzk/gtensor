#include <tuple>
#include <iostream>
#include <sstream>
#include "catch.hpp"
#include "helpers_for_testing.hpp"

namespace test_helpers_for_testing{
struct A{};
struct B{};
struct C{};
struct D{};
struct E{};
struct F{};

template<typename...Ts>
struct type_list{
    using type = type_list<Ts...>;
    static constexpr std::size_t size = sizeof...(Ts);
};

template<typename F, typename S> struct type_pair{
    using first_type = F;
    using second_type = S;
};

}   //end of namespace test_helpers_for_testing

TEST_CASE("test_list_concat","[test_helpers_for_testing]"){
    using helpers_for_testing::list_concat;
    using test_helpers_for_testing::type_list;
    using test_helpers_for_testing::A;
    using test_helpers_for_testing::B;
    using test_helpers_for_testing::C;
    using test_helpers_for_testing::D;
    using test_helpers_for_testing::E;
    using test_helpers_for_testing::F;

    using l1 = type_list<A>;
    using l2 = type_list<B,C>;
    using l3 = type_list<D,E,F>;
    using l4 = type_list<>;

    REQUIRE(std::is_same_v<list_concat<l1,l2>::type, type_list<A,B,C>>);
    REQUIRE(std::is_same_v<list_concat<l2,l1>::type, type_list<B,C,A>>);
    REQUIRE(std::is_same_v<list_concat<l4,l1>::type, type_list<A>>);
    REQUIRE(std::is_same_v<list_concat<l3,l4,l1,l2>::type, type_list<D,E,F,A,B,C>>);
}

TEST_CASE("test_types_cross_product_with_type_list","[test_helpers_for_testing]"){
    using helpers_for_testing::cross_product;
    using test_helpers_for_testing::type_list;
    using test_helpers_for_testing::type_pair;
    using test_helpers_for_testing::A;
    using test_helpers_for_testing::B;
    using test_helpers_for_testing::C;
    using test_helpers_for_testing::D;
    using test_helpers_for_testing::E;
    using test_helpers_for_testing::F;

    REQUIRE(std::is_same_v<
        cross_product<type_list, type_list<A,B,C>, type_list<D,E,F>>::type ,
        type_list<type_list<A,D>,type_list<A,E>,type_list<A,F>,type_list<B,D>,type_list<B,E>,type_list<B,F>,type_list<C,D>,type_list<C,E>,type_list<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<type_pair, type_list<A,B,C>, type_list<D,E,F>>::type ,
        type_list<type_pair<A,D>,type_pair<A,E>,type_pair<A,F>,type_pair<B,D>,type_pair<B,E>,type_pair<B,F>,type_pair<C,D>,type_pair<C,E>,type_pair<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<type_list, type_list<A,A,C>, type_list<D,E,E>>::type ,
        type_list<type_list<A,D>,type_list<A,E>,type_list<A,E>,type_list<A,D>,type_list<A,E>,type_list<A,E>,type_list<C,D>,type_list<C,E>,type_list<C,E>> >
    );
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<A>, type_list<B>>::type , type_list<type_list<A,B>>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<>, type_list<B>>::type , type_list<>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<>, type_list<>>::type , type_list<>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<A,B,C>, type_list<>>::type , type_list<>>);
}

TEST_CASE("test_types_cross_product_with_tuple","[test_helpers_for_testing]"){
    using helpers_for_testing::cross_product;
    using test_helpers_for_testing::A;
    using test_helpers_for_testing::B;
    using test_helpers_for_testing::C;
    using test_helpers_for_testing::D;
    using test_helpers_for_testing::E;
    using test_helpers_for_testing::F;

    REQUIRE(std::is_same_v<
        cross_product<std::tuple, std::tuple<A,B,C>, std::tuple<D,E,F>>::type ,
        std::tuple<std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,F>,std::tuple<B,D>,std::tuple<B,E>,std::tuple<B,F>,std::tuple<C,D>,std::tuple<C,E>,std::tuple<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<std::pair, std::tuple<A,B,C>, std::tuple<D,E,F>>::type ,
        std::tuple<std::pair<A,D>,std::pair<A,E>,std::pair<A,F>,std::pair<B,D>,std::pair<B,E>,std::pair<B,F>,std::pair<C,D>,std::pair<C,E>,std::pair<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<std::tuple, std::tuple<A,A,C>, std::tuple<D,E,E>>::type ,
        std::tuple<std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,E>,std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,E>,std::tuple<C,D>,std::tuple<C,E>,std::tuple<C,E>> >
    );
    REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<A>, std::tuple<B>>::type , std::tuple<std::tuple<A,B>>>);
    REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<>, std::tuple<B>>::type , std::tuple<>>);
    REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<>, std::tuple<>>::type , std::tuple<>>);
    REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<A,B,C>, std::tuple<>>::type , std::tuple<>>);
}

TEST_CASE("test_apply_by_element","[test_helpers_for_testing]"){
    using helpers_for_testing::apply_by_element;

    auto tests = std::make_tuple(
        [](const auto& t){REQUIRE(t == t);},
        [](const auto& t){REQUIRE((t+0) == t);},
        [](const auto& t){REQUIRE(t+1 > t);},
        [](const auto& t){REQUIRE(t-1 < t);}
    );

    auto test_data = std::make_tuple(
        int{0},
        int{1},
        float{0},
        float{1},
        double{0},
        double{1}
    );

    auto apply_tests = [&test_data](auto& test){
        apply_by_element(test, test_data);
    };
    apply_by_element(apply_tests, tests);
}

namespace test_tuple{

template<typename T> struct type_tree_to_str{
    auto operator()()const{return std::string{typeid(T).name()};}
};
template<template<typename...> typename L, typename...Ts>
struct type_tree_to_str<L<Ts...>>{
    auto operator()()const{
        std::stringstream ss{};
        ss<<"[";
        ((ss<<type_tree_to_str<Ts>{}()),...);
        ss<<"]";
        return ss.str();
    }
};

template<template<typename...> typename TypeListIndexer>
struct type_list_indexer_wrapper{
    template<typename...Ts> using type_list_indexer = TypeListIndexer<Ts...>;
};

}   //end of namespace test_tuple

TEMPLATE_TEST_CASE("test_type_list_indexer","[test_helpers_for_testing]",
    test_tuple::type_list_indexer_wrapper<helpers_for_testing::tuple_details::type_list_indexer_2>,
    test_tuple::type_list_indexer_wrapper<helpers_for_testing::tuple_details::type_list_indexer_4>
)
{
    //using helpers_for_testing::tuple_details::type_list_indexer_4;
    //using type_list
    SECTION("list_1")
    {
        REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<void>::template at<0>, void>);
        REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<void*>::template at<0>, void*>);
        REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<int>::template at<0>, int>);
        REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<int*>::template at<0>, int*>);
        REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<double>::template at<0>, double>);
        REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<double&&>::template at<0>, double&&>);
        REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<const double&>::template at<0>, const double&>);
        REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<std::string>::template at<0>, std::string>);
    }
    SECTION("list_20")
    {
        using indexer_20_type = typename TestType::template type_list_indexer<void,int,float&,const double,std::vector<int>,int&&,double,double,int,int,
            const int**,std::string,char,char,int,void,void*,const int&,double,char>;

        REQUIRE(std::is_same_v<typename indexer_20_type::template at<0>, void>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<1>, int>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<2>, float&>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<3>, const double>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<4>, std::vector<int>>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<5>, int&&>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<6>, double>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<7>, double>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<8>, int>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<9>, int>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<10>, const int**>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<11>, std::string>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<12>, char>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<13>, char>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<14>, int>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<15>, void>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<16>, void*>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<17>, const int&>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<18>, double>);
        REQUIRE(std::is_same_v<typename indexer_20_type::template at<19>, char>);
    }
}

TEST_CASE("test_tuple","[test_helpers_for_testing]"){
    using helpers_for_testing::tuple;
    int i{};

    tuple<> t0{};
    tuple<int> t1{1};
    tuple<const int> ct1{1};
    tuple<const int> cct1{i};
    tuple<int,double> t2{1,2};
    tuple<int,double> t2_def{};
    tuple<int,double,std::string> t3{1,2,"3"};
    tuple<int,double,std::string> t3_def{};
    tuple<int*,const double*,std::string> t_with_ptr{};
    tuple<int&,double> my_t_with_ref{i,2};
    tuple<const int&,double> my_t_with_const_ref{i,2};
    tuple<const int&,double> my_t_with_const_ref1{1,2};
    tuple<int&&,double> my_t_rval_ref{std::move(i),2};
    //tuple<int&&,double> my_t_rval_ref1{i,2}; //must not compile

    std::reference_wrapper<const int> ref{i};
    //std::reference_wrapper<const int> ref1{1};

    std::tuple<int&,double> t_with_lval_ref{i,2};
    std::tuple<const int&,double> t_with_const_lval_ref{1,2};
    //std::tuple<int&&,double> t_with_rval_ref{i,2};
    std::tuple<int&&,double> t_with_rval_ref1{1,2};
    std::tuple<int&&,double> t_with_rval_ref2{std::move(i),2};
}