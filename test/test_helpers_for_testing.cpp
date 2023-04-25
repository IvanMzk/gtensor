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

// TEST_CASE("test_list_concat","[test_helpers_for_testing]"){
//     using helpers_for_testing::list_concat;
//     using test_helpers_for_testing::type_list;
//     using test_helpers_for_testing::A;
//     using test_helpers_for_testing::B;
//     using test_helpers_for_testing::C;
//     using test_helpers_for_testing::D;
//     using test_helpers_for_testing::E;
//     using test_helpers_for_testing::F;

//     using l1 = type_list<A>;
//     using l2 = type_list<B,C>;
//     using l3 = type_list<D,E,F>;
//     using l4 = type_list<>;

//     REQUIRE(std::is_same_v<list_concat<l1,l2>::type, type_list<A,B,C>>);
//     REQUIRE(std::is_same_v<list_concat<l2,l1>::type, type_list<B,C,A>>);
//     REQUIRE(std::is_same_v<list_concat<l4,l1>::type, type_list<A>>);
//     REQUIRE(std::is_same_v<list_concat<l3,l4,l1,l2>::type, type_list<D,E,F,A,B,C>>);
// }

// TEST_CASE("test_types_cross_product_with_type_list","[test_helpers_for_testing]"){
//     using helpers_for_testing::cross_product;
//     using test_helpers_for_testing::type_list;
//     using test_helpers_for_testing::type_pair;
//     using test_helpers_for_testing::A;
//     using test_helpers_for_testing::B;
//     using test_helpers_for_testing::C;
//     using test_helpers_for_testing::D;
//     using test_helpers_for_testing::E;
//     using test_helpers_for_testing::F;

//     REQUIRE(std::is_same_v<
//         cross_product<type_list, type_list<A,B,C>, type_list<D,E,F>>::type ,
//         type_list<type_list<A,D>,type_list<A,E>,type_list<A,F>,type_list<B,D>,type_list<B,E>,type_list<B,F>,type_list<C,D>,type_list<C,E>,type_list<C,F>> >
//     );
//     REQUIRE(std::is_same_v<
//         cross_product<type_pair, type_list<A,B,C>, type_list<D,E,F>>::type ,
//         type_list<type_pair<A,D>,type_pair<A,E>,type_pair<A,F>,type_pair<B,D>,type_pair<B,E>,type_pair<B,F>,type_pair<C,D>,type_pair<C,E>,type_pair<C,F>> >
//     );
//     REQUIRE(std::is_same_v<
//         cross_product<type_list, type_list<A,A,C>, type_list<D,E,E>>::type ,
//         type_list<type_list<A,D>,type_list<A,E>,type_list<A,E>,type_list<A,D>,type_list<A,E>,type_list<A,E>,type_list<C,D>,type_list<C,E>,type_list<C,E>> >
//     );
//     REQUIRE(std::is_same_v<cross_product<type_list, type_list<A>, type_list<B>>::type , type_list<type_list<A,B>>>);
//     REQUIRE(std::is_same_v<cross_product<type_list, type_list<>, type_list<B>>::type , type_list<>>);
//     REQUIRE(std::is_same_v<cross_product<type_list, type_list<>, type_list<>>::type , type_list<>>);
//     REQUIRE(std::is_same_v<cross_product<type_list, type_list<A,B,C>, type_list<>>::type , type_list<>>);
// }

// TEST_CASE("test_types_cross_product_with_tuple","[test_helpers_for_testing]"){
//     using helpers_for_testing::cross_product;
//     using test_helpers_for_testing::A;
//     using test_helpers_for_testing::B;
//     using test_helpers_for_testing::C;
//     using test_helpers_for_testing::D;
//     using test_helpers_for_testing::E;
//     using test_helpers_for_testing::F;

//     REQUIRE(std::is_same_v<
//         cross_product<std::tuple, std::tuple<A,B,C>, std::tuple<D,E,F>>::type ,
//         std::tuple<std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,F>,std::tuple<B,D>,std::tuple<B,E>,std::tuple<B,F>,std::tuple<C,D>,std::tuple<C,E>,std::tuple<C,F>> >
//     );
//     REQUIRE(std::is_same_v<
//         cross_product<std::pair, std::tuple<A,B,C>, std::tuple<D,E,F>>::type ,
//         std::tuple<std::pair<A,D>,std::pair<A,E>,std::pair<A,F>,std::pair<B,D>,std::pair<B,E>,std::pair<B,F>,std::pair<C,D>,std::pair<C,E>,std::pair<C,F>> >
//     );
//     REQUIRE(std::is_same_v<
//         cross_product<std::tuple, std::tuple<A,A,C>, std::tuple<D,E,E>>::type ,
//         std::tuple<std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,E>,std::tuple<A,D>,std::tuple<A,E>,std::tuple<A,E>,std::tuple<C,D>,std::tuple<C,E>,std::tuple<C,E>> >
//     );
//     REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<A>, std::tuple<B>>::type , std::tuple<std::tuple<A,B>>>);
//     REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<>, std::tuple<B>>::type , std::tuple<>>);
//     REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<>, std::tuple<>>::type , std::tuple<>>);
//     REQUIRE(std::is_same_v<cross_product<std::tuple, std::tuple<A,B,C>, std::tuple<>>::type , std::tuple<>>);
// }

// TEST_CASE("test_apply_by_element","[test_helpers_for_testing]"){
//     using helpers_for_testing::apply_by_element;

//     auto tests = std::make_tuple(
//         [](const auto& t){REQUIRE(t == t);},
//         [](const auto& t){REQUIRE((t+0) == t);},
//         [](const auto& t){REQUIRE(t+1 > t);},
//         [](const auto& t){REQUIRE(t-1 < t);}
//     );

//     auto test_data = std::make_tuple(
//         int{0},
//         int{1},
//         float{0},
//         float{1},
//         double{0},
//         double{1}
//     );

//     auto apply_tests = [&test_data](auto& test){
//         apply_by_element(test, test_data);
//     };
//     apply_by_element(apply_tests, tests);
// }

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

// TEMPLATE_TEST_CASE("test_type_list_indexer","[test_helpers_for_testing]",
//     test_tuple::type_list_indexer_wrapper<helpers_for_testing::tuple_details::type_list_indexer_2>,
//     test_tuple::type_list_indexer_wrapper<helpers_for_testing::tuple_details::type_list_indexer_4>
// )
// {
//     //using helpers_for_testing::tuple_details::type_list_indexer_4;
//     //using type_list
//     SECTION("list_1")
//     {
//         REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<void>::template at<0>, void>);
//         REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<void*>::template at<0>, void*>);
//         REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<int>::template at<0>, int>);
//         REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<int*>::template at<0>, int*>);
//         REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<double>::template at<0>, double>);
//         REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<double&&>::template at<0>, double&&>);
//         REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<const double&>::template at<0>, const double&>);
//         REQUIRE(std::is_same_v<typename TestType::template type_list_indexer<std::string>::template at<0>, std::string>);
//     }
//     SECTION("list_20")
//     {
//         using indexer_20_type = typename TestType::template type_list_indexer<void,int,float&,const double,std::vector<int>,int&&,double,double,int,int,
//             const int**,std::string,char,char,int,void,void*,const int&,double,char>;

//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<0>, void>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<1>, int>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<2>, float&>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<3>, const double>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<4>, std::vector<int>>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<5>, int&&>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<6>, double>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<7>, double>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<8>, int>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<9>, int>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<10>, const int**>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<11>, std::string>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<12>, char>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<13>, char>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<14>, int>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<15>, void>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<16>, void*>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<17>, const int&>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<18>, double>);
//         REQUIRE(std::is_same_v<typename indexer_20_type::template at<19>, char>);
//     }
// }

// TEST_CASE("test_tuple_size","[test_helpers_for_testing]")
// {
//     using helpers_for_testing::tuple;
//     using helpers_for_testing::tuple_size_v;

//     REQUIRE(tuple_size_v<tuple<>> == 0);
//     REQUIRE(tuple_size_v<tuple<void>> == 1);
//     REQUIRE(tuple_size_v<tuple<int>> == 1);
//     REQUIRE(tuple_size_v<tuple<void,void>> == 2);
//     REQUIRE(tuple_size_v<tuple<int,double,std::string>> == 3);
//     REQUIRE(tuple_size_v<tuple<int&,double*,std::string&&,void,void>> == 5);
// }

// TEST_CASE("test_tuple_element","[test_helpers_for_testing]")
// {
//     using helpers_for_testing::tuple;
//     using helpers_for_testing::tuple_element_t;

//     REQUIRE(std::is_same_v<tuple_element_t<0,tuple<void*>>,void*>);
//     REQUIRE(std::is_same_v<tuple_element_t<0,tuple<const void*>>,const void*>);
//     REQUIRE(std::is_same_v<tuple_element_t<0,tuple<int>>,int>);
//     REQUIRE(std::is_same_v<tuple_element_t<0,tuple<const int>>,const int>);
//     REQUIRE(std::is_same_v<tuple_element_t<0,tuple<int&,double,char>>,int&>);
//     REQUIRE(std::is_same_v<tuple_element_t<1,tuple<int,double&&,char>>,double&&>);
//     REQUIRE(std::is_same_v<tuple_element_t<2,tuple<int,double,char*>>,char*>);
//     REQUIRE(std::is_same_v<tuple_element_t<2,tuple<int,double,char**>>,char**>);
//     REQUIRE(std::is_same_v<tuple_element_t<2,tuple<int,double,const char*>>,const char*>);
//     REQUIRE(std::is_same_v<tuple_element_t<1,tuple<int,const double&,char>>,const double&>);
//     REQUIRE(std::is_same_v<tuple_element_t<1,tuple<int,const double&&,char>>,const double&&>);
// }

// TEMPLATE_TEST_CASE("test_tuple_get","[test_helpers_for_testing]",
//     (std::integral_constant<std::size_t,0>),
//     (std::integral_constant<std::size_t,1>),
//     (std::integral_constant<std::size_t,2>),
//     (std::integral_constant<std::size_t,3>),
//     (std::integral_constant<std::size_t,4>),
//     (std::integral_constant<std::size_t,5>),
//     (std::integral_constant<std::size_t,6>),
//     (std::integral_constant<std::size_t,7>),
//     (std::integral_constant<std::size_t,8>),
//     (std::integral_constant<std::size_t,9>)
// )
// {
//     using helpers_for_testing::tuple;
//     using helpers_for_testing::get;
//     using tuple_type = tuple<int,const char,int&,const double&,std::string,float&&,const int&&,char*,const char*,const char*const>;
//     using std_tuple_type = std::tuple<int,const char,int&,const double&,std::string,float&&,const int&&,char*,const char*,const char*const>;
//     static constexpr std::size_t I = TestType::value;

//     SECTION("test_tuple_get_result_type")
//     {
//         //lvalue argument
//         REQUIRE(std::is_same_v<decltype(get<I>(std::declval<tuple_type&>())), decltype(std::get<I>(std::declval<std_tuple_type&>()))>);
//         //const lvalue argument
//         REQUIRE(std::is_same_v<decltype(get<I>(std::declval<const tuple_type&>())), decltype(std::get<I>(std::declval<const std_tuple_type&>()))>);
//         //rvalue argument
//         REQUIRE(std::is_same_v<decltype(get<I>(std::declval<tuple_type>())), decltype(std::get<I>(std::declval<std_tuple_type>()))>);
//         //const rvalue argument
//         REQUIRE(std::is_same_v<decltype(get<I>(std::declval<const tuple_type>())), decltype(std::get<I>(std::declval<const std_tuple_type>()))>);
//     }
//     SECTION("test_tuple_get_result_value")
//     {
//         int i{0};
//         double d{1};
//         float f{2};
//         char c{3};
//         tuple_type test_tuple{1,c,i,d,"abcd",std::move(f),std::move(i),&c,&c,&c};
//         std_tuple_type test_std_tuple{1,c,i,d,"abcd",std::move(f),std::move(i),&c,&c,&c};
//         // tuple_type test_tuple{1,2,i,d,"abcd",std::move(f),std::move(i),&c,&c,&c};
//         // std_tuple_type test_std_tuple{1,2,i,d,"abcd",std::move(f),std::move(i),&c,&c,&c};
//         //lvalue argument
//         REQUIRE(get<I>(test_tuple) == std::get<I>(test_std_tuple));
//         //const lvalue argument
//         REQUIRE(get<I>(static_cast<const tuple_type&>(test_tuple)) == std::get<I>(static_cast<const std_tuple_type&>(test_std_tuple)));
//         //rvalue argument
//         REQUIRE(get<I>(static_cast<tuple_type&&>(test_tuple)) == std::get<I>(static_cast<std_tuple_type&&>(test_std_tuple)));
//         //const rvalue argument
//         REQUIRE(get<I>(static_cast<const tuple_type&&>(test_tuple)) == std::get<I>(static_cast<const std_tuple_type&&>(test_std_tuple)));
//     }
// }

// TEST_CASE("test_tuple_operator==","[test_helpers_for_testing]")
// {
//     using helpers_for_testing::tuple;

//     int i1{1};
//     int i2{2};
//     std::string s_abc{"abc"};
//     std::string s_def{"def"};

//     REQUIRE(tuple<>{} == tuple<>{});
//     REQUIRE(tuple<int>{} == tuple<int>{});
//     REQUIRE(tuple<int>{} == tuple<double>{});
//     REQUIRE(tuple<int>{1} == tuple<int>{1});
//     REQUIRE(tuple<int>{1} == tuple<double>{1});
//     REQUIRE(tuple<int,double>{1,2} == tuple<int,double>{1,2});
//     REQUIRE(tuple<int,int>{1,2} == tuple<double,double>{1,2});
//     REQUIRE(tuple<int&>{i1} == tuple<int&>{i1});
//     REQUIRE(tuple<int&&>{std::move(i1)} == tuple<int&&>{std::move(i1)});
//     REQUIRE(tuple<std::string&>{s_abc} == tuple<std::string>{"abc"});
//     REQUIRE(tuple<int*>{&i1} == tuple<int*>{&i1});
//     REQUIRE(tuple<std::string, const int&, int*>{s_abc,i1,&i1} == tuple<std::string, double, int*>{"abc",1,&i1});

//     REQUIRE(tuple<int>{0} != tuple<int>{1});
//     REQUIRE(tuple<int,std::string>{0,"abc"} != tuple<int,std::string>{0,"def"});
//     REQUIRE(tuple<int,double>{0,1} != tuple<double,int>{1,0});
//     REQUIRE(tuple<int&>{i1} != tuple<int&>{i2});
//     REQUIRE(tuple<int&&>{std::move(i1)} != tuple<int&&>{std::move(i2)});
//     REQUIRE(tuple<std::string&>{s_abc} != tuple<std::string>{"def"});
//     REQUIRE(tuple<int*>{&i1} != tuple<int*>{&i2});
//     REQUIRE(tuple<std::string, const int&, int*>{"abc",i1,&i1} != tuple<std::string&, double, int*>{s_def,1,&i1});
//     REQUIRE(tuple<std::string, const int&, int*>{"def",i1,&i1} != tuple<std::string&, double, int*>{s_def,2,&i1});
//     REQUIRE(tuple<std::string, const int&, int*>{"def",i1,&i1} != tuple<std::string&, double, int*>{s_def,1,&i2});
// }

//add tests:
//default constructo
//converting copy assignment
//convereting move assignment
//destructor
//exception cleanup

TEST_CASE("test_tuple_converting_args_constructor","[test_helpers_for_testing]")
{
    using helpers_for_testing::tuple;
    using helpers_for_testing::get;

    SECTION("Types>1_Args>1"){
        int i{0};
        std::vector<int> v{1,2,3};
        tuple<double, void*, std::string, std::vector<int>> result{i,&i,"abc",std::move(v)};
        REQUIRE(get<0>(result) == i);
        REQUIRE(get<1>(result) == &i);
        REQUIRE(get<2>(result) == std::string{"abc"});
        REQUIRE(get<3>(result) == std::vector<int>{1,2,3});
        REQUIRE(v.empty());
    }
    SECTION("Types==1_Args==1_tuple>1"){
        struct abc{
            int i0_,i1_;
            abc(const tuple<int,int>& arg):
                i0_{get<0>(arg)},
                i1_{get<1>(arg)}
            {}
        };
        tuple<int,int> arg{1,2};
        tuple<abc> result{arg};
        REQUIRE(get<0>(result).i0_ == 1);
        REQUIRE(get<0>(result).i1_ == 2);
    }
    SECTION("Types==1_Args==1_tuple==1"){
        struct abc{
            int i0_{};
            int i_{};
            abc(const tuple<int>& arg):
                i0_{get<0>(arg)}
            {}
            abc(int arg):
                i_{arg}
            {}
        };
        tuple<int> arg{2};
        tuple<abc> result{arg};
        REQUIRE(get<0>(result).i0_ == 2);
        REQUIRE(get<0>(result).i_ == int{});
    }
}

TEST_CASE("test_tuple_copy_operations","[test_helpers_for_testing]")
{
    using helpers_for_testing::tuple;
    using helpers_for_testing::get;

    SECTION("test_tuple_copy_constructor")
    {
        int i{1};
        double d{2};
        using tuple_type = tuple<int,const int,int&,int&&,std::string,const double&, const double&&>;
        tuple_type test_tuple{3,4,i,std::move(i),"abc",d,std::move(d)};
        const tuple_type& cr_test_tuple{test_tuple};
        tuple_type tuple_copy1{test_tuple};
        tuple_type tuple_copy2{cr_test_tuple};
        REQUIRE(test_tuple == tuple_copy1);
        REQUIRE(test_tuple == tuple_copy2);
        get<4>(tuple_copy1) = "def";
        REQUIRE(test_tuple != tuple_copy1);
        REQUIRE(test_tuple == tuple_copy2);
        i = 3;
        REQUIRE(get<2>(test_tuple) == i);
        REQUIRE(get<2>(tuple_copy1) == i);
        REQUIRE(get<3>(test_tuple) == i);
        REQUIRE(get<3>(tuple_copy1) == i);

        tuple<> empty_tuple{};
        tuple<> empty_tuple_copy = empty_tuple;
        REQUIRE(empty_tuple == empty_tuple_copy);
    }
    SECTION("test_tuple_copy_assignment")
    {
        using tuple_type = tuple<int,double,std::string>;
        tuple_type rhs{1,2,"abc"};
        tuple_type lhs{};
        REQUIRE(lhs != rhs);
        lhs = rhs;
        REQUIRE(lhs == rhs);

        tuple<> empty_rhs{};
        tuple<> empty_lhs{};
        empty_lhs = empty_rhs;
        REQUIRE(empty_lhs == empty_rhs);
    }
    SECTION("test_tuple_converting_copy_constructor")
    {
        SECTION("Types>1")
        {
            double d{};
            tuple<int,double*> src{1,&d};
            tuple<double,void*> from_src{src};
            REQUIRE(from_src == tuple<double,void*>{1,&d});

            const tuple<int,double*> const_src{1,&d};
            tuple<double,void*> from_const_src{const_src};
            REQUIRE(from_const_src == tuple<double,void*>{1,&d});
        }
        SECTION("Types==1")
        {
            struct Int
            {
                int i_;
                Int(int i__):
                    i_{i__}
                {}
                bool operator==(const Int& other)const{return i_==other.i_;}
            };
            tuple<int> srs{2};
            tuple<Int> from_src{srs};
            REQUIRE(get<0>(from_src) == Int{2});

            const tuple<int> const_src{2};
            tuple<Int> from_const_src{const_src};
            REQUIRE(get<0>(from_const_src) == Int{2});
        }
    }
}

TEST_CASE("test_tuple_move_operations","[test_helpers_for_testing]")
{
    using helpers_for_testing::tuple;
    using helpers_for_testing::get;

    SECTION("test_tuple_move_constructor"){
        using tuple_type = tuple<int,std::vector<int>>;
        tuple_type test_tuple{1,{1,2,3}};
        tuple_type tuple_move = std::move(test_tuple);
        REQUIRE(tuple_move == tuple_type{1,{1,2,3}});
        REQUIRE(get<1>(test_tuple).empty() == true);

        tuple<> empty_tuple{};
        tuple<> empty_tuple_move = std::move(empty_tuple);
        REQUIRE(empty_tuple == tuple<>{});
    }
    SECTION("test_tuple_move_assignment"){
        using tuple_type = tuple<int,std::vector<int>>;
        tuple_type rhs{1,{1,2,3}};
        tuple_type lhs{};
        lhs = std::move(rhs);
        REQUIRE(lhs == tuple_type{1,{1,2,3}});
        REQUIRE(get<1>(rhs).empty() == true);

        tuple<> empty_rhs{};
        tuple<> empty_lhs{};
        empty_lhs = empty_rhs;
        REQUIRE(empty_lhs == tuple<>{});
    }
    SECTION("test_tuple_converting_move_constructor")
    {
        SECTION("Types>1")
        {
            double d{};
            tuple<int,double*,std::vector<int>> src{1,&d,{1,2,3}};
            tuple<double,void*,std::vector<int>> from_src{std::move(src)};
            REQUIRE(from_src == tuple<double,void*,std::vector<int>>{1,&d,{1,2,3}});
            REQUIRE(get<2>(src).empty());
        }
        SECTION("Types==1")
        {
            struct abc
            {
                std::vector<char> s_;
                abc(std::initializer_list<char> s__):
                    s_{s__}
                {}
                auto empty()const{return s_.empty();}
            };
            struct def
            {
                std::vector<char> s_;
                def(std::initializer_list<char> s__):
                    s_{s__}
                {}
                def(abc&& other):
                    s_{std::move(other.s_)}
                {}
                bool operator==(const def& other)const{return s_==other.s_;}
            };
            tuple<abc> src{{'a','b','c'}};
            tuple<def> from_src{std::move(src)};
            REQUIRE(get<0>(from_src) == def{'a','b','c'});
            REQUIRE(get<0>(src).empty());
        }
    }
}

