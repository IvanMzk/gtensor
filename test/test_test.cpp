#include <memory>
#include <tuple>
#include <list>
#include <iostream>
#include <sstream>
#include <variant>
#include <regex>
#include <functional>
#include "catch.hpp"
#include "test.hpp"


namespace template_depth{

template<std::size_t First, std::size_t Last>
struct sum_range{
    static constexpr std::size_t value = Last+sum_range<First,Last-1>::value;
};
template<std::size_t First>
struct sum_range<First,First>{
    static constexpr std::size_t value = First;
};

template<std::size_t First, std::size_t Last>
struct sum_range_v1{
    static constexpr std::size_t value = sum_range_v1<First, (First+Last)/2>::value + sum_range_v1<(First+Last)/2 + 1, Last>::value;
};

template<std::size_t First>
struct sum_range_v1<First,First>{
    static constexpr std::size_t value = First;
};

auto sum_range_test(std::size_t first, std::size_t last){
    return (first+last)*(last-first+1)/2;
}


struct A{};
template<typename T>
struct B{};


template<std::size_t Depth, typename T>
struct bmaker{  
    using type = typename bmaker<Depth-1,B<T>>::type;
};
template<typename T>
struct bmaker<0,T>{  
    using type = T;
};


template<typename> struct config{};
struct op{};

template<typename ValT, template<typename> typename Cfg>
struct ten{
    int i;
    ten():i{1}{}
    //ten() = default;
};
template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
struct exp{
    exp():i{2}{}
    int i;
    //exp() = default;
    //std::tuple<Ops...> operands{};
    std::pair<Ops...> operands{};
    exp(const Ops&...operands_):        
        operands{operands_...}
    {}    
};

template<std::size_t Depth, typename T1, typename T2>
struct cmaker{  
    using type = typename cmaker<Depth-1,T1,exp<float,config,op,T1,T2>>::type;
};
template<typename T1, typename T2>
struct cmaker<0,T1,T2>{  
    using type = T2;
};

template<typename ValT, template<typename> typename Cfg>
auto operator+(const ten<ValT,Cfg>& t1,  const ten<ValT,Cfg>& t2){    
    return exp<ValT,Cfg,op,ten<ValT,Cfg>,ten<ValT,Cfg>>{t1,t2};
}
template<typename ValT, template<typename> typename Cfg, typename O, typename...Ops>
auto operator+(const ten<ValT,Cfg>& t,  const exp<ValT,Cfg,O,Ops...>& e){    
    return exp<ValT,Cfg,op,ten<ValT,Cfg>,exp<ValT,Cfg,O,Ops...>>{t,e};
}
template<typename ValT, template<typename> typename Cfg, typename O, typename...Ops>
auto operator+(const exp<ValT,Cfg,O,Ops...>& e, const ten<ValT,Cfg>& t){    
    return exp<ValT,Cfg,op,exp<ValT,Cfg,O,Ops...>,ten<ValT,Cfg>>{e,t};
}

template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return make_asymmetric_tree<Depth-1>(t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

}   //end of namespace template_depth

namespace member_access{

    struct B{};    

    template<typename T>
    struct A
    {
        T t;
        A(T& t_):
            t{t_}
        {std::cout<<std::endl<<"A(T& t_):";}
        A(const T& t_):
            t{t_}
        {std::cout<<std::endl<<"A(const T& t_):";}
        A(T&& t_):
            t{std::move(t_)}
        {std::cout<<std::endl<<"A(T&& t_):";}

        auto f(){return t;}
    };
}   //end of namespace pair_member_move

namespace static_list{

    template<std::size_t, std::size_t> struct node;
    template<> struct node<0,0>{};

    template<std::size_t I>
    struct node<I, 0>{
        node<I-1, 0> next_;        
        auto next(){return next_;}
    };

}


namespace binary_tree_dispatch{

struct A{};
struct B{};
struct C{};
struct D{};
struct E{};
struct F{};
template<typename F, typename S> struct type_pair{};

template<typename...Ts>
struct type_list{
    using type = type_list<Ts...>;
    static constexpr std::size_t size = sizeof...(Ts); 
    static auto to_str(){return std::to_string(sizeof...(Ts)) + std::regex_replace(typeid(type).name(), std::regex{"struct binary_tree_dispatch::"}, "");}
};

template<typename, typename> struct list_concat;
template<typename...Us, typename...Vs> 
struct list_concat<type_list<Us...>,type_list<Vs...>>{
    using type = type_list<Us...,Vs...>;
};

template<template <typename...> typename, typename, typename> struct cross_product;
template<template <typename...> typename PairT, typename U, typename...Us, typename...Vs> 
struct cross_product<PairT, type_list<U, Us...>, type_list<Vs...>>{
    using cross_u_vs = type_list<PairT<U,Vs>...>;
    using cross_us_vs = typename cross_product<PairT, type_list<Us...>, type_list<Vs...>>::type;
    using type = typename list_concat<cross_u_vs, cross_us_vs>::type;
};
template<template <typename...> typename PairT, typename...Vs>
struct cross_product<PairT, type_list<>, type_list<Vs...>>{    
    using type = type_list<>;
};





struct add{std::string to_str()const{return "add";}};
struct sub{std::string to_str()const{return "sub";}};
struct mul{std::string to_str()const{return "mul";}};

template<typename F, typename...Wks>
struct ew{
    std::tuple<Wks...> wks;
    template<typename...Args>
    ew(Args&&...args):
        wks{std::forward<Args>(args)...}
    {}
};


struct sw{};

struct w{};

struct snode{
    using w_types = type_list<sw>;
    auto create_w()const{
        std::variant<sw>{sw{}};
    }
    std::string to_str()const{return "snode";}
};

// template<std::size_t max_size,  typename...Ops>
// class w_types_traits{
//     static constexpr size = (Ops::w_types::size*...);
//     //size<max_size
//     template<bool> struct selector{using type = typename list_concat<type_list<sw>, typename cross_product<ew_alias, typename Ops::w_types...>::type>::type;};
//     //size>max_size
//     template<> struct selector<false>{};
// public:    
// };

template<typename F, typename...Ops>
struct enode{
    static constexpr std::size_t max_w_types_size = 100;
    template<typename...Us> using ew_alias = ew<F, Us...>;
    static constexpr std::size_t w_types_size = (Ops::w_types::size*...);
    template<bool> struct w_types_traits{using type = typename list_concat<type_list<sw>, typename cross_product<ew_alias, typename Ops::w_types...>::type>::type;};
    template<> struct w_types_traits<false>{using type = type_list<w>;};
    using w_types = typename w_types_traits<(w_types_size<max_w_types_size)>::type;
    
    //using w_types = typename list_concat<type_list<sw>, typename cross_product<ew_alias, typename Ops::w_types...>::type>::type;

    std::tuple<Ops...> ops;
    template<typename...Args>
    explicit enode(const Args&...args):
        ops{args...}
    {}
    auto op0()const{return std::get<0>(ops);}
    auto op1()const{return std::get<1>(ops);}    
    
    template<typename, typename> struct make_variant_type;
    template<typename...Vs, typename...Us> struct make_variant_type<std::variant<Vs...>, std::variant<Us...>>
    {
        
    };
    std::string to_str()const{
        std::stringstream ss{};
        ss<<"enode< "<<F{}.to_str()+" "<<std::apply([](const auto&...args){return ((args.to_str()+" ")+...);}, ops)<<" >";
        return ss.str();
    }     
};


auto operator+(const snode& op1, const snode& op2){return enode<add, snode,snode>{op1,op2};}
template<typename...Ts> auto operator+(const snode& op1, const enode<Ts...>& op2){return enode<add, snode,enode<Ts...>>{op1,op2};}
template<typename...Ts> auto operator+(const enode<Ts...>& op1, const snode& op2){return enode<add, enode<Ts...>,snode>{op1,op2};}
template<typename...Ts, typename...Us> auto operator+(const enode<Ts...>& op1, const enode<Us...>& op2){return enode<add, enode<Ts...>,enode<Us...>>{op1,op2};}

auto operator-(const snode& op1, const snode& op2){return enode<sub, snode,snode>{op1,op2};}
template<typename...Ts> auto operator-(const snode& op1, const enode<Ts...>& op2){return enode<sub, snode,enode<Ts...>>{op1,op2};}
template<typename...Ts> auto operator-(const enode<Ts...>& op1, const snode& op2){return enode<sub, enode<Ts...>,snode>{op1,op2};}
template<typename...Ts, typename...Us> auto operator-(const enode<Ts...>& op1, const enode<Us...>& op2){return enode<sub, enode<Ts...>,enode<Us...>>{op1,op2};}

auto operator*(const snode& op1, const snode& op2){return enode<mul, snode,snode>{op1,op2};}
template<typename...Ts> auto operator*(const snode& op1, const enode<Ts...>& op2){return enode<mul, snode,enode<Ts...>>{op1,op2};}
template<typename...Ts> auto operator*(const enode<Ts...>& op1, const snode& op2){return enode<mul, enode<Ts...>,snode>{op1,op2};}
template<typename...Ts, typename...Us> auto operator*(const enode<Ts...>& op1, const enode<Us...>& op2){return enode<mul, enode<Ts...>,enode<Us...>>{op1,op2};}


}   //end of namespace binary_tree_dispatch


TEST_CASE("template_depth","[template_depth]"){

    using template_depth::sum_range;
    using template_depth::sum_range_v1;
    using template_depth::sum_range_test;
    using template_depth::A;
    using template_depth::B;
    using template_depth::bmaker;
    using template_depth::cmaker;
    using template_depth::config;
    using template_depth::ten;
    using template_depth::exp;
    using template_depth::make_asymmetric_tree;


    REQUIRE(sum_range<0,0>::value == 0);
    REQUIRE(sum_range<0,1>::value == 1);
            
    REQUIRE(sum_range<501,1000>::value == sum_range_test(501,1000));
    REQUIRE(sum_range<0,500>::value + sum_range<501,1000>::value == sum_range_test(0,1000));
    REQUIRE(sum_range_v1<0,1000>::value == sum_range_test(0,1000));


    auto b =  bmaker<100,A>::type{};
    
    

    ten<int,config> t;
    auto e = t+t;
    auto e1 = t+e;    
    auto e2 = e+t;
    auto ee = make_asymmetric_tree<200>(t,t); 
    auto tt = ee.operands.first;
    auto ttt = std::get<0>(ee.operands);
    auto eee = ee;    
}

TEST_CASE("member_access","[member_access]"){
    using member_access::B;
    using member_access::A;
    
    auto p = std::make_pair(B{},B{});
    auto& ref_p = p;
    const auto& cref_p = p;
    REQUIRE(std::is_same_v<decltype(p.first), B>);
    REQUIRE(std::is_same_v<decltype((p.first)), B&>);
    REQUIRE(std::is_same_v<decltype(ref_p.first), B>);
    REQUIRE(std::is_same_v<decltype((ref_p.first)), B&>);
    REQUIRE(std::is_same_v<decltype(cref_p.first), B>);
    REQUIRE(std::is_same_v<decltype((cref_p.first)), const B&>);
    
    A<B>{p.first};
    A<B>{ref_p.first};
    A<B>{cref_p.first};
    
    A<B>{std::move(p.first)};
    A<B>{std::move(ref_p.first)};
    A<B>{std::move(cref_p.first)};

    A<B> a{B{}};
    A<B>{a.f()};
     
}

TEST_CASE("static_list","[static_list]"){
    using static_list::node;
    node<5,0> l{};
}


TEST_CASE("test_cross_product","[binary_tree_dispatch]"){
    using binary_tree_dispatch::type_list;
    using binary_tree_dispatch::type_pair;
    using binary_tree_dispatch::cross_product;
    using binary_tree_dispatch::A;
    using binary_tree_dispatch::B;
    using binary_tree_dispatch::C;
    using binary_tree_dispatch::D;
    using binary_tree_dispatch::E;
    using binary_tree_dispatch::F;
            
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


TEST_CASE("test_w_types","[binary_tree_dispatch]"){
    using binary_tree_dispatch::snode;
    using binary_tree_dispatch::enode;
    using binary_tree_dispatch::add;
    using binary_tree_dispatch::sub;
    using binary_tree_dispatch::mul;
    using binary_tree_dispatch::sw;
    using binary_tree_dispatch::ew;
    using binary_tree_dispatch::type_list;

    snode t{};
    REQUIRE(std::is_same_v<typename snode::w_types, type_list<sw>>);
    REQUIRE(std::is_same_v<typename decltype(t+t)::w_types, type_list<sw, ew<add,sw,sw>>>);
    REQUIRE(std::is_same_v<typename decltype(t+t+t)::w_types, type_list< sw, ew<add,sw,sw>, ew<add,ew<add,sw,sw>,sw> >>);
    REQUIRE(std::is_same_v<typename decltype(t+t*t)::w_types, type_list< sw, ew<add,sw,sw>, ew<add, sw, ew<mul,sw,sw>> >>);
    REQUIRE(std::is_same_v<typename decltype((t-t)*(t+t))::w_types, 
        type_list< 
            sw, 
            ew< mul,sw,sw >,
            ew<mul, sw, ew<add, sw,sw>>,
            ew<mul, ew<sub, sw,sw>, sw>, 
            ew<mul, ew<sub, sw,sw>, ew<add, sw,sw>>
        >>
    );

    auto e1 = t+t;
    auto e2 = e1+e1;
    auto e3 = e2+e2;
    auto e4 = e3+e3;
    auto e5 = e4+e4;
    auto e6 = e5*e5;
    std::cout<<std::endl<<decltype(e3)::w_types::to_str();
    std::cout<<std::endl<<decltype(e4)::w_types::to_str();
    std::cout<<std::endl<<decltype(e5)::w_types::to_str();
    std::cout<<std::endl<<decltype(e6)::w_types::to_str();
}
