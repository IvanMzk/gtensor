#include <memory>
#include <tuple>
#include <list>
#include <iostream>
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