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
#include "benchmark_helpers.hpp"
#include "tensor_init_list.hpp"


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


namespace test_cross_product{
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

}   //end of namespace cross_product


namespace binary_tree_dispatch{

using test_cross_product::type_list;
using test_cross_product::list_concat;
using test_cross_product::cross_product;

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

template<typename F, typename...Ops>
struct enode{
    static constexpr std::size_t max_w_types_size = 100;
    template<typename...Us> using ew_alias = ew<F, Us...>;
    static constexpr std::size_t w_types_size = (Ops::w_types::size*...);
    template<bool> struct w_types_traits{using type = typename list_concat<type_list<sw>, typename cross_product<ew_alias, typename Ops::w_types...>::type>::type;};
    template<> struct w_types_traits<false>{using type = type_list<w>;};
    using w_types = typename w_types_traits<(w_types_size<max_w_types_size)>::type;


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

namespace binary_tree_with_engine{

using test_cross_product::type_list;
using test_cross_product::list_concat;
using test_cross_product::cross_product;

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
struct node_base{};

class e_engine{
protected:
    using root_type = node_base;
    e_engine() = default;
    e_engine(const root_type* root__):
        root_{root__}
    {}
    auto root()const{return root_;}
    void set_root(const root_type* root__){root_ = root__;}

private:
    const root_type* root_;
};

class stor_engine : public e_engine{
public:
    using w_types = type_list<sw>;
    explicit stor_engine(const e_engine::root_type* root):
        e_engine{root}
    {}
};


template<typename F, typename...Ops>
class e_eval_engine : public e_engine
{
    static constexpr std::size_t max_w_types_size = 100;
    template<typename...Us> using ew_alias = ew<F, Us...>;
    static constexpr std::size_t w_types_size = (Ops::engine_type::w_types::size*...);
    template<bool> struct w_types_traits{using type = typename list_concat<type_list<sw>, typename cross_product<ew_alias, typename Ops::engine_type::w_types...>::type>::type;};
    template<> struct w_types_traits<false>{using type = type_list<w>;};
    using e_engine::root_type;

    //std::tuple<Ops...> ops;
    std::pair<Ops...> ops;
public:
    using w_types = typename w_types_traits<(w_types_size<max_w_types_size)>::type;
    template<typename...Args>
    explicit e_eval_engine(const root_type* root, Args&&...args):
        e_engine{root},
        ops{std::forward<Args>(args)...}
    {}
};


template<typename EngineT = stor_engine>
class snode : public node_base
{
public:
    using engine_type = EngineT;
    std::string to_str()const{return "snode";}
private:
    engine_type engine_{this};
};

template<typename EngineT>
class enode : public node_base
{
public:
    using engine_type = EngineT;
    template<typename...Args>
    explicit enode(Args&&...args):
        engine_{this, std::forward<Args>(args)...}
    {}
private:
    engine_type engine_;
};

template<typename ImplT = snode<stor_engine>>
class node{
public:
    using impl_type = ImplT;
    node() = default;
    node(impl_type&& impl__):
        impl_{std::move(impl__)}
    {}
    auto impl()const{return impl_;}
private:
    impl_type impl_;
};


template<typename ImplT1, typename ImplT2> auto operator+(const node<ImplT1>& op1, const node<ImplT2>& op2){
    using engine_type = e_eval_engine<add, ImplT1, ImplT2>;
    using impl_type = enode<engine_type>;
    return node<impl_type>{impl_type{op1.impl(),op2.impl()}};
}

}   //end of namespace binary_tree_with_engine


namespace binary_tree_with_engine_and_node_frame{

using test_cross_product::type_list;
using test_cross_product::list_concat;
using test_cross_product::cross_product;
using gtensor::detail::nested_initializer_list_type;

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
struct node_base{};

class e_engine{
protected:
    using root_type = node_base;
    e_engine() = default;
    e_engine(const root_type* root__):
        root_{root__}
    {}
    auto root()const{return root_;}

private:
    const root_type* root_;
public:
    void set_root(const root_type* root__){root_ = root__;}
};

class stor_engine : public e_engine{
public:
    using w_types = type_list<sw>;
    stor_engine() = default;
    explicit stor_engine(const e_engine::root_type* root):
        e_engine{root}
    {}
    template<typename Nested>
    stor_engine(const e_engine::root_type* root, std::initializer_list<Nested> init_data):
        e_engine{root}
    {
        std::cout<<std::endl<<"stor_engine(const e_engine::root_type* root, std::initializer_list<Nested> init_data):";
    }
};

template<typename F, typename...Ops>
class e_eval_engine : public e_engine
{
    static constexpr std::size_t max_w_types_size = 100;
    template<typename...Us> using ew_alias = ew<F, Us...>;
    static constexpr std::size_t w_types_size = (Ops::engine_type::w_types::size*...);
    template<bool> struct w_types_traits{using type = typename list_concat<type_list<sw>, typename cross_product<ew_alias, typename Ops::engine_type::w_types...>::type>::type;};
    template<> struct w_types_traits<false>{using type = type_list<w>;};
    using e_engine::root_type;
    std::tuple<std::shared_ptr<Ops>...> ops;
    //std::pair<std::shared_ptr<Ops>...> ops;
public:
    using w_types = typename w_types_traits<(w_types_size<max_w_types_size)>::type;
    template<typename...Args, std::enable_if_t<sizeof...(Args)==sizeof...(Ops),int> = 0 >
    explicit e_eval_engine(Args&&...args):
        ops{std::forward<Args>(args)...}
    {}
    template<typename...Args>
    explicit e_eval_engine(root_type* root, Args&&...args):
        ops{std::forward<Args>(args)...},
        e_engine{root}
    {}
};



template<typename EngineT>
class node_frame : public node_base
{
public:
    using engine_type = EngineT;

    //the problem with forwarding when we have more then one member to initialize
    template<typename...Args>
    explicit node_frame(Args&&...args):
        engine_{this, std::forward<Args>(args)...}
    {}

    explicit node_frame(engine_type&& engine__):
        engine_{std::move(engine__)}
    {engine_.set_root(this);}
private:
    engine_type engine_;
};

template<typename ImplT = node_frame<stor_engine>>
class node
{
    using impl_type = ImplT;
    //using default_type = node_frame<stor_engine>;
    using default_type = impl_type;
    std::shared_ptr<impl_type> impl_;

    template<typename Nested>
    node(std::initializer_list<Nested> init_data, int):
        impl_{std::make_shared<impl_type>(init_data)}
    {}

public:
    //create default impl
    template<typename D = default_type, std::enable_if_t<std::is_convertible_v<D,impl_type> ,int> = 0 >
    node():
        impl_{std::make_shared<default_type>()}
    {}

    node(typename nested_initializer_list_type<int,1>::type init_data):node(init_data,0){}
    node(typename nested_initializer_list_type<int,2>::type init_data):node(init_data,0){}

    //get impl from outside
    node(std::shared_ptr<impl_type>&& impl__):
        impl_{std::move(impl__)}
    {}
    auto impl()const{return impl_;}
};

// template<typename E1, typename E2> auto operator+(const node_frame<E1>& op1, const node_frame<E2>& op2){
//     using engine_type = e_eval_engine<add, node_frame<E1>, node_frame<E2>>;
//     return node_frame<engine_type>{engine_type{op1,op2}};
// }

template<typename Impl1, typename Impl2> auto operator+(const node<Impl1>& op1, const node<Impl2>& op2){
    using engine_type = e_eval_engine<add, Impl1, Impl2>;
    using impl_type = node_frame<engine_type>;
    return node<impl_type>(std::make_shared<impl_type>(op1.impl(),op2.impl()));
}
template<typename E1, typename E2> auto operator*(const node_frame<E1>& op1, const node_frame<E2>& op2){
    using engine_type = e_eval_engine<mul, node_frame<E1>, node_frame<E2>>;
    return node_frame<engine_type>{op1,op2};
}

}   //end of namespace binary_tree_with_engine_and_node_frame

namespace binary_tree_with_engine_and_node_frame_without_shared_ptr{

using test_cross_product::type_list;
using test_cross_product::list_concat;
using test_cross_product::cross_product;

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
struct node_base{};

class e_engine{
protected:
    using root_type = node_base;
    e_engine() = default;
    e_engine(const root_type* root__):
        root_{root__}
    {}
    auto root()const{return root_;}

private:
    const root_type* root_;
public:
    void set_root(const root_type* root__){root_ = root__;}
};

class stor_engine : public e_engine{
public:
    using w_types = type_list<sw>;
    stor_engine() = default;
    explicit stor_engine(const e_engine::root_type* root):
        e_engine{root}
    {}
};

template<typename F, typename...Ops>
class e_eval_engine : public e_engine
{
    static constexpr std::size_t max_w_types_size = 100;
    template<typename...Us> using ew_alias = ew<F, Us...>;
    static constexpr std::size_t w_types_size = (Ops::engine_type::w_types::size*...);
    template<bool> struct w_types_traits{using type = typename list_concat<type_list<sw>, typename cross_product<ew_alias, typename Ops::engine_type::w_types...>::type>::type;};
    template<> struct w_types_traits<false>{using type = type_list<w>;};
    using e_engine::root_type;
    std::tuple<Ops...> ops;
    //std::pair<Ops...> ops;
public:
    using w_types = typename w_types_traits<(w_types_size<max_w_types_size)>::type;
    template<typename...Args, std::enable_if_t<sizeof...(Args)==sizeof...(Ops),int> = 0 >
    explicit e_eval_engine(Args&&...args):
        ops{std::forward<Args>(args)...}
    {}
    template<typename...Args>
    explicit e_eval_engine(root_type* root, Args&&...args):
        ops{std::forward<Args>(args)...},
        e_engine{root}
    {}
};

template<typename EngineT>
class node_frame : public node_base
{
public:
    using engine_type = EngineT;
    template<typename...Args>
    explicit node_frame(Args&&...args):
        engine_{this, std::forward<Args>(args)...}
    {}

    explicit node_frame(engine_type&& engine__):
        engine_{std::move(engine__)}
    {engine_.set_root(this);}
private:
    engine_type engine_;
};

template<typename ImplT = node_frame<stor_engine>>
class node
{
    using impl_type = ImplT;
    using default_type = node_frame<stor_engine>;
    impl_type impl_;
public:
    //create default impl
    template<typename D = default_type, std::enable_if_t<std::is_convertible_v<D,impl_type> ,int> = 0 >
    node():
        impl_{default_type{}}
    {}
    //get impl from outside
    node(impl_type&& impl__):
        impl_{std::move(impl__)}
    {}
    auto impl()const{return impl_;}
};

template<typename Impl1, typename Impl2> auto operator+(const node<Impl1>& op1, const node<Impl2>& op2){
    using engine_type = e_eval_engine<add, Impl1, Impl2>;
    using impl_type = node_frame<engine_type>;
    return node<impl_type>{impl_type{op1.impl(),op2.impl()}};
}

}   //end of namespace binary_tree_with_engine_and_node_frame_without_shared_ptr

namespace lambda_universal_reference{

    template<typename T>
    bool is_same_as_decayed(T&& arg){
        return std::is_same_v<std::decay_t<T> , T>;
    }
    template<typename T>
    bool is_same_as_decayed_ref(T&& arg){
        return std::is_same_v<std::decay_t<T>& , T>;
    }
    template<typename T>
    bool is_decltype_arg_same_as_decayed(T&& arg){
        return std::is_same_v<std::decay_t<T> , decltype(arg)>;
    }
    template<typename T>
    bool is_decltype_arg_same_as_decayed_ref(T&& arg){
        return std::is_same_v<std::decay_t<T>& , decltype(arg)>;
    }
    template<typename T>
    bool is_decltype_arg_same_as_decayed_rref(T&& arg){
        return std::is_same_v<std::decay_t<T>&& , decltype(arg)>;
    }

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

TEST_CASE("test_cross_product","[test_cross_product]"){
    using test_cross_product::type_list;
    using test_cross_product::type_pair;
    using test_cross_product::cross_product;
    using test_cross_product::A;
    using test_cross_product::B;
    using test_cross_product::C;
    using test_cross_product::D;
    using test_cross_product::E;
    using test_cross_product::F;

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
}

TEST_CASE("binary_tree_with_engine","[binary_tree_with_engine]"){
    using binary_tree_with_engine::node;
    using binary_tree_with_engine::snode;
    using binary_tree_with_engine::enode;
    using benchmark_helpers::make_asymmetric_tree;

    constexpr std::size_t asymmetric_tree_depth = 5;
    auto e = make_asymmetric_tree<asymmetric_tree_depth>(node{},node{});
    REQUIRE(decltype(e)::impl_type::engine_type::w_types::size == asymmetric_tree_depth+1);
}

TEST_CASE("binary_tree_with_engine_and_node_frame","[binary_tree_with_engine_and_node_frame]"){
    using binary_tree_with_engine_and_node_frame::node_frame;
    using binary_tree_with_engine_and_node_frame::node;
    using binary_tree_with_engine_and_node_frame::stor_engine;
    using snode = node_frame<stor_engine>;
    using benchmark_helpers::make_asymmetric_tree;

    auto s = node<>{1,2,3};
    auto s1 = node<>{{1,2,3}};
    auto s2 = node<>{{1,2,3},{4,5,6}};

    constexpr std::size_t asymmetric_tree_depth = 100;
    auto e = make_asymmetric_tree<asymmetric_tree_depth>(node<>{}, node<>{});
    REQUIRE(decltype(e.impl())::element_type::engine_type::w_types::size == asymmetric_tree_depth+1);
}

TEST_CASE("binary_tree_with_engine_and_node_frame_without_shared_ptr","[binary_tree_with_engine_and_node_frame_without_shared_ptr]"){
    using binary_tree_with_engine_and_node_frame_without_shared_ptr::node_frame;
    using binary_tree_with_engine_and_node_frame_without_shared_ptr::node;
    using binary_tree_with_engine_and_node_frame_without_shared_ptr::stor_engine;
    using snode = node_frame<stor_engine>;
    using benchmark_helpers::make_asymmetric_tree;

    constexpr std::size_t asymmetric_tree_depth = 5;
    auto e = make_asymmetric_tree<asymmetric_tree_depth>(node<>{}, node<>{});
    REQUIRE(decltype(e.impl())::engine_type::w_types::size == asymmetric_tree_depth+1);
}





TEST_CASE("lambda_universal_reference", "[lambda_universal_reference]"){
    using lambda_universal_reference::is_same_as_decayed;
    using lambda_universal_reference::is_same_as_decayed_ref;
    using lambda_universal_reference::is_decltype_arg_same_as_decayed_ref;
    using lambda_universal_reference::is_decltype_arg_same_as_decayed_rref;
    using lambda_universal_reference::is_decltype_arg_same_as_decayed;

    int x{};
    REQUIRE(is_same_as_decayed(4));
    REQUIRE(is_same_as_decayed_ref(x));

    REQUIRE(is_decltype_arg_same_as_decayed_ref(x));
    REQUIRE(is_decltype_arg_same_as_decayed_rref(4));

    auto f = [](auto&& arg){return is_same_as_decayed(std::forward<decltype(arg)>(arg));};
    auto g = [](auto&& arg){return is_same_as_decayed_ref(std::forward<decltype(arg)>(arg));};

    REQUIRE(f(4));
    REQUIRE(g(x));

}