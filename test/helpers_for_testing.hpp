#ifndef HELPERS_FOR_TESTING_HPP_
#define HELPERS_FOR_TESTING_HPP_

#include <tuple>
#include <functional>

namespace helpers_for_testing{

template<typename...> struct list_concat;
template<template<typename...> typename L, typename...Us>
struct list_concat<L<Us...>>{
    using type = L<Us...>;
};
template<template<typename...> typename L, typename...Us, typename...Vs>
struct list_concat<L<Us...>,L<Vs...>>{
    using type = L<Us...,Vs...>;
};
template<typename T1, typename T2, typename...Tail>
struct list_concat<T1,T2,Tail...>{
    using type = typename list_concat<typename list_concat<T1,T2>::type, Tail...>::type;
};

template<template <typename...> typename, typename, typename> struct cross_product;
template<template <typename...> typename PairT, template<typename...> typename L, typename U, typename...Us, typename...Vs>
struct cross_product<PairT, L<U, Us...>, L<Vs...>>{
    using cross_u_vs = L<PairT<U,Vs>...>;
    using cross_us_vs = typename cross_product<PairT, L<Us...>, L<Vs...>>::type;
    using type = typename list_concat<cross_u_vs, cross_us_vs>::type;
};
template<template <typename...> typename PairT, template<typename...> typename L, typename...Vs>
struct cross_product<PairT, L<>, L<Vs...>>{
    using type = L<>;
};

//apply f to each element of t
template<typename F, typename Tuple, std::size_t...I>
inline auto apply_by_element(F&& f, Tuple&& t, std::index_sequence<I...>){
    if constexpr(std::disjunction_v<std::is_void<decltype(std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))))>...>){
        (std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))),...);
    }else{
        return std::make_tuple(std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t)))...);
    }
}
template<typename F, typename Tuple>
inline auto apply_by_element(F&& f, Tuple&& t){
    using tuple_type = std::decay_t<Tuple>;
    return apply_by_element(std::forward<F>(f), std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size_v<tuple_type>>{});
}

//safe cmp of signed,unsigned integrals
template<typename T, typename U>
inline constexpr bool cmp_equal(T t, U u){
    using UT = std::make_unsigned_t<T>;
    using UU = std::make_unsigned_t<U>;
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
        return t==u;
    else if constexpr (std::is_signed_v<T>)
        return t<0 ? false : UT(t) == u;
    else
        return u<0 ? false : t == UU(u);
}
template<typename T, typename U>
inline constexpr bool cmp_not_equal(T t, U u){
    return !cmp_equal(t, u);
}
template<typename T, typename U>
inline constexpr bool cmp_less(T t, U u){
    using UT = std::make_unsigned_t<T>;
    using UU = std::make_unsigned_t<U>;
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
        return t<u;
    else if constexpr (std::is_signed_v<T>)
        return t<0 ? true : UT(t) < u;
    else
        return u<0 ? false : t < UU(u);
}
template<typename T, typename U>
inline constexpr bool cmp_greater(T t, U u){
    return cmp_less(u, t);
}
template<typename T, typename U>
inline constexpr bool cmp_less_equal(T t, U u){
    return !cmp_greater(t, u);
}
template<typename T, typename U>
inline constexpr bool cmp_greater_equal(T t, U u){
    return !cmp_less(t, u);
}


namespace tuple_details{

template<typename T>
    class lvalue_ref_wrapper
    {
        T* wrapped_;
        static std::true_type can_bound(T&);
        static std::false_type can_bound(...);
    public:
        template<typename U, std::enable_if_t<decltype(can_bound(std::declval<U>()))::value&&!std::is_convertible_v<std::decay_t<U>,lvalue_ref_wrapper>,int> =0>
        explicit lvalue_ref_wrapper(U&& u):
            wrapped_{&u}
        {}

    };
    template<typename T>
    class rvalue_ref_wrapper
    {
        T* wrapped_;
        static std::true_type can_bound(T&&);
        static std::false_type can_bound(...);
    public:
        template<typename U, std::enable_if_t<decltype(can_bound(std::declval<U>()))::value&&!std::is_convertible_v<std::decay_t<U>,rvalue_ref_wrapper>,int> =0>
        explicit rvalue_ref_wrapper(U&& u):
            wrapped_{&u}
        {}
    };
    template<typename T> struct type_adapter{using type = T;};
    template<typename T> struct type_adapter<T&>{using type = lvalue_ref_wrapper<T>;};
    template<typename T> struct type_adapter<T&&>{using type = rvalue_ref_wrapper<T>;};

    //type list indexing helpers
    template<typename, typename...> struct split_list_2;
    template<template<typename...> typename L, typename...Vs, typename T0, typename T1, typename...Us>
    struct split_list_2<L<Vs...>,T0,T1,Us...>{
        using type = typename split_list_2<L<Vs...,L<T0,T1>>,Us...>::type;
    };
    template<template<typename...> typename L, typename...Vs, typename...Us>
    struct split_list_2<L<Vs...>,Us...>{
        using type = L<Vs...,L<Us...>>;
    };
    template<template<typename...> typename L, typename...Vs>
    struct split_list_2<L<Vs...>>{
        using type = L<Vs...>;
    };

    template<typename, typename...> struct split_list_4;
    template<template<typename...> typename L, typename...Vs, typename T0, typename T1, typename T2, typename T3, typename...Us>
    struct split_list_4<L<Vs...>,T0,T1,T2,T3,Us...>{
        using type = typename split_list_4<L<Vs...,L<T0,T1,T2,T3>>,Us...>::type;
    };
    template<template<typename...> typename L, typename...Vs, typename...Us>
    struct split_list_4<L<Vs...>,Us...>{
        using type = L<Vs...,L<Us...>>;
    };
    template<template<typename...> typename L, typename...Vs>
    struct split_list_4<L<Vs...>>{
        using type = L<Vs...>;
    };

    //ListSize is number of types in type list to be indexed
    //NodeSize is max size of list that is made by split_list that must be equal to index template specializations
    template<std::size_t ListSize, std::size_t NodeSize, std::size_t D=0, std::size_t N=NodeSize>
    static constexpr std::size_t make_type_tree_depth(){
        if constexpr (N >= ListSize || ListSize == 0){
            return D;
        }else{
            return make_type_tree_depth<ListSize, NodeSize, D+1, N*NodeSize>();
        }
    }

    template<std::size_t N>
    static constexpr std::size_t log2(){
        if constexpr (N == 1){
            return 0;
        }else{
            return 1+log2<N/2>();
        }
    }

    template<template<typename...> typename Splitter, std::size_t Depth, typename L> struct make_type_tree;
    template<template<typename...> typename Splitter, std::size_t Depth, template<typename...> typename L, typename...Us>
    struct make_type_tree<Splitter,Depth,L<Us...>>{
        using type = typename make_type_tree<Splitter, Depth-1, typename Splitter<L<>,Us...>::type>::type;
    };
    template<template<typename...> typename Splitter, template<typename...> typename L, typename...Us> struct make_type_tree<Splitter,0,L<Us...>>{
        using type = L<Us...>;
    };

    template<typename...Ts>
    struct indexed{
        template<template<typename...> typename F>
        using f = F<Ts...>;
    };

    template<std::size_t> struct index;
    template<> struct index<0>{
        template<typename T0,typename...>
        using f = T0;
    };
    template<> struct index<1>{
        template<typename T0, typename T1, typename...>
        using f = T1;
    };
    template<> struct index<2>{
        template<typename T0, typename T1, typename T2, typename...>
        using f = T2;
    };
    template<> struct index<3>{
        template<typename T0, typename T1, typename T2, typename T3, typename...>
        using f = T3;
    };

    template<typename U, std::size_t I, std::size_t NodeSize, std::size_t D>
    struct lookup_type_tree{
        static constexpr std::size_t shift = log2<NodeSize>();
        using type = typename lookup_type_tree<typename U::template f<index<(I>>D*shift)&(NodeSize-1)>::template f>, I, NodeSize, D-1>::type;
    };
    template<typename U, std::size_t I, std::size_t NodeSize>
    struct lookup_type_tree<U,I,NodeSize,0>{
        using type = typename U::template f<index<I&(NodeSize-1)>::template f>;
    };

    template<typename...Types>
    class type_list_indexer_2
    {
        template<typename...Us> using splitter = split_list_2<Us...>;
        static constexpr std::size_t node_size = 2; //must be power of two, be less or equal to index template specializations and split_list max list size
        static constexpr std::size_t list_size = sizeof...(Types);
        static constexpr std::size_t type_tree_depth = make_type_tree_depth<list_size,node_size>();
        using type_tree = typename make_type_tree<splitter, type_tree_depth, indexed<Types...>>::type;
    public:
        template<std::size_t I>
        using at = typename lookup_type_tree<type_tree, I, node_size, type_tree_depth>::type;
    };

    template<typename...Types>
    class type_list_indexer_4
    {
        template<typename...Us> using splitter = split_list_4<Us...>;
        static constexpr std::size_t list_size = sizeof...(Types);
        static constexpr std::size_t node_size = 4; //must be power of two, be less or equal to index template specializations and split_list max list size
        static constexpr std::size_t type_tree_depth = make_type_tree_depth<list_size,node_size>();
        using type_tree = typename make_type_tree<splitter, type_tree_depth, indexed<Types...>>::type;
    public:
        template<std::size_t I>
        using at = typename lookup_type_tree<type_tree, I, node_size, type_tree_depth>::type;
    };
}   //end of namespace tuple_details

template<typename...Types>
class basic_tuple
{
    using size_type = std::size_t;
public:

    ~basic_tuple()
    {
        destroy_elements(std::make_integer_sequence<size_type, sizeof...(Types)>{});
    }
    //default constructor
    basic_tuple()
    {
        init_elements_default(std::make_integer_sequence<size_type, sizeof...(Types)>{});
    }
    //converting constructors
    template<typename Arg, std::enable_if_t<!std::is_convertible_v<Arg,basic_tuple>,int> =0>
    explicit basic_tuple(Arg&& arg)
    {
        static_assert(sizeof...(Types) == 1);
        init_elements(std::make_integer_sequence<size_type, sizeof...(Types)>{}, std::forward<Arg>(arg));
    }
    template<typename...Args>
    basic_tuple(Args&&...args)
    {
        static_assert(sizeof...(Types) == sizeof...(Args));
        init_elements(std::make_integer_sequence<size_type, sizeof...(Types)>{}, std::forward<Args>(args)...);
    }
    //copy,move operations
    basic_tuple(const basic_tuple& other)
    {
        copy_elements(std::make_integer_sequence<size_type, sizeof...(Types)>{}, other.elements_);
    }
    basic_tuple& operator=(const basic_tuple& other)
    {
        copy_assign_elements(std::make_integer_sequence<size_type, sizeof...(Types)>{}, other.elements_);
        return *this;
    }
    basic_tuple(basic_tuple&& other)
    {
        move_elements(std::make_integer_sequence<size_type, sizeof...(Types)>{}, other.elements_);
    }
    basic_tuple& operator=(basic_tuple&& other)
    {
        move_assign_elements(std::make_integer_sequence<size_type, sizeof...(Types)>{}, other.elements_);
        return *this;
    }

    //add converting copy,move operations

private:

    template<size_type I>
    void* get_(){
        return static_cast<void*>(elements_+offsets_[I]);
    }

    static constexpr size_type size(){
        if constexpr (sizeof...(Types) == 0){
            return 0;
        }else{
            return (...+sizeof(Types));
        }
    }

    template<size_type I, typename Type_, typename...Types_>
    static constexpr size_type make_offset(){
        if constexpr (I == 0){
            return 0;
        }else{
            return sizeof(Type_)+make_offset<I-1,Types_...>();
        }
    }

    template<size_type I>
    static constexpr size_type make_offset(){
        static_assert(I < sizeof...(Types));
        return make_offset<I, Types...>();
    }

    template<size_type...I>
    static constexpr auto make_offsets(){
        return std::array<size_type, size()>{make_offset<I>()...};
    }

    template<size_type...I>
    void destroy_elements(std::integer_sequence<size_type, I...>){
        ((reinterpret_cast<Types*>(elements_+offsets_[I])->~Types()),...);
    }
    template<size_type...I, typename...Args>
    void init_elements(std::integer_sequence<size_type, I...>, Args&&...args){
        ((new(static_cast<void*>(elements_+offsets_[I])) Types(std::forward<Args>(args))),...);
    }
    template<size_type...I>
    void init_elements_default(std::integer_sequence<size_type, I...>){
        ((new(static_cast<void*>(elements_+offsets_[I])) Types{}),...);
    }
    template<size_type...I>
    void copy_elements(std::integer_sequence<size_type, I...>, const std::byte* other_elements_){
        ((new(static_cast<void*>(elements_+offsets_[I])) Types(*reinterpret_cast<const Types*>(other_elements_+offsets_[I]))),...);
    }
    template<size_type...I>
    void copy_assign_elements(std::integer_sequence<size_type, I...>, const std::byte* other_elements_){
        ((*reinterpret_cast<Types*>(elements_+offsets_[I]) = *reinterpret_cast<const Types*>(other_elements_+offsets_[I])),...);
    }
    template<size_type...I>
    void move_elements(std::integer_sequence<size_type, I...>, std::byte* other_elements_){
        ((new(static_cast<void*>(elements_+offsets_[I])) Types(std::move(*reinterpret_cast<Types*>(other_elements_+offsets_[I])))),...);
    }
    template<size_type...I>
    void move_assign_elements(std::integer_sequence<size_type, I...>, std::byte* other_elements_){
        ((*reinterpret_cast<Types*>(elements_+offsets_[I]) = std::move(*reinterpret_cast<Types*>(other_elements_+offsets_[I]))),...);
    }

    static constexpr std::array<size_type, size()> offsets_{make_offsets()};
    std::byte elements_[size()];
};

template<typename...Types> using tuple = basic_tuple<typename tuple_details::type_adapter<Types>::type...>;



}   //end of namespace helpers_for_testing

#endif