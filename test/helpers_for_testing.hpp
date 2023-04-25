#ifndef HELPERS_FOR_TESTING_HPP_
#define HELPERS_FOR_TESTING_HPP_

#include <tuple>
#include <array>
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

template<typename> inline constexpr bool always_false = false;

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
        lvalue_ref_wrapper(const lvalue_ref_wrapper&) = default;
        //cant assign to reference
        lvalue_ref_wrapper& operator=(const lvalue_ref_wrapper&) = delete;
        lvalue_ref_wrapper& operator=(lvalue_ref_wrapper&&) = delete;
        explicit operator T&()const{return *wrapped_;}
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
        rvalue_ref_wrapper(const rvalue_ref_wrapper&) = default;
        //cant assign to reference
        rvalue_ref_wrapper& operator=(const rvalue_ref_wrapper&) = delete;
        rvalue_ref_wrapper& operator=(rvalue_ref_wrapper&&) = delete;
        explicit operator T&&()const{return static_cast<T&&>(*wrapped_);}
        //must allow to emulate reference collapsing
        explicit operator T&()const{return *wrapped_;}
    };
    template<typename T> struct type_adapter{using type = T;};
    template<typename T> struct type_adapter<T&>{using type = lvalue_ref_wrapper<T>;};
    template<typename T> struct type_adapter<T&&>{using type = rvalue_ref_wrapper<T>;};
    template<typename T> struct type_adapter<lvalue_ref_wrapper<T>>{using type = T&;};
    template<typename T> struct type_adapter<rvalue_ref_wrapper<T>>{using type = T&&;};
    template<typename T> using type_adapter_t = typename type_adapter<T>::type;

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

template<typename> struct tuple_size;
template<std::size_t, typename> struct tuple_element;
template<typename T> inline constexpr std::size_t tuple_size_v = tuple_size<T>::value;
template<std::size_t I, typename T> using tuple_element_t = typename tuple_element<I,T>::type;

template<typename...Types>
class tuple
{
    template<typename T> using type_adapter_t = tuple_details::type_adapter_t<T>;
public:
    using size_type = std::size_t;
    static constexpr size_type tuple_size = sizeof...(Types);
    using type_list_indexer = tuple_details::type_list_indexer_4<Types...>;
    ~tuple()
    {
        destroy_elements(std::make_integer_sequence<size_type, tuple_size>{});
    }
    //default constructor
    tuple()
    {
        init_elements_default(std::make_integer_sequence<size_type, tuple_size>{});
    }
    //direct constructor, must be template to disambiguate with default constructor for tuple<>
    template<typename = void>
    explicit tuple(const Types&...args)
    {
        std::cout<<std::endl<<"explicit tuple(const Types&...args)"<<sizeof...(Types);
        init_elements(std::make_integer_sequence<size_type, tuple_size>{}, args...);
    }
    //copy,move operations
    tuple(const tuple& other)
    {
        std::cout<<std::endl<<"tuple(const tuple& other)";
        copy_elements_(*this, other, std::make_integer_sequence<size_type, tuple_size>{});
    }
    tuple& operator=(const tuple& other)
    {
        std::cout<<std::endl<<"tuple& operator=(const tuple& other)";
        copy_assign_elements_(*this, other, std::make_integer_sequence<size_type, tuple_size>{});
        return *this;
    }
    tuple(tuple&& other)
    {
        std::cout<<std::endl<<"tuple(tuple&& other)";
        move_elements_(*this, std::move(other), std::make_integer_sequence<size_type, tuple_size>{});
    }
    tuple& operator=(tuple&& other)
    {
        std::cout<<std::endl<<"tuple& operator=(tuple&& other)";
        move_assign_elements_(*this, std::move(other), std::make_integer_sequence<size_type, tuple_size>{});
        return *this;
    }
    //converting constructors
    template<typename, typename...> struct forward_args : std::false_type{};
    template<typename...Ts, typename...Args> struct forward_args<tuple<Ts...>,Args...> : std::bool_constant<sizeof...(Ts)==sizeof...(Args)>{};
    template<typename T,typename Arg> struct forward_args<tuple<T>,Arg> :
        std::bool_constant<std::is_constructible_v<T,Arg>&&!std::is_same_v<tuple<T>,std::remove_cv_t<std::remove_reference_t<Arg>>>>{};

    template<typename...Args, std::enable_if_t<forward_args<tuple,Args...>::value,int> =0>
    explicit tuple(Args&&...args)
    {
        std::cout<<std::endl<<"explicit tuple(Args&&...args)"<<sizeof...(Args);
        init_elements(std::make_integer_sequence<size_type, tuple_size>{}, std::forward<Args>(args)...);
    }

    template<typename,typename> struct copy_convert_tuple : std::false_type{};
    template<typename V> struct copy_convert_tuple<V,V> : std::false_type{};
    template<typename...Ts, typename...Us> struct copy_convert_tuple<tuple<Ts...>,tuple<Us...>> : std::bool_constant<sizeof...(Ts)==sizeof...(Us)>{};
    template<typename T, typename U> struct copy_convert_tuple<tuple<T>,tuple<U>> :
        std::bool_constant<!(std::is_same_v<T,U>||std::is_convertible_v<const tuple<U>&,T>||std::is_constructible_v<T,const tuple<U>&>)>{};

    template<typename...Us, std::enable_if_t<copy_convert_tuple<tuple, tuple<Us...>>::value,int> = 0>
    explicit tuple(const tuple<Us...>& other)
    {
        std::cout<<std::endl<<"tuple(const tuple<Ts...>& other)";
        copy_elements_(*this, other, std::make_integer_sequence<size_type, tuple_size>{});
    }

    template<typename, typename> struct move_convert_tuple : std::false_type{};
    template<typename V> struct move_convert_tuple<V,V> : std::false_type{};
    template<typename...Ts, typename...Us> struct move_convert_tuple<tuple<Ts...>,tuple<Us...>> : std::bool_constant<sizeof...(Ts)==sizeof...(Us)>{};
    template<typename T, typename U> struct move_convert_tuple<tuple<T>,tuple<U>> :
        std::bool_constant<!(std::is_same_v<T,U>||std::is_convertible_v<tuple<U>,T>||std::is_constructible_v<T,tuple<U>>)>{};

    template<typename...Us, std::enable_if_t<move_convert_tuple<tuple, tuple<Us...>>::value,int> = 0>
    explicit tuple(tuple<Us...>&& other)
    {
        std::cout<<std::endl<<"tuple(tuple<Ts...>&& other)";
        move_elements_(*this, std::move(other), std::make_integer_sequence<size_type, tuple_size>{});
    }
    //converting assignment
    template<typename,typename> struct assign_convert : std::false_type{};
    template<typename V> struct assign_convert<V,V> : std::false_type{};
    template<typename...Ts, typename...Us> struct assign_convert<tuple<Ts...>,tuple<Us...>> : std::bool_constant<sizeof...(Ts)==sizeof...(Us)>{};

    template<typename...Us, std::enable_if_t<assign_convert<tuple, tuple<Us...>>::value,int> = 0>
    tuple& operator=(const tuple<Us...>& other)
    {
        std::cout<<std::endl<<"tuple& operator=(const tuple<Us...>& other)";
        copy_assign_elements_(*this, other, std::make_integer_sequence<size_type, tuple_size>{});
        return *this;
    }
    template<typename...Us, std::enable_if_t<assign_convert<tuple, tuple<Us...>>::value,int> = 0>
    tuple& operator=(tuple<Us...>&& other)
    {
        std::cout<<std::endl<<"tuple& operator=(tuple<Us...>&& other)";
        copy_assign_elements_(*this, std::move(other), std::make_integer_sequence<size_type, tuple_size>{});
        return *this;
    }

    //add swap
    //add make_tuple
    //add concat_tuple

    template<typename...Ts,typename...Vs, std::size_t...I> friend void copy_elements_(tuple<Ts...>& lhs, const tuple<Vs...>& rhs, std::integer_sequence<std::size_t, I...>);
    template<typename...Ts,typename...Vs, std::size_t...I> friend void copy_assign_elements_(tuple<Ts...>& lhs, const tuple<Vs...>& rhs, std::integer_sequence<std::size_t, I...>);
    template<typename...Ts,typename...Vs, std::size_t...I> friend void move_elements_(tuple<Ts...>& lhs, tuple<Vs...>&& rhs, std::integer_sequence<std::size_t, I...>);
    template<typename...Ts,typename...Vs, std::size_t...I> friend void move_assign_elements_(tuple<Ts...>& lhs, tuple<Vs...>&& rhs, std::integer_sequence<std::size_t, I...>);
    template<typename...Ts,typename...Vs, std::size_t...I> friend bool equals_(const tuple<Ts...>& lhs, const tuple<Vs...>& rhs, std::integer_sequence<std::size_t, I...>);
    template<size_type I, typename...Ts> friend tuple_element_t<I,tuple<Ts...>>& get(tuple<Ts...>&);
    template<size_type I, typename...Ts> friend const tuple_element_t<I,tuple<Ts...>>& get(const tuple<Ts...>&);
    template<size_type I, typename...Ts> friend tuple_element_t<I,tuple<Ts...>>&& get(tuple<Ts...>&&);
    template<size_type I, typename...Ts> friend const tuple_element_t<I,tuple<Ts...>>&& get(const tuple<Ts...>&&);

private:
    template<typename U>
    static constexpr size_type size_of_type(){
        if constexpr (std::is_void_v<U>){
            return 0;
        }else{
            return sizeof(U);
        }
    }
    static constexpr size_type size(){
        if constexpr (tuple_size == 0){
            return 0;
        }else{
            return (...+size_of_type<type_adapter_t<Types>>());
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
        static_assert(I < tuple_size);
        return make_offset<I, type_adapter_t<Types>...>();
    }
    template<size_type...I>
    static constexpr auto make_offsets(std::integer_sequence<size_type, I...>){
        return std::array<size_type, size()>{make_offset<I>()...};
    }

    template<size_type I>
    void* get_(){
        return elements_.data()+offsets_[I];
    }
    template<size_type I>
    const void* get_()const{
        return elements_.data()+offsets_[I];
    }

    template<typename InnerType>
    void destroy_element(InnerType* p){
        p->~InnerType();
    }
    template<typename InnerType>
    void destroy_element(size_type i, size_type n, InnerType* p){
        if (i<n){
            p->~InnerType();
        }
    }
    template<size_type...I>
    void destroy_elements(std::integer_sequence<size_type, I...>){
        (destroy_element(reinterpret_cast<type_adapter_t<Types>*>(get_<I>())),...);
    }
    template<size_type...I>
    void destroy_first_n_elements(size_type n, std::integer_sequence<size_type, I...>){
        (destroy_element(I, n, reinterpret_cast<type_adapter_t<Types>*>(get_<I>())),...);
    }

    template<std::size_t I, typename ThisElementType, typename OtherElementType>
    void emplace_element(void* this_place, OtherElementType&& other_element){
        try{
            new(this_place) ThisElementType(std::forward<OtherElementType>(other_element));
        }catch(...){
            destroy_first_n_elements(I, std::make_integer_sequence<size_type,tuple_size>{});
        }
    }
    template<std::size_t I, typename ThisElementType>
    void emplace_element_default(void* this_place){
        try{
            new(this_place) ThisElementType{};
        }catch(...){
            destroy_first_n_elements(I, std::make_integer_sequence<size_type,tuple_size>{});
        }
    }
    template<size_type...I, typename...Args>
    void init_elements(std::integer_sequence<size_type, I...>, Args&&...args){
        (emplace_element<I,type_adapter_t<Types>>(get_<I>(),std::forward<Args>(args)),...);
    }

    template<size_type...I>
    void init_elements_default(std::integer_sequence<size_type, I...>){
        (emplace_element_default<I,type_adapter_t<Types>>(get_<I>()),...);
    }

    static constexpr std::array<size_type, size()> offsets_{make_offsets(std::make_integer_sequence<size_type, tuple_size>{})};
    std::array<std::byte,size()> elements_;
};

//tuple_size
template<typename...Ts> struct tuple_size<tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)>{};
//tuple_element
template<std::size_t I, typename T> struct tuple_element<I,const T>{
    using type = std::add_const_t<typename tuple_element<I,T>::type>;
};
template<std::size_t I> struct tuple_element<I,tuple<>>{
    static_assert(tuple_details::always_false<std::integral_constant<std::size_t,I>>, "tuple index out of bounds");
};
template<std::size_t I, typename...Ts> struct tuple_element<I,tuple<Ts...>>{
    using type = typename tuple<Ts...>::type_list_indexer::template at<I>;
};
//get by index
template<std::size_t I, typename...Ts>
tuple_element_t<I,tuple<Ts...>>& get(tuple<Ts...>& t){
    using element_type = tuple_element_t<I,tuple<Ts...>>;
    return static_cast<element_type&>(*reinterpret_cast<tuple_details::type_adapter_t<element_type>*>(t.template get_<I>()));
}
template<std::size_t I, typename...Ts>
const tuple_element_t<I,tuple<Ts...>>& get(const tuple<Ts...>& t){
    using element_type = tuple_element_t<I,tuple<Ts...>>;
    return static_cast<const element_type&>(*reinterpret_cast<const tuple_details::type_adapter_t<element_type>*>(t.template get_<I>()));
}
template<std::size_t I, typename...Ts>
tuple_element_t<I,tuple<Ts...>>&& get(tuple<Ts...>&& t){
    using element_type = tuple_element_t<I,tuple<Ts...>>;
    return static_cast<element_type&&>(*reinterpret_cast<tuple_details::type_adapter_t<element_type>*>(t.template get_<I>()));
}
template<std::size_t I, typename...Ts>
const tuple_element_t<I,tuple<Ts...>>&& get(const tuple<Ts...>&& t){
    using element_type = tuple_element_t<I,tuple<Ts...>>;
    return static_cast<const element_type&&>(*reinterpret_cast<const tuple_details::type_adapter_t<element_type>*>(t.template get_<I>()));
}
//tuple helper friends
template<typename...Ts,typename...Vs, std::size_t...I>
void copy_elements_(tuple<Ts...>& this_, const tuple<Vs...>& other_, std::integer_sequence<std::size_t, I...>){
    (this_.template emplace_element<I,tuple_details::type_adapter_t<Ts>>(this_.template get_<I>(), *reinterpret_cast<const tuple_details::type_adapter_t<Vs>*>(other_.template get_<I>())),...);
}
template<typename...Ts,typename...Vs, std::size_t...I>
void copy_assign_elements_(tuple<Ts...>& this_, const tuple<Vs...>& other_, std::integer_sequence<std::size_t, I...>){
    ((*reinterpret_cast<tuple_details::type_adapter_t<Ts>*>(this_.template get_<I>()) = *reinterpret_cast<const tuple_details::type_adapter_t<Vs>*>(other_.template get_<I>())),...);
}
template<typename...Ts,typename...Vs, std::size_t...I>
void move_elements_(tuple<Ts...>& this_, tuple<Vs...>&& other_, std::integer_sequence<std::size_t, I...>){
    (this_.template emplace_element<I,tuple_details::type_adapter_t<Ts>>(this_.template get_<I>(), std::move(*reinterpret_cast<tuple_details::type_adapter_t<Vs>*>(other_.template get_<I>()))),...);
}
template<typename...Ts,typename...Vs, std::size_t...I>
void move_assign_elements_(tuple<Ts...>& this_, tuple<Vs...>&& other_, std::integer_sequence<std::size_t, I...>){
    ((*reinterpret_cast<tuple_details::type_adapter_t<Ts>*>(this_.template get_<I>()) = std::move(*reinterpret_cast<tuple_details::type_adapter_t<Vs>*>(other_.template get_<I>()))),...);
}
template<typename...Ts,typename...Vs, std::size_t...I>
bool equals_(const tuple<Ts...>& lhs, const tuple<Vs...>& rhs, std::integer_sequence<std::size_t, I...>){
    return (...&&(static_cast<const Ts&>(*reinterpret_cast<const tuple_details::type_adapter_t<Ts>*>(lhs.template get_<I>())) ==
    static_cast<const Vs&>(*reinterpret_cast<const tuple_details::type_adapter_t<Vs>*>(rhs.template get_<I>()))));
}
//tuple operators
template<typename...Ts,typename...Vs>
bool operator==(const tuple<Ts...>& lhs, const tuple<Vs...>& rhs){
    static_assert(sizeof...(Ts) == sizeof...(Vs), "cannot compare tuples of different sizes");
    return equals_(lhs,rhs,std::make_integer_sequence<std::size_t, sizeof...(Ts)>{});
}
template<typename...Ts,typename...Vs>
bool operator!=(const tuple<Ts...>& lhs, const tuple<Vs...>& rhs){
    return !(lhs==rhs);
}

}   //end of namespace helpers_for_testing

#endif