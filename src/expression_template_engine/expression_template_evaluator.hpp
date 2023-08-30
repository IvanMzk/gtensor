#ifndef EXPRESSION_TEMPLATE_EVALUATOR_HPP_
#define EXPRESSION_TEMPLATE_EVALUATOR_HPP_

#include <type_traits>
#include "common.hpp"
namespace gtensor{

namespace detail{

template<std::size_t I, typename F, typename...Ts, std::enable_if_t<(I==sizeof...(Ts)),int> =0>
ALWAYS_INLINE void apply_per_element(const F&, std::tuple<Ts...>&){}

template<std::size_t I=0, typename F, typename...Ts, std::enable_if_t<(I<sizeof...(Ts)),int> =0>
ALWAYS_INLINE void apply_per_element(const F& f, std::tuple<Ts...>& t){
    f(std::get<I>(t));
    apply_per_element<I+1>(f,t);
}

}

//expression template evaluator
template<typename Config, typename F, typename...Walkers>
class expression_template_walker
{
    template<typename...Ts> using tuple_type = std::tuple<Ts...>;
    using sequence_type = std::make_index_sequence<sizeof...(Walkers)>;
public:
    using config_type = Config;
    using dim_type = typename Config::dim_type;
    using index_type = typename Config::index_type;
    using shape_type = typename Config::shape_type;
    using size_type = typename shape_type::size_type;
    using value_type = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<F>()(*std::declval<Walkers>()...))>>;

    template<typename F_> struct forward_args : std::bool_constant<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<F_>>,expression_template_walker>>{};

    template<typename F_, typename...Walkers_, std::enable_if_t<forward_args<F_>::value,int> =0>
    explicit expression_template_walker(F_&& f__, Walkers_&&...walkers__):
        f_{std::forward<F_>(f__)},
        walkers_{std::forward<Walkers_>(walkers__)...}
    {}

    ALWAYS_INLINE void walk(const dim_type& axis, const index_type& steps){
        auto f = [&axis,&steps](auto& w){w.walk(axis,steps);};
        detail::apply_per_element(f,walkers_);
    }
    ALWAYS_INLINE void walk_back(const dim_type& axis, const index_type& steps){
        auto f = [&axis,&steps](auto& w){w.walk_back(axis,steps);};
        detail::apply_per_element(f,walkers_);
    }
    ALWAYS_INLINE void step(const dim_type& axis){
        auto f = [&axis](auto& w){w.step(axis);};
        detail::apply_per_element(f,walkers_);
    }
    ALWAYS_INLINE void step_back(const dim_type& axis){
        auto f = [&axis](auto& w){w.step_back(axis);};
        detail::apply_per_element(f,walkers_);
    }
    ALWAYS_INLINE void reset(const dim_type& axis){
        auto f = [&axis](auto& w){w.reset(axis);};
        detail::apply_per_element(f,walkers_);
    }
    ALWAYS_INLINE void reset_back(const dim_type& axis){
        auto f = [&axis](auto& w){w.reset_back(axis);};
        detail::apply_per_element(f,walkers_);
    }
    ALWAYS_INLINE void reset_back(){
        auto f = [](auto& w){w.reset_back();};
        detail::apply_per_element(f,walkers_);

    }
    ALWAYS_INLINE void update_offset(){
        auto f = [](auto& w){w.update_offset();};
        detail::apply_per_element(f,walkers_);
    }
    ALWAYS_INLINE decltype(auto) operator*()const{
        return deref_helper(sequence_type{});
    }
private:
    template<std::size_t...I>
    ALWAYS_INLINE decltype(auto) deref_helper(std::index_sequence<I...>)const{
        return f_(*std::get<I>(walkers_)...);
    }

    F f_;
    tuple_type<Walkers...> walkers_;
};

template<typename Config, typename F, typename...Indexers>
class expression_template_trivial_indexer
{
    template<typename...Ts> using tuple_type = std::tuple<Ts...>;
    using sequence_type = std::make_index_sequence<sizeof...(Indexers)>;
public:
    using config_type = Config;
    using dim_type = typename Config::dim_type;
    using index_type = typename Config::index_type;
    using shape_type = typename Config::shape_type;
    using value_type = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<F>()(std::declval<Indexers>()[std::declval<index_type>()]...))>>;

    template<typename F_> struct forward_args : std::bool_constant<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<F_>>,expression_template_trivial_indexer>>{};

    template<typename F_, typename...Indexers_, std::enable_if_t<forward_args<F_>::value,int> =0>
    explicit expression_template_trivial_indexer(F_&& f__, Indexers_&&...indexers__):
        f_{std::forward<F_>(f__)},
        indexers_{std::forward<Indexers_>(indexers__)...}
    {}
    ALWAYS_INLINE decltype(auto) operator[](const index_type& idx)const{
        return subscript_helper(idx, sequence_type{});
    }
private:
    template<std::size_t...I>
    ALWAYS_INLINE decltype(auto) subscript_helper(const index_type& idx, std::index_sequence<I...>)const{
        return f_(std::get<I>(indexers_)[idx]...);
    }

    F f_;
    tuple_type<Indexers...> indexers_;
};

}   //end of namespace gtensor
#endif