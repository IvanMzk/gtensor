#ifndef EXPRESSION_TEMPLATE_EVALUATOR_HPP_
#define EXPRESSION_TEMPLATE_EVALUATOR_HPP_

#include <type_traits>

namespace gtensor{
//expression template evaluator
template<typename Config, typename F, typename...Walkers>
class expression_template_walker
{
    template<typename...Ts> using tuple_type = std::tuple<Ts...>;
    using sequence_type = std::make_index_sequence<sizeof...(Walkers)>;
    using result_type = decltype(std::declval<F>()(*std::declval<Walkers>()...));
public:
    using config_type = Config;
    using dim_type = typename Config::dim_type;
    using index_type = typename Config::index_type;
    using shape_type = typename Config::shape_type;

    template<typename F_> struct forward_args : std::bool_constant<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<F_>>,expression_template_walker>>{};

    template<typename F_, typename...Walkers_, std::enable_if_t<forward_args<F_>::value,int> =0>
    explicit expression_template_walker(F_&& f__, Walkers_&&...walkers__):
        f_{std::forward<F_>(f__)},
        walkers_{std::forward<Walkers_>(walkers__)...}
    {}

    void walk(const dim_type& axis, const index_type& steps){
        walk_helper(axis,steps,sequence_type{});
    }
    void walk_back(const dim_type& axis, const index_type& steps){
        walk_back_helper(axis,steps,sequence_type{});
    }
    void step(const dim_type& axis){
        step_helper(axis,sequence_type{});
    }
    void step_back(const dim_type& axis){
        step_back_helper(axis,sequence_type{});
    }
    void reset(const dim_type& axis){
        reset_helper(axis,sequence_type{});
    }
    void reset_back(const dim_type& axis){
        reset_back_helper(axis,sequence_type{});
    }
    void reset_back(){
        reset_back_helper(sequence_type{});
    }
    void update_offset(){
        update_offset_helper(sequence_type{});
    }
    result_type operator*()const{
        return deref_helper(sequence_type{});
    }
private:
    template<std::size_t...I>
    void walk_helper(const dim_type& axis, const index_type& steps, std::index_sequence<I...>){
        (std::get<I>(walkers_).walk(axis,steps),...);
    }
    template<std::size_t...I>
    void walk_back_helper(const dim_type& axis, const index_type& steps, std::index_sequence<I...>){
        (std::get<I>(walkers_).walk_back(axis,steps),...);
    }
    template<std::size_t...I>
    void step_helper(const dim_type& axis, std::index_sequence<I...>){
        (std::get<I>(walkers_).step(axis),...);
    }
    template<std::size_t...I>
    void step_back_helper(const dim_type& axis, std::index_sequence<I...>){
        (std::get<I>(walkers_).step_back(axis),...);
    }
    template<std::size_t...I>
    void reset_helper(const dim_type& axis, std::index_sequence<I...>){
        (std::get<I>(walkers_).reset(axis),...);
    }
    template<std::size_t...I>
    void reset_back_helper(const dim_type& axis, std::index_sequence<I...>){
        (std::get<I>(walkers_).reset_back(axis),...);
    }
    template<std::size_t...I>
    void reset_back_helper(std::index_sequence<I...>){
        (std::get<I>(walkers_).reset_back(),...);
    }
    template<std::size_t...I>
    void update_offset_helper(std::index_sequence<I...>){
        (std::get<I>(walkers_).update_offset(),...);
    }
    template<std::size_t...I>
    result_type deref_helper(std::index_sequence<I...>)const{
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

    template<typename F_> struct forward_args : std::bool_constant<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<F_>>,expression_template_trivial_indexer>>{};

    template<typename F_, typename...Indexers_, std::enable_if_t<forward_args<F_>::value,int> =0>
    explicit expression_template_trivial_indexer(F_&& f__, Indexers_&&...indexers__):
        f_{std::forward<F_>(f__)},
        indexers_{std::forward<Indexers_>(indexers__)...}
    {}
    decltype(auto) operator[](const index_type& idx)const{
        return subscript_helper(idx, sequence_type{});
    }
private:
    template<std::size_t...I>
    decltype(auto) subscript_helper(const index_type& idx, std::index_sequence<I...>)const{
        return f_(std::get<I>(indexers_)[idx]...);
    }

    F f_;
    tuple_type<Indexers...> indexers_;
};


}   //end of namespace gtensor
#endif