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

    void walk(const dim_type& direction, const index_type& steps){
        walk_helper(direction,steps,sequence_type{});
    }
    void step(const dim_type& direction){
        step_helper(direction,sequence_type{});
    }
    void step_back(const dim_type& direction){
        step_back_helper(direction,sequence_type{});
    }
    void reset(const dim_type& direction){
        reset_helper(direction,sequence_type{});
    }
    void reset_back(const dim_type& direction){
        reset_back_helper(direction,sequence_type{});
    }
    void reset_back(){
        reset_back_helper(sequence_type{});
    }
    result_type operator*()const{
        return deref_helper(sequence_type{});
    }
private:
    template<std::size_t...I>
    void walk_helper(const dim_type& direction, const index_type& steps, std::index_sequence<I...>){
        (std::get<I>(walkers_).walk(direction,steps),...);
    }
    template<std::size_t...I>
    void step_helper(const dim_type& direction, std::index_sequence<I...>){
        (std::get<I>(walkers_).step(direction),...);
    }
    template<std::size_t...I>
    void step_back_helper(const dim_type& direction, std::index_sequence<I...>){
        (std::get<I>(walkers_).step_back(direction),...);
    }
    template<std::size_t...I>
    void reset_helper(const dim_type& direction, std::index_sequence<I...>){
        (std::get<I>(walkers_).reset(direction),...);
    }
    template<std::size_t...I>
    void reset_back_helper(const dim_type& direction, std::index_sequence<I...>){
        (std::get<I>(walkers_).reset_back(direction),...);
    }
    template<std::size_t...I>
    void reset_back_helper(std::index_sequence<I...>){
        (std::get<I>(walkers_).reset_back(),...);
    }
    template<std::size_t...I>
    result_type deref_helper(std::index_sequence<I...>)const{
        return f_(*std::get<I>(walkers_)...);
    }

    F f_;
    tuple_type<Walkers...> walkers_;
};

}   //end of namespace gtensor
#endif