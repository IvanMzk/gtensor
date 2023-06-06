#ifndef EXPRESSION_TEMPLATE_CORE_HPP_
#define EXPRESSION_TEMPLATE_CORE_HPP_

#include "descriptor.hpp"
#include "expression_template_evaluator.hpp"

namespace gtensor{

namespace detail{

// template<typename...> struct common_order;
// template<typename Config>
// struct common_order<Config>{
//     using type = typename Config::order;
// };
// template<typename Config, typename Order, typename...Orders>
// struct common_order<Config,Order,Orders...>{
//     using type = std::conditional_t<
//         std::conjunction_v<std::is_same<Order,Orders>...>,
//         Order,
//         typename Config::order
//     >;
// };
// template<typename Config, typename...Orders> using common_order_t = typename common_order<Config,Orders...>::type;

}   //end of namespace detail


//expression_template_core represents expression on basic_tensor operands
//F is expression operation, must provide operator()() on arguments that are operands elements
//Operands are specializations of basic_tensor
template<typename Config, typename F, typename...Operands>
class expression_template_core
{
    using common_order = detail::common_order_t<Config,typename Operands::order...>;
    using descriptor_type = basic_descriptor<Config,common_order>;
    template<typename...Ts> using tuple_type = std::tuple<Ts...>;
    using sequence_type = std::make_index_sequence<sizeof...(Operands)>;
public:
    using order = common_order;
    using value_type = std::decay_t<decltype(std::declval<F>()(std::declval<typename Operands::value_type&>()...))>;
    using config_type = config::extend_config_t<Config,value_type>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;

    template<typename F_> struct forward_args : std::bool_constant<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<F_>>,expression_template_core>>{};

    template<typename F_, typename...Operands_, std::enable_if_t<forward_args<F_>::value,int> =0>
    explicit expression_template_core(F_&& f__, Operands_&&...operands__):
        descriptor_{detail::make_broadcast_shape<shape_type>(operands__.shape()...)},
        f_{std::forward<F_>(f__)},
        operands_{std::forward<Operands_>(operands__)...}
    {}
    const descriptor_type& descriptor()const{
        return descriptor_;
    }
    auto create_walker(dim_type max_dim){
        return create_walker_helper(*this,max_dim,sequence_type{});
    }
    auto create_walker(dim_type max_dim)const{
        return create_walker_helper(*this,max_dim,sequence_type{});
    }
private:
    template<typename U, std::size_t...I>
    static auto create_walker_helper(U& instance, dim_type max_dim, std::index_sequence<I...>){
        return expression_template_walker<config_type,F,decltype(std::get<I>(instance.operands_).create_walker(max_dim))...>{
            instance.f_,
            std::get<I>(instance.operands_).create_walker(max_dim)...
        };
    }
    descriptor_type descriptor_;
    F f_;
    tuple_type<Operands...> operands_;
};

}   //end of namespace gtensor
#endif