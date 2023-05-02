#ifndef EXPRESSION_TEMPLATE_CORE_HPP_
#define EXPRESSION_TEMPLATE_CORE_HPP_

#include "descriptor.hpp"
#include "expression_template_evaluator.hpp"

namespace gtensor{

//expression template core represents expression on basic_tensor operands
//F is expression operation, must provide operator()() on arguments that are operands elements
//Operands are specializations of basic_tensor
template<typename Config, typename F, typename...Operands>
class expression_template_core
{
    using descriptor_type = basic_descriptor<Config>;
    template<typename...Ts> using tuple_type = std::tuple<Ts...>;
public:
    using config_type = Config;
    using value_type = std::decay_t<decltype(std::declval<F>()(std::declval<typename Operands::value_type>()...))>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;

    template<typename F_, typename...Operands_>
    explicit expression_template_core(F_&& f__, Operands_&&...operands__):
        descriptor_{detail::make_broadcast_shape<shape_type>(operands__.shape()...)},
        f_{std::forward<F_>(f__)},
        operands_{std::forward<Operands_>(operands__)...}
    {}
    const descriptor_type& descriptor()const{return descriptor_;}
    auto create_walker(dim_type max_dim){return create_walker_helper(*this,max_dim);}
    auto create_walker(dim_type max_dim)const{return create_walker_helper(*this,max_dim);}
private:
    template<typename U, std::size_t...I>
    static auto create_walker_helper(U& instance, dim_type max_dim, std::index_sequence<I...>){
        return expression_template_walker<Config,F,decltype(get<I>(instance.operands_).create_walker(max_dim))...>{
            instance.f_,
            get<I>(instance.operands_).create_walker(max_dim)...
        };
    }
    descriptor_type descriptor_;
    F f_;
    tuple_type<Operands...> operands_;
};

}   //end of namespace gtensor
#endif