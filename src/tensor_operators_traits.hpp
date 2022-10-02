#ifndef TENSOR_OPERATORS_TRAITS_HPP_
#define TENSOR_OPERATORS_TRAITS_HPP_

#include "expression_template_operators.hpp"

#define BINARY_OPERATOR_DISPATCHER(NAME,OP)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
static inline auto NAME(const tensor<ValT1, CfgT, ImplT1>& t1, const tensor<ValT2, CfgT, ImplT2>& t2){\
    using operator_tag = OP;\
    using config_type = CfgT;\
    using operator_impl_type = typename tensor_operator_traits::tensor_operator_selector<typename config_type::engine, operator_tag>::type;\
    static_assert(!std::is_same_v<operator_impl_type,tensor_operator_traits::not_defined_tag>);\
    return operator_impl_type{}(t1,t2,t1.impl(),t2.impl());\
}


namespace gtensor{

namespace tensor_operator_traits{

using gtensor::config::tag;
enum class operators : std::size_t{not_defined,add,sub,mul,div,equal,greater,less};
using not_defined_tag = tag<operators, operators::not_defined>;
using add_tag = tag<operators, operators::add>;
using sub_tag = tag<operators, operators::sub>;
using mul_tag = tag<operators, operators::mul>;
using div_tag = tag<operators, operators::div>;
using equal_tag = tag<operators, operators::equal>;
using greater_tag = tag<operators, operators::greater>;
using less_tag = tag<operators, operators::less>;

template<typename,typename> struct tensor_operator_selector;
template<typename OpT> struct tensor_operator_selector<gtensor::config::engine_expression_template, OpT>{
    template<typename> struct selector{using type = not_defined_tag;};
    template<> struct selector<add_tag>{using type = expression_template_operators::operator_add;};
    template<> struct selector<sub_tag>{using type = expression_template_operators::operator_sub;};
    template<> struct selector<mul_tag>{using type = expression_template_operators::operator_mul;};
    template<> struct selector<div_tag>{using type = expression_template_operators::operator_div;};
    template<> struct selector<greater_tag>{using type = expression_template_operators::operator_greater;};
    template<> struct selector<less_tag>{using type = expression_template_operators::operator_less;};
    using type = typename selector<OpT>::type;
};

}   //end of namespace tensor_operator_traits

struct tensor_operators_dispatcher{

    BINARY_OPERATOR_DISPATCHER(operator_add_dispatcher, tensor_operator_traits::add_tag);
    BINARY_OPERATOR_DISPATCHER(operator_sub_dispatcher, tensor_operator_traits::sub_tag);
    BINARY_OPERATOR_DISPATCHER(operator_mul_dispatcher, tensor_operator_traits::mul_tag);
    BINARY_OPERATOR_DISPATCHER(operator_div_dispatcher, tensor_operator_traits::div_tag);
    BINARY_OPERATOR_DISPATCHER(operator_greater_dispatcher, tensor_operator_traits::greater_tag);

};

}   //end of namespace gtensor

#endif