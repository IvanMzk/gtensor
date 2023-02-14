#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_

#include "expression_template_operators.hpp"

#define BINARY_TENSOR_OPERATOR(NAME,TAG)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
inline auto NAME(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){return tensor_operators::binary_dispatcher(op1,op2,TAG{});}

namespace gtensor{
namespace tensor_operator_traits{

using gtensor::config::tag;
enum class operators : std::size_t{not_defined,assign,add,sub,mul,div,equal,greater,less,logic_and,logic_or};
using not_defined_tag = tag<operators, operators::not_defined>;
using assign_tag = tag<operators, operators::assign>;
using add_tag = tag<operators, operators::add>;
using sub_tag = tag<operators, operators::sub>;
using mul_tag = tag<operators, operators::mul>;
using div_tag = tag<operators, operators::div>;
using equal_tag = tag<operators, operators::equal>;
using greater_tag = tag<operators, operators::greater>;
using less_tag = tag<operators, operators::less>;
using logic_and_tag = tag<operators, operators::logic_and>;
using logic_or_tag = tag<operators, operators::logic_or>;


template<typename Engine, typename OperatorTag> struct tensor_operator_selector;
//expression template engine operators dispatching
template<typename OperatorTag> struct tensor_operator_selector<gtensor::config::engine_expression_template, OperatorTag>{
    template<typename Tag, typename Dummy> struct selector{using type = not_defined_tag;};
    template<typename D> struct selector<assign_tag,D>{using type = expression_template_operators::operator_assign;};
    template<typename D> struct selector<add_tag,D>{using type = expression_template_operators::operator_add;};
    template<typename D> struct selector<sub_tag,D>{using type = expression_template_operators::operator_sub;};
    template<typename D> struct selector<mul_tag,D>{using type = expression_template_operators::operator_mul;};
    template<typename D> struct selector<div_tag,D>{using type = expression_template_operators::operator_div;};
    template<typename D> struct selector<greater_tag,D>{using type = expression_template_operators::operator_greater;};
    template<typename D> struct selector<less_tag,D>{using type = expression_template_operators::operator_less;};
    template<typename D> struct selector<equal_tag,D>{using type = expression_template_operators::operator_equal;};
    template<typename D> struct selector<logic_and_tag,D>{using type = expression_template_operators::operator_logic_and;};
    template<typename D> struct selector<logic_or_tag,D>{using type = expression_template_operators::operator_logic_or;};
    using type = typename selector<OperatorTag,void>::type;
};

}   //end of namespace tensor_operator_traits

struct tensor_operators{
    template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT, typename Tag>
    static inline auto binary_dispatcher(const tensor<ValT1, CfgT, ImplT1>& t1, const tensor<ValT2, CfgT, ImplT2>& t2, Tag){
        using operator_impl_type = typename tensor_operator_traits::tensor_operator_selector<typename CfgT::host_engine, Tag>::type;
        static_assert(!std::is_same_v<operator_impl_type,tensor_operator_traits::not_defined_tag>);
        return operator_impl_type{}(t1.impl(),t2.impl());
    }
};

BINARY_TENSOR_OPERATOR(operator+, tensor_operator_traits::add_tag);
BINARY_TENSOR_OPERATOR(operator-, tensor_operator_traits::sub_tag);
BINARY_TENSOR_OPERATOR(operator*, tensor_operator_traits::mul_tag);
BINARY_TENSOR_OPERATOR(operator/, tensor_operator_traits::div_tag);
BINARY_TENSOR_OPERATOR(operator>, tensor_operator_traits::greater_tag);
BINARY_TENSOR_OPERATOR(operator<, tensor_operator_traits::less_tag);
BINARY_TENSOR_OPERATOR(operator==, tensor_operator_traits::equal_tag);
BINARY_TENSOR_OPERATOR(operator&&, tensor_operator_traits::logic_and_tag);
BINARY_TENSOR_OPERATOR(operator||, tensor_operator_traits::logic_or_tag);
BINARY_TENSOR_OPERATOR(operator_assign, tensor_operator_traits::assign_tag);

template<typename ValT, typename CfgT, typename ImplT1, typename ImplT2>
static inline auto equals(const tensor<ValT, CfgT, ImplT1>& t1, const tensor<ValT, CfgT, ImplT2>& t2){
    return std::is_same_v<std::decay_t<decltype(t1)>, std::decay_t<decltype(t2)>> && static_cast<const void*>(&t1) == static_cast<const void*>(&t2) ||
        t1.shape() == t2.shape() && std::equal(t1.begin(), t1.end(), t2.begin());
}



}   //end of namespace gtensor

#endif