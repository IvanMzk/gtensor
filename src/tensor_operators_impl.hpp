#ifndef TENSOR_OPERATORS_IMPL_HPP_
#define TENSOR_OPERATORS_IMPL_HPP_

#include "forward_decl.hpp"
#include "operations.hpp"
#include "evaluating_tensor.hpp"
#include "expression_template_engine.hpp"

#define BINARY_OPERATOR_IMPL(NAME,OP)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
static inline auto NAME(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){\
    using operation_type = OP;\
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));\
    using exp_operand1_type = ImplT1;\
    using exp_operand2_type = ImplT2;\
    using engine_type = expression_template_elementwise_engine<result_type, CfgT, operation_type, exp_operand1_type, exp_operand2_type>;\
    using exp_type = evaluating_tensor<result_type, CfgT, operation_type, engine_type, exp_operand1_type, exp_operand2_type>;\
    return tensor<result_type,CfgT, exp_type>{std::make_shared<exp_type>(engine_type{}, op1.impl(),op2.impl())};\
}


namespace gtensor{

struct tensor_operators_impl{

    BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
    BINARY_OPERATOR_IMPL(operator_sub_impl, gtensor::binary_operations::sub);
    BINARY_OPERATOR_IMPL(operator_mul_impl, gtensor::binary_operations::mul);
    BINARY_OPERATOR_IMPL(operator_div_impl, gtensor::binary_operations::div);
    BINARY_OPERATOR_IMPL(operator_ge_impl, gtensor::binary_operations::ge);


};


}   //end of namespace gtensor

#endif