#ifndef TENSOR_OPERATORS_IMPL_HPP_
#define TENSOR_OPERATORS_IMPL_HPP_

#include "forward_decl.hpp"
#include "operations.hpp"
#include "evaluating_tensor.hpp"

#define BINARY_OPERATOR_IMPL(NAME,OP,FACTORY)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
static inline auto NAME(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){\
    using operation_type = OP;\
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));\
    using walker_factory_type = FACTORY<result_type, CfgT>;\
    using exp_operand1_type = ImplT1;\
    using exp_operand2_type = ImplT2;\
    using exp_type = evaluating_tensor<result_type, CfgT, operation_type, walker_factory_type, exp_operand1_type, exp_operand2_type>;\
    return tensor<result_type,CfgT, exp_type>{std::make_shared<exp_type>(op1.impl(),op2.impl())};\
}


namespace gtensor{

struct tensor_operators_impl{

    BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add, evaluating_walker_factory);
    BINARY_OPERATOR_IMPL(operator_sub_impl, gtensor::binary_operations::sub, evaluating_walker_factory);
    BINARY_OPERATOR_IMPL(operator_mul_impl, gtensor::binary_operations::mul, evaluating_walker_factory);
    BINARY_OPERATOR_IMPL(operator_div_impl, gtensor::binary_operations::div, evaluating_walker_factory);
    BINARY_OPERATOR_IMPL(operator_ge_impl, gtensor::binary_operations::ge, evaluating_walker_factory);


};


}   //end of namespace gtensor

#endif