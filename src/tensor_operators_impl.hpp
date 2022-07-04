#ifndef TENSOR_OPERATORS_IMPL_HPP_
#define TENSOR_OPERATORS_IMPL_HPP_

#include "forward_decl.hpp"
#include "operations.hpp"
#include "impl_expression.hpp"

#define BINARY_OPERATOR_IMPL(NAME,OP)\
template<typename ValT1, typename ValT2, template<typename> typename Cfg>\
static inline auto NAME(const tensor<ValT1, Cfg>& op1, const tensor<ValT2, Cfg>& op2){\
    using operation_type = OP;\
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));\
    using exp_operand1_type = std::shared_ptr<tensor_impl_base<ValT1,Cfg>>;\
    using exp_operand2_type = std::shared_ptr<tensor_impl_base<ValT2,Cfg>>;\
    using exp_type = expression_impl<result_type, operation_type, Cfg, exp_operand1_type, exp_operand2_type>;\
    return tensor<result_type,Cfg>{std::make_shared<exp_type>(op1.get_impl(),op2.get_impl())};\
}


namespace gtensor{

struct tensor_operators_impl{

    BINARY_OPERATOR_IMPL(operator_plus_impl, gtensor::binary_operations::add);
    BINARY_OPERATOR_IMPL(operator_ge_impl, gtensor::binary_operations::ge);


};


}   //end of namespace gtensor

#endif