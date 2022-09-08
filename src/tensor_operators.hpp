#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_

#include "tensor_operators_impl.hpp"

#define BINARY_OPERATOR(NAME,IMPL)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
auto NAME(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){return IMPL(op1,op2);}

namespace gtensor{

    BINARY_OPERATOR(operator+, tensor_operators_impl::operator_add_impl);
    BINARY_OPERATOR(operator-, tensor_operators_impl::operator_sub_impl);
    BINARY_OPERATOR(operator*, tensor_operators_impl::operator_mul_impl);
    BINARY_OPERATOR(operator/, tensor_operators_impl::operator_div_impl);
    BINARY_OPERATOR(operator>, tensor_operators_impl::operator_ge_impl);
    
}   //end of namespace gtensor



#endif