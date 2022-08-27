#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_

#include "tensor_operators_impl.hpp"

namespace gtensor{

    template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>
    auto operator+(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){
        return tensor_operators_impl::operator_add_impl(op1,op2);
    }
    template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>
    auto operator-(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){
        return tensor_operators_impl::operator_sub_impl(op1,op2);
    }
    template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>
    auto operator*(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){
        return tensor_operators_impl::operator_mul_impl(op1,op2);
    }
    template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>
    auto operator/(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){
        return tensor_operators_impl::operator_div_impl(op1,op2);
    }
    template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>
    auto operator>(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){
        return tensor_operators_impl::operator_ge_impl(op1,op2);
    }
    
}   //end of namespace gtensor



#endif