#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_

#include "tensor_operators_impl.hpp"

namespace gtensor{

    template<typename ValT1, typename ValT2, template<typename> typename Cfg>
    auto operator+(const tensor<ValT1, Cfg>& op1, const tensor<ValT2, Cfg>& op2){
        return tensor_operators_impl::operator_plus_impl(op1,op2);
    }
    template<typename ValT1, typename ValT2, template<typename> typename Cfg>
    auto operator>(const tensor<ValT1, Cfg>& op1, const tensor<ValT2, Cfg>& op2){
        return tensor_operators_impl::operator_ge_impl(op1,op2);
    }
    
}   //end of namespace gtensor



#endif