#ifndef TENSOR_OPERATORS_IMPL_HPP_
#define TENSOR_OPERATORS_IMPL_HPP_

#include "forward_decl.hpp"
#include "operations.hpp"

namespace gtensor{

struct tensor_operators_impl{

    template<typename ValT1, typename ValT2, template<typename> typename Cfg>
    static auto operator_plus_impl(const tensor<ValT1, Cfg>& op1, const tensor<ValT2, Cfg>& op2){
        using operation_type = operation_add;        
        using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));
        using exp_type = expression_impl<result_type, operation_type, Cfg, tensor<ValT1, Cfg>, tensor<ValT2, Cfg>>;
        return tensor<result_type,Cfg>{std::make_shared<exp_type>(op1,op2)};
    }

};


}   //end of namespace gtensor

#endif