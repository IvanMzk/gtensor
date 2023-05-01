#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_


#define BINARY_TENSOR_OPERATOR(NAME,TAG)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
inline auto NAME(const tensor<ValT1, CfgT, ImplT1>& op1, const tensor<ValT2, CfgT, ImplT2>& op2){return tensor_operator_dispatcher::binary_dispatcher(op1,op2,TAG{});}

namespace gtensor{


template<typename Impl1, typename Impl2>
static inline auto equals(const basic_tensor<Impl1>& t1, const basic_tensor<Impl2>& t2){
    if constexpr (std::is_same_v<Impl1,Impl2>){
        if (static_cast<const void*>(&t1) == static_cast<const void*>(&t2)){
            return true;
        }
    }
    return t1.shape() == t2.shape() && std::equal(t1.begin(), t1.end(), t2.begin());
}



}   //end of namespace gtensor

#endif