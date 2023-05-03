#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_

#include "type_selector.hpp"
#include "operation.hpp"
#include "expression_template_operators.hpp"

#define GTENSOR_UNARY_TENSOR_OPERATOR(NAME,F)\
template<typename Impl>\
inline auto NAME(const basic_tensor<Impl>& t){\
    return n_operator(F{},t);\
}\
template<typename Impl>\
inline auto NAME(basic_tensor<Impl>&& t){\
    return n_operator(F{},std::move(t));\
}

#define GTENSOR_BINARY_TENSOR_OPERATOR(NAME,F)\
template<typename Impl1, typename Impl2>\
inline auto NAME(const basic_tensor<Impl1>& t1, const basic_tensor<Impl2>& t2){\
    return n_operator(F{},t1,t2);\
}\
template<typename Impl1, typename Impl2>\
inline auto NAME(basic_tensor<Impl1>&& t1, const basic_tensor<Impl2>& t2){\
    return n_operator(F{},std::move(t1),t2);\
}\
template<typename Impl1, typename Impl2>\
inline auto NAME(const basic_tensor<Impl1>& t1, basic_tensor<Impl2>&& t2){\
    return n_operator(F{},t1,std::move(t2));\
}\
template<typename Impl1, typename Impl2>\
inline auto NAME(basic_tensor<Impl1>&& t1, basic_tensor<Impl2>&& t2){\
    return n_operator(F{},std::move(t1),std::move(t2));\
}

namespace gtensor{

template<typename F, typename Operand, typename...Operands>
inline auto n_operator(F&& f, Operand&& operand, Operands&&...operands){
    using config_type = typename std::decay_t<Operand>::config_type;
    using operation_type = std::decay_t<F>;
    return n_operator_selector_t<config_type, operation_type>{}(
        std::forward<F>(f),
        std::forward<Operand>(operand),
        std::forward<Operands>(operands)...
    );
}

template<typename Impl1, typename Impl2>
static inline auto equals(const basic_tensor<Impl1>& t1, const basic_tensor<Impl2>& t2){
    if constexpr (std::is_same_v<Impl1,Impl2>){
        if (static_cast<const void*>(&t1) == static_cast<const void*>(&t2)){
            return true;
        }
    }
    return t1.shape() == t2.shape() && std::equal(t1.begin(), t1.end(), t2.begin());
}

template<typename...Ts>
std::ostream& operator<<(std::ostream& os, const basic_tensor<Ts...>& t){return os<<str(t);}

// add,+);
// sub,-);
// mul,*);
// div,/);
// greater,>);
// less,<);
// equal,==);
// logic_and,&&);
// logic_or,||);


GTENSOR_UNARY_TENSOR_OPERATOR(operator+,operations::unary_plus);
GTENSOR_UNARY_TENSOR_OPERATOR(operator-,operations::unary_minus);

GTENSOR_BINARY_TENSOR_OPERATOR(operator+,operations::add);
GTENSOR_BINARY_TENSOR_OPERATOR(operator-,operations::sub);
GTENSOR_BINARY_TENSOR_OPERATOR(operator*,operations::mul);
GTENSOR_BINARY_TENSOR_OPERATOR(operator/,operations::div);
GTENSOR_BINARY_TENSOR_OPERATOR(operator>,operations::greater);
GTENSOR_BINARY_TENSOR_OPERATOR(operator<,operations::less);
GTENSOR_BINARY_TENSOR_OPERATOR(operator==,operations::equal);
GTENSOR_BINARY_TENSOR_OPERATOR(operator&&,operations::logic_and);
GTENSOR_BINARY_TENSOR_OPERATOR(operator||,operations::logic_or);


}   //end of namespace gtensor

#endif