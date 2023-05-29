#ifndef EXPRESSION_TEMPLATE_OPERATORS_HPP_
#define EXPRESSION_TEMPLATE_OPERATORS_HPP_

#include "tensor.hpp"
#include "expression_template_core.hpp"

namespace gtensor{

namespace detail{

enum class binary_operation_scalar_order{scalar_first,scalar_second};
using scalar_first_type = std::integral_constant<binary_operation_scalar_order, binary_operation_scalar_order::scalar_first>;
using scalar_second_type = std::integral_constant<binary_operation_scalar_order, binary_operation_scalar_order::scalar_second>;
template<typename F, typename Scalar, typename Order>
class binary_operation_scalar_wrapper{
    F f_;
    Scalar scalar_;
    static constexpr bool is_scalar_first = Order::value == binary_operation_scalar_order::scalar_first;
public:
    template<typename F_, typename Scalar_>
    binary_operation_scalar_wrapper(F_&& f__, Scalar_&& scalar__):
        f_{std::forward<F_>(f__)},
        scalar_{std::forward<Scalar_>(scalar__)}
    {}
    template<typename U>
    std::conditional_t<is_scalar_first, decltype(f_(scalar_,std::declval<U>())), decltype(f_(std::declval<U>(),scalar_))> operator()(U&& u)const{
        if constexpr (is_scalar_first){
            return f_(scalar_,std::forward<U>(u));
        }else{
            return f_(std::forward<U>(u),scalar_);
        }
    }
};

template<typename, typename T, std::enable_if_t<is_tensor_v<std::remove_cv_t<std::remove_reference_t<T>>>,int> =0>
inline auto&& forward_as_tensor(T&& t){
    return std::forward<T>(t);
}
template<typename Config, typename T, std::enable_if_t<!is_tensor_v<std::remove_cv_t<std::remove_reference_t<T>>>,int> =0>
inline auto forward_as_tensor(T&& t){
    using value_type = std::remove_cv_t<std::remove_reference_t<T>>;
    return tensor<value_type,gtensor::config::c_order,Config>(t);
}

}   //end of namespace detail

template<typename F>
class expression_template_operator{

    template<typename F_, typename Operand, typename...Operands>
    static auto n_operator_(F_&& f, Operand&& operand, Operands&&...operands){
        using config_type = typename std::decay_t<Operand>::config_type;
        using implementation_type = tensor_implementation<
            expression_template_core<config_type, std::remove_reference_t<F_>, std::remove_reference_t<Operand>, std::remove_reference_t<Operands>...>
        >;
        return basic_tensor<implementation_type>{
            std::make_shared<implementation_type>(
                std::forward<F_>(f),
                std::forward<Operand>(operand),
                std::forward<Operands>(operands)...
            )
        };
    }

public:
    //optimized binary n_operator
    template<typename F_, typename Operand1, typename Operand2>
    static auto n_operator(F_&& f, Operand1&& operand1, Operand2&& operand2){
        static_assert(std::is_same_v<F,std::decay_t<F_>>);
        static constexpr bool is_operand1_tensor = detail::is_tensor_v<std::decay_t<Operand1>>;
        static constexpr bool is_operand2_tensor = detail::is_tensor_v<std::decay_t<Operand2>>;
        static_assert(is_operand1_tensor||is_operand2_tensor);
        if constexpr (is_operand1_tensor&&is_operand2_tensor){
            return n_operator_(std::forward<F_>(f), std::forward<Operand1>(operand1), std::forward<Operand2>(operand2));
        }else if constexpr (is_operand1_tensor){
            using operation_type = detail::binary_operation_scalar_wrapper<F,std::remove_reference_t<Operand2>,detail::scalar_second_type>;
            return n_operator_(operation_type{std::forward<F_>(f),std::forward<Operand2>(operand2)},std::forward<Operand1>(operand1));
        }else{
            using operation_type = detail::binary_operation_scalar_wrapper<F,std::remove_reference_t<Operand1>,detail::scalar_first_type>;
            return n_operator_(operation_type{std::forward<F_>(f),std::forward<Operand1>(operand1)},std::forward<Operand2>(operand2));
        }
    }

    //generalized broadcast operator
    //makes tensor that represents applying operation F on operands element-wise
    //operands shapes must be broadcastable
    //scalar operands is allowed
    //actual evaluation happens only when result tensor is iterated
    template<typename F_, typename...Operands>
    static auto n_operator(F_&& f, Operands&&...operands){
        static_assert(std::is_same_v<F,std::decay_t<F_>>);
        using config_type = typename detail::first_tensor_type_t<std::remove_cv_t<std::remove_reference_t<Operands>>...>::config_type;
        return n_operator_(std::forward<F_>(f), detail::forward_as_tensor<config_type>(std::forward<Operands>(operands))...);
    }

    //generalized broadcast assign
    //shapes of lhs and rhs must be broadcastable, scalar rhs is allowed
    //assign operation is defined by F
    //not lazys
    template<typename F_, typename Rhs, typename...Ts>
    static basic_tensor<Ts...>& a_operator(F_&& f, basic_tensor<Ts...>& lhs, Rhs&& rhs){
        static_assert(std::is_same_v<F,std::decay_t<F_>>);
        auto tmp = n_operator(std::forward<F_>(f),lhs,std::forward<Rhs>(rhs));
        for (auto it = tmp.begin(), last = tmp.end(); it!=last; ++it){
            (void)*it;
        }
        return lhs;
    }
};

}   //end of namespace gtensor
#endif