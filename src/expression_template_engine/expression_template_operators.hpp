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
public:
    template<typename F_, typename Scalar_>
    binary_operation_scalar_wrapper(F_&& f__, Scalar_&& scalar__):
        f_{std::forward<F_>(f__)},
        scalar_{std::forward<Scalar_>(scalar__)}
    {}
    template<typename U>
    auto operator()(U&& u)const{
        if constexpr (Order::value == binary_operation_scalar_order::scalar_first){
            return f_(scalar_,std::forward<U>(u));
        }else{
            return f_(std::forward<U>(u),scalar_);
        }
    }
};

}

template<typename F>
class expression_template_n_operator{

    template<typename F_, typename Operand, typename...Operands>
    auto n_operator(F_&& f, Operand&& operand, Operands&&...operands){
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
    template<typename F_, typename Operand1, typename Operand2>
    auto operator()(F_&& f, Operand1&& operand1, Operand2&& operand2){
        static_assert(std::is_same_v<F,std::decay_t<F_>>);
        static constexpr bool is_operand1_tensor = detail::is_tensor_v<std::decay_t<Operand1>>;
        static constexpr bool is_operand2_tensor = detail::is_tensor_v<std::decay_t<Operand2>>;
        static_assert(is_operand1_tensor||is_operand2_tensor);
        if constexpr (is_operand1_tensor&&is_operand2_tensor){
            return n_operator(std::forward<F_>(f), std::forward<Operand1>(operand1), std::forward<Operand2>(operand2));
        }else if constexpr (is_operand1_tensor){
            using operation_type = detail::binary_operation_scalar_wrapper<F,std::remove_reference_t<Operand2>,detail::scalar_second_type>;
            return n_operator(operation_type{std::forward<F_>(f),std::forward<Operand2>(operand2)},std::forward<Operand1>(operand1));
        }else{
            using operation_type = detail::binary_operation_scalar_wrapper<F,std::remove_reference_t<Operand1>,detail::scalar_first_type>;
            return n_operator(operation_type{std::forward<F_>(f),std::forward<Operand1>(operand1)},std::forward<Operand2>(operand2));
        }
    }

    template<typename F_, typename Operand, typename...Operands>
    auto operator()(F_&& f, Operand&& operand, Operands&&...operands){
        static_assert(std::is_same_v<F,std::decay_t<F_>>);
        return n_operator(std::forward<F_>(f), std::forward<Operand>(operand), std::forward<Operands>(operands)...);
    }
};

}
#endif