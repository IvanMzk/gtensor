#ifndef EXPRESSION_TEMPLATE_OPERATORS_HPP_
#define EXPRESSION_TEMPLATE_OPERATORS_HPP_

#include "tensor.hpp"
#include "expression_template_core.hpp"

namespace gtensor{

template<typename F>
class expression_template_n_operator{
public:
    template<typename F_, typename Operand, typename...Operands>
    auto operator()(F_&& f, Operand&& operand, Operands&&...operands){
        static_assert(std::is_same_v<F,std::decay_t<F_>>);
        using config_type = typename std::decay_t<Operand>::config_type;
        using implementation_type = tensor_implementation<
            expression_template_core<config_type, F, std::decay_t<Operand>, std::decay_t<Operands>...>
        >;
        return basic_tensor<implementation_type>{
            std::make_shared<implementation_type>(
                std::forward<F_>(f),
                std::forward<Operand>(operand),
                std::forward<Operands>(operands)...
            )
        };
    }
};

}
#endif