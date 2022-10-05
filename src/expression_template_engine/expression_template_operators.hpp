#ifndef EXPRESSION_TEMPLATE_OPERATORS_HPP_
#define EXPRESSION_TEMPLATE_OPERATORS_HPP_

#include "forward_decl.hpp"
#include "tensor.hpp"
#include "engine.hpp"

#define EXPRESSION_TEMPLATE_UNARY_OPERATION(NAME, OP)\
struct NAME{\
    template <typename T>\
    auto operator()(T&& arg) const {\
        return (OP std::forward<T>(arg));\
    }\
};

#define EXPRESSION_TEMPLATE_BINARY_OPERATION(NAME, OP)\
struct NAME{\
    template <typename T1, typename T2>\
    auto operator()(T1&& arg1, T2&& arg2) const {\
        return (std::forward<T1>(arg1) OP std::forward<T2>(arg2));\
    }\
};

#define EXPRESSION_TEMPLATE_BINARY_OPERATOR(NAME,OP)\
struct NAME{\
    template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
    auto operator()(const tensor<ValT1, CfgT, ImplT1>&, const tensor<ValT2, CfgT, ImplT2>&, std::shared_ptr<ImplT1>&& op1, std::shared_ptr<ImplT2>&& op2){\
        using operation_type = OP;\
        using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));\
        using operand1_type = ImplT1;\
        using operand2_type = ImplT2;\
        using engine_type = typename detail::evaluating_engine_traits<typename CfgT::engine, result_type, CfgT, operation_type, operand1_type, operand2_type>::type;\
        using impl_type = evaluating_tensor<engine_type>;\
        return tensor<result_type,CfgT, impl_type>{std::make_shared<impl_type>(operation_type{}, std::move(op1),std::move(op2))};\
    }\
};

namespace gtensor{

namespace expression_template_binary_operations{

struct assign{
    template <typename T1, typename T2>
    void operator()(T1&& arg1, T2&& arg2)const{(std::forward<T1>(arg1)=std::forward<T2>(arg2));}
};

EXPRESSION_TEMPLATE_BINARY_OPERATION(add,+);
EXPRESSION_TEMPLATE_BINARY_OPERATION(sub,-);
EXPRESSION_TEMPLATE_BINARY_OPERATION(mul,*);
EXPRESSION_TEMPLATE_BINARY_OPERATION(div,/);
EXPRESSION_TEMPLATE_BINARY_OPERATION(greater,>);
EXPRESSION_TEMPLATE_BINARY_OPERATION(less,<);
EXPRESSION_TEMPLATE_BINARY_OPERATION(equal,==);
EXPRESSION_TEMPLATE_BINARY_OPERATION(logic_and,&&);
EXPRESSION_TEMPLATE_BINARY_OPERATION(logic_or,||);

}   //end of namespace binary_operations

namespace expression_template_unary_operations{

EXPRESSION_TEMPLATE_UNARY_OPERATION(ident,+);
EXPRESSION_TEMPLATE_UNARY_OPERATION(neg,-);

}   //end of namespace unary_operations

namespace expression_template_operators{

    EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator_add, expression_template_binary_operations::add);
    EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator_sub, expression_template_binary_operations::sub);
    EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator_mul, expression_template_binary_operations::mul);
    EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator_div, expression_template_binary_operations::div);
    EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator_greater, expression_template_binary_operations::greater);
    EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator_less, expression_template_binary_operations::less);
    EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator_equal, expression_template_binary_operations::equal);

    struct operator_assign{
        template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>
        auto operator()(const tensor<ValT1, CfgT, ImplT1>&, const tensor<ValT2, CfgT, ImplT2>&, std::shared_ptr<ImplT1>&& lhs, std::shared_ptr<ImplT2>&& rhs){
            using operation_type = expression_template_binary_operations::assign;
            using result_type = void;
            using operand1_type = ImplT1;
            using operand2_type = ImplT2;
            using engine_type = typename detail::evaluating_engine_traits<typename CfgT::engine, result_type, CfgT, operation_type, operand1_type, operand2_type>::type;
            auto lhs_size{lhs->size()};
            auto assigning = evaluating_tensor<engine_type>{operation_type{}, std::move(lhs),std::move(rhs)};
            if (lhs_size < assigning.size())
            {throw broadcast_exception("shapes are not assign broadcastable, lhs would be assigned multiple times per element");}
            for(auto it = assigning.engine().begin(), end = assigning.engine().end(); it!=end; ++it){*it;}
        }
    };

}   //end of namespace expression_template_operators


}   //end of namespace gtensor

#endif