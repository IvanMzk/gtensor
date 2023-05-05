#ifndef OPERATION_HPP_
#define OPERATION_HPP_

#define GTENSOR_UNARY_OPERATION(NAME, OP)\
struct NAME{\
    template<typename T>\
    auto operator()(T&& arg)const{\
        return (OP std::forward<T>(arg));\
    }\
};

#define GTENSOR_BINARY_OPERATION(NAME, OP)\
struct NAME{\
    template<typename T1, typename T2>\
    auto operator()(T1&& arg1, T2&& arg2)const{\
        return (std::forward<T1>(arg1) OP std::forward<T2>(arg2));\
    }\
};

#define GTENSOR_ASSIGN_OPERATION(NAME, OP)\
struct NAME{\
    template<typename T1, typename T2>\
    void operator()(T1&& arg1, T2&& arg2)const{\
        std::forward<T1>(arg1) OP std::forward<T2>(arg2);\
    }\
};

#define GTENSOR_OPERATION_TAG(NAME) struct NAME{};

namespace gtensor{
namespace operations{

//arithmetic
GTENSOR_UNARY_OPERATION(unary_plus,+);
GTENSOR_UNARY_OPERATION(unary_minus,-);
GTENSOR_BINARY_OPERATION(add,+);
GTENSOR_BINARY_OPERATION(sub,-);
GTENSOR_BINARY_OPERATION(mul,*);
GTENSOR_BINARY_OPERATION(div,/);
GTENSOR_BINARY_OPERATION(mod,%);

//bitwise
GTENSOR_UNARY_OPERATION(bitwise_not,~);
GTENSOR_BINARY_OPERATION(bitwise_and,&);
GTENSOR_BINARY_OPERATION(bitwise_or,|);
GTENSOR_BINARY_OPERATION(bitwise_xor,^);
GTENSOR_BINARY_OPERATION(bitwise_lshift,<<);
GTENSOR_BINARY_OPERATION(bitwise_rshift,>>);

//comparison
GTENSOR_BINARY_OPERATION(equal,==);
GTENSOR_BINARY_OPERATION(not_equal,!=);
GTENSOR_BINARY_OPERATION(greater,>);
GTENSOR_BINARY_OPERATION(greater_equal,>=);
GTENSOR_BINARY_OPERATION(less,<);
GTENSOR_BINARY_OPERATION(less_equal,<=);

//logical
GTENSOR_UNARY_OPERATION(logic_not,!);
GTENSOR_BINARY_OPERATION(logic_and,&&);
GTENSOR_BINARY_OPERATION(logic_or,||);

//asignment
GTENSOR_ASSIGN_OPERATION(assign,=);
GTENSOR_ASSIGN_OPERATION(assign_add,+=);
GTENSOR_ASSIGN_OPERATION(assign_sub,-=);
GTENSOR_ASSIGN_OPERATION(assign_mul,*=);
GTENSOR_ASSIGN_OPERATION(assign_div,/=);
GTENSOR_ASSIGN_OPERATION(assign_mod,%=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_and,&=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_or,|=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_xor,^=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_lshift,<<=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_rshift,>>=);

}   //end of nemespace operations
}   //end of namespace gtensor
#endif