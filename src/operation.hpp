#ifndef OPERATION_HPP_
#define OPERATION_HPP_

#define GTENSOR_UNARY_OPERATION(NAME, OP)\
struct NAME{\
    template <typename T>\
    auto operator()(T&& arg) const {\
        return (OP std::forward<T>(arg));\
    }\
};

#define GTENSOR_BINARY_OPERATION(NAME, OP)\
struct NAME{\
    template <typename T1, typename T2>\
    auto operator()(T1&& arg1, T2&& arg2) const {\
        return (std::forward<T1>(arg1) OP std::forward<T2>(arg2));\
    }\
};

#define GTENSOR_OPERATION_TAG(NAME) struct NAME{};

namespace gtensor{
namespace operations{

GTENSOR_UNARY_OPERATION(unary_plus,+);
GTENSOR_UNARY_OPERATION(unary_minus,-);

GTENSOR_BINARY_OPERATION(add,+);
GTENSOR_BINARY_OPERATION(sub,-);
GTENSOR_BINARY_OPERATION(mul,*);
GTENSOR_BINARY_OPERATION(div,/);
GTENSOR_BINARY_OPERATION(greater,>);
GTENSOR_BINARY_OPERATION(less,<);
GTENSOR_BINARY_OPERATION(equal,==);
GTENSOR_BINARY_OPERATION(logic_and,&&);
GTENSOR_BINARY_OPERATION(logic_or,||);

GTENSOR_OPERATION_TAG(assign);

}   //end of nemespace operations
}   //end of namespace gtensor
#endif