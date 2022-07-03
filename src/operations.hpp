#ifndef OPERATIONS_HPP_
#define OPERATIONS_HPP_

#include <iostream>

namespace gtensor{

namespace binary_operations{
#define BINARY_OPERATION(NAME, OP)\
struct NAME{template <typename T1, typename T2> auto operator()(T1&& arg1, T2&& arg2) const {return (std::forward<T1>(arg1) OP std::forward<T2>(arg2));}};

BINARY_OPERATION(add,+);
BINARY_OPERATION(sub,-);
BINARY_OPERATION(mul,*);
BINARY_OPERATION(div,/);
BINARY_OPERATION(ge,>);

BINARY_OPERATION(logic_and,&&);
BINARY_OPERATION(logic_or,||);

}

namespace unary_operations{
#define UNARY_OPERATION(NAME, OP)\
struct NAME{template <typename T> auto operator()(T&& arg) const {return (OP std::forward<T>(arg));}};

UNARY_OPERATION(ident,+);
UNARY_OPERATION(neg,-);

}



}

#endif