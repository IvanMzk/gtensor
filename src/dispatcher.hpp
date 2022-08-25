#ifndef DISPATCHER_HPP_
#define DISPATCHER_HPP_

#include "tensor_base.hpp"

namespace gtensor{

namespace detail{

class dispatch_exception : public std::runtime_error{
    public: dispatch_exception(const char* what):runtime_error(what){}
};

template<typename ValT, typename CfgT>
class dispatcher{      
    template<typename FirstT, typename ValT2, typename CfgT, typename F>
    static auto dispatch_second(F& f, const FirstT& first, const tensor_base<ValT2, CfgT>& second){
        if (second.is_storage())
        {
            return f(first, *second.as_storing());
        }
        else if (second.tensor_kind() == tensor_kinds::expression)
        {                     
            return f(first, *second.as_evaluating());
        }
        else if (auto v = second.as_viewing_evaluating())
        {            
            return f(first, *v);
        }        
        else
        {
            throw dispatch_exception("type is not supported by dispatcher");
        }
    }        
    template<typename ValT1, typename ValT2, typename CfgT, typename F>
    static auto dispatch_first(F& f, const tensor_base<ValT1, CfgT>& first, const tensor_base<ValT2, CfgT>& second){
        if (first.is_storage())
        {
            return dispatch_second(f, *first.as_storing(), second);
        }
        else if (first.tensor_kind() == tensor_kinds::expression)
        {            
            return dispatch_second(f, *first.as_evaluating(), second);
        }        
        else if (auto v = first.as_viewing_evaluating())
        {
            return dispatch_second(f, *v, second);
        }
        else
        {
            throw dispatch_exception("type is not supported by dispatcher");
        }
    }    
public:    
    template<typename ValT1, typename ValT2, typename CfgT, typename F>
    static auto call(F& f, const tensor_base<ValT1, CfgT>& first, const tensor_base<ValT2, CfgT>& second){
        return dispatch_first(f,first,second);
    }
};

// template<typename ValT, typename CfgT>
// class dispatcher{      
//     template<typename FirstT, typename ValT2, typename CfgT, typename F>
//     static auto dispatch_second(F& f, const FirstT& first, const tensor_base<ValT2, CfgT>& second){
//         if (second.is_storage())
//         {
//             return f(first, *second.as_storing());
//         }
//         else if (second.tensor_kind() == tensor_kinds::expression)
//         {
//             if (second.is_trivial()){
//                 return f(first, *second.as_evaluating_trivial());
//             }            
//             else{
//                 return f(first, *second.as_evaluating());
//             }
//         }
//         else if (auto v = second.as_viewing_evaluating()){            
//             return f(first, *v);
//         }        
//         else
//         {
//             throw dispatch_exception("type is not supported by dispatcher");
//         }
//     }        
//     template<typename ValT1, typename ValT2, typename CfgT, typename F>
//     static auto dispatch_first(F& f, const tensor_base<ValT1, CfgT>& first, const tensor_base<ValT2, CfgT>& second){
//         if (first.is_storage())
//         {
//             return dispatch_second(f, *first.as_storing(), second);
//         }
//         else if (first.tensor_kind() == tensor_kinds::expression)
//         {
//             if (first.is_trivial()){
//                 return dispatch_second(f, *first.as_evaluating_trivial(), second);
//             }            
//             else{
//                 return dispatch_second(f, *first.as_evaluating(), second);
//             }
//         }        
//         else if (auto v = first.as_viewing_evaluating()){
//             return dispatch_second(f, *v, second);
//         }
//         else
//         {
//             throw dispatch_exception("type is not supported by dispatcher");
//         }
//     }    
// public:    
//     template<typename ValT1, typename ValT2, typename CfgT, typename F>
//     static auto call(F& f, const tensor_base<ValT1, CfgT>& first, const tensor_base<ValT2, CfgT>& second){
//         return dispatch_first(f,first,second);
//     }
// };

}   //end of namespace detail

}   //end of namespace gtensor


#endif