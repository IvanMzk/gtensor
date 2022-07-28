#ifndef DISPATCHER_HPP_
#define DISPATCHER_HPP_

#include "impl_tensor_base.hpp"

namespace gtensor{

namespace detail{

class dispatch_exception : public std::runtime_error{
    public: dispatch_exception(const char* what):runtime_error(what){}
};

template<typename ValT, template<typename> typename Cfg>
class dispatcher{
    using base_type = tensor_impl_base<ValT,Cfg>;    

    template<typename FirstT, typename F>
    static auto dispatch_second(F& f, const FirstT& first, const base_type& second){
        if (second.is_storage())
        {
            return f(first, *second.as_storage_tensor());
        }
        else if (second.tensor_kind() == tensor_kinds::expression)
        {
            if (second.is_trivial()){
                return f(first, *second.as_expression_trivial());
            }            
            else{
                return f(first, *second.as_expression());
            }
        }
        else if (second.tensor_kind() == tensor_kinds::view)
        {
            return f(first, *second.as_view());
        }
        else
        {
            throw dispatch_exception("type is not supported by dispatcher");
        }
    }        
    template<typename F>
    static auto dispatch_first(F& f, const base_type& first, const base_type& second){
        if (first.is_storage())
        {
            return dispatch_second(f, *first.as_storage_tensor(), second);
        }
        else if (first.tensor_kind() == tensor_kinds::expression)
        {
            if (first.is_trivial()){
                return dispatch_second(f, *first.as_expression_trivial(), second);
            }            
            else{
                return dispatch_second(f, *first.as_expression(), second);
            }
        }
        else if (first.tensor_kind() == tensor_kinds::view)
        {
            return dispatch_second(f, *first.as_view(), second);
        }
        else
        {
            throw dispatch_exception("type is not supported by dispatcher");
        }
    }    
public:    
    template<typename F>
    static auto call(F& f, const base_type& first, const base_type& second){
        return dispatch_first(f,first,second);
    }
};

}   //end of namespace detail

}   //end of namespace gtensor


#endif