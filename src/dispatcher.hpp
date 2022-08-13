#ifndef DISPATCHER_HPP_
#define DISPATCHER_HPP_

#include "tensor_base.hpp"

namespace gtensor{

namespace detail{

class dispatch_exception : public std::runtime_error{
    public: dispatch_exception(const char* what):runtime_error(what){}
};

template<typename ValT, template<typename> typename Cfg>
class dispatcher{
    using base_type = tensor_base<ValT,Cfg>;    

    template<typename FirstT, typename F>
    static auto dispatch_second(F& f, const FirstT& first, const base_type& second){
        if (second.is_storage())
        {
            return f(first, *second.as_storing());
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
        else if (auto v = second.as_view())
        {
            if (v->view_root_kind() == tensor_kinds::expression){
                if (second.is_trivial()){
                    throw dispatch_exception("type is not supported by dispatcher");
                }else{
                    return f(first, *second.as_view_expression());
                }
            }else{
                throw dispatch_exception("type is not supported by dispatcher");
            }
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
            return dispatch_second(f, *first.as_storing(), second);
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
        else if (auto v = first.as_view())
        {
            if (v->view_root_kind() == tensor_kinds::expression){
                if (first.is_trivial()){
                    throw dispatch_exception("type is not supported by dispatcher");
                }else{
                    return dispatch_second(f, *first.as_view_expression(), second);
                }
            }
            else{
                throw dispatch_exception("type is not supported by dispatcher");
            }
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