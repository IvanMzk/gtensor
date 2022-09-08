#ifndef IMPL_EXPRESSION_HPP_
#define IMPL_EXPRESSION_HPP_

#include "tensor_base.hpp"
#include "descriptor.hpp"
#include "iterator.hpp"

namespace gtensor{

class broadcast_exception : public std::runtime_error{
    public: broadcast_exception(const char* what):runtime_error(what){}
};

namespace detail{

/*
* create broadcast shape
* parameters: shapes to broadcast
* exception if shapes are not broadcastable
*/
template<typename ShT>
inline ShT broadcast(const ShT& shape1, const ShT& shape2){
    using shape_type = ShT;
    using index_type = typename ShT::value_type;
    if (shape1.size() == 0 || shape2.size() == 0){
        throw broadcast_exception("shapes are not broadcastable");
    }else{
        bool b{shape1.size() < shape2.size()};
        const shape_type& shorter{ b ? shape1 : shape2};
        const shape_type& longer{b ? shape2 : shape1};
        shape_type res(longer.size());    
        auto shorter_begin{shorter.begin()};
        auto shorter_end{shorter.end()};
        auto longer_begin{longer.begin()};
        auto longer_end{longer.end()};
        auto res_end{res.end()};
        while(shorter_begin!=shorter_end){
            const index_type& i{*--shorter_end};
            const index_type& j{*--longer_end};
            if (i==index_type(1)){
                *--res_end = j;
            }
            else if (j==index_type(1) || i==j){
                *--res_end = i;
            }                    
            else{                        
                throw broadcast_exception("shapes are not broadcastable");
            }
        }
        while(longer_begin!=longer_end){
            *--res_end = *--longer_end;
        }
        return res;        
    }
}

template<typename T> inline constexpr bool is_valid_operand = false;
template<typename...T> inline constexpr bool is_valid_operand<std::shared_ptr<tensor_base<T...>>> = true;
template<typename...Ops> inline constexpr bool is_valid_operands = (is_valid_operand<Ops>&&...);

template<typename ValT, typename CfgT>
auto de_wrap(const std::shared_ptr<tensor_wrapper<ValT,CfgT>>& wrapper){return wrapper->impl();}
template<typename ImplT>
auto de_wrap(const std::shared_ptr<ImplT>& impl){return impl;}


}   //end of namespace detail


template<typename ValT, typename CfgT, typename F, typename EngineT, typename...Ops>
class evaluating_tensor : 
    public tensor_base<ValT,CfgT>,
    //public evaluating_base<ValT,CfgT>,
    //public evaluating_trivial_base<ValT,CfgT>,
    public converting_base<CfgT>    
{    
public:
    using value_type = ValT;
    using engine_type = EngineT;    
private:    
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using descriptor_type = descriptor_with_libdivide<CfgT>;
    using iterator_type = multiindex_iterator<ValT,CfgT,walker<ValT,CfgT>>; 
    static_assert((std::is_convertible_v<Ops*, tensor_base<typename Ops::value_type, CfgT>*>&&...));
    static_assert(std::is_convertible_v<engine_type*, engine_root_accessor<evaluating_tensor,ValT,CfgT,F,EngineT,Ops...>*>);
    //static_assert(std::is_convertible_v<engine_type*, evaluating_engine_root_accessor<engine_type,ValT,CfgT,F,Ops...>*>);

    descriptor_type descriptor_;
    F f;
    std::tuple<std::shared_ptr<tensor_base<typename Ops::value_type, CfgT> >...> operands_;
    engine_type engine_;
    //friend engine_type;
    
    // walker<ValT,CfgT> create_evaluating_walker()const override{
    //     return engine_.create_walker();
    // }    
    // indexer<ValT,CfgT> create_evaluating_indexer()const override{
    //     return engine_.create_indexer();
    // }
    index_type view_index_convert(const index_type& idx)const override{return idx;}

    bool is_storage()const override{return is_cached();}
    bool is_cached()const override{return false;}
    
    //const evaluating_base<ValT,CfgT>* as_evaluating()const override{return static_cast<const evaluating_base<ValT,CfgT>*>(this);}
    //const evaluating_trivial_base<ValT,CfgT>* as_evaluating_trivial()const override{return static_cast<const evaluating_trivial_base<ValT,CfgT>*>(this);}
    const converting_base* as_converting()const override{return static_cast<const converting_base*>(this);}
protected:
        
    const auto& concrete_descriptor()const{return descriptor_;}
    template<std::size_t I>
    auto& operand(){return std::get<I>(operands_);}

public:            
    
    template<typename E, typename...O>
    explicit evaluating_tensor(E&& engine__, O&&...operands__):
        descriptor_{detail::broadcast(operands__->shape()...)},
        operands_{std::forward<O>(operands__)...},
        engine_{std::forward<E>(engine__)}
    {
        engine_.set_root(this);
    }

    const engine_type& engine()const override{return engine_;}
    const auto& operands()const{return operands_;}
    detail::tensor_kinds tensor_kind()const override{return detail::tensor_kinds::expression;}
    const descriptor_base<CfgT>& descriptor()const override{return descriptor_;}
    index_type size()const override{return descriptor_.size();}
    index_type dim()const override{return descriptor_.dim();}
    const shape_type& shape()const override{return descriptor_.shape();}
    const shape_type& strides()const override{return descriptor_.strides();}
    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor_.to_str(); return "}";}();
        return ss.str();
    }    
};


}   //end of namespace gtensor



#endif