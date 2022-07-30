#ifndef IMPL_EXPRESSION_HPP_
#define IMPL_EXPRESSION_HPP_

#include "shareable_storage.hpp"
#include "impl_tensor_base.hpp"
#include "stensor_descriptor.hpp"
#include "walker_factory.hpp"
#include "impl_multiindex_iterator.hpp"

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

/*
* is expression is trivial broadcast i.e. shapes of all nodes in expression tree is same
* flat index access without walkers is used to evaluate broadcast expression
* stensor and view are trivial
*/
template<typename...T, typename...Ops>
inline bool is_trivial(const expression_impl<T...>& root, const std::tuple<Ops...>& root_operands){
    return is_trivial_helper(root,root_operands,std::make_index_sequence<sizeof...(Ops)>{});
}
template<typename...T, typename...Ops, std::size_t...I>
inline bool is_trivial_helper(const expression_impl<T...>& root, const std::tuple<Ops...>& root_operands, std::index_sequence<I...>){
    return ((root.size()==std::get<I>(root_operands)->size())&&...) && (is_trivial_operand(std::get<I>(root_operands))&&...);
}
template<typename T>
inline bool is_trivial_operand(const T& operand){
    if (auto e = operand->as_expression()){
        return e->is_trivial(); 
    }else{
        return true;
    }
}



template<typename T> inline constexpr bool is_valid_operand = false;
template<typename...T> inline constexpr bool is_valid_operand<std::shared_ptr<tensor_impl_base<T...>>> = true;
template<typename...Ops> inline constexpr bool is_valid_operands = (is_valid_operand<Ops>&&...);

}   //end of namespace detail



template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class expression_impl : 
    public tensor_impl_base<ValT,Cfg>,
    public expression_impl_base<ValT,Cfg>,
    public trivial_impl_base<ValT,Cfg>,
    public storage_tensor_impl_base<ValT,Cfg>,
    public view_index_converter<ValT,Cfg>
{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using descriptor_type = stensor_descriptor<value_type, Cfg>;
    using storage_type = typename config_type::storage_type;
    using slices_collection_type = typename config_type::slices_collection_type;     
    using iterator_type = multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>; 
    static_assert(detail::is_valid_operands<Ops...>);

    descriptor_type descriptor;
    std::tuple<Ops...> operands;
    F f{};
    storage_type cache{};    

    // template<typename C = config_type, std::enable_if_t<detail::is_mode_div_native<C> ,int> =0 >
    // auto create_iterator(const index_type& i)const{
    //     return i==0 ? iterator_type{create_walker(), shape(), strides()} : iterator_type{create_walker(), shape(), strides(), i};
    // }
    // template<typename C = config_type, std::enable_if_t<detail::is_mode_div_libdivide<C> ,int> =0 >
    // auto create_iterator(const index_type& i)const{
    //     return i==0 ? iterator_type{create_walker(), shape(), descriptor.strides_libdivide()} : iterator_type{create_walker(), shape(), descriptor.strides_libdivide(), i};
    // } 
    template<std::size_t...I>
    value_type trivial_at_helper(const index_type& idx, std::index_sequence<I...>)const{return f(std::get<I>(operands)->trivial_at(idx)...);} 
    
    storage_walker_impl<ValT,Cfg> create_storage_walker()const override{
        return storage_walker_factory<ValT,Cfg>::create_walker(shape(),strides(),cache.data());
    }
    walker<ValT,Cfg> create_evaluating_walker()const override{
        return evaluating_walker_factory<ValT,Cfg>::create_walker(shape(),f,operands);
    }
    ewalker_trivial_impl<ValT,Cfg> create_trivial_walker()const override{
        return trivial_walker_factory<ValT,Cfg>::create_walker(shape(), strides(), *this);
    }

    index_type view_index_convert(const index_type& idx)const override{return idx;}
    bool is_storage()const override{return is_cached();}
    const value_type* storage_data()const override{return cache.data();}
    
    const expression_impl_base<ValT,Cfg>* as_expression()const override{return static_cast<const expression_impl_base<ValT,Cfg>*>(this);}
    const trivial_impl_base<ValT,Cfg>* as_expression_trivial()const override{return static_cast<const trivial_impl_base<ValT,Cfg>*>(this);}
    const storage_tensor_impl_base<ValT,Cfg>* as_storage_tensor()const override{return static_cast<const storage_tensor_impl_base<ValT,Cfg>*>(this);}
    const view_index_converter<ValT,Cfg>* as_index_converter()const override{return static_cast<const view_index_converter<ValT,Cfg>*>(this);}

public:            
    explicit expression_impl(Ops&...operands_):
        descriptor{detail::broadcast(operands_->shape()...)},
        operands{operands_...}
    {}
    expression_impl(const expression_impl& other):
        descriptor{other.descriptor},
        operands{other.operands},
        f{other.f},
        cache{other.cache}
    {}
    expression_impl(expression_impl&& other):
        descriptor{std::move(other.descriptor)},
        operands{std::move(other.operands)},
        f{std::move(other.f)},
        cache{std::move(other.cache)}
    {}

    detail::tensor_kinds tensor_kind()const override{return detail::tensor_kinds::expression;}
    index_type size()const override{return descriptor.size();}
    index_type dim()const override{return descriptor.dim();}
    const shape_type& shape()const override{return descriptor.shape();}
    const shape_type& strides()const override{return descriptor.strides();}
    bool is_cached()const override{return cache.size();}
    bool is_trivial()const override{return detail::is_trivial(*this,operands);}
    // iterator_type begin()const{return create_iterator(0);}    
    // iterator_type end()const{return create_iterator(size());}
    value_type trivial_at(const index_type& idx)const override{return trivial_at_helper(idx,std::make_index_sequence<sizeof...(Ops)>{});}

    
    

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor.to_str(); return "}";}();
        return ss.str();
    }
};


}   //end of namespace gtensor



#endif