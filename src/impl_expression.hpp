#ifndef IMPL_EXPRESSION_HPP_
#define IMPL_EXPRESSION_HPP_

#include "shareable_storage.hpp"
#include "impl_tensor_base.hpp"
#include "stensor_descriptor.hpp"
#include "walker_factory.hpp"

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
ShT broadcast(const ShT& shape1, const ShT& shape2){
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
template<typename...T> inline constexpr bool is_valid_operand<std::shared_ptr<tensor_impl_base<T...>>> = true;
template<typename...Ops> inline constexpr bool is_valid_operands = (is_valid_operand<Ops>&&...);

}   //end of namespace detail



template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class expression_impl : public tensor_impl_base<ValT,Cfg>{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using descriptor_type = stensor_descriptor<value_type, Cfg>;
    using storage_type = typename config_type::storage_type;
    using slices_collection_type = typename config_type::slices_collection_type; 
    using walker_factory_type = walker_factory<ValT,Cfg>; 
    static_assert(detail::is_valid_operands<Ops...>);

    descriptor_type descriptor;
    std::tuple<Ops...> operands;
    F f{};
    storage_type cache{};
    walker_factory_type walker_maker;
public:            
    explicit expression_impl(Ops&...operands_):
        descriptor{detail::broadcast(operands_->shape()...)},
        operands{operands_...},
        walker_maker{*this, descriptor, f, cache, operands}
    {}
    expression_impl(const expression_impl& other):
        descriptor{other.descriptor},
        operands{other.operands},
        f{other.f},
        cache{other.cache},
        walker_maker{*this, descriptor, f, cache, operands}
    {}
    expression_impl(expression_impl&& other):
        descriptor{std::move(other.descriptor)},
        operands{std::move(other.operands)},
        f{std::move(other.f)},
        cache{std::move(other.cache)},
        walker_maker{*this, descriptor, f, cache, operands}
    {}



    index_type size()const override{return descriptor.size();}
    index_type dim()const override{return descriptor.dim();}
    const shape_type& shape()const override{return descriptor.shape();}
    const shape_type& strides()const override{return descriptor.strides();}
    bool is_cached()const{return cache.size();}
    bool is_trivial()const {return is_cached();}

    walker<ValT,Cfg> create_walker()const override{return walker_maker.create_walker();}
    std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_transpose(const shape_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_subdim(const shape_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_reshape(const shape_type&)const override{return nullptr;}

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor.to_str(); return "}";}();
        return ss.str();
    }
};


}   //end of namespace gtensor



#endif