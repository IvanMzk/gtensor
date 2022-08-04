#ifndef IMPL_VWALKER_HPP_
#define IMPL_VWALKER_HPP_

#include "impl_walker_base.hpp"
#include "impl_tensor_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class view_expression_walker_impl : 
    public walker_impl_base<ValT, Cfg>,
    private basic_walker<ValT, Cfg, typename Cfg<ValT>::index_type>
{
    using base_basic_walker = basic_walker<ValT, Cfg, typename Cfg<ValT>::index_type>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    mutable evaluating_storage<ValT,Cfg> estorage;
    const view_index_converter<ValT,Cfg>* converter;
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<view_expression_walker_impl<ValT,Cfg>>(*this);}

public:        
    view_expression_walker_impl(const shape_type& shape_,  const shape_type& strides_, const index_type& offset_, const view_index_converter<ValT,Cfg>* converter_, evaluating_storage<ValT,Cfg> estorage_):
        base_basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, offset_},
        converter{converter_},
        estorage{std::move(estorage_)}
    {}    
    
    void walk(const index_type& direction, const index_type& steps)override{base_basic_walker::walk(direction,steps);}
    void step(const index_type& direction)override{base_basic_walker::step(direction);}
    void step_back(const index_type& direction)override{base_basic_walker::step_back(direction);}
    void reset(const index_type& direction)override{base_basic_walker::reset(direction);}
    void reset()override{base_basic_walker::reset();}    
    value_type operator*()const override{        
        return estorage[converter->convert(base_basic_walker::cursor())];
    }
};

}   //end of namespace gtensor


#endif