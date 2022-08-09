#ifndef IMPL_SWALKER_HPP_
#define IMPL_SWALKER_HPP_

#include "impl_walker_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class storage_walker_impl : 
    public walker_impl_base<ValT, Cfg>,
    private basic_walker<ValT,Cfg, const ValT*>
{   
    using base_basic_walker = basic_walker<ValT, Cfg, const ValT*>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<storage_walker_impl<ValT,Cfg>>(*this);}

public:    
    storage_walker_impl(const shape_type& shape_, const shape_type& strides_,  const value_type* data_):
        base_basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, data_}        
    {}
    void walk(const index_type& direction, const index_type& steps)override{base_basic_walker::walk(direction,steps);}
    void step(const index_type& direction)override{base_basic_walker::step(direction);}
    void step_back(const index_type& direction)override{base_basic_walker::step_back(direction);}
    void reset(const index_type& direction)override{base_basic_walker::reset(direction);}
    void reset()override{base_basic_walker::reset();}
    value_type operator*() const override{return *cursor();}
};

template<typename ValT, template<typename> typename Cfg>
class storage_walker_inline_impl :    
    private basic_walker<ValT,Cfg, const ValT*>
{   
    using base_basic_walker = basic_walker<ValT, Cfg, const ValT*>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

public:    
    storage_walker_inline_impl(const shape_type& shape_, const shape_type& strides_,  const value_type* data_):
        base_basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, data_}        
    {}
    void walk(const index_type& direction, const index_type& steps){base_basic_walker::walk(direction,steps);}
    void step(const index_type& direction){base_basic_walker::step(direction);}
    void step_back(const index_type& direction){base_basic_walker::step_back(direction);}
    void reset(const index_type& direction){base_basic_walker::reset(direction);}
    void reset(){base_basic_walker::reset();}
    value_type operator*() const {return *cursor();}
};

}   //end of namespace gtensor

#endif