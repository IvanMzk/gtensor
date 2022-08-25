#ifndef IMPL_SWALKER_HPP_
#define IMPL_SWALKER_HPP_

#include "walker_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail

template<typename ValT, typename CfgT>
class storage_walker :    
    private basic_walker<ValT,CfgT, const ValT*>
{   
    using base_basic_walker = basic_walker<ValT, CfgT, const ValT*>;    
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

public:    
    storage_walker(const shape_type& shape_, const shape_type& strides_,  const value_type* data_):
        base_basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, data_}        
    {}
    void walk(const index_type& direction, const index_type& steps){base_basic_walker::walk(direction,steps);}
    void step(const index_type& direction){base_basic_walker::step(direction);}
    void step_back(const index_type& direction){base_basic_walker::step_back(direction);}
    void reset(const index_type& direction){base_basic_walker::reset(direction);}
    void reset(){base_basic_walker::reset();}
    value_type operator*() const {return *cursor();}
    value_type operator[](const index_type& idx)const{return *(offset()+idx);}
};

template<typename ValT, typename CfgT>
class storage_walker_polymorphic : 
    public walker_base<ValT, CfgT>,
    private storage_walker<ValT,CfgT>
{   
    using base_storage_walker = storage_walker<ValT, CfgT>;    
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    
    std::unique_ptr<walker_base<ValT,CfgT>> clone()const override{return std::make_unique<storage_walker_polymorphic>(*this);}

public:    
    storage_walker_polymorphic(const shape_type& shape_, const shape_type& strides_,  const value_type* data_):
        base_storage_walker{shape_, strides_, data_}        
    {}
    storage_walker_polymorphic(const base_storage_walker& base_):
        base_storage_walker{base_}        
    {}
    void walk(const index_type& direction, const index_type& steps)override{base_storage_walker::walk(direction,steps);}
    void step(const index_type& direction)override{base_storage_walker::step(direction);}
    void step_back(const index_type& direction)override{base_storage_walker::step_back(direction);}
    void reset(const index_type& direction)override{base_storage_walker::reset(direction);}
    void reset()override{base_storage_walker::reset();}
    value_type operator*() const override{return base_storage_walker::operator*();}
};

}   //end of namespace gtensor

#endif