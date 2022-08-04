#ifndef IMPL_SWALKER_HPP_
#define IMPL_SWALKER_HPP_

#include "impl_walker_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class storage_walker_impl : 
    public walker_impl_base<ValT, Cfg>,
    private basic_walker<ValT,Cfg>
{   
    using basic_type = basic_walker<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    const value_type* data;
    const value_type* cursor{data};

    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<storage_walker_impl<ValT,Cfg>>(*this);}

public:    
    storage_walker_impl(const shape_type& shape_, const shape_type& strides_,  const value_type* data_):
        basic_type{static_cast<index_type>(shape_.size()), shape_, strides_},
        data{data_}
    {}
    
    void walk(const index_type& direction, const index_type& steps) override{
        if (can_walk(direction)){
            cursor+=steps*strides_element(direction);
        }   
    }
    void step(const index_type& direction) override{
        if (can_walk(direction)){
            cursor+=strides_element(direction);
        }
    }
    void step_back(const index_type& direction) override{        
        if (can_walk(direction)){
            cursor-=strides_element(direction);
        }            
    }
    void reset(const index_type& direction) override{
        if (can_walk(direction)){
            cursor-=(shape_element(direction)-1)*strides_element(direction);
        }
    }
    void reset() override{cursor = data;}
    
    value_type operator*() const override{return *cursor;}
};

}   //end of namespace gtensor

#endif