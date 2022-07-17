#ifndef IMPL_VWALKER_HPP_
#define IMPL_VWALKER_HPP_

#include "impl_walker_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg, typename DescT ,typename StorT>
class vwalker_impl : public walker_impl_base<ValT, Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = StorT;
    using descriptor_type = DescT;

    const descriptor_type* descriptor; 
    const storage_type* elements;
    const shape_type* shape{descriptor->shape()};
    const shape_type* strides{descriptor->cstrides()};
    index_type cursor{0};
    index_type dim{descriptor->dim()};

    auto shape_element(const index_type direction)const{return (*shape)[direction];}
    auto strides_element(const index_type direction)const{return (*strides)[direction];}
    bool can_walk(const index_type& direction)const{return direction < dim && shape_element(direction) != index_type(1);}
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<vwalker_impl<ValT,Cfg>>(*this);}

public:    
    vwalker_impl(const descriptor_type& descriptor_,  const storage_type& elements_):
        descriptor{&descriptor_},
        elements{elements_}
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
    void reset() override{cursor = 0;}
    
    value_type operator*() const override{return elements->operator[](descriptor->convert_by_prev(cursor));}
};


}   //end of namespace gtensor


#endif