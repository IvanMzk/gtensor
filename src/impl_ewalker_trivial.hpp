#ifndef IMPL_EWALKER_TRIVIAL_HPP_
#define IMPL_EWALKER_TRIVIAL_HPP_

#include "impl_walker_base.hpp"
#include "impl_tensor_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class ewalker_trivial_impl : public walker_impl_base<ValT, Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    const shape_type* shape;
    const shape_type* strides;
    const tensor_impl_base<ValT,Cfg>* parent;
    index_type cursor{0};
    index_type dim{static_cast<index_type>(shape->size())};
        

    auto shape_element(const index_type direction)const{return (*shape)[direction];}
    auto strides_element(const index_type direction)const{return (*strides)[direction];}
    bool can_walk(const index_type& direction)const{return direction < dim && shape_element(direction) != index_type(1);}
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<ewalker_trivial_impl<ValT,Cfg>>(*this);}

public:    
    ewalker_trivial_impl(const shape_type& shape_, const shape_type& strides_,  const tensor_impl_base<ValT,Cfg>& parent_):
        shape{&shape_},
        strides{&strides_},
        parent{&parent_}
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
    
    value_type operator*() const override{return parent->trivial_at(cursor);}
};

}   //end of namespace gtensor

#endif