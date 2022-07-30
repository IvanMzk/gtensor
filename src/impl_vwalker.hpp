#ifndef IMPL_VWALKER_HPP_
#define IMPL_VWALKER_HPP_

#include "impl_walker_base.hpp"
#include "libdivide_helper.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail

/*
* abstract base class for view of expression walker class, view of view walker class
* operator* not implemented
*/
template<typename ValT, template<typename> typename Cfg>
class vwalker_impl : public walker_impl_base<ValT, Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

    const shape_type* shape;
    const shape_type* strides;
    index_type offset;
    index_type dim{static_cast<index_type>(shape->size())};
    index_type cursor{offset};

    auto shape_element(const index_type direction)const{return (*shape)[direction];}
    auto strides_element(const index_type direction)const{return (*strides)[direction];}
    bool can_walk(const index_type& direction)const{return direction < dim && shape_element(direction) != index_type(1);}
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<vwalker_impl<ValT,Cfg>>(*this);}


public:    
    vwalker_impl(const shape_type& shape_,  const shape_type& strides_, const index_type& offset_):
        shape{&shape_},
        strides{&strides_},
        offset{offset_}
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
    void reset() override{cursor = offset;}
    value_type operator*() const override{        
        return 0;
    }
};

template<typename ValT, template<typename> typename Cfg>
class view_expression_walker_impl : 
    vwalker_impl<ValT, Cfg>,
    detail::reference_libdivide_extension<ValT,Cfg,typename Cfg<ValT>::div_mode>
{
    using base_type = vwalker_impl<ValT, Cfg>;
    using base_strides_libdivide = detail::reference_libdivide_extension<ValT,Cfg,typename Cfg<ValT>::div_mode>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

    walker<ValT,Cfg> parent_walker;
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<view_expression_walker_impl<ValT,Cfg>>(*this);}
    const auto& strides_libdivide()const{return base_strides_libdivide::dividers_libdivide();}

public:    
    template<typename C = config_type, std::enable_if_t<detail::is_mode_div_native<C> ,int> = 0 >
    view_expression_walker_impl(const shape_type& shape_,  const shape_type& strides_, const index_type& offset_, walker<ValT,Cfg>&& parent_walker_):
        base_type{shape_,strides_,offset_},
        base_strides_libdivide{},
        parent_walker{std::move(parent_walker_)}
    {}
    // template<typename C = config_type, std::enable_if_t<detail::is_mode_div_libdivide<C> ,int> = 0 >
    // view_expression_walker_impl(const shape_type& shape_,  const shape_type& strides_, const detail::libdivide_vector<index_type>& strides_libdivide_  const index_type& offset_, walker<ValT,Cfg>&& parent_walker_):
    //     base_type{shape_,strides_,offset_},
    //     base_strides_libdivide{strides_libdivide_},
    //     parent_walker{std::move(parent_walker_)}
    // {}
    using base_type::walk;
    using base_type::step;
    using base_type::step_back;
    using base_type::reset;
    
    value_type operator*() const override{        
        return 0;
    }
};

}   //end of namespace gtensor


#endif