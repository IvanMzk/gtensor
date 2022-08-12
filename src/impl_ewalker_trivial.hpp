#ifndef IMPL_EWALKER_TRIVIAL_HPP_
#define IMPL_EWALKER_TRIVIAL_HPP_

#include "impl_walker_base.hpp"
#include "tensor_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class ewalker_trivial_impl : 
    public walker_impl_base<ValT, Cfg>,
    private basic_walker<ValT, Cfg, typename Cfg<ValT>::index_type>
{
    using base_basic_walker = basic_walker<ValT, Cfg, typename Cfg<ValT>::index_type>;
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    const tensor_base<ValT,Cfg>* parent;
        
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<ewalker_trivial_impl<ValT,Cfg>>(*this);}

public:    
    ewalker_trivial_impl(const shape_type& shape_, const shape_type& strides_,  const tensor_base<ValT,Cfg>& parent_):
        base_basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, index_type{0}},
        parent{&parent_}
    {}
        
    void walk(const index_type& direction, const index_type& steps)override{base_basic_walker::walk(direction,steps);}
    void step(const index_type& direction)override{base_basic_walker::step(direction);}
    void step_back(const index_type& direction)override{base_basic_walker::step_back(direction);}
    void reset(const index_type& direction)override{base_basic_walker::reset(direction);}
    void reset()override{base_basic_walker::reset();}
    value_type operator*() const override{return parent->trivial_at(cursor());}
};

}   //end of namespace gtensor

#endif