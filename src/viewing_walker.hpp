#ifndef VIEWING_WALKER_HPP_
#define VIEWING_WALKER_HPP_

#include "walker_base.hpp"
#include "tensor_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, typename CfgT>
class viewing_evaluating_walker : 
    public walker_base<ValT, CfgT>,
    private basic_walker<ValT, CfgT, typename CfgT::index_type>
{    
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    mutable indexer<ValT,CfgT> estorage;
    const converting_base<ValT,CfgT>* converter;
    std::unique_ptr<walker_base<ValT,CfgT>> clone()const override{return std::make_unique<viewing_evaluating_walker>(*this);}

public:        
    viewing_evaluating_walker(const shape_type& shape_,  const shape_type& strides_, const index_type& offset_, const converting_base<ValT,CfgT>* converter_, indexer<ValT,CfgT> estorage_):
        basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, offset_},
        converter{converter_},
        estorage{std::move(estorage_)}
    {}    
    
    void walk(const index_type& direction, const index_type& steps)override{basic_walker::walk(direction,steps);}
    void step(const index_type& direction)override{basic_walker::step(direction);}
    void step_back(const index_type& direction)override{basic_walker::step_back(direction);}
    void reset(const index_type& direction)override{basic_walker::reset(direction);}
    void reset()override{basic_walker::reset();}    
    value_type operator*()const override{        
        return estorage[converter->convert(basic_walker::cursor())];
    }
};

}   //end of namespace gtensor


#endif