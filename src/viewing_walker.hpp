#ifndef VIEWING_WALKER_HPP_
#define VIEWING_WALKER_HPP_

#include "walker_base.hpp"
#include "tensor_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class viewing_evaluating_walker : 
    public walker_base<ValT, Cfg>,
    private basic_walker<ValT, Cfg, typename Cfg<ValT>::index_type>
{
    using base_basic_walker = basic_walker<ValT, Cfg, typename Cfg<ValT>::index_type>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    mutable evaluating_indexer<ValT,Cfg> estorage;
    const converting_base<ValT,Cfg>* converter;
    std::unique_ptr<walker_base<ValT,Cfg>> clone()const override{return std::make_unique<viewing_evaluating_walker<ValT,Cfg>>(*this);}

public:        
    viewing_evaluating_walker(const shape_type& shape_,  const shape_type& strides_, const index_type& offset_, const converting_base<ValT,Cfg>* converter_, evaluating_indexer<ValT,Cfg> estorage_):
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