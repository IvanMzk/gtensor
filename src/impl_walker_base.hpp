#ifndef IMPL_WALKER_BASE_HPP_
#define IMPL_WALKER_BASE_HPP_

namespace gtensor{

template<typename ValT, template<typename> typename Cfg>
class walker_impl_base{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
public:
    virtual ~walker_impl_base(){}
    virtual void walk(index_type direction, index_type steps) = 0;
    virtual void step(index_type direction) = 0;
    virtual void step_back(index_type direction) = 0;
    virtual void reset(index_type direction) = 0;
    virtual void reset() = 0;        
    virtual value_type operator*() const = 0;
};



}   //end of namespace gtensor

#endif