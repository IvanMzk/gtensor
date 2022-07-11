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


template<typename ValT, template<typename> typename Cfg>
class walker{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using impl_base_type = walker_impl_base<ValT, Cfg>;
    
    std::unique_ptr<impl_base_type> impl;
public:    
    walker(std::unique_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}
    void walk(const index_type& direction, const index_type& steps){impl->walk(direction,steps);}
    void step(const index_type& direction){impl->step(direction);}
    void step_back(const index_type& direction){impl->step_back(direction);}
    void reset(const index_type& direction){impl->reset(direction);}
    void reset(){impl->reset();}    
    value_type operator*() const{return impl->operator*();}    
};


}   //end of namespace gtensor

#endif