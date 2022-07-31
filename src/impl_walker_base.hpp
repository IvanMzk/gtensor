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
    virtual void walk(const index_type& direction, const index_type& steps) = 0;
    virtual void step(const index_type& direction) = 0;
    virtual void step_back(const index_type& direction) = 0;
    virtual void reset(const index_type& direction) = 0;
    virtual void reset() = 0;        
    virtual value_type operator*() const = 0;
    virtual std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const = 0;
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
    walker(const walker& other):
        impl{other.impl->clone()}
    {}
    walker(walker&& other) = default;
            
    walker& walk(const index_type& direction, const index_type& steps){
        impl->walk(direction,steps);
        return *this;
    }
    walker& step(const index_type& direction){
        impl->step(direction);
        return *this;
    }
    walker& step_back(const index_type& direction){
        impl->step_back(direction);
        return *this;
    }
    walker& reset(const index_type& direction){
        impl->reset(direction);
        return *this;
    }
    walker& reset(){
        impl->reset();
        return *this;
    }    
    value_type operator*() const{return impl->operator*();}    
};

template<typename ValT, template<typename> typename Cfg>
class evaluating_storage_impl_base{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;    
public:
    virtual ~evaluating_storage_impl_base(){}    
    virtual std::unique_ptr<evaluating_storage_impl_base<ValT,Cfg>> clone()const = 0;
    virtual value_type operator[](const index_type&)const = 0;
};

template<typename ValT, template<typename> typename Cfg>
class evaluating_storage{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using impl_base_type = evaluating_storage_impl_base<ValT, Cfg>;
    
    std::unique_ptr<impl_base_type> impl;
public:    
    evaluating_storage(std::unique_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}
    evaluating_storage(const evaluating_storage& other):
        impl{other.impl->clone()}
    {}
    evaluating_storage(evaluating_storage&& other) = default;
            
    value_type operator[](const index_type& idx){return impl->operator[](idx);}
};


}   //end of namespace gtensor

#endif