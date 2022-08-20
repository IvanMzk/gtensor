#ifndef WALKER_BASE_HPP_
#define WALKER_BASE_HPP_

namespace gtensor{

namespace detail{

template<typename IdxT>
inline bool can_walk(const IdxT& direction, const IdxT& dim, const IdxT& direction_dim){
    return direction < dim && direction_dim != IdxT(1);
}

template<typename IdxT, typename ShT>
class shape_inverter
{    
    using index_type = IdxT;
    using shape_type = ShT;
    
    const index_type* shape_last;

public:
    shape_inverter(const shape_type& shape_):
        shape_last{shape_.data()+shape_.size()-1}
    {}       

    //direction must be in range [0,dim-1]
    //0 direction corresponding to last shape element - direction with minimal stride
    //1 direction corresponding to shape element befor last
    //...
    //dim-1 direction correcponding to 0 shape element - direction with max stride    
    index_type element(const index_type& direction)const{return *(shape_last-direction);}
};

}   //end of namespace detail

template<typename ValT, template<typename> typename Cfg, typename CursorT>
class basic_walker
{
    using config_type = Cfg<ValT>;        
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    index_type dim;
    detail::shape_inverter<index_type, shape_type> shape;
    detail::shape_inverter<index_type, shape_type> strides;
    CursorT offset;
    CursorT cursor_{offset};

protected:
    basic_walker(const index_type& dim_, const shape_type& shape_, const shape_type& strides_, const CursorT& offset_):
        dim{dim_},
        shape{shape_},
        strides{strides_},
        offset{offset_}
    {}    
    
    void walk(const index_type& direction, const index_type& steps){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_+=steps*strides.element(direction);
        }   
    }
    void step(const index_type& direction){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_+=strides.element(direction);
        }
    }
    void step_back(const index_type& direction){        
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_-=strides.element(direction);
        }            
    }
    void reset(const index_type& direction){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_-=(shape.element(direction)-1)*strides.element(direction);
        }
    }
    void reset(){cursor_ = offset;}
    
    CursorT cursor()const{return cursor_;}
};

template<typename ValT, template<typename> typename Cfg>
class walker_base{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;    
public:
    virtual ~walker_base(){}
    virtual void walk(const index_type& direction, const index_type& steps) = 0;
    virtual void step(const index_type& direction) = 0;
    virtual void step_back(const index_type& direction) = 0;
    virtual void reset(const index_type& direction) = 0;
    virtual void reset() = 0;        
    virtual value_type operator*() const = 0;
    virtual std::unique_ptr<walker_base<ValT,Cfg>> clone()const = 0;
};


template<typename ValT, template<typename> typename Cfg>
class walker{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using impl_base_type = walker_base<ValT, Cfg>;
    
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
class evaluating_indexer_base{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;    
public:
    virtual ~evaluating_indexer_base(){}    
    virtual std::unique_ptr<evaluating_indexer_base<ValT,Cfg>> clone(int)const = 0;
    virtual value_type operator[](index_type) = 0;
};

template<typename ValT, template<typename> typename Cfg>
class evaluating_indexer{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using impl_base_type = evaluating_indexer_base<ValT, Cfg>;
    
    std::unique_ptr<impl_base_type> impl;
public:    
    evaluating_indexer(std::unique_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}
    evaluating_indexer(const evaluating_indexer& other):
        impl{other.impl->clone(0)}
    {}
    evaluating_indexer(evaluating_indexer&& other) = default;
            
    value_type operator[](index_type idx){return impl->operator[](idx);}
};


}   //end of namespace gtensor

#endif