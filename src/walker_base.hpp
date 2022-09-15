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

template<typename ValT, typename CfgT, typename CursorT>
class basic_walker
{
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    index_type dim;
    detail::shape_inverter<index_type, shape_type> shape;
    detail::shape_inverter<index_type, shape_type> strides;
    CursorT offset_;
    CursorT cursor_{offset_};

protected:
    basic_walker(const index_type& dim_, const shape_type& shape_, const shape_type& strides_, const CursorT& offset__):
        dim{dim_},
        shape{shape_},
        strides{strides_},
        offset_{offset__}
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
    void reset(){cursor_ = offset_;}

    bool can_walk(const index_type& direction)const{return detail::can_walk(direction, dim, shape.element(direction));}
    CursorT cursor()const{return cursor_;}
    CursorT offset()const{return offset_;}
};

template<typename ValT, typename CfgT>
class walker_base{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
public:
    virtual ~walker_base(){}
    virtual void walk(const index_type& direction, const index_type& steps) = 0;
    virtual void step(const index_type& direction) = 0;
    virtual void step_back(const index_type& direction) = 0;
    virtual void reset(const index_type& direction) = 0;
    virtual void reset() = 0;
    virtual value_type operator*() const = 0;
    virtual std::unique_ptr<walker_base<ValT,CfgT>> clone()const = 0;
};

template<typename ValT, typename CfgT>
class walker_trivial_base : public walker_base<ValT, CfgT>{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
public:
    virtual ~walker_trivial_base(){}
    virtual value_type operator[](const index_type& idx) const = 0;
};

template<typename ValT, typename CfgT, typename ImplT>
class walker_polymorphic : public walker_base<ValT, CfgT>
{
    using impl_type = ImplT;
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    impl_type impl_;

    std::unique_ptr<walker_base<ValT,CfgT>> clone()const override{return std::make_unique<walker_polymorphic>(*this);}

public:
    walker_polymorphic(impl_type&& impl__):
        impl_{std::move(impl__)}
    {}

    void walk(const index_type& direction, const index_type& steps)override{return impl_.walk(direction,steps);}
    void step(const index_type& direction)override{return impl_.step(direction);}
    void step_back(const index_type& direction)override{return impl_.step_back(direction);}
    void reset(const index_type& direction)override{return impl_.reset(direction);}
    void reset()override{return impl_.reset();}
    value_type operator*() const override{return impl_.operator*();}
};

template<typename ValT, typename CfgT>
class walker{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using impl_base_type = walker_base<ValT, CfgT>;
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
    auto& as_trivial()const{return static_cast<const walker_trivial_base<ValT,CfgT>&>(*impl.get());}
};

template<typename ValT, typename CfgT>
class indexer_base{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
public:
    virtual ~indexer_base(){}
    virtual std::unique_ptr<indexer_base<ValT,CfgT>> clone(int)const = 0;
    virtual value_type operator[](index_type) = 0;
};

template<typename ValT, typename CfgT>
class indexer{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using impl_base_type = indexer_base<ValT, CfgT>;

    std::unique_ptr<impl_base_type> impl;
public:
    indexer(std::unique_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}
    indexer(const indexer& other):
        impl{other.impl->clone(0)}
    {}
    indexer(indexer&& other) = default;

    value_type operator[](index_type idx){return impl->operator[](idx);}
};


}   //end of namespace gtensor

#endif