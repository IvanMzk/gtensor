#ifndef WALKER_BASE_HPP_
#define WALKER_BASE_HPP_

namespace gtensor{

namespace detail{

template<typename IdxT>
constexpr inline bool can_walk_eval(const IdxT&, const IdxT&, const IdxT&){
    return true;
}
template<typename IdxT>
constexpr inline bool can_walk(const IdxT& direction, const IdxT& dim, const IdxT& direction_dim){
    //return direction < dim;
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
    detail::shape_inverter<index_type, shape_type> reset_strides;
    CursorT offset_;
    CursorT cursor_{offset_};

protected:
    basic_walker(const index_type& dim_, const shape_type& shape_, const shape_type& strides_, const shape_type& reset_strides_, const CursorT& offset__):
        dim{dim_},
        shape{shape_},
        strides{strides_},
        reset_strides{reset_strides_},
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
            cursor_-=reset_strides.element(direction);
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
class walker_trivial_root_base : public walker_base<ValT, CfgT>{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
public:
    virtual ~walker_trivial_root_base(){}
    virtual value_type operator[](const index_type& idx) const = 0;
};

template<typename ValT, typename CfgT>
class indexer_base{
protected:
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
public:
    virtual ~indexer_base(){}
    virtual std::unique_ptr<indexer_base> clone()const = 0;
    virtual value_type operator[](const index_type&)const = 0;
};

}   //end of namespace gtensor

#endif