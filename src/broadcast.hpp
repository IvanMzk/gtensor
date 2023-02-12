#ifndef BROADCAST_HPP_
#define BROADCAST_HPP_

namespace gtensor{

class broadcast_exception : public std::runtime_error{
    public: broadcast_exception(const char* what):runtime_error(what){}
};

namespace detail{

/*
* create broadcast shape
* parameters: shapes to broadcast
* exception if shapes are not broadcastable
*/
template<typename ShT, typename...Ts>
inline auto broadcast_shape(const Ts&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    if (std::min({shapes.size()...}) == 0){
        throw broadcast_exception("shapes are not broadcastable");
    }else{
        auto res = shape_type(std::max({shapes.size()...}),index_type(0));
        broadcast_shape_helper(res, shapes...);
        return res;
    }
}
template<typename ShT, typename T, typename...Ts>
inline void broadcast_shape_helper(ShT& res, const T& shape, const Ts&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    auto res_it = res.end();
    auto shape_it = shape.end();
    auto shape_begin = shape.begin();
    while(shape_it!=shape_begin){
        const index_type& r{*--res_it};
        const index_type& s{*--shape_it};
        if (r==index_type(0) || r==index_type(1)){
            *res_it = s;
        }
        else if (s!=index_type(1) && s!=r){
            throw broadcast_exception("shapes are not broadcastable");
        }
    }
    broadcast_shape_helper(res, shapes...);
}
template<typename ShT>
inline void broadcast_shape_helper(ShT&){}

template<typename IdxT>
constexpr inline bool can_walk(const IdxT& direction, const IdxT& dim, const IdxT& direction_dim){
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

//broadcast walker cursor moving functionality
//CursorT is type of cursor, operations like cursor+n, cursor+=n, cursor-n, cursor-=n, where n is index_type must be defined
//e.g. CursorT can be integral type or can be random access iterator type
template<typename CfgT, typename CursorT>
class walker_common
{
public:
    using cursor_type = CursorT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    walker_common(const index_type& dim_, const shape_type& shape_, const shape_type& strides_, const shape_type& reset_strides_, const cursor_type& offset__):
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
    void reset_back(const index_type& direction){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_+=reset_strides.element(direction);
        }
    }
    void reset(){cursor_ = offset_;}

    bool can_walk(const index_type& direction)const{return detail::can_walk(direction, dim, shape.element(direction));}
    cursor_type cursor()const{return cursor_;}
    cursor_type offset()const{return offset_;}

private:
    index_type dim;
    detail::shape_inverter<index_type, shape_type> shape;
    detail::shape_inverter<index_type, shape_type> strides;
    detail::shape_inverter<index_type, shape_type> reset_strides;
    cursor_type offset_;
    cursor_type cursor_{offset_};
};

//broadcast walker
//IndexerT is indexer type, must provide subscription operator
template<typename CfgT, typename IndexerT>
class walker
{
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using indexer_type = IndexerT;
    using result_type = typename indexer_type::result_type;

    walker_common<CfgT, index_type> index_walker;
    indexer_type indexer;
public:
    walker(const shape_type& shape_,  const shape_type& strides_, const shape_type& reset_strides_, const index_type& offset_, const indexer_type& indexer_):
        index_walker{static_cast<index_type>(shape_.size()), shape_, strides_, reset_strides_, offset_},
        indexer{indexer_}
    {}

    void walk(const index_type& direction, const index_type& steps){index_walker.walk(direction,steps);}
    void step(const index_type& direction){index_walker.step(direction);}
    void step_back(const index_type& direction){index_walker.step_back(direction);}
    void reset(const index_type& direction){index_walker.reset(direction);}
    void reset_back(const index_type& direction){index_walker.reset_back(direction);}
    void reset(){index_walker.reset();}
    result_type operator*()const{return indexer[index_walker.cursor()];}
};

}   //end of namespace gtensor

#endif