#ifndef BROADCAST_HPP_
#define BROADCAST_HPP_

#include "libdivide_helper.hpp"
#include "descriptor.hpp"

namespace gtensor{
class broadcast_exception : public std::runtime_error{
public:
    explicit broadcast_exception(const char* what):
        runtime_error(what)
    {}
};

namespace detail{

/*
* create broadcast shape
* parameters: shapes to broadcast
* exception if shapes are not broadcastable
*/
template<typename ShT>
inline void broadcast_shape_helper(ShT&){}
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
        if (r==index_type(-1) || r==index_type(1)){
            *res_it = s;
        }
        else if (s!=index_type(1) && s!=r){
            throw broadcast_exception("shapes are not broadcastable");
        }
    }
    broadcast_shape_helper(res, shapes...);
}
template<typename ShT, typename...Ts>
inline auto broadcast_shape(const Ts&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    if (std::min({shapes.size()...}) == 0){
        throw broadcast_exception("shapes are not broadcastable");
    }else{
        auto res = shape_type(std::max({shapes.size()...}),index_type(-1));
        broadcast_shape_helper(res, shapes...);
        return res;
    }
}

template<typename SizeT, typename IdxT>
constexpr inline bool can_walk(const SizeT& direction, const SizeT& dim, const IdxT& direction_dim){
    return direction < dim && direction_dim != IdxT(1);
}

template<typename CfgT>
class shape_inverter
{
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using size_type = typename CfgT::size_type;

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
    index_type element(const size_type& direction)const{return *(shape_last-direction);}
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
    using size_type = typename CfgT::size_type;
    using shape_type = typename CfgT::shape_type;

    walker_common(const size_type& dim_, const shape_type& shape_, const shape_type& strides_, const shape_type& reset_strides_, const cursor_type& offset__):
        dim{dim_},
        shape{shape_},
        strides{strides_},
        reset_strides{reset_strides_},
        offset_{offset__}
    {}

    void walk(const size_type& direction, const index_type& steps){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_+=steps*strides.element(direction);
        }
    }
    void step(const size_type& direction){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_+=strides.element(direction);
        }
    }
    void step_back(const size_type& direction){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_-=strides.element(direction);
        }
    }
    void reset(const size_type& direction){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_-=reset_strides.element(direction);
        }
    }
    void reset_back(const size_type& direction){
        if (detail::can_walk(direction, dim, shape.element(direction))){
            cursor_+=reset_strides.element(direction);
        }
    }
    void reset(){cursor_ = offset_;}

    bool can_walk(const size_type& direction)const{return detail::can_walk(direction, dim, shape.element(direction));}
    cursor_type cursor()const{return cursor_;}
    cursor_type offset()const{return offset_;}

private:
    size_type dim;
    detail::shape_inverter<CfgT> shape;
    detail::shape_inverter<CfgT> strides;
    detail::shape_inverter<CfgT> reset_strides;
    cursor_type offset_;
    cursor_type cursor_{offset_};
};

//broadcast walker
//IndexerT is indexer type, must provide subscription operator
template<typename CfgT, typename IndexerT>
class walker
{
    using index_type = typename CfgT::index_type;
    using size_type = typename CfgT::size_type;
    using shape_type = typename CfgT::shape_type;
    using indexer_type = IndexerT;
    using result_type = typename indexer_type::result_type;

    walker_common<CfgT, index_type> index_walker;
    indexer_type indexer;
public:
    walker(const shape_type& shape_,  const shape_type& strides_, const shape_type& reset_strides_, const index_type& offset_, const indexer_type& indexer_):
        index_walker{static_cast<size_type>(shape_.size()), shape_, strides_, reset_strides_, offset_},
        indexer{indexer_}
    {}

    void walk(const size_type& direction, const index_type& steps){index_walker.walk(direction,steps);}
    void step(const size_type& direction){index_walker.step(direction);}
    void step_back(const size_type& direction){index_walker.step_back(direction);}
    void reset(const size_type& direction){index_walker.reset(direction);}
    void reset_back(const size_type& direction){index_walker.reset_back(direction);}
    void reset(){index_walker.reset();}
    result_type operator*()const{return indexer[index_walker.cursor()];}
};

/*

//variant with shape reverse, with modified strides, only subscription
shape_type strides = ...;   //init in constructor
index_type strides_last_idx = strides.size()-1; //init in constructor
index_type dim = ...; //init in constructor
void step_old(const size_type& direction){
    if (direction < dim){
        cursor+=strides[strides_last_idx - direction];
    }
}

//variant with shape reverse, with modified strides
index_type* strides_last = ...; //init in constructor
index_type dim = ...; /init in constructor
void step_old(const size_type& direction){
    if (direction < dim){
        cursor+=*(strides_last - direction);
    }
}




//variant with direction offset, without shape reverse, with modified strides
index_type direction_offset = ...;  //init in constructor
index_type* strides = ...; //init in constructor
void step_new(const size_type& direction){
    if (direction >= direction_offset){ //can move in direction
        direction-=direction_offset;
        cursor+=*(strides + direction);
    }
}
//variant with direction offset, without shape reverse, with modified strides, subscription only
index_type direction_offset = ...;  //init in constructor
shape_type strides = ...; //init in constructor
void step_new(const size_type& direction){
    if (direction >= direction_offset){ //can move in direction
        cursor+=strides[direction-direction_offset];
    }
}

//variant with strides as param
void step_new(const size_type& direction, const shape_type& strides){
    if (direction >= direction_offset){ //can move in direction
        cursor+=strides[direction-direction_offset];
    }
}

*/

template<typename CfgT, typename Walker>
class walker_forward_adapter
{
protected:
    using config_type = CfgT;
    using walker_type = Walker;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using size_type = typename config_type::size_type;

    const size_type dim_;
    const detail::shape_inverter<CfgT> shape_;
    walker_type walker_;
    shape_type index_;

public:
    template<typename Walker_>
    walker_forward_adapter(const shape_type& shape__, Walker_&& walker__):
        dim_(shape__.size()),
        shape_{shape__},
        walker_{std::forward<Walker_>(walker__)},
        index_(dim_, index_type{0})
    {}
    const auto& index()const{return index_;}
    const auto& walker()const{return walker_;}
    auto& walker(){return walker_;}
    bool next(){
        size_type direction{0}; //start from direction with min stride
        auto index_it = index_.end();
        for(;direction!=dim_;++direction){
            if (*--index_it == shape_.element(direction)-index_type{1}){   //direction at their max
                *index_it = index_type{0};
                walker_.reset(direction);
            }else{  //can next on direction
                ++(*index_it);
                walker_.step(direction);
                return true;
            }
        }
        return false;
    }
};

template<typename CfgT, typename Walker>
class walker_bidirectional_adapter : public walker_forward_adapter<CfgT, Walker>
{
protected:
    using walker_forward_adapter_base = walker_forward_adapter<CfgT, Walker>;
    using typename walker_forward_adapter_base::config_type;
    using typename walker_forward_adapter_base::walker_type;
    using typename walker_forward_adapter_base::shape_type;
    using typename walker_forward_adapter_base::index_type;
    using typename walker_forward_adapter_base::size_type;
    using walker_forward_adapter_base::walker_;
    using walker_forward_adapter_base::dim_;
    using walker_forward_adapter_base::index_;
    using walker_forward_adapter_base::shape_;

    index_type overflow_{0};
public:
    using walker_forward_adapter_base::walker_forward_adapter_base;

    bool next(){
        if (walker_forward_adapter_base::next()){
            return true;
        }else{
            if (overflow_ == index_type{-1}){
                ++overflow_;
                return true;
            }else{
                ++overflow_;
                return false;
            }
        }
    }
    bool prev(){
        size_type direction{0}; //start from direction with min stride
        auto index_it = index_.end();
        for(;direction!=dim_;++direction){
            if (*--index_it == index_type{0}){   //direction at their min
                *index_it = shape_.element(direction)-index_type{1};
                walker_.reset_back(direction);
            }else{  //can prev on direction
                --(*index_it);
                walker_.step_back(direction);
                return true;
            }
        }
        if (overflow_ == index_type{1}){
            --overflow_;
            return true;
        }else{
            --overflow_;
            return false;
        }
    }
};

template<typename CfgT, typename Walker>
class walker_iterator_adapter : public walker_bidirectional_adapter<CfgT, Walker>
{
    using walker_bidirectional_adapter_base = walker_bidirectional_adapter<CfgT, Walker>;
    using typename walker_bidirectional_adapter_base::config_type;
    using typename walker_bidirectional_adapter_base::walker_type;
    using typename walker_bidirectional_adapter_base::shape_type;
    using typename walker_bidirectional_adapter_base::index_type;
    using typename walker_bidirectional_adapter_base::size_type;
    using strides_div_type = typename detail::strides_div_traits<CfgT>::type;
    using walker_bidirectional_adapter_base::walker_;
    using walker_bidirectional_adapter_base::dim_;
    using walker_bidirectional_adapter_base::index_;
    const strides_div_type* strides_;
public:
    template<typename Walker_>
    walker_iterator_adapter(const shape_type& shape__, const strides_div_type& strides__ ,Walker_&& walker__):
        walker_bidirectional_adapter_base(shape__,walker__),
        strides_{&strides__}
    {}
    void move(index_type n){
        walker_.reset();
        auto strides_it = strides_->begin();
        auto strides_end = strides_->end();
        auto index_it = index_.begin();
        size_type direction{dim_};
        for(;strides_it!=strides_end; ++strides_it,++index_it){
            --direction;
            auto steps = detail::divide(n,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(direction,steps);
            }
            *index_it = steps;
        }
    }
};


}   //end of namespace gtensor

#endif