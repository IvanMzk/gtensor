#ifndef DATA_ACCESSOR_HPP_
#define DATA_ACCESSOR_HPP_

#include <type_traits>
#include <numeric>
#include <algorithm>
#include <iterator>
#include "descriptor.hpp"

namespace gtensor{

namespace detail{

//Axes is container or scalar
//result is also container or scalar of axes
template<typename Config, typename DimT, typename Axes>
auto make_axes(const DimT& dim, const Axes& axes_){
    if constexpr (detail::is_container_v<Axes>){
        using dim_type = typename Config::dim_type;
        using res_type = typename Config::template shape<dim_type>;
        res_type res{};
        detail::reserve(res,axes_.size());
        std::transform(axes_.begin(),axes_.end(),std::back_inserter(res),[dim](const auto& axis_){return make_axis(dim,axis_);});
        return res;
    }else{
        return make_axis(dim,axes_);
    }
}

//Axes is container or scalar, axes should be result of make_axes
//result is container of permuted axes
template<typename Config, typename DimT, typename Axes>
auto make_range_traverser_axes_map(const DimT& dim, const Axes& axes){
    using dim_type = typename Config::dim_type;
    using res_type = typename Config::template shape<dim_type>;
    res_type res(dim,dim_type{0});
    std::iota(res.begin(),res.end(),dim_type{0});
    if constexpr (detail::is_container_v<Axes>){
        std::stable_partition(res.begin(),res.end(),
            [&axes](const auto& axis){
                const auto last = axes.end();
                return std::find_if(axes.begin(),last,[axis](const auto& a){return axis == a;}) != last;
            }
        );
    }else{  //axes scalar
        auto tmp = *(res.begin()+axes);
        std::copy_backward(res.begin(),res.begin()+axes,res.begin()+axes+1);
        *res.begin() = tmp;
    }
    return res;
}

//make shape for given axes permutation
template<typename ShT, typename AxesMap>
auto make_range_traverser_shape(const ShT& shape, const AxesMap& axes_map){
    const auto dim = detail::make_dim(shape);
    ShT res{};
    detail::reserve(res,dim);
    std::transform(axes_map.begin(),axes_map.end(),std::back_inserter(res),[&shape](const auto& a){return shape[a];});
    return res;
}

//make strides_div for walker_random_access_traverser
template<typename Config, typename ShT, typename DimT, typename Order>
auto make_range_traverser_strides_div(const ShT& shape, const DimT& axes_size, Order order){
    using res_type = strides_div_t<Config>;
    using res_value_type = typename res_type::value_type;
    const auto dim = detail::make_dim(shape);
    res_type res(dim,res_value_type{1});
    make_strides(shape.begin(),shape.begin()+axes_size,res.begin(),res.begin()+axes_size,order);
    make_strides(shape.begin()+axes_size,shape.end(),res.begin()+axes_size,res.end(),order);
    return res;
}

}   //end of namespace detail

//basic indexer is data accessor that uses flat index to address data
//indexers can be chained to make data view
template<typename...> class basic_indexer;
//Should be the first indexer in chain
//Parent is data storage that provide subscript operator
template<typename Parent>
class basic_indexer<Parent&>
{
    using parent_type = Parent;
    parent_type* parent_;
public:
    template<typename Parent_> struct enable_parent_ : std::conjunction<
        std::is_lvalue_reference<Parent_>,
        std::negation<std::is_same<std::remove_cv_t<std::remove_reference_t<Parent_>>, basic_indexer>>
    >{};

    template<typename Parent_, std::enable_if_t<enable_parent_<Parent_>::value,int> =0>
    explicit basic_indexer(Parent_&& parent__):
        parent_{&parent__}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
        return (*parent_)[i];
    }
};
//shouldn't be the first indexer in the chain
//Indexer is type of previous indexer in the chain
template<typename Indexer>
class basic_indexer<Indexer>
{
    static_assert(!std::is_reference_v<Indexer>);
    using indexer_type = Indexer;
    indexer_type indexer_;
public:
    template<typename Indexer_, std::enable_if_t<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Indexer_>>, basic_indexer>,int> =0>
    explicit basic_indexer(Indexer_&& indexer__):
        indexer_{std::forward<Indexer_>(indexer__)}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
        return indexer_[i];
    }
};
//map data elements using converter
//Indexer is type of previous indexer in the chain or data storage
//Converter is flat index mapper, must provide operator()() that take index as parameter and return mapped index
template<typename Indexer, typename Converter>
class basic_indexer<Indexer, Converter&>
{
    using indexer_type = basic_indexer<Indexer>;
    using Converter_type = Converter;
public:
    template<typename Indexer_, typename Converter_, std::enable_if_t<std::is_lvalue_reference_v<Converter_> ,int> = 0>
    basic_indexer(Indexer_&& indexer__,  Converter_&& converter__):
        indexer_{std::forward<Indexer_>(indexer__)},
        converter_{&converter__}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
        return indexer_[converter_->operator()(i)];
    }
private:
    indexer_type indexer_;
    const Converter_type* converter_;
};
template<typename Indexer, typename Converter>
class basic_indexer<Indexer, Converter>
{
    static_assert(!std::is_reference_v<Converter>);
    using indexer_type = basic_indexer<Indexer>;
    using Converter_type = Converter;
public:
    template<typename Indexer_, typename Converter_>
    basic_indexer(Indexer_&& indexer__,  Converter_&& converter__):
        indexer_{std::forward<Indexer_>(indexer__)},
        converter_{std::forward<Converter_>(converter__)}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
        return indexer_[converter_(i)];
    }
private:
    indexer_type indexer_;
    Converter_type converter_;
};

//walker-indexer adaptes walker to indexer interface
//Order can be c_order or f_order, specifies how to treat index parameter of subscript operator
//strides must be in this order
template<typename Walker, typename Order>
class walker_indexer
{
    static_assert(!std::is_reference_v<Walker>);
    using walker_type = Walker;
    using config_type = typename walker_type::config_type;
    using dim_type = typename walker_type::dim_type;
    using index_type = typename walker_type::index_type;
    using strides_div_type = detail::strides_div_t<config_type>;

    const strides_div_type* strides_;
    mutable walker_type walker_;
public:
    template<typename Walker_>
    walker_indexer(const strides_div_type& strides__, Walker_&& walker__):
        strides_{&strides__},
        walker_{std::forward<Walker_>(walker__)}
    {}
    decltype(auto) operator[](index_type i)const{
        walker_.reset_back();
        subscript_helper(i, Order{});
        return *walker_;
    }
private:
    decltype(auto) subscript_helper(index_type i, gtensor::config::c_order)const{
        dim_type direction{0};
        for(auto strides_it = strides_->begin(), srtides_end=strides_->end(); strides_it!=srtides_end; ++strides_it, ++direction){
            auto steps = detail::divide(i,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(direction,steps);
            }
        }
    }
    decltype(auto) subscript_helper(index_type i, gtensor::config::f_order)const{
        auto direction = static_cast<dim_type>(strides_->size());
        for(auto strides_it = strides_->end(), strides_first=strides_->begin(); strides_it!=strides_first;){
            --strides_it;
            --direction;
            auto steps = detail::divide(i,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(direction,steps);
            }
        }
    }
};
//iterator-indexer adaptes iterator to indexer interface
template<typename Iterator>
class iterator_indexer
{
    static_assert(!std::is_reference_v<Iterator>);
    using iterator_type = Iterator;
    using difference_type = typename std::iterator_traits<iterator_type>::difference_type;
    iterator_type iterator_;
public:
    //iterator__ argument should point to the first element
    template<typename Iterator_, std::enable_if_t<!std::is_convertible_v<std::decay_t<Iterator_>, iterator_indexer>,int> =0>
    explicit iterator_indexer(Iterator_&& iterator__):
        iterator_{std::forward<Iterator_>(iterator__)}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
        static_assert(std::is_convertible_v<U,difference_type>);
        iterator_type tmp = iterator_;
        std::advance(tmp, i);
        return *tmp;
    }
};

//walker is indexer adapter that allows address data elements using multidimensional index
//Cursor is responsible for storing flat position, it may have semantic of integral type or random access iterator
template<typename Config, typename Cursor>
class walker_common
{
public:
    using cursor_type = Cursor;
    using index_type = typename Config::index_type;
    using dim_type = typename Config::dim_type;
    using shape_type = typename Config::shape_type;

    walker_common(const shape_type& adapted_strides__, const shape_type& reset_strides__, const cursor_type& offset__):
        adapted_strides_{&adapted_strides__},
        reset_strides_{&reset_strides__},
        offset_{offset__},
        cursor_{offset__}
    {}
    //axis must be in range [0,dim-1]
    void walk(const dim_type& axis, const index_type& steps){
        cursor_+=steps*(*adapted_strides_)[axis];
    }
    void walk_back(const dim_type& axis, const index_type& steps){
        cursor_-=steps*(*adapted_strides_)[axis];
    }
    void step(const dim_type& axis){
        cursor_+=(*adapted_strides_)[axis];
    }
    void step_back(const dim_type& axis){
        cursor_-=(*adapted_strides_)[axis];
    }
    void reset(const dim_type& axis){
        cursor_+=(*reset_strides_)[axis];
    }
    void reset_back(const dim_type& axis){
        cursor_-=(*reset_strides_)[axis];
    }
    void reset_back(){cursor_ = offset_;}
    void update_offset(){offset_+=cursor_;}
    dim_type dim()const{return detail::make_dim(*adapted_strides_);}
    cursor_type cursor()const{return cursor_;}
    cursor_type offset()const{return offset_;}
private:
    const shape_type* adapted_strides_;
    const shape_type* reset_strides_;
    cursor_type offset_;
    cursor_type cursor_;
};

//Indexer is adaptee
template<typename Config, typename Indexer>
class indexer_walker
{
    using indexer_type = Indexer;
public:
    using config_type = Config;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;

    template<typename Indexer_>
    indexer_walker(const shape_type& adapted_strides_, const shape_type& reset_strides_, const index_type& offset_, Indexer_&& indexer_):
        impl{adapted_strides_, reset_strides_, offset_},
        indexer{std::forward<Indexer_>(indexer_)}
    {}
    void walk(const dim_type& axis, const index_type& steps){
        impl.walk(axis,steps);
    }
    void walk_back(const dim_type& axis, const index_type& steps){
        impl.walk_back(axis,steps);
    }
    void step(const dim_type& axis){
        impl.step(axis);
    }
    void step_back(const dim_type& axis){impl.step_back(axis);}
    void reset(const dim_type& axis){impl.reset(axis);}
    void reset_back(const dim_type& axis){impl.reset_back(axis);}
    void reset_back(){impl.reset_back();}
    void update_offset(){impl.update_offset();}
    dim_type dim(){return impl.dim();}
    decltype(auto) operator*()const{return indexer[impl.cursor()];}
private:
    walker_common<config_type, index_type> impl;
    indexer_type indexer;
};

//walker decorators
template<typename BaseWalker>
class offsetting_walker : private BaseWalker
{
    using base_walker_type = BaseWalker;
public:
    using typename base_walker_type::config_type;
    using typename base_walker_type::shape_type;
    using typename base_walker_type::index_type;
    using typename base_walker_type::dim_type;

    //offset.size() equals number of subscripts given to make slice view
    template<typename...Args>
    offsetting_walker(const shape_type& offset,Args&&...args):
        base_walker_type{std::forward<Args>(args)...}
    {
        const auto n = detail::make_dim(offset);
        for (dim_type axis=0; axis!=n; ++axis){
            base_walker_type::walk(axis,offset[axis]);
        }
        base_walker_type::update_offset();
    }
    using base_walker_type::walk;
    using base_walker_type::walk_back;
    using base_walker_type::step;
    using base_walker_type::step_back;
    using base_walker_type::reset;
    using base_walker_type::reset_back;
    using base_walker_type::operator*;
    using base_walker_type::update_offset;
};

template<typename BaseWalker>
class mapping_axes_walker : private BaseWalker
{
    using base_walker_type = BaseWalker;
    using base_config_type = typename base_walker_type::config_type;
    using axes_map_type = typename base_config_type::template shape<typename base_config_type::dim_type>;
public:
    using typename base_walker_type::config_type;
    using typename base_walker_type::shape_type;
    using typename base_walker_type::index_type;
    using typename base_walker_type::dim_type;

    //if a is axis of view then axes_map_[a] is corresponding axis of view's parent
    //axes_map_.size() always equals to view dim
    template<typename...Args>
    mapping_axes_walker(const axes_map_type& axes_map__, Args&&...args):
        base_walker_type{std::forward<Args>(args)...},
        axes_map_{&axes_map__}
    {}
    dim_type dim()const{return detail::make_dim(*axes_map_);}
    void walk(const dim_type& axis, const index_type& steps){base_walker_type::walk(map_axis(axis),steps);}
    void walk_back(const dim_type& axis, const index_type& steps){base_walker_type::walk_back(map_axis(axis),steps);}
    void step(const dim_type& axis){
        base_walker_type::step(map_axis(axis));
    }
    void step_back(const dim_type& axis){base_walker_type::step_back(map_axis(axis));}
    void reset(const dim_type& axis){base_walker_type::reset(map_axis(axis));}
    void reset_back(const dim_type& axis){base_walker_type::reset_back(map_axis(axis));}
    void reset_back(){base_walker_type::reset_back();}
    using base_walker_type::operator*;
    using base_walker_type::update_offset;
private:
    dim_type map_axis(const dim_type& axis){
        return (*axes_map_)[axis];
    }
    const axes_map_type* axes_map_;
};

template<typename BaseWalker>
class scaling_walker : private BaseWalker
{
    using base_walker_type = BaseWalker;
public:
    using typename base_walker_type::config_type;
    using typename base_walker_type::shape_type;
    using typename base_walker_type::index_type;
    using typename base_walker_type::dim_type;

    //if a is axis of view then step_scale_[a] is steps number along a in parent that corresponds single step along a in view
    //step_scale_.size() always equals to view dim
    template<typename...Args>
    scaling_walker(const shape_type& step_scale__, Args&&...args):
        base_walker_type{std::forward<Args>(args)...},
        step_scale_{&step_scale__}
    {}
    dim_type dim()const{return detail::make_dim(*step_scale_);}
    void walk(const dim_type& axis, const index_type& steps){base_walker_type::walk(axis,steps*step_scale(axis));}
    void walk_back(const dim_type& axis, const index_type& steps){base_walker_type::walk_back(axis,steps*step_scale(axis));}
    void step(const dim_type& axis){base_walker_type::walk(axis,step_scale(axis));}
    void step_back(const dim_type& axis){base_walker_type::walk_back(axis,step_scale(axis));}
    using base_walker_type::reset;
    using base_walker_type::reset_back;
    using base_walker_type::operator*;
    using base_walker_type::update_offset;
private:

    index_type step_scale(const dim_type& axis)const{
        return (*step_scale_)[axis];
    }
    const shape_type* step_scale_;
};

template<typename BaseWalker>
class resetting_walker : private BaseWalker
{
    using base_walker_type = BaseWalker;
public:
    using typename base_walker_type::config_type;
    using typename base_walker_type::shape_type;
    using typename base_walker_type::index_type;
    using typename base_walker_type::dim_type;

    //offset.size() equals number of subscripts given to make slice view
    template<typename...Args>
    resetting_walker(const shape_type& shape_,Args&&...args):
        base_walker_type{std::forward<Args>(args)...},
        shape{&shape_}
    {}
    using base_walker_type::dim;
    void walk(const dim_type& axis, const index_type& steps){
        if (can_move_on_axis(axis)){
            base_walker_type::walk(axis,steps);
        }
    }
    void walk_back(const dim_type& axis, const index_type& steps){
        if (can_move_on_axis(axis)){
            base_walker_type::walk_back(axis,steps);
        }
    }
    void step(const dim_type& axis){
        if (can_move_on_axis(axis)){
            base_walker_type::step(axis);
        }
    }
    void step_back(const dim_type& axis){
        if (can_move_on_axis(axis)){
            base_walker_type::step_back(axis);
        }
    }
    void reset(const dim_type& axis){
        base_walker_type::walk(axis,shape_element(axis)-index_type{1});
    }
    void reset_back(const dim_type& axis){
        base_walker_type::walk_back(axis,shape_element(axis)-index_type{1});
    }
    void reset_back(){
        base_walker_type::reset_back();
    }
    using base_walker_type::operator*;
    using base_walker_type::update_offset;
private:
    index_type shape_element(const dim_type& axis)const{
        return (*shape)[axis];
    }
    bool can_move_on_axis(const dim_type& axis)const{
        return shape_element(axis)>index_type{1};
    }
    const shape_type* shape;
};

template<typename BaseWalker>
class axes_correction_walker : private BaseWalker
{
    using base_walker_type = BaseWalker;
public:
    using typename base_walker_type::config_type;
    using typename base_walker_type::shape_type;
    using typename base_walker_type::index_type;
    using typename base_walker_type::dim_type;

    template<typename...Args>
    axes_correction_walker(const dim_type& max_dim,Args&&...args):
        base_walker_type{std::forward<Args>(args)...},
        dim_offset_{max_dim-base_walker_type::dim()}
    {}

    void walk(const dim_type& axis, const index_type& steps){
        if (can_move_on_axis(axis)){
            base_walker_type::walk(make_axis(axis),steps);
        }
    }
    void walk_back(const dim_type& axis, const index_type& steps){
        if (can_move_on_axis(axis)){
            base_walker_type::walk_back(make_axis(axis),steps);
        }
    }
    void step(const dim_type& axis){
        if (can_move_on_axis(axis)){
            base_walker_type::step(make_axis(axis));
        }
    }
    void step_back(const dim_type& axis){
        if (can_move_on_axis(axis)){
            base_walker_type::step_back(make_axis(axis));
        }
    }
    void reset(const dim_type& axis){
        if (can_move_on_axis(axis)){
            base_walker_type::reset(make_axis(axis));
        }
    }
    void reset_back(const dim_type& axis){
        if (can_move_on_axis(axis)){
            base_walker_type::reset_back(make_axis(axis));
        }
    }
    void reset_back(){
        base_walker_type::reset_back();
    }
    using base_walker_type::operator*;
    using base_walker_type::update_offset;
    using base_walker_type::dim;
private:
    bool can_move_on_axis(const dim_type& axis)const{
        return axis >= dim_offset_;
    }
    dim_type make_axis(const dim_type& axis)const{
        return axis - dim_offset_;
    }
    dim_type dim_offset_;
};

//walker_traverser implement algorithms to iterate walker using given shape
//traverse shape may be not native walker shape but shapes must be broadcastable
template<typename Config, typename Walker>
class walker_forward_traverser
{
public:
    using config_type = Config;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
protected:
    using walker_type = Walker;

    const shape_type* shape_;
    dim_type dim_;
    walker_type walker_;
    shape_type index_;
public:
    template<typename Walker_>
    walker_forward_traverser(const shape_type& shape__, Walker_&& walker__):
        shape_{&shape__},
        dim_(detail::make_dim(shape__)),
        walker_{std::forward<Walker_>(walker__)},
        index_(dim_, index_type{0})
    {}
    dim_type axis_min()const{return 0;}
    dim_type axis_max()const{return dim_;}
    const auto& index()const{return index_;}
    const auto& walker()const{return walker_;}
    auto& walker(){return walker_;}
    decltype(auto) operator*()const{return *walker_;}
    //traverse over range [axis_min,axis_max)
    template<typename Order>
    bool next(const dim_type& amin, const dim_type& amax){
        ASSERT_ORDER(Order);
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            return next_c(amin, amax);
        }else{
            return next_f(amin, amax);
        }
    }
    template<typename Order>
    bool next(){
        return next<Order>(axis_min(), axis_max());
    }
private:
    bool next_c(const dim_type& axis_min, dim_type axis_max){
        auto index_it = index_.begin()+axis_max;
        while(axis_max!=axis_min){
            --index_it;
            --axis_max;
            if (next_on_axis(axis_max, *index_it)){
                return true;
            }
        }
        return false;
    }
    bool next_f(dim_type axis_min, const dim_type& axis_max){
        auto index_it = index_.begin()+axis_min;
        for (;axis_min!=axis_max; ++axis_min,++index_it){
            if (next_on_axis(axis_min, *index_it)){
                return true;
            }
        }
        return false;
    }
    bool next_on_axis(dim_type axis, index_type& index){
        if (index == (*shape_)[axis]-index_type{1}){   //axis at their max
            index = index_type{0};
            walker_.reset_back(axis);
        }else{  //can next on axis
            ++index;
            walker_.step(axis);
            return true;
        }
        return false;
    }
};

//traverse walker in specified axes range
template<typename Config, typename Walker>
class walker_forward_range_traverser : public walker_forward_traverser<Config, Walker>
{
    using base_type = walker_forward_traverser<Config,Walker>;
    public:
    using typename base_type::config_type;
    using typename base_type::shape_type;
    using typename base_type::index_type;
    using typename base_type::dim_type;
protected:
    dim_type axis_min_;
    dim_type axis_max_;
    using walker_type = Walker;

public:
    template<typename Walker_>
    walker_forward_range_traverser(const shape_type& shape__, Walker_&& walker__, const dim_type& axis_min__, const dim_type& axis_max__):
        base_type{shape__,std::forward<Walker_>(walker__)},
        axis_min_{axis_min__},
        axis_max_{axis_max__}
    {}
    dim_type axis_min()const{return axis_min_;}
    dim_type axis_max()const{return axis_max_;}
    template<typename Order>
    bool next(){
        return base_type::template next<Order>(axis_min(), axis_max());
    }
};

//Base can be walker_forward_traverser or walker_forward_range_traverser
template<typename Base>
class walker_bidirectional_traverser : public Base
{
    using base_type = Base;
protected:
    using base_type::walker_;
    using base_type::dim_;
    using base_type::index_;
    using base_type::shape_;
public:
    using typename base_type::config_type;
    using typename base_type::dim_type;
    using typename base_type::index_type;
    using typename base_type::shape_type;
    using base_type::base_type;
    using base_type::axis_min;
    using base_type::axis_max;

    template<typename Order>
    bool prev(){
        ASSERT_ORDER(Order);
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            return prev_c();
        }else{
            return prev_f();
        }
    }
    void to_last(){
        for (auto axis=axis_min(); axis!=axis_max(); ++axis){
            auto dec_axis_size = (*shape_)[axis]-index_type{1};
            walker_.walk(axis, dec_axis_size-index_[axis]);
            index_[axis] = dec_axis_size;
        }
    }
    void to_first(){
        for (auto axis=axis_min(); axis!=axis_max(); ++axis){
            walker_.walk_back(axis, index_[axis]);
            index_[axis] = 0;
        }
    }
private:
    bool prev_c(){
        auto amax = axis_max();
        auto index_it = index_.begin()+amax;
        for (const auto amin=axis_min(); amax!=amin;){
            --index_it;
            --amax;
            if (prev_on_axis(amax, *index_it)){
                return true;
            }
        }
        return false;
    }
    bool prev_f(){
        auto amin = axis_min();
        auto index_it = index_.begin()+amin;
        for (const auto amax=axis_max(); amin!=amax; ++amin,++index_it){
            if (prev_on_axis(amin, *index_it)){
                return true;
            }
        }
        return false;
    }
    bool prev_on_axis(const dim_type& axis, index_type& index){
        if (index == index_type{0}){   //axis at their min
            index = (*shape_)[axis]-index_type{1};
            walker_.reset(axis);
        }else{  //can prev on axis
            --index;
            walker_.step_back(axis);
            return true;
        }
        return false;
    }
};

//Base is specialization of walker_bidirectional_traverser
template<typename Base, typename Order>
class walker_random_access_traverser : public Base
{
    using base_type = Base;
    using base_type::walker_;
    using base_type::index_;
    using base_type::dim_;
    using base_type::axis_min;
    using base_type::axis_max;
    using base_type::to_first;
    using strides_div_type = detail::strides_div_t<typename base_type::config_type>;
public:
    using typename base_type::config_type;
    using typename base_type::dim_type;
    using typename base_type::index_type;
    using typename base_type::shape_type;
    //depending on Base, args is empty or axis_min,axis_max
    template<typename Walker_, typename...Args>
    walker_random_access_traverser(const shape_type& shape__, const strides_div_type& strides__, Walker_&& walker__, const Args&...args):
        base_type(shape__, std::forward<Walker_>(walker__), args...),
        strides_{&strides__}
    {}
    bool next(){return base_type::template next<Order>();}
    bool prev(){return base_type::template prev<Order>();}
    //n must be in range [0,size-1], where size = make_size(shape__)
    void move(index_type n){
        to_first();
        move_(n);
    }
private:
    void move_(index_type n){
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            move_c(n);
        }else{
            move_f(n);
        }
    }
    void move_c(index_type n){
        auto index_it = index_.begin()+axis_min();
        auto strides_it = strides_->begin()+axis_min();
        for (dim_type axis=axis_min(), last=axis_max(); axis!=last; ++axis,++index_it){
            auto steps = detail::divide(n,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(axis,steps);
            }
            *index_it = steps;
            ++strides_it;
        }
    }
    void move_f(index_type n){
        auto index_it = index_.begin()+axis_max();
        auto strides_it = strides_->begin()+axis_max();
        for (dim_type axis=axis_max(), first=axis_min(); axis!=first;){
            --axis;
            --index_it;
            --strides_it;
            auto steps = detail::divide(n,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(axis,steps);
            }
            *index_it = steps;
        }
    }

    const strides_div_type* strides_;
};

}   //end of namespace gtensor
#endif