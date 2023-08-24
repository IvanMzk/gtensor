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

//traverser helpers
#define NEXT_ON_AXIS(axis)\
auto& i = *(index_first+axis);\
if (i == *(shape_first+axis)-1){\
    i = 0;\
    walker.reset_back(axis);\
}else{\
    ++i;\
    walker.step(axis);\
    return true;\
}

#define PREV_ON_AXIS(axis)\
auto& i = *(index_first+axis);\
if (i == 0){\
    i = *(shape_first+axis)-1;\
    walker.reset(axis);\
}else{\
    --i;\
    walker.step_back(axis);\
    return true;\
}

//traverse all axes
template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
bool next_c(Walker& walker, const IdxIt index_first, const ShIt shape_first, DimT dim){
    while(dim!=0){
        --dim;
        NEXT_ON_AXIS(dim);
    }
    return false;
}

template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
bool next_f(Walker& walker, const IdxIt index_first, const ShIt shape_first, const DimT& dim){
    using dim_type = typename std::iterator_traits<ShIt>::difference_type;
    for (dim_type a=0; a!=dim; ++a){
        NEXT_ON_AXIS(a);
    }
    return false;
}

template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
bool prev_c(Walker& walker, const IdxIt index_first, const ShIt shape_first, DimT dim){
    while(dim!=0){
        --dim;
        PREV_ON_AXIS(dim);
    }
    return false;
}

template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
bool prev_f(Walker& walker, const IdxIt index_first, const ShIt shape_first, const DimT& dim){
    using dim_type = typename std::iterator_traits<ShIt>::difference_type;
    for (dim_type a=0; a!=dim; ++a){
        PREV_ON_AXIS(a);
    }
    return false;
}

//traverse axes range
template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
bool next_c(Walker& walker, const IdxIt index_first, const ShIt shape_first, const DimT& axis_min, DimT axis_max){
    while(axis_max!=axis_min){
        --axis_max;
        NEXT_ON_AXIS(axis_max);
    }
    return false;
}

template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
bool next_f(Walker& walker, const IdxIt index_first, const ShIt shape_first, DimT axis_min, const DimT& axis_max){
    for (;axis_min!=axis_max; ++axis_min){
        NEXT_ON_AXIS(axis_min);
    }
    return false;
}

template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
bool prev_c(Walker& walker, const IdxIt index_first, const ShIt shape_first, const DimT& axis_min, DimT axis_max){
    while(axis_max!=axis_min){
        --axis_max;
        PREV_ON_AXIS(axis_max);
    }
    return false;
}

template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
bool prev_f(Walker& walker, const IdxIt index_first, const ShIt shape_first, DimT axis_min, const DimT& axis_max){
    for (;axis_min!=axis_max; ++axis_min){
        PREV_ON_AXIS(axis_min);
    }
    return false;
}

#undef NEXT_ON_AXIS
#undef PREV_ON_AXIS

#define TO_LAST_ON_AXIS(axis)\
auto& i = *(index_first+axis);\
auto dec_axis_size = *(shape_first+axis)-1;\
walker.walk(axis, dec_axis_size-i);\
i = dec_axis_size;

//move on all axes
template<typename Walker, typename IdxIt>
void to_first(Walker& walker, IdxIt index_first, const IdxIt index_last){
    walker.reset_back();
    for (;index_first!=index_last; ++index_first){
        *index_first=0;
    }
}
template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
void to_last(Walker& walker, const IdxIt index_first, const ShIt shape_first, DimT dim){
    while(dim!=0){
        --dim;
        TO_LAST_ON_AXIS(dim);
    }
}
//move on axes range
template<typename Walker, typename IdxIt, typename DimT>
void to_first(Walker& walker, const IdxIt index_first, DimT axis_min, const DimT& axis_max){
    for (;axis_min!=axis_max; ++axis_min){
        auto& i = *(index_first+axis_min);
        walker.walk_back(axis_min, i);
        i = 0;
    }
}
template<typename Walker, typename IdxIt, typename ShIt, typename DimT>
void to_last(Walker& walker, const IdxIt index_first, const ShIt shape_first, DimT axis_min, const DimT& axis_max){
    for (;axis_min!=axis_max; ++axis_min){
        TO_LAST_ON_AXIS(axis_min);
    }
}

#undef TO_LAST_ON_AXIS

#define ADVANCE_ON_AXIS(axis)\
auto steps = detail::divide(n,*(strides_first+axis));\
if (steps!=0){\
    walker.walk(axis,steps);\
}\
*(index_first+axis) = steps;\

//move on axes range
template<typename Walker, typename IdxIt,  typename StIt, typename DimT, typename IdxT>
void advance_c(Walker& walker, IdxIt index_first, StIt strides_first, DimT axis_min, const DimT& axis_max, IdxT n){
    for (;axis_min!=axis_max; ++axis_min){
        ADVANCE_ON_AXIS(axis_min);
    }
}
template<typename Walker, typename IdxIt,  typename StIt, typename DimT, typename IdxT>
void advance_f(Walker& walker, IdxIt index_first, StIt strides_first, const DimT& axis_min, DimT axis_max, IdxT n){
    while(axis_max!=axis_min){
        --axis_max;
        ADVANCE_ON_AXIS(axis_max);
    }
}

#undef ADVANCE_ON_AXIS

template<typename T, typename=void> inline constexpr bool is_range_traverser_v = false;
template<typename T> inline constexpr bool is_range_traverser_v<T,std::void_t<decltype(std::declval<T>().axis_min()),decltype(std::declval<T>().axis_max())>> = true;

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

//walker is adapter that allows address data elements using multidimensional index
//Cursor is responsible for storing flat position, it may have semantic of integral type or random access iterator
template<typename Config, typename Cursor>
class cursor_walker
{
    using shape_iterator_type = typename Config::shape_type::const_iterator;
    struct iter_{using type = decltype(*std::declval<Cursor>());};
    struct cur_{using type = const Cursor&;};
    using result_type = typename std::conditional_t<detail::is_iterator_v<Cursor>,iter_,cur_>::type;
public:
    using config_type = Config;
    using cursor_type = Cursor;
    using index_type = typename Config::index_type;
    using dim_type = typename Config::dim_type;
    using shape_type = typename Config::shape_type;

    cursor_walker(const shape_type& adapted_strides__, const shape_type& reset_strides__, const cursor_type& offset__):
        adapted_strides_it_{adapted_strides__.begin()},
        reset_strides_it_{reset_strides__.begin()},
        offset_{offset__},
        cursor_{offset__},
        dim_{detail::make_dim(adapted_strides__)}
    {}
    //axis must be in range [0,dim-1]
    void walk(const dim_type& axis, const index_type& steps){
        cursor_+=steps**(adapted_strides_it_+axis);
    }
    void walk_back(const dim_type& axis, const index_type& steps){
        cursor_-=steps**(adapted_strides_it_+axis);
    }
    void step(const dim_type& axis){
        cursor_+=*(adapted_strides_it_+axis);
    }
    void step_back(const dim_type& axis){
        cursor_-=*(adapted_strides_it_+axis);
    }
    void reset(const dim_type& axis){
        cursor_+=*(reset_strides_it_+axis);
    }
    void reset_back(const dim_type& axis){
        cursor_-=*(reset_strides_it_+axis);
    }
    void reset_back(){
        cursor_ = offset_;
    }
    void update_offset(){
        offset_+=(cursor_-offset_);
    }
    dim_type dim()const{
        return dim_;
    }
    result_type operator*()const{
        if constexpr (detail::is_iterator_v<Cursor>){
            return *cursor_;
        }else{
            return cursor_;
        }
    }
private:
    shape_iterator_type adapted_strides_it_;
    shape_iterator_type reset_strides_it_;
    cursor_type offset_;
    cursor_type cursor_;
    dim_type dim_;
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
    void step_back(const dim_type& axis){
        impl.step_back(axis);
    }
    void reset(const dim_type& axis){
        impl.reset(axis);
    }
    void reset_back(const dim_type& axis){
        impl.reset_back(axis);
    }
    void reset_back(){
        impl.reset_back();
    }
    void update_offset(){
        impl.update_offset();
    }
    dim_type dim(){
        return impl.dim();
    }
    decltype(auto) operator*()const{
        return indexer[*impl];
    }
private:
    cursor_walker<config_type, index_type> impl;
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
        if (axis>=dim_offset_){
            base_walker_type::walk(axis-dim_offset_,steps);
        }
    }
    void walk_back(const dim_type& axis, const index_type& steps){
        if (axis>=dim_offset_){
            base_walker_type::walk_back(axis-dim_offset_,steps);
        }
    }
    void step(const dim_type& axis){
        if (axis>=dim_offset_){
            base_walker_type::step(axis-dim_offset_);
        }
    }
    void step_back(const dim_type& axis){
        if (axis>=dim_offset_){
            base_walker_type::step_back(axis-dim_offset_);
        }
    }
    void reset(const dim_type& axis){
        if (axis>=dim_offset_){
            base_walker_type::reset(axis-dim_offset_);
        }
    }
    void reset_back(const dim_type& axis){
        if (axis>=dim_offset_){
            base_walker_type::reset_back(axis-dim_offset_);
        }
    }
    void reset_back(){
        base_walker_type::reset_back();
    }
    using base_walker_type::operator*;
    using base_walker_type::update_offset;
    using base_walker_type::dim;
private:
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
    const auto& index()const{return index_;}
    const auto& walker()const{return walker_;}
    auto& walker(){return walker_;}
    decltype(auto) operator*()const{return *walker_;}
    template<typename Order>
    bool next(){
        ASSERT_ORDER(Order);
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            return detail::next_c(walker_,index_.begin(),shape_->begin(),dim_);
        }else{
            return detail::next_f(walker_,index_.begin(),shape_->begin(),dim_);
        }
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
    using base_type::shape_;
    using base_type::dim_;
    using base_type::walker_;
    using base_type::index_;
    dim_type axis_min_;
    dim_type axis_max_;

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
        ASSERT_ORDER(Order);
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            return detail::next_c(walker_,index_.begin(),shape_->begin(),axis_min_,axis_max_);
        }else{
            return detail::next_f(walker_,index_.begin(),shape_->begin(),axis_min_,axis_max_);
        }
    }
};

//Base can be walker_forward_traverser or walker_forward_range_traverser
template<typename Base>
class walker_bidirectional_traverser : public Base
{
    using base_type = Base;
protected:
    using base_type::shape_;
    using base_type::dim_;
    using base_type::walker_;
    using base_type::index_;
    static constexpr bool is_range_traverser = detail::is_range_traverser_v<base_type>;
public:
    using typename base_type::config_type;
    using typename base_type::dim_type;
    using typename base_type::index_type;
    using typename base_type::shape_type;
    using base_type::base_type;

    template<typename Order>
    bool prev(){
        ASSERT_ORDER(Order);
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            if constexpr (is_range_traverser){
                return detail::prev_c(walker_,index_.begin(),shape_->begin(),base_type::axis_min(),base_type::axis_max());
            }else{
                return detail::prev_c(walker_,index_.begin(),shape_->begin(),dim_);
            }
        }else{
            if constexpr (is_range_traverser){
                return detail::prev_f(walker_,index_.begin(),shape_->begin(),base_type::axis_min(),base_type::axis_max());
            }else{
                return detail::prev_f(walker_,index_.begin(),shape_->begin(),dim_);
            }
        }
    }

    void to_last(){
        if constexpr (is_range_traverser){
            detail::to_last(walker_,index_.begin(),shape_->begin(),base_type::axis_min(),base_type::axis_max());
        }else{
            detail::to_last(walker_,index_.begin(),shape_->begin(),dim_);
        }
    }
    void to_first(){
        if constexpr (is_range_traverser){
            detail::to_first(walker_,index_.begin(),base_type::axis_min(),base_type::axis_max());
        }else{
            detail::to_first(walker_,index_.begin(),index_.end());
        }
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
    using base_type::to_first;
    using base_type::is_range_traverser;
    using strides_div_type = detail::strides_div_t<typename base_type::config_type>;

    const strides_div_type* strides_;
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
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            if constexpr (is_range_traverser){
                return detail::advance_c(walker_,index_.begin(),strides_->begin(),base_type::axis_min(),base_type::axis_max(),n);
            }else{
                return detail::advance_c(walker_,index_.begin(),strides_->begin(),dim_type{0},dim_,n);
            }
        }else{
            if constexpr (is_range_traverser){
                return detail::advance_f(walker_,index_.begin(),strides_->begin(),base_type::axis_min(),base_type::axis_max(),n);
            }else{
                return detail::advance_f(walker_,index_.begin(),strides_->begin(),dim_type{0},dim_,n);
            }
        }
    }
};

}   //end of namespace gtensor
#endif