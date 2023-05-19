#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "type_selector.hpp"
#include "iterator.hpp"
#include "tensor.hpp"

namespace gtensor{

class reduce_exception : public std::runtime_error{
public:
    explicit reduce_exception(const char* what):
        runtime_error(what)
    {}
};

namespace detail{

template<typename ShT, typename DimT>
auto check_reduce_args(const ShT& shape, const DimT& direction){
    using dim_type = DimT;
    const dim_type dim = shape.size();
    if (direction >= dim){
        throw reduce_exception("bad reduce direction");
    }
}
template<typename ShT, typename DimT>
auto make_reduce_shape(const ShT& shape, const DimT& direction){
    using dim_type = DimT;
    using shape_type = ShT;
    dim_type dim = shape.size();
    shape_type res(--dim);
    auto shape_stop = shape.begin()+direction;
    std::copy(shape.begin(), shape_stop, res.begin());
    std::copy(++shape_stop, shape.end(), res.begin()+direction);
    return res;
}

template<typename ShT, typename DimT, typename IdxT>
auto check_slide_args(const ShT& shape, const DimT& direction, const IdxT& window_size){
    using dim_type = DimT;
    using index_type = IdxT;
    const dim_type dim = shape.size();
    if (direction >= dim){
        throw reduce_exception("bad slide direction");
    }
    index_type direction_size = shape[direction];
    if (direction_size > 0){
        if (window_size > direction_size || window_size <= index_type{0}){
            throw reduce_exception("bad sliding window size");
        }
    }
}
template<typename ShT, typename DimT, typename IdxT>
auto make_slide_shape(const ShT& shape, const DimT& direction, const IdxT& window_size, const IdxT& window_step){
    using index_type = IdxT;
    using shape_type = ShT;
    shape_type res(shape);
    index_type direction_size = shape[direction];
    if (direction_size != index_type{0}){
        index_type result_direction_size = (direction_size - window_size)/window_step + index_type{1};
        res[direction] = result_direction_size;
    }
    return res;
}

//random access iterator, use walker data accessor
template<typename Config, typename Walker>
class reduce_iterator
{
protected:
    using walker_type = Walker;
    using index_type = typename Config::index_type;
    using dim_type = typename Config::dim_type;
    using result_type = decltype(*std::declval<walker_type>());
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = index_type;
    using value_type = typename detail::iterator_internals_selector<result_type>::value_type;
    using pointer = typename detail::iterator_internals_selector<result_type>::pointer;
    using reference = typename detail::iterator_internals_selector<result_type>::reference;
    using const_reference = typename detail::iterator_internals_selector<result_type>::const_reference;

    //assuming usual stoarge subscript operator semantic i.e. subscript index in range [0,size()-1]:
    //begin should be constructed with zero flat_index_ argument, end with size() flat_index_argument
    template<typename Walker_>
    reduce_iterator(Walker_&& walker_, const dim_type& reduce_direction_, const difference_type& flat_index_):
        walker{std::forward<Walker_>(walker_)},
        reduce_direction{reduce_direction_},
        flat_index{flat_index_}
    {}
    reduce_iterator& operator+=(difference_type n){
        advance(n);
        return *this;
    }
    reduce_iterator& operator++(){
        walker.step(reduce_direction);
        ++flat_index;
        return *this;
    }
    reduce_iterator& operator--(){
        walker.step_back(reduce_direction);
        --flat_index;
        return *this;
    }
    result_type operator[](difference_type n)const{return *(*this+n);}
    result_type operator*() const{return *walker;}
    inline difference_type friend operator-(const reduce_iterator& lhs, const reduce_iterator& rhs){return lhs.flat_index - rhs.flat_index;}
private:
    void advance(difference_type n){
        walker.walk(reduce_direction, n);
        flat_index+=n;
    }
    walker_type walker;
    const dim_type reduce_direction;
    difference_type flat_index;
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_PLUS(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_MINUS(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_EQUAL(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_NOT_EQUAL(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_GREATER(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_LESS(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_GREATER_EQUAL(reduce_iterator);
GTENSOR_ITERATOR_OPERATOR_LESS_EQUAL(reduce_iterator);

template<typename Config, typename Walker>
class walker_reduce_traverser
{
    using config_type = Config;
    using walker_type = Walker;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using reduce_iterator_type = reduce_iterator<config_type,walker_type>;

    const shape_type* shape_;
    walker_type walker_;
    const dim_type reduce_direction_;
    const dim_type dim_;
    shape_type index_;

public:
    template<typename Walker_>
    walker_reduce_traverser(const shape_type& shape__, Walker_&& walker__, const dim_type direction__):
        shape_{&shape__},
        walker_{std::forward<Walker_>(walker__)},
        reduce_direction_{direction__},
        dim_{shape_->size()},
        index_(dim_, index_type{0})
    {}

    bool next(){
        auto direction = dim_;
        auto index_it = index_.end();
        while(direction!=dim_type{0}){
            --direction;
            --index_it;
            if (direction == reduce_direction_){
                continue;
            }else{
                if (*index_it == (*shape_)[direction]-index_type{1}){   //direction at their max
                    *index_it = index_type{0};
                    walker_.reset_back(direction);
                }else{  //can next on direction
                    ++(*index_it);
                    walker_.step(direction);
                    return true;
                }
            }
        }
        return false;
    }

    //create reduce iterator
    auto begin(){
        return reduce_iterator_type{walker_,reduce_direction_,0};
    }
    auto end(){
        return reduce_iterator_type{walker_,reduce_direction_,(*shape_)[reduce_direction_]};
    }
};

}   //end of namespace detail

class reducer
{
    template<typename F, typename...Ts>
    static auto reduce_(const basic_tensor<Ts...>& parent, const typename basic_tensor<Ts...>::dim_type& direction, F reduce_f){
        using parent_type = basic_tensor<Ts...>;
        using value_type = typename parent_type::value_type;
        using config_type = typename parent_type::config_type;
        using dim_type = typename config_type::dim_type;
        using index_type = typename config_type::index_type;
        using walker_reduce_traverser_type = detail::walker_reduce_traverser<config_type, decltype(parent.create_walker())>;
        using reduce_iterator_type = decltype(std::declval<walker_reduce_traverser_type>().begin());
        using res_value_type = std::decay_t<decltype(reduce_f(std::declval<reduce_iterator_type>(),std::declval<reduce_iterator_type>()))>;
        const auto& pshape = parent.shape();
        detail::check_reduce_args(pshape, direction);
        auto res = tensor<res_value_type,config_type>{detail::make_reduce_shape(pshape, direction)};
        index_type reduce_direction_size = pshape[direction];
        if (!res.empty()){
            if (reduce_direction_size == index_type{0}){    //fill with default
                if constexpr (std::is_default_constructible_v<value_type>){
                    detail::fill(res.begin(), res.end(), value_type{});
                }else{
                    throw reduce_exception("reduce can't fill result, value_type is not default constructible");
                }
            }else{
                auto pdim = parent.dim();
                if (pdim == dim_type{1}){
                    *res.begin() = reduce_f(parent.begin(), parent.end());
                }else{
                    auto reduce_traverser = walker_reduce_traverser_type{pshape, parent.create_walker(), direction};
                    auto res_it = res.begin();
                    do{
                        *res_it = reduce_f(reduce_traverser.begin(),reduce_traverser.end());
                        ++res_it;
                    }while(reduce_traverser.next());
                }
            }
        }
        return res;
    }

    template<typename F, typename...Ts, typename...Args>
    static auto slide_(
        const basic_tensor<Ts...>& parent,
        const typename basic_tensor<Ts...>::dim_type& direction,
        F slide_f,
        const typename basic_tensor<Ts...>::index_type& window_size,
        const typename basic_tensor<Ts...>::index_type& window_step,
        Args&&...args
    ){
        using parent_type = basic_tensor<Ts...>;
        using value_type = typename parent_type::value_type;
        using config_type = typename parent_type::config_type;
        using dim_type = typename config_type::dim_type;
        const auto& pshape = parent.shape();
        detail::check_slide_args(pshape, direction, window_size);
        auto res = tensor<value_type,config_type>{detail::make_slide_shape(pshape, direction, window_size, window_step)};
        if (!res.empty()){
            auto pdim = parent.dim();
            if (pdim == dim_type{1}){
                slide_f(parent.begin(), parent.end(), res.begin(), res.end(), window_size,window_step,std::forward<Args>(args)...);
            }else{
                auto parent_reduce_traverser = detail::walker_reduce_traverser<config_type, decltype(parent.create_walker())>{pshape, parent.create_walker(), direction};
                auto result_reduce_traverser = detail::walker_reduce_traverser<config_type, decltype(res.create_walker())>{res.shape(), res.create_walker(), direction};
                do{
                    //0first,1last,2dst_first,3dst_last,4window_size,5window_step,6args
                    slide_f(
                        parent_reduce_traverser.begin(),
                        parent_reduce_traverser.end(),
                        result_reduce_traverser.begin(),
                        result_reduce_traverser.end(),
                        window_size,
                        window_step,
                        std::forward<Args>(args)...
                    );
                    result_reduce_traverser.next();
                }while(parent_reduce_traverser.next());
            }
        }
        return res;
    }
public:
    //interface
    template<typename F, typename...Ts>
    static auto reduce(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type& direction, F f){
        return reduce_(t,direction,f);
    }
    template<typename F, typename...Ts, typename...Args>
    static auto slide(
        const basic_tensor<Ts...>& t,
        const typename basic_tensor<Ts...>::dim_type& direction,
        F f,
        const typename basic_tensor<Ts...>::index_type& window_size,
        const typename basic_tensor<Ts...>::index_type& window_step,
        Args&&...args
    ){
        return slide_(t,direction,f,window_size,window_step,std::forward<Args>(args)...);
    }
};

//F is reduce functor that takes iterators range of data to be reduced as arguments and return scalar - reduction result
//F call operator must be defined like this: template<typename It> Ret operator()(It first, It last){...}
template<typename F, typename...Ts>
auto reduce(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type& direction, F f){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce(t, direction, f);
}

//F is slide functor that takes arguments: iterators range of data to be slided, dst iterators range, sliding parameters
//F call operator must be defined like this:
//template<typename It,typename DstIt,typename IdxT,typename...Args> void operator()(It first, It last, DstIt dfirst, DstIt dlast, IdxT window_size, IdxT window_step, Args&&...args){...}
//where Args is application specific arguments
template<typename F, typename...Ts, typename...Args>
auto slide(
    const basic_tensor<Ts...>& t,
    const typename basic_tensor<Ts...>::dim_type& direction,
    F f,
    const typename basic_tensor<Ts...>::index_type& window_size,
    const typename basic_tensor<Ts...>::index_type& window_step,
    Args&&...args
){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::slide(t, direction, f, window_size, window_step,std::forward<Args>(args)...);
}

}   //end of namespace gtensor
#endif