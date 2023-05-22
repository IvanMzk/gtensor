#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "type_selector.hpp"
#include "common.hpp"
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

template<typename ShT>
auto check_reduce_args(const ShT& shape, const typename ShT::size_type& direction){
    using dim_type = typename ShT::size_type;
    const dim_type dim = shape.size();
    if (direction >= dim){
        throw reduce_exception("invalid reduce direction: direction is out of bounds");
    }
}
template<typename ShT, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename ShT::size_type>,int> =0>
auto check_reduce_args(const ShT& shape, const Container& directions){
    using dim_type = typename ShT::size_type;
    const dim_type dim = shape.size();
    const dim_type directions_number = static_cast<dim_type>(directions.size());
    if (directions_number > dim){
        throw reduce_exception("invalid reduce directions: too many directions");
    }
    auto it=directions.begin();
    auto last=directions.end();
    while(it!=last){
        const dim_type& direction = static_cast<dim_type>(*it);
        if (direction >= dim || direction < dim_type{0}){
            throw reduce_exception("invalid reduce directions: direction is out of bounds");
        }
        ++it;
        if (std::find(it, last, direction) != last){
            throw reduce_exception("invalid reduce directions: duplicates in directions");
        }
    }
}

template<typename ShT>
auto make_reduce_shape(const ShT& shape, const typename ShT::size_type& direction, bool keep_dims){
    using shape_type = ShT;
    using dim_type = typename ShT::size_type;
    using index_type = typename ShT::value_type;
    if (keep_dims){
        shape_type res(shape);
        res[direction] = index_type{1};
        return res;
    }else{
        dim_type dim = shape.size();
        shape_type res(--dim);
        auto shape_stop = shape.begin()+direction;
        std::copy(shape.begin(), shape_stop, res.begin());
        std::copy(++shape_stop, shape.end(), res.begin()+direction);
        return res;
    }
}
template<typename ShT, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename ShT::size_type>,int> =0>
auto make_reduce_shape(const ShT& shape, const Container& directions, bool keep_dims){
    using shape_type = ShT;
    using dim_type = typename ShT::size_type;
    using index_type = typename ShT::value_type;
    using directions_value_type = typename Container::value_type;
    const dim_type dim = shape.size();
    const dim_type directions_number = static_cast<dim_type>(directions.size());
    if (keep_dims){
        if (directions_number == dim_type{0}){  //all directions
            return shape_type(dim, index_type{1});
        }else{
            shape_type res(shape);
            for (auto it=directions.begin(), last=directions.end(); it!=last; ++it){
                res[*it] = index_type{1};
            }
            return res;
        }
    }else{
        if (directions_number == dim_type{0}){  //all directions
            return shape_type{};
        }else{
            shape_type res{};
            res.reserve(dim - directions_number);
            auto directions_first = directions.begin();
            auto directions_last = directions.end();
            for(dim_type d{0}; d!=dim; ++d){
                if (std::find(directions_first, directions_last, static_cast<directions_value_type>(d)) == directions_last){
                    res.push_back(shape[d]);
                }
            }
            return res;
        }
    }
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

template<typename ShT>
auto make_slide_direction_size(const ShT& shape, const typename ShT::size_type& direction){
    return shape[direction];
}

template<typename ShT, typename Directions>
auto make_reduce_directions_size(const ShT& shape, const typename ShT::value_type& size, const Directions& directions){
    using dim_type = typename ShT::size_type;
    using index_type = typename ShT::value_type;
    if constexpr (detail::is_container_of_type_v<Directions,dim_type>){
        if (directions.size()==0){
            return size;
        }else{
            index_type directions_size{1};
            for (auto it=directions.begin(), last=directions.end(); it!=last; ++it){
                directions_size*=shape[static_cast<dim_type>(*it)];
            }
            return directions_size;
        }
    }else{
        return make_slide_direction_size(shape,directions);
    }
}

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

template<typename Config, typename Traverser>
class reduce_directions_iterator
{
protected:
    using traverser_type = Traverser;
    using index_type = typename Config::index_type;
    using dim_type = typename Config::dim_type;
    using shape_type = typename Config::shape_type;
    using result_type = decltype(*std::declval<traverser_type>().walker());
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = index_type;
    using value_type = typename detail::iterator_internals_selector<result_type>::value_type;
    using pointer = typename detail::iterator_internals_selector<result_type>::pointer;
    using reference = typename detail::iterator_internals_selector<result_type>::reference;
    using const_reference = typename detail::iterator_internals_selector<result_type>::const_reference;

    //assuming usual stoarge subscript operator semantic i.e. subscript index in range [0,size()-1]:
    //begin should be constructed with zero flat_index_ argument, end with size() flat_index_argument
    template<typename Traverser_>
    reduce_directions_iterator(Traverser_&& traverser_, const difference_type& flat_index_):
        traverser{std::forward<Traverser_>(traverser_)},
        flat_index{flat_index_}
    {}
    reduce_directions_iterator& operator++(){
        traverser.next();
        ++flat_index;
        return *this;
    }
    reduce_directions_iterator& operator--(){
        traverser.prev();
        --flat_index;
        return *this;
    }
    bool operator==(const reduce_directions_iterator& other){
        return flat_index == other.flat_index;
    }
    bool operator!=(const reduce_directions_iterator& other){
        return !(*this == other);
    }
    result_type operator*() const{return *traverser.walker();}
private:
    traverser_type traverser;
    difference_type flat_index;
};

GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(reduce_directions_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(reduce_directions_iterator);

template<typename Config, typename Directions>
class reduce_traverse_predicate
{
    using config_type = Config;
    using directions_type = Directions;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    static_assert(detail::is_container_of_type_v<directions_type,dim_type> || std::is_convertible_v<directions_type,dim_type>);

    const directions_type* directions_;
    const bool inverse_;

    bool is_in_directions(const dim_type& d)const{
        if constexpr (detail::is_container_of_type_v<directions_type,dim_type>){
            if (directions_->size()==0){
                return true;
            }else{
                const auto last = directions_->end();
                return std::find_if(directions_->begin(),last,[&d](const auto& dir){return d == static_cast<dim_type>(dir);}) != last;
            }
        }else{
            return d == static_cast<dim_type>(*directions_);
        }
    }

    bool apply_inverse(bool b)const{
        return inverse_ != b;
    }

public:
    reduce_traverse_predicate(const directions_type& directions__, bool inverse__):
        directions_{&directions__},
        inverse_{inverse__}
    {}

    bool operator()(const dim_type& d)const{
        return apply_inverse(is_in_directions(d));
    }
};

template<typename Traverser>
auto slide_begin(const Traverser& traverser, const typename Traverser::dim_type& direction){
    using config_type = typename Traverser::config_type;
    using index_type = typename config_type::index_type;
    using walker_type = std::decay_t<decltype(traverser.walker())>;

    return reduce_iterator<config_type,walker_type>{traverser.walker(),direction,index_type{0}};
}
template<typename Traverser>
auto slide_end(const Traverser& traverser, const typename Traverser::dim_type& direction, const typename Traverser::index_type& direction_size){
    using config_type = typename Traverser::config_type;
    using walker_type = std::decay_t<decltype(traverser.walker())>;

    auto walker = traverser.walker();
    walker.reset(direction);
    walker.step(direction);
    return reduce_iterator<config_type,walker_type>{std::move(walker), direction, direction_size};
}

template<typename ShT, typename Traverser, typename Directions>
auto reduce_begin(const ShT& shape, const Traverser& traverser, const Directions& directions){
    using config_type = typename Traverser::config_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using walker_type = std::decay_t<decltype(traverser.walker())>;
    using traverse_predicate_type = detail::reduce_traverse_predicate<config_type, Directions>;
    using traverser_type = gtensor::walker_bidirectional_traverser<config_type, walker_type, traverse_predicate_type>;

    if constexpr (detail::is_container_of_type_v<Directions,dim_type>){
        return reduce_directions_iterator<config_type, traverser_type>{
            traverser_type{shape, traverser.walker(), traverse_predicate_type{directions, false}},
            index_type{0}
        };
    }else{
        return slide_begin(traverser, directions);
    }
}
template<typename ShT, typename Traverser, typename Directions>
auto reduce_end(const ShT& shape, const Traverser& traverser, const Directions& directions, const typename Traverser::index_type& directions_size){
    using config_type = typename Traverser::config_type;
    using dim_type = typename config_type::dim_type;
    using walker_type = std::decay_t<decltype(traverser.walker())>;
    using traverse_predicate_type = detail::reduce_traverse_predicate<config_type, Directions>;
    using traverser_type = gtensor::walker_bidirectional_traverser<config_type, walker_type, traverse_predicate_type>;

    if constexpr (detail::is_container_of_type_v<Directions,dim_type>){
        traverser_type directions_traverser{shape, traverser.walker(), traverse_predicate_type{directions, false}};
        directions_traverser.to_last();
        directions_traverser.next();
        return reduce_directions_iterator<config_type, traverser_type>{std::move(directions_traverser),directions_size};
    }else{
        return slide_end(traverser,directions,directions_size);
    }
}

}   //end of namespace detail

class reducer
{
    template<typename F, typename Directions, typename...Ts, typename...Args>
    static auto reduce_(const basic_tensor<Ts...>& parent, const Directions& directions, F reduce_f, bool keep_dims, Args&&...args){
        using parent_type = basic_tensor<Ts...>;
        using value_type = typename parent_type::value_type;
        using config_type = typename parent_type::config_type;
        using dim_type = typename config_type::dim_type;
        using index_type = typename config_type::index_type;

        const auto& pshape = parent.shape();
        detail::check_reduce_args(pshape, directions);
        auto res = tensor<value_type,config_type>{detail::make_reduce_shape(pshape, directions, keep_dims)};
        bool reduce_zero_size_direction{false};
        if (parent.size() == index_type{0}){    //check if reduce zero size direction
            if constexpr (detail::is_container_of_type_v<Directions,dim_type>){
                if (directions.size()==0){
                    reduce_zero_size_direction = true;
                }else{
                    for(const auto& d : directions){
                        if (pshape[d] == index_type{0}){
                            reduce_zero_size_direction = true;
                            break;
                        }
                    }
                }
            }else if constexpr (std::is_convertible_v<Directions,dim_type>){
                if (pshape[directions] == index_type{0}){
                    reduce_zero_size_direction = true;
                }
            }else{
                static_assert(detail::always_false<Directions>, "invalid directions argument");
            }
        }
        if (!res.empty()){
            if (reduce_zero_size_direction){    //fill with default
                if constexpr (std::is_default_constructible_v<value_type>){
                    detail::fill(res.begin(), res.end(), value_type{});
                }else{
                    throw reduce_exception("reduce can't fill result, value_type is not default constructible");
                }
            }else{
                auto pdim = parent.dim();
                if (pdim == dim_type{1}){
                    *res.begin() = reduce_f(parent.begin(), parent.end(), std::forward<Args>(args)...);
                }else{
                    using traverse_predicate_type = detail::reduce_traverse_predicate<config_type, Directions>;
                    traverse_predicate_type traverse_predicate{directions, true};
                    walker_bidirectional_traverser<config_type, decltype(parent.create_walker()), traverse_predicate_type> traverser{pshape, parent.create_walker(), traverse_predicate};
                    const auto directions_size = detail::make_reduce_directions_size(pshape,parent.size(),directions);
                    auto res_it = res.begin();
                    do{
                        *res_it = reduce_f(detail::reduce_begin(pshape,traverser,directions), detail::reduce_end(pshape,traverser,directions,directions_size), std::forward<Args>(args)...);
                        ++res_it;
                    }while(traverser.next());
                }
            }
        }
        return res;
    }

    // template<typename F, typename...Ts>
    // static auto reduce_(const basic_tensor<Ts...>& parent, const typename basic_tensor<Ts...>::dim_type& direction, F reduce_f, bool keep_dims){
    //     using parent_type = basic_tensor<Ts...>;
    //     using value_type = typename parent_type::value_type;
    //     using config_type = typename parent_type::config_type;
    //     using dim_type = typename config_type::dim_type;
    //     using index_type = typename config_type::index_type;
    //     using walker_reduce_traverser_type = detail::walker_reduce_traverser<config_type, decltype(parent.create_walker())>;
    //     using reduce_iterator_type = decltype(std::declval<walker_reduce_traverser_type>().begin());
    //     using res_value_type = std::decay_t<decltype(reduce_f(std::declval<reduce_iterator_type>(),std::declval<reduce_iterator_type>()))>;
    //     const auto& pshape = parent.shape();
    //     detail::check_reduce_args(pshape, direction);
    //     auto res = tensor<res_value_type,config_type>{detail::make_reduce_shape(pshape, direction, keep_dims)};
    //     index_type reduce_direction_size = pshape[direction];
    //     if (!res.empty()){
    //         if (reduce_direction_size == index_type{0}){    //fill with default
    //             if constexpr (std::is_default_constructible_v<value_type>){
    //                 detail::fill(res.begin(), res.end(), value_type{});
    //             }else{
    //                 throw reduce_exception("reduce can't fill result, value_type is not default constructible");
    //             }
    //         }else{
    //             auto pdim = parent.dim();
    //             if (pdim == dim_type{1}){
    //                 *res.begin() = reduce_f(parent.begin(), parent.end());
    //             }else{
    //                 auto reduce_traverser = walker_reduce_traverser_type{pshape, parent.create_walker(), direction};
    //                 auto res_it = res.begin();
    //                 do{
    //                     *res_it = reduce_f(reduce_traverser.begin(),reduce_traverser.end());
    //                     ++res_it;
    //                 }while(reduce_traverser.next());
    //             }
    //         }
    //     }
    //     return res;
    // }

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
                using traverse_predicate_type = detail::reduce_traverse_predicate<config_type, dim_type>;
                traverse_predicate_type traverse_predicate{direction, true};
                walker_bidirectional_traverser<config_type, decltype(parent.create_walker()), traverse_predicate_type> parent_traverser{pshape, parent.create_walker(), traverse_predicate};
                const auto& res_shape = res.shape();
                walker_bidirectional_traverser<config_type, decltype(res.create_walker()), traverse_predicate_type> res_traverser{res_shape, res.create_walker(), traverse_predicate};
                const auto parent_direction_size = detail::make_slide_direction_size(pshape,direction);
                const auto res_direction_size = detail::make_slide_direction_size(res_shape,direction);
                do{
                    //0first,1last,2dst_first,3dst_last,4window_size,5window_step,6args
                    slide_f(
                        detail::slide_begin(parent_traverser,direction),
                        detail::slide_end(parent_traverser,direction,parent_direction_size),
                        detail::slide_begin(res_traverser,direction),
                        detail::slide_end(res_traverser,direction,res_direction_size),
                        window_size,
                        window_step,
                        std::forward<Args>(args)...
                    );
                    res_traverser.next();
                }while(parent_traverser.next());
            }
        }
        return res;
    }
public:
    //interface
    template<typename F, typename Directions, typename...Ts, typename...Args>
    static auto reduce(const basic_tensor<Ts...>& t, const Directions& directions, F f, bool keep_dims, Args&&...args){
        return reduce_(t,directions,f,keep_dims,std::forward<Args>(args)...);
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
template<typename F, typename Directions, typename...Ts, typename...Args>
auto reduce(const basic_tensor<Ts...>& t, const Directions& directions, F f, bool keep_dim, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce(t, directions, f, keep_dim, std::forward<Args>(args)...);
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