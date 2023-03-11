#ifndef COMBINE_HPP_
#define COMBINE_HPP_

#include <type_traits>
#include <stdexcept>
//#include

namespace gtensor{

class combine_exception : public std::runtime_error{
public:
    explicit combine_exception(const char* what):
        runtime_error(what)
    {}
};

namespace detail{

template<typename SizeT, typename ShT, typename...ShTs>
void check_stack_args(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using size_type = SizeT;
    size_type dim = shape.size();
    if (dim > size_type{0} && direction > dim){
        throw combine_exception{"bad stack direction"};
    }
    if constexpr (sizeof...(ShTs) > 0){
        if (!((shape==shapes)&&...)){
            throw combine_exception{"tensors to stack must have equal shapes"};
        }
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
void check_concatenate_args(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using size_type = SizeT;
    size_type dim = shape.size();
    if (dim > size_type{0} && direction >= dim){
        throw combine_exception{"bad concatenate direction"};
    }
    if constexpr (sizeof...(ShTs) > 0){
        if (!((dim==static_cast<size_type>(shapes.size()))&&...)){
            throw combine_exception{"tensors to concatenate must have equal shapes"};
        }
        for (size_type d{0}; d!=dim; ++d){
            if (!((shape[d]==shapes[d])&&...)){
                if (d!=direction){
                    throw combine_exception{"tensors to concatenate must have equal shapes"};
                }
            }
        }
    }
}

template<typename SizeT, typename ShT>
auto make_stack_shape(const SizeT& direction, const ShT& shape, const typename ShT::value_type& tensors_number){
    using size_type = SizeT;
    using shape_type = ShT;
    size_type pdim = shape.size();
    if (pdim == size_type{0}){
        return shape_type{};
    }else{
        shape_type res(pdim+size_type{1});
        std::copy(shape.begin(), shape.begin()+direction, res.begin());
        std::copy(shape.begin()+direction, shape.end(), res.begin()+direction+size_type{1});
        res[direction] = tensors_number;
        return res;
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
auto make_concatenate_shape(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    using size_type = SizeT;
    size_type dim = shape.size();
    if (dim == size_type{0}){
        return shape_type{};
    }else{
        index_type direction_size{shape[direction]};
        ((direction_size+=shapes[direction]),...);
        shape_type res{shape};
        res[direction]=direction_size;
        return res;
    }
}

template<typename SizeT, typename ShT>
auto make_stack_block_size(const SizeT& direction, const ShT& shape){
    using size_type = SizeT;
    using index_type = typename ShT::value_type;
    size_type pdim = shape.size();
    if (pdim == size_type{0}){
        return index_type{0};
    }else{
        return std::accumulate(shape.begin()+direction, shape.end(), index_type{1}, std::multiplies{});
    }
}

template<typename SizeT, typename ShT, typename ResultIt, typename...It>
auto fill_stack(const SizeT& direction, const ShT& shape, const typename ShT::value_type& size, ResultIt res_it, It...it){
    using index_type = typename ShT::value_type;
    index_type block_size = make_stack_block_size(direction, shape);
    index_type i{0};
    auto filler = [i, block_size, res_it](auto& it) mutable {
        auto i_ = i;
        i += block_size;
        for (;i_!=i; ++i_, ++res_it, ++it){
            *res_it = *it;
        }
    };
    index_type iterations_number = size/block_size;
    for (;i!=iterations_number; ++i){
        (filler(it),...);
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
auto make_concatenate_block_size(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using size_type = SizeT;
    using index_type = typename ShT::value_type;
    index_type block_size_ = std::accumulate(shape.begin()+direction+size_type{1}, shape.end(), index_type{1}, std::multiplies{});
    return std::make_tuple(shape[direction]*block_size_, shapes[direction]*block_size_...);
}

template<typename SizeT, typename IdxT, typename ResultIt, typename ImplT, typename...ImplTs>
class concatenate_filler
{
    using index_type = IdxT;
    using block_size_type = decltype(make_concatenate_block_size(std::declval<SizeT>(), std::declval<ImplT>().shape(), std::declval<ImplTs>().shape()...));
    block_size_type block_sizes_;
    ResultIt res_it_;
public:
    concatenate_filler(const SizeT& direction__, ResultIt res_it__, const ImplT& t__, const ImplTs&...ts__):
        block_sizes_{make_concatenate_block_size(direction__, t__.shape(), ts__.shape()...)},
        res_it_{res_it__}
    {}
    template<std::size_t I, typename It>
    void fill(It& it){
        index_type block_size = std::get<I>(block_sizes_);
        for (index_type i{0}; i!=block_size; ++i, ++it, ++res_it_){
            *res_it_ = *it;
        }
    }
};

template<typename SizeT, typename ResultIt, std::size_t...I, typename ImplT, typename...ImplTs>
auto fill_concatenate(const SizeT& direction,  ResultIt res_it, std::index_sequence<I...>, const ImplT& t, const ImplTs&...ts){
    using shape_type = std::decay_t<decltype(t.shape())>;
    using index_type = typename shape_type::value_type;
    using filler_type = concatenate_filler<SizeT, index_type, ResultIt, ImplT, ImplTs...>;
    auto iters = std::make_tuple(t.engine().begin(), ts.engine().begin()...);
    filler_type filler{direction, res_it, t, ts...};
    index_type iterations_number = std::accumulate(t.shape().begin(), t.shape().begin()+direction, index_type{1}, std::multiplies{});
    for (index_type i{0}; i!=iterations_number; ++i){
        ((filler.template fill<I>(std::get<I>(iters))),...);
    }
}

}   //end of namespace detail

class combiner{
//join tensors along new direction, tensors must have the same shape
template<typename SizeT, typename ImplT, typename...ImplTs>
static auto stack_(const SizeT& direction, const ImplT& t, const ImplTs&...ts){
    using config_type = typename ImplT::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename ImplT::value_type, typename ImplTs::value_type...>;
    using res_impl_type = storage_tensor<typename detail::storage_engine_traits<typename config_type::host_engine,config_type,typename config_type::template storage<res_value_type>>::type>;
    const auto& shape = t.shape();
    detail::check_stack_args(direction, shape, ts.shape()...);
    index_type tensors_number = sizeof...(ImplTs) + 1;
    auto res_shape = detail::make_stack_shape(direction,shape,tensors_number);
    if constexpr (sizeof...(ImplTs) == 0){
        return tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape), t.engine().begin(), t.engine().end());
    }else{
        auto res = tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape), res_value_type{});
        if (res.size() > index_type{0}){
            detail::fill_stack(direction, shape, t.size(), res.begin(), t.engine().begin(), ts.engine().begin()...);
        }
        return res;
    }
}
template<typename SizeT, typename...Us, typename...Ts>
static auto stack_(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return stack_(direction, t.impl_ref(), ts.impl_ref()...);
}

//join tensors along existing direction, tensors must have the same shape except concatenate direction
template<typename SizeT, typename ImplT, typename...ImplTs>
static auto concatenate_(const SizeT& direction, const ImplT& t, const ImplTs&...ts){
    using config_type = typename ImplT::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename ImplT::value_type, typename ImplTs::value_type...>;
    using res_impl_type = storage_tensor<typename detail::storage_engine_traits<typename config_type::host_engine,config_type,typename config_type::template storage<res_value_type>>::type>;
    detail::check_concatenate_args(direction, t.shape(), ts.shape()...);
    auto res_shape = detail::make_concatenate_shape(direction, t.shape(), ts.shape()...);
    if constexpr (sizeof...(ImplTs) == 0){
        return tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape),t.engine().begin(),t.engine().end());
    }else{
        auto res = tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape), res_value_type{});
        if (res.size() > index_type{0}){
            detail::fill_concatenate(direction, res.begin(), std::make_index_sequence<sizeof...(ImplTs) + 1>{}, t, ts...);
        }
        return res;
    }
}
template<typename SizeT, typename...Us, typename...Ts>
static auto concatenate_(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return concatenate_(direction, t.impl_ref(), ts.impl_ref()...);
}


template<typename SizeT, typename...Us, typename...Ts>
friend auto stack(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts);
template<typename SizeT, typename...Us, typename...Ts>
friend auto concatenate(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts);

};  //end of class combiner

template<typename SizeT, typename...Us, typename...Ts>
auto stack(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return combiner::stack_(direction, t, ts...);
}
template<typename SizeT, typename...Us, typename...Ts>
auto concatenate(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return combiner::concatenate_(direction, t, ts...);
}

}   //end of namespace gtensor

#endif