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
    check_concatenate_args(direction, shape, shapes...);
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
    if (block_size > index_type{0}){
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
}

}   //end of namespace detail

//join tensors along new direction, tensors must have the same shape
template<typename SizeT, typename T, typename...Ts>
auto stack(const SizeT& direction, const T& t, const Ts&...ts){
    using config_type = typename T::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename T::value_type, typename Ts::value_type...>;
    using res_impl_type = storage_tensor<typename detail::storage_engine_traits<typename config_type::host_engine,config_type,typename config_type::template storage<res_value_type>>::type>;
    const auto& shape = t.shape();
    detail::check_stack_args(direction, shape, ts.shape()...);
    index_type tensors_number = sizeof...(Ts) + 1;
    auto res_shape = detail::make_stack_shape(direction,shape,tensors_number);
    if constexpr (sizeof...(Ts) == 0){
        return tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape),t.begin(),t.end());
    }else{
        auto res = tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape),res_value_type{});
        detail::fill_stack(direction, shape, t.size(), res.begin(), t.begin(), ts.begin()...);
        return res;
    }
}

//join tensors along existing direction, tensors must have the same shape except concatenate direction
template<typename SizeT, typename T, typename...Ts>
auto concatenate(const SizeT& direction, const T& t, const Ts&...ts){
    using config_type = typename T::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename T::value_type, typename Ts::value_type...>;
    using res_impl_type = storage_tensor<typename detail::storage_engine_traits<typename config_type::host_engine,config_type,typename config_type::template storage<res_value_type>>::type>;
    const auto& shape = t.shape();
    detail::check_concatenate_args(direction, shape, ts.shape()...);
    if constexpr (sizeof...(Ts) == 0){
        return tensor<res_value_type, config_type, res_impl_type>::make_tensor(shape,t.begin(),t.end());
    }else{
        // index_type tensors_number = sizeof...(Ts) + 1;
        // auto res = tensor<res_value_type, config_type, res_impl_type>::make_tensor(detail::make_stack_shape(shape,direction,tensors_number),res_value_type{});
        // detail::fill_stack(direction, shape, t.size(), res.begin(), t.begin(), ts.begin()...);
        // return res;
    }
}

}   //end of namespace gtensor

#endif