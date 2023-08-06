#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include <iostream>
#include <numeric>
#include <sstream>
#include "common.hpp"
#include "libdivide_helper.hpp"
#include "exception.hpp"

namespace gtensor{

namespace detail{

template<typename Config>
struct strides_div_type_
{
    template<typename, typename> struct selector_;
    //native division
    template<typename Dummy> struct selector_<gtensor::config::mode_div_native, Dummy>
    {
        using type = typename Config::shape_type;
    };
    //libdivide division
    template<typename Dummy> struct selector_<config::mode_div_libdivide, Dummy>
    {
        using type = libdivide_dividers_t<Config, typename Config::index_type>;
    };
    using type = typename selector_<typename Config::div_mode, void>::type;
};
template<typename Config> using strides_div_t = typename strides_div_type_<Config>::type;

template<typename T> struct change_order{using type = gtensor::config::c_order;};
template<> struct change_order<gtensor::config::c_order>{using type = gtensor::config::f_order;};
template<> struct change_order<gtensor::config::f_order>{using type = gtensor::config::c_order;};
template<typename Order> using change_order_t = typename change_order<Order>::type;

//makes broadcast shape, throws if shapes are not broadcastable
template<typename ShT>
inline void make_broadcast_shape_helper(ShT&){}
template<typename ShT, typename T>
inline void make_broadcast_shape_helper(ShT& res, const T& shape){
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
            throw value_error("shapes are not broadcastable");
        }
    }
}
template<typename ShT, typename T, typename...Ts>
inline void make_broadcast_shape_helper(ShT& res, const T& shape, const Ts&...shapes){
    make_broadcast_shape_helper(res, shape);
    make_broadcast_shape_helper(res, shapes...);
}
template<typename ShT, typename...Ts>
inline auto make_broadcast_shape(const Ts&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    auto res = shape_type(std::max({shapes.size()...}),index_type(-1));
    make_broadcast_shape_helper(res, shapes...);
    return res;
}
template<typename ShT, typename Container>
inline auto make_broadcast_shape_container(const Container& shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    using dim_type = typename shape_type::difference_type;
    dim_type max_dim{0};
    for (auto it = shapes.begin(), last = shapes.end(); it!=last; ++it){
        const auto& shape = unwrap_shape(*it);
        dim_type dim = static_cast<dim_type>(shape.size());
        max_dim = std::max(max_dim,dim);
    }
    auto res = shape_type(max_dim,index_type(-1));
    for (auto it = shapes.begin(), last = shapes.end(); it!=last; ++it){
        const auto& shape = unwrap_shape(*it);
        make_broadcast_shape_helper(res,shape);
    }
    return res;
}

template<typename T>
inline T make_shape_element(const T& shape_element){
    using shape_element_type = T;
    return shape_element==shape_element_type{0} ? shape_element_type{1}:shape_element;
}

template<typename It, typename DstIt, typename Order>
inline void make_strides(It shape_first, It shape_last, DstIt strides_first, DstIt strides_last, Order, typename std::iterator_traits<It>::value_type min_stride = 1){
    using result_value_type = typename std::iterator_traits<DstIt>::value_type;
    if (strides_first!=strides_last){
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            *--strides_last = result_value_type(min_stride);
            while(strides_last != strides_first){
                min_stride *= make_shape_element(*--shape_last);
                *--strides_last = result_value_type(min_stride);
            }
        }else if constexpr (std::is_same_v<Order,gtensor::config::f_order>){
            *strides_first = result_value_type(min_stride);
            for(++strides_first; strides_first!=strides_last; ++strides_first,++shape_first){
                min_stride *= make_shape_element(*shape_first);
                *strides_first = result_value_type(min_stride);
            }
        }else{
            static_assert(always_false<Order>,"invalid Order argument");
        }
    }
}

template<typename ResT, typename ShT, typename Order>
inline ResT make_strides(const ShT& shape, Order order, typename ShT::value_type min_stride = typename ShT::value_type(1)){
    using result_type = ResT;
    using result_value_type = typename result_type::value_type;
    result_type res(shape.size(), result_value_type(1));
    make_strides(shape.begin(),shape.end(),res.begin(),res.end(),order);
    return res;
}

template<typename ShT, typename Order>
inline ShT make_strides(const ShT& shape, Order order, typename ShT::value_type min_stride = typename ShT::value_type(1)){
    return make_strides<ShT,ShT>(shape, order, min_stride);
}

template<typename Config, typename ShT, typename Order>
inline auto make_strides_div(const ShT& shape, Order order){
    using div_mode = typename Config::div_mode;
    if constexpr (std::is_same_v<div_mode, gtensor::config::mode_div_native>){
        return make_strides(shape, order);
    }else if constexpr (std::is_same_v<div_mode, gtensor::config::mode_div_libdivide>){
        return make_strides<strides_div_t<Config>>(shape,order);
    }else{
        static_assert(always_false<Order>,"invalid div_mode");
    }
}

template<typename ShT>
inline auto make_reset_strides(const ShT& shape, const ShT& strides){
    using index_type = typename ShT::value_type;
    if (!std::empty(shape)){
        ShT res(shape.size());
        std::transform(
            shape.begin(),
            shape.end(),
            strides.begin(),
            res.begin(),
            [](const auto& shape_element, const auto& strides_element){
                return (make_shape_element(shape_element)-index_type(1))*strides_element;
            }
        );
        return res;
    }
    else{
        return ShT{};
    }
}

template<typename ShT>
inline auto make_adapted_strides(const ShT& shape, const ShT& strides){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    shape_type res{strides};
    auto shape_it = shape.begin();
    for (auto res_it = res.begin(); res_it!=res.end(); ++res_it,++shape_it){
        if (*shape_it == index_type{1}){
            *res_it = index_type{0};
        }
    }
    return res;
}

template<typename ShT>
inline auto make_size(const ShT& shape, const ShT& strides){
    using index_type = typename ShT::value_type;
    return std::empty(shape) ? index_type(1) : shape.front()*strides.front();
}

template<typename IdxT, typename ShT>
inline IdxT make_size(const ShT& shape){
    using index_type = IdxT;
    static constexpr bool is_shape = detail::is_container_of_type_v<ShT,index_type>;
    static_assert(is_shape || std::is_convertible_v<ShT,index_type>);
    if constexpr (is_shape){
        return std::accumulate(shape.begin(),shape.end(),index_type(1),std::multiplies<index_type>{});
    }else{
        return shape;
    }
}

template<typename ShT>
inline auto make_size(const ShT& shape){
    using index_type = typename ShT::value_type;
    return make_size<index_type>(shape);
}

template<typename ShT>
auto shape_to_str(const ShT& shape){
    std::stringstream ss{};
    const char* sep = "";
    ss<<"("<<[&](){for(const auto& i : shape){ss<<sep<<i; if(sep[0] == '\0'){sep=",";}} return ")";}();
    return ss.str();
}

template<typename T, typename ShT, std::enable_if_t<detail::is_container_v<std::remove_cv_t<std::remove_reference_t<ShT>>>,int> =0>
auto make_shape_of_type(ShT&& shape) -> std::conditional_t<std::is_same_v<T,std::decay_t<ShT>>,ShT&&,T>{
    if constexpr (std::is_same_v<T,std::decay_t<ShT>>){
        return std::forward<ShT>(shape);
    }else{
        return T(shape.begin(),shape.end());
    }
}
template<typename T, typename IdxT>
T make_shape_of_type(std::initializer_list<IdxT> shape){
    return T(shape.begin(),shape.end());
}
template<typename T, typename IdxT, std::enable_if_t<std::is_convertible_v<IdxT,typename T::value_type>,int> =0>
T make_shape_of_type(const IdxT& size){
    using index_type = typename T::value_type;
    return T{static_cast<index_type>(size)};
}

//converts flat index to flat index given strides and converting strides
template<typename Order, typename StT, typename CStT, typename IdxT>
auto flat_to_flat(const StT& strides, const CStT& cstrides, IdxT offset, IdxT idx){
    using index_type = IdxT;
    if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
        auto st_it = strides.begin();
        auto cst_it = cstrides.begin();
        while(idx != index_type(0)){
            offset += *cst_it*divide(idx,*st_it);
            ++st_it;
            ++cst_it;
        }
    }else if constexpr (std::is_same_v<Order,gtensor::config::f_order>){
        auto st_it = strides.end();
        auto cst_it = cstrides.end();
        while(idx != index_type(0)){
            offset += *--cst_it*divide(idx,*--st_it);
        }
    }else{
        static_assert(always_false<Order>,"invalid Order argument");
    }
    return offset;
}

template<typename Config, typename Mode> class strides_div_extension;

template<typename Config>
class strides_div_extension<Config,gtensor::config::mode_div_libdivide>
{
    using shape_type = typename Config::shape_type;
    using index_type = typename Config::index_type;
    using strides_div_type = detail::strides_div_t<Config>;
    strides_div_type strides_div_;
    strides_div_type changing_order_strides_div_;
protected:
    strides_div_extension() = default;
    template<typename Order>
    strides_div_extension(const shape_type& shape__, const shape_type& strides__, Order):
        strides_div_{detail::make_libdivide_dividers<Config>(strides__)},
        changing_order_strides_div_{detail::make_strides_div<Config>(shape__, detail::change_order_t<Order>{})}
    {}
    const auto&  strides_div()const{return strides_div_;}
    const auto&  changing_order_strides_div()const{return changing_order_strides_div_;}
};

template<typename Config>
class strides_div_extension<Config,gtensor::config::mode_div_native>
{
    using shape_type = typename Config::shape_type;
    shape_type changing_order_strides_div_;
protected:
    strides_div_extension() = default;
    template<typename Order>
    strides_div_extension(const shape_type& shape__, const shape_type&, Order):
        changing_order_strides_div_{detail::make_strides(shape__,detail::change_order_t<Order>{})}
    {}
    const auto&  changing_order_strides_div()const{return changing_order_strides_div_;}
};

template<typename Config>
class strides_extension
{
    using shape_type = typename Config::shape_type;
    shape_type strides_;
    shape_type adapted_strides_;
    shape_type reset_strides_;

protected:
    strides_extension() = default;
    template<typename Order>
    strides_extension(const shape_type& shape__, Order):
        strides_{detail::make_strides(shape__, Order{})},
        adapted_strides_{detail::make_adapted_strides(shape__,strides_)},
        reset_strides_{detail::make_reset_strides(shape__,strides_)}
    {}
    const auto&  strides()const{return strides_;}
    const auto&  adapted_strides()const{return adapted_strides_;}
    const auto&  reset_strides()const{return reset_strides_;}
};

template<typename Config>
class descriptor_strides :
    private strides_extension<Config>,
    private strides_div_extension<Config, typename Config::div_mode>
{
    using strides_extension_base = strides_extension<Config>;
    using strides_div_extension_base = strides_div_extension<Config, typename Config::div_mode>;
    using shape_type = typename Config::shape_type;

    const auto& strides_div(gtensor::config::mode_div_libdivide)const{return strides_div_extension_base::strides_div();}
    const auto& strides_div(gtensor::config::mode_div_native)const{return strides_extension_base::strides();}
public:
    descriptor_strides() = default;
    template<typename Order>
    descriptor_strides(const shape_type& shape__, Order order__):
        strides_extension_base{shape__, order__},
        strides_div_extension_base{shape__, strides_extension_base::strides(), order__}
    {}
    const auto& strides_div()const{return strides_div(typename Config::div_mode{});}
    const auto& strides()const{return strides_extension_base::strides();}
    const auto& adapted_strides()const{return strides_extension_base::adapted_strides();}
    const auto& reset_strides()const{return strides_extension_base::reset_strides();}
    const auto& changing_order_strides_div()const{return strides_div_extension_base::changing_order_strides_div();}
};

}   //end of namespace detail

//common staff
template<typename Config>
class descriptor_common
{
public:
    using config_type = Config;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    auto size()const{return size_;}
    auto dim()const{return detail::make_dim(shape_);}
    const auto& shape()const{return shape_;}
    const auto& strides_div()const{return strides_.strides_div();}
    const auto& changing_order_strides_div()const{return strides_.changing_order_strides_div();}
    const auto& strides()const{return strides_.strides();}
    const auto& adapted_strides()const{return strides_.adapted_strides();}
    const auto& reset_strides()const{return strides_.reset_strides();}

    template<typename ShT, typename Order>
    explicit descriptor_common(ShT&& shape__, Order):
        shape_{detail::make_shape_of_type<shape_type>(std::forward<ShT>(shape__))},
        strides_{shape_, Order{}}
    {}
private:
    shape_type shape_;
    detail::descriptor_strides<Config> strides_;
    index_type size_{detail::make_size(shape_)};
};

//descriptors
template<typename Config, typename Order>
class basic_descriptor
{
    ASSERT_ORDER(Order);
    descriptor_common<Config> impl_;
public:
    using config_type = Config;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using strides_div_type = detail::strides_div_t<config_type>;

    template<typename ShT, std::enable_if_t<!std::is_convertible_v<std::decay_t<ShT>, basic_descriptor>,int> =0 >
    explicit basic_descriptor(ShT&& shape__):
        impl_{std::forward<ShT>(shape__), Order{}}
    {}
    dim_type dim()const{return impl_.dim();}
    index_type size()const{return impl_.size();}
    const shape_type& shape()const{return impl_.shape();}
    const strides_div_type& strides_div(gtensor::config::c_order)const{return strides_div_helper<gtensor::config::c_order>();}
    const strides_div_type& strides_div(gtensor::config::f_order)const{return strides_div_helper<gtensor::config::f_order>();}
    const strides_div_type& strides_div()const{return strides_div_helper<Order>();}
    const shape_type& strides()const{return impl_.strides();}
    const shape_type& adapted_strides()const{return impl_.adapted_strides();}
    const shape_type& reset_strides()const{return impl_.reset_strides();}
private:
    template<typename Order_>
    const strides_div_type& strides_div_helper()const{
        if constexpr (std::is_same_v<Order_,Order>){
            return impl_.strides_div();
        }else{
            return impl_.changing_order_strides_div();
        }
    }
};

template<typename Config, typename Order>
class transpose_descriptor : public basic_descriptor<Config,Order>
{
    using basic_descriptor_base = basic_descriptor<Config,Order>;
    using typename basic_descriptor_base::dim_type;
    using axes_map_type = typename Config::template shape<dim_type>;
public:
    using typename basic_descriptor_base::index_type;
    using typename basic_descriptor_base::shape_type;

    template<typename ShT, typename Container>
    transpose_descriptor(Container&& axes_map__,ShT&& shape__):
        basic_descriptor_base{std::forward<ShT>(shape__)},
        axes_map_{detail::make_shape_of_type<axes_map_type>(std::forward<Container>(axes_map__))}
    {}
    const axes_map_type& axes_map()const{return axes_map_;}
private:
    axes_map_type axes_map_;
};

template<typename Base>
class descriptor_w_offset : public Base
{
    using base_type = Base;
public:
    using typename base_type::index_type;
    using typename base_type::shape_type;

    template<typename Container, typename...Args>
    descriptor_w_offset(Container&& offset__, Args&&...args__):
        base_type{std::forward<Args>(args__)...},
        offset_{detail::make_shape_of_type<shape_type>(std::forward<Container>(offset__))}
    {}
    const shape_type& offset()const{return offset_;}
private:
    shape_type offset_;
};

template<typename Base>
class descriptor_w_scale : public Base
{
    using base_type = Base;
public:
    using typename base_type::index_type;
    using typename base_type::shape_type;

    template<typename Container, typename...Args>
    descriptor_w_scale(Container&& scale__, Args&&...args__):
        base_type{std::forward<Args>(args__)...},
        scale_{detail::make_shape_of_type<shape_type>(std::forward<Container>(scale__))}
    {}
    const shape_type& scale()const{return scale_;}
private:
    shape_type scale_;
};

}   //end of namespace gtensor
#endif