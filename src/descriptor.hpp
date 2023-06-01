#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include <iostream>
#include <numeric>
#include <sstream>
#include "common.hpp"
#include "libdivide_helper.hpp"

namespace gtensor{

class broadcast_exception : public std::runtime_error{
public:
    explicit broadcast_exception(const char* what):
        runtime_error(what)
    {}
};

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

template<typename> struct change_order;
template<> struct change_order<gtensor::config::c_order>{using type = gtensor::config::f_order;};
template<> struct change_order<gtensor::config::f_order>{using type = gtensor::config::c_order;};
template<typename Order> using change_order_t = typename change_order<Order>::type;

//makes broadcast shape, throws if shapes are not broadcastable
template<typename ShT>
inline void make_broadcast_shape_helper(ShT&){}
template<typename ShT, typename T, typename...Ts>
inline void make_broadcast_shape_helper(ShT& res, const T& shape, const Ts&...shapes){
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

template<typename T>
inline T make_shape_element(const T& shape_element){
    using shape_element_type = T;
    return shape_element==shape_element_type{0} ? shape_element_type{1}:shape_element;
}
template<typename ResT, typename ShT, typename Order>
inline ResT make_strides(const ShT& shape, Order, typename ShT::value_type min_stride = typename ShT::value_type(1)){
    using result_type = ResT;
    using result_value_type = typename result_type::value_type;
    if (!std::empty(shape)){
        result_type res(shape.size(), result_value_type());
        auto shape_first = shape.begin();
        auto shape_last = shape.end();
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            auto res_it = res.end();
            *--res_it = result_value_type(min_stride);
            for(;res_it != res.begin();){
                min_stride *= make_shape_element(*--shape_last);
                *--res_it = result_value_type(min_stride);
            }
        }else if constexpr (std::is_same_v<Order,gtensor::config::f_order>){
            auto res_it = res.begin();
            *res_it = result_value_type(min_stride);
            for(++res_it;res_it != res.end();++res_it,++shape_first){
                min_stride *= make_shape_element(*shape_first);
                *res_it = result_value_type(min_stride);
            }
        }else{
            static_assert(always_false<Order>,"invalid Order argument");
        }
        return res;
    }
    else{
        return result_type{};
    }
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
inline auto make_size(const ShT& shape){
    using index_type = IdxT;
    return std::accumulate(shape.begin(),shape.end(),index_type(1),std::multiplies<index_type>{});
}

template<typename ShT>
inline auto make_size(const ShT& shape){
    using index_type = typename ShT::value_type;
    return make_size<index_type>(shape);
}

template<typename ShT>
auto shape_to_str(const ShT& shape){
    std::stringstream ss{};
    ss<<"("<<[&](){for(const auto& i : shape){ss<<i<<",";} return ")";}();
    return ss.str();
}

template<typename T, typename ShT>
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

template<typename Descriptor>
auto make_order_converter(const Descriptor& desc){
    return [&desc](const auto& idx){
        return desc.convert_order_inline(idx);
    };
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

//descriptor abstract interface
template<typename Config>
class descriptor_base{
public:
    using config_type = Config;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using strides_div_type = detail::strides_div_t<config_type>;

    virtual ~descriptor_base(){}
    virtual index_type convert(const index_type& idx)const = 0;
    virtual index_type convert_order(const index_type& idx)const = 0;
    virtual dim_type dim()const = 0;
    virtual index_type size()const = 0;
    virtual index_type offset()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const strides_div_type& strides_div()const = 0; //strides optimized for division
    virtual const strides_div_type& strides_div(gtensor::config::c_order)const = 0; //strides optimized for division
    virtual const strides_div_type& strides_div(gtensor::config::f_order)const = 0; //strides optimized for division
    virtual const shape_type& strides()const = 0;
    virtual const shape_type& adapted_strides()const = 0;
    virtual const shape_type& reset_strides()const = 0;
    virtual const shape_type& cstrides()const = 0;
};

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

//descriptors implementation
template<typename Config, typename Order>
class basic_descriptor : public descriptor_base<Config>
{
    ASSERT_ORDER(Order);
    using descriptor_base_type = descriptor_base<Config>;
    descriptor_common<Config> impl_;
public:
    using typename descriptor_base_type::config_type;
    using typename descriptor_base_type::index_type;
    using typename descriptor_base_type::dim_type;
    using typename descriptor_base_type::shape_type;
    using typename descriptor_base_type::strides_div_type;

    template<typename ShT, std::enable_if_t<!std::is_convertible_v<std::decay_t<ShT>, basic_descriptor>,int> =0 >
    explicit basic_descriptor(ShT&& shape__):
        impl_{std::forward<ShT>(shape__), Order{}}
    {}
    dim_type dim()const override{return impl_.dim();}
    index_type size()const override{return impl_.size();}
    const shape_type& shape()const override{return impl_.shape();}
    const strides_div_type& strides_div(gtensor::config::c_order)const override{return strides_div_helper<gtensor::config::c_order>();}
    const strides_div_type& strides_div(gtensor::config::f_order)const override{return strides_div_helper<gtensor::config::f_order>();}
    const strides_div_type& strides_div()const override{return strides_div_helper<Order>();}
    const shape_type& strides()const override{return impl_.strides();}
    const shape_type& adapted_strides()const override{return impl_.adapted_strides();}
    const shape_type& reset_strides()const override{return impl_.reset_strides();}
    index_type offset()const override{return index_type{0};}
    const shape_type& cstrides()const override{return strides();}
    index_type convert(const index_type& idx)const override{return operator()(idx);}
    index_type convert_order(const index_type& idx)const override{return convert_order_inline(idx);}
    index_type operator()(const index_type& idx)const{return idx;}
    index_type convert_order_inline(const index_type& idx)const{
        using changed_order_type = detail::change_order_t<Order>;
        return detail::flat_to_flat<changed_order_type>(strides_div_helper<changed_order_type>(),strides(),offset(),idx);
    }
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
class descriptor_with_offset : public basic_descriptor<Config,Order>
{
    using basic_descriptor_base = basic_descriptor<Config,Order>;
public:
    using typename basic_descriptor_base::index_type;
    using typename basic_descriptor_base::shape_type;

    template<typename ShT>
    descriptor_with_offset(ShT&& shape__, index_type offset__):
        basic_descriptor_base{std::forward<ShT>(shape__)},
        offset_{offset__}
    {}
    index_type offset()const override{return offset_;}
    index_type convert(const index_type& idx)const override{return operator()(idx);}
    index_type operator()(const index_type& idx)const{return idx+offset_;}
private:
    index_type offset_;
};

template<typename Config, typename Order>
class converting_descriptor : public descriptor_with_offset<Config,Order>
{
    using descriptor_with_offset_base = descriptor_with_offset<Config,Order>;
public:
    using typename descriptor_with_offset_base::index_type;
    using typename descriptor_with_offset_base::shape_type;
    using descriptor_with_offset_base::shape;
    using descriptor_with_offset_base::offset;
    using descriptor_with_offset_base::strides_div;

    template<typename ShT, typename StT>
    converting_descriptor(ShT&& shape__, StT&& cstrides__,  const index_type& offset__):
        descriptor_with_offset_base{std::forward<ShT>(shape__), offset__},
        cstrides_{std::forward<StT>(cstrides__)}
    {}
    const shape_type& cstrides()const override{
        return cstrides_;
    }
    index_type convert(const index_type& idx)const override{
        return operator()(idx);
    }
    index_type operator()(const index_type& idx)const{
        return detail::flat_to_flat<Order>(strides_div(),cstrides(),offset(),idx);
    }
private:
    shape_type cstrides_;
};

template<typename Config, typename Order>
class mapping_descriptor : public basic_descriptor<Config,Order>
{
    using basic_descriptor_base = basic_descriptor<Config,Order>;
    using index_map_type = typename Config::index_map_type;
public:
    using typename basic_descriptor_base::shape_type;
    using typename basic_descriptor_base::index_type;

    mapping_descriptor() = default;
    template<typename ShT, typename Map>
    mapping_descriptor(ShT&& shape__, Map&& index_map__):
        basic_descriptor_base{std::forward<ShT>(shape__)},
        index_map_{std::forward<Map>(index_map__)}
    {}
    index_type convert(const index_type& idx)const override{return operator()(idx);}
    index_type operator()(const index_type& idx)const{return index_map_[idx];}
private:
    index_map_type index_map_;
};

}   //end of namespace gtensor
#endif