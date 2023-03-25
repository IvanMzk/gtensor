#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include <iostream>
#include <numeric>
#include <sstream>
#include "libdivide_helper.hpp"

namespace gtensor{
namespace detail{

//select type of strides_div
template<typename CfgT>
class strides_div_traits
{
    using config_type = CfgT;
    //native division
    template<typename, typename>
    struct selector_
    {
        using type = typename config_type::shape_type;
    };
    //libdivide division
    template<typename Dummy>
    struct selector_<config::mode_div_libdivide, Dummy>
    {
        using type = typename libdivide_container_selector<config_type>::template container<typename config_type::index_type>;
    };
public:
    using type = typename selector_<typename config_type::div_mode, void>::type;
};

template<typename T>
inline T make_shape_element(const T& shape_element){
    using shape_element_type = T;
    return shape_element==shape_element_type{0} ? shape_element_type{1}:shape_element;
}

template<typename ResT, typename ShT>
inline ResT make_strides(const ShT& shape, typename ShT::value_type min_stride = typename ShT::value_type(1)){
    using result_type = ResT;
    using result_value_type = typename result_type::value_type;
    if (!std::empty(shape)){
        result_type res(shape.size(), result_value_type());
        auto shape_begin = shape.begin();
        auto shape_it = shape.end();
        auto res_end{res.end()};
        *--res_end = result_value_type(min_stride);
        while (--shape_it!=shape_begin){
            min_stride *= make_shape_element(*shape_it);
            *--res_end = result_value_type(min_stride);
        }
        return res;
    }
    else{
        return result_type{};
    }
}
template<typename ShT>
inline ShT make_strides(const ShT& shape, typename ShT::value_type min_stride = typename ShT::value_type(1)){
    return make_strides<ShT,ShT>(shape,min_stride);
}

template<typename ShT, typename CfgT>
inline auto make_strides_div(const ShT& shape, CfgT, gtensor::config::mode_div_libdivide){
    return make_strides<typename strides_div_traits<CfgT>::type>(shape);
}
template<typename ShT, typename CfgT>
inline auto make_strides_div(const ShT& shape, CfgT, gtensor::config::mode_div_native){
    return make_strides(shape);
}
template<typename CfgT, typename ShT>
inline auto make_strides_div(const ShT& shape){
    return make_strides_div(shape, CfgT{}, typename CfgT::div_mode{});
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
inline auto make_size(const ShT& shape, const ShT& strides){
    using index_type = typename ShT::value_type;
    return std::empty(shape) ? index_type(0) : shape.front()*strides.front();
}

template<typename IdxT, typename ShT>
inline auto make_size(const ShT& shape){
    using index_type = IdxT;
    if (std::empty(shape)){
        return index_type(0);
    }else{
        return std::accumulate(shape.begin(),shape.end(),index_type(1),std::multiplies<index_type>{});
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

//convert flat index to multi index given strides, ShT result multiindex type, must specialize explicitly
template<typename ShT, typename StT, typename IdxT>
auto flat_to_multi(const StT& strides, const IdxT& idx){
    using shape_type = ShT;
    using index_type = IdxT;
    shape_type res(strides.size(), index_type(0));
    index_type idx_{idx};
    auto st_it = strides.begin();
    auto res_it = res.begin();
    while(idx_ != index_type(0)){
        *res_it = divide(idx_,*st_it);
        ++st_it,++res_it;
    }
    return res;
}

//converts flat index to flat index given strides and converting strides
template<typename StT, typename CStT, typename IdxT>
auto flat_to_flat(const StT& strides, const CStT& cstrides, const IdxT& offset, const IdxT& idx){
    using index_type = IdxT;
    index_type res{offset};
    index_type idx_{idx};
    auto st_it = strides.begin();
    auto cst_it = cstrides.begin();
    while(idx_ != index_type(0)){
        res += *cst_it*divide(idx_,*st_it);
        ++st_it;
        ++cst_it;
    }
    return res;
}

//converts multi index to flat index given converting strides
template<typename ShT>
auto convert_index(const ShT& cstrides, const typename ShT::value_type& offset, const ShT& idx){
    return std::inner_product(idx.begin(), idx.end(), cstrides.begin(), offset);
}

template<typename CfgT, typename Mode> class strides_div_extension;

template<typename CfgT>
class strides_div_extension<CfgT,gtensor::config::mode_div_libdivide>
{
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
    using strides_div_type = typename detail::strides_div_traits<CfgT>::type;
    strides_div_type strides_div_;
protected:
    strides_div_extension() = default;
    strides_div_extension(const shape_type& strides__):
        strides_div_{detail::make_libdivide_container<CfgT>(strides__)}
    {}
    const auto&  strides_div()const{return strides_div_;}
};

template<typename CfgT>
class strides_div_extension<CfgT,gtensor::config::mode_div_native>
{
    using shape_type = typename CfgT::shape_type;
protected:
    strides_div_extension() = default;
    strides_div_extension(const shape_type&)
    {}
};

template<typename CfgT>
class strides_extension
{
    using shape_type = typename CfgT::shape_type;
    shape_type strides_;
    shape_type reset_strides_;
protected:
    strides_extension() = default;
    strides_extension(const shape_type& shape__):
        strides_{detail::make_strides(shape__)},
        reset_strides_{detail::make_reset_strides(shape__,strides_)}
    {}
    const auto&  strides()const{return strides_;}
    const auto&  reset_strides()const{return reset_strides_;}
};

template<typename CfgT>
class descriptor_strides :
    private strides_extension<CfgT>,
    private strides_div_extension<CfgT, typename CfgT::div_mode>
{
    using strides_extension_base = strides_extension<CfgT>;
    using strides_div_extension_base = strides_div_extension<CfgT, typename CfgT::div_mode>;
    using shape_type = typename CfgT::shape_type;

    const auto& strides_div(gtensor::config::mode_div_libdivide)const{return strides_div_extension_base::strides_div();}
    const auto& strides_div(gtensor::config::mode_div_native)const{return strides_extension_base::strides();}
public:
    descriptor_strides() = default;
    descriptor_strides(const shape_type& shape__):
        strides_extension_base{shape__},
        strides_div_extension_base{strides_extension_base::strides()}
    {}
    const auto& strides_div()const{return strides_div(typename CfgT::div_mode{});}
    const auto& strides()const{return strides_extension_base::strides();}
    const auto& reset_strides()const{return strides_extension_base::reset_strides();}
};

}   //end of namespace detail

//descriptor abstract interface
template<typename CfgT>
class descriptor_base{
public:
    using config_type = CfgT;
    using index_type = typename config_type::index_type;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using strides_div_type = typename detail::strides_div_traits<config_type>::type;

    virtual ~descriptor_base(){}
    virtual index_type convert(const shape_type& idx)const = 0;
    virtual index_type convert(const index_type& idx)const = 0;
    virtual size_type dim()const = 0;
    virtual index_type size()const = 0;
    virtual index_type offset()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const strides_div_type& strides_div()const = 0; //strides optimized for division
    virtual const shape_type& strides()const = 0;
    virtual const shape_type& reset_strides()const = 0;
    virtual const shape_type& cstrides()const = 0;
    virtual const shape_type& reset_cstrides()const = 0;
};

//common staff
template<typename CfgT>
class descriptor_common
{
public:
    using config_type = CfgT;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    auto size()const{return size_;}
    auto dim()const{return shape_.size();}
    const auto& shape()const{return shape_;}
    const auto& strides_div()const{return strides_.strides_div();}
    const auto& strides()const{return strides_.strides();}
    const auto& reset_strides()const{return strides_.reset_strides();}

    template<typename ShT, std::enable_if_t<!std::is_convertible_v<std::decay_t<ShT>, descriptor_common>,int> =0 >
    explicit descriptor_common(ShT&& shape__):
        shape_{detail::make_shape_of_type<shape_type>(std::forward<ShT>(shape__))}
    {}
private:
    shape_type shape_;
    index_type size_{detail::make_size(shape_)};
    detail::descriptor_strides<CfgT> strides_{shape_};
};

//descriptors implementation
template<typename CfgT>
class basic_descriptor : public descriptor_base<CfgT>
{
    using descriptor_base_type = descriptor_base<CfgT>;
    descriptor_common<CfgT> impl_;
public:
    using typename descriptor_base_type::config_type;
    using typename descriptor_base_type::index_type;
    using typename descriptor_base_type::size_type;
    using typename descriptor_base_type::shape_type;
    using typename descriptor_base_type::strides_div_type;

    template<typename ShT, std::enable_if_t<!std::is_convertible_v<std::decay_t<ShT>, basic_descriptor>,int> =0 >
    explicit basic_descriptor(ShT&& shape__):
        impl_{std::forward<ShT>(shape__)}
    {}
    size_type dim()const override{return impl_.dim();}
    index_type size()const override{return impl_.size();}
    const shape_type& shape()const override{return impl_.shape();}
    const strides_div_type& strides_div()const override{return impl_.strides_div();}
    const shape_type& strides()const override{return impl_.strides();}
    const shape_type& reset_strides()const override{return impl_.reset_strides();}
    index_type offset()const override{return index_type{0};}
    const shape_type& cstrides()const override{return strides();}
    const shape_type& reset_cstrides()const override{return reset_strides();}
    index_type convert(const index_type& idx)const override{return idx;}
    index_type convert(const shape_type& idx)const override{return detail::convert_index(cstrides(),offset(),idx);}
};

template<typename CfgT>
class descriptor_with_offset : public basic_descriptor<CfgT>
{
    using basic_descriptor_base = basic_descriptor<CfgT>;
public:
    using typename basic_descriptor_base::index_type;
    using typename basic_descriptor_base::shape_type;

    template<typename ShT>
    descriptor_with_offset(ShT&& shape__, index_type offset__):
        basic_descriptor_base{std::forward<ShT>(shape__)},
        offset_{offset__}
    {}
    index_type offset()const override{return offset_;}
    index_type convert(const index_type& idx)const override{return idx+offset_;}
    index_type convert(const shape_type& idx)const override{return detail::convert_index(basic_descriptor_base::cstrides(),offset(),idx);}
private:
    index_type offset_;
};

template<typename CfgT>
class converting_descriptor : public descriptor_with_offset<CfgT>
{
    using descriptor_with_offset_base = descriptor_with_offset<CfgT>;
public:
    using typename descriptor_with_offset_base::index_type;
    using typename descriptor_with_offset_base::shape_type;
    using descriptor_with_offset_base::shape;
    using descriptor_with_offset_base::offset;
    using descriptor_with_offset_base::strides_div;

    template<typename ShT, typename StT>
    converting_descriptor(ShT&& shape__, StT&& cstrides__,  const index_type& offset__):
        descriptor_with_offset_base{std::forward<ShT>(shape__), offset__},
        cstrides_{std::forward<StT>(cstrides__)},
        reset_cstrides_{detail::make_reset_strides(shape(),cstrides_)}
    {}
    const shape_type& cstrides()const override{return cstrides_;}
    const shape_type& reset_cstrides()const override{return reset_cstrides_;}
    index_type convert(const shape_type& idx)const override{return detail::convert_index(cstrides(),offset(),idx);}
    index_type convert(const index_type& idx)const override{return detail::flat_to_flat(strides_div(),cstrides(),offset(),idx);}
private:
    shape_type cstrides_;
    shape_type reset_cstrides_;
};

}   //end of namespace gtensor

#endif