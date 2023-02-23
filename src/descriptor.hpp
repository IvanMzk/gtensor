#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include <iostream>
#include <numeric>
#include <sstream>
#include "libdivide_helper.hpp"

namespace gtensor{
namespace detail{

template<typename ShT>
inline ShT make_strides(const ShT& shape, typename ShT::value_type min_stride = typename ShT::value_type(1)){
    if (!shape.empty()){
        ShT res(shape.size(), min_stride);
        auto shape_begin{shape.begin()};
        auto shape_end{shape.end()};
        auto res_end{res.end()};
        --res_end;
        while (shape_begin!=--shape_end){
            min_stride*=*shape_end;
            *--res_end = min_stride;
        }
        return res;
    }
    else{
        return ShT{};
    }
}

template<typename ShT>
inline auto make_reset_strides(const ShT& shape, const ShT& strides){
    if (!shape.empty()){
        ShT res(shape.size());
        std::transform(
            shape.begin(),
            shape.end(),
            strides.begin(),
            res.begin(),
            [](const auto& shape_element, const auto& strides_element){return (shape_element-1)*strides_element;}
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
    return shape.empty() ? index_type(0) : shape.front()*strides.front();
}

template<typename IdxT, typename ShT>
inline auto make_size(const ShT& shape){
    using index_type = IdxT;
    if (shape.size() == 0){
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

template<typename ShT>
auto convert_index(const ShT& cstrides, const typename ShT::value_type& offset, const ShT& idx){
    return std::inner_product(idx.begin(), idx.end(), cstrides.begin(), offset);
}

template<typename CfgT>
struct strides_div_traits{
    template<typename, typename> struct selector{using type = typename CfgT::shape_type;};    //native division
    template<typename Dummy> struct selector<config::mode_div_libdivide, Dummy>{using type = libdivide_vector<typename CfgT::index_type>;};  //libdivide division
    using type = typename selector<typename CfgT::div_mode, void>::type;
};

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
        strides_div_{detail::make_libdivide_vector(strides__)}
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
    const shape_type strides_;
    const shape_type reset_strides_;
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
    using shape_type = typename config_type::shape_type;
    using strides_div_type = typename detail::strides_div_traits<config_type>::type;

    virtual index_type convert(const shape_type& idx)const = 0;
    virtual index_type convert(const index_type& idx)const = 0;
    virtual index_type dim()const = 0;
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
    auto size()const{return detail::make_size(shape(),strides());}
    auto dim()const{return shape_.size();}
    const auto& shape()const{return shape_;}
    const auto& strides_div()const{return strides_.strides_div();}
    const auto& strides()const{return strides_.strides();}
    const auto& reset_strides()const{return strides_.reset_strides();}

    descriptor_common() = default;
    template<typename ShT, std::enable_if_t<!std::is_convertible_v<std::decay_t<ShT>, descriptor_common>,int> =0 >
    explicit descriptor_common(ShT&& shape__):
        shape_{detail::make_shape_of_type<shape_type>(std::forward<ShT>(shape__))},
        strides_{shape_}
    {}
private:
    const shape_type shape_;
    const detail::descriptor_strides<CfgT> strides_;
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
    using typename descriptor_base_type::shape_type;
    using typename descriptor_base_type::strides_div_type;
    basic_descriptor() = default;
    template<typename ShT, std::enable_if_t<!std::is_convertible_v<std::decay_t<ShT>, basic_descriptor>,int> =0 >
    explicit basic_descriptor(ShT&& shape__):
        impl_{std::forward<ShT>(shape__)}
    {}

    index_type dim()const override{return impl_.dim();}
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
    descriptor_with_offset() = default;
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
    converting_descriptor() = default;
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

template<typename CfgT, typename MapT = typename CfgT::shape_type>
class mapping_descriptor : public basic_descriptor<CfgT>
{
    using basic_descriptor_base = basic_descriptor<CfgT>;
public:
    using map_type = MapT;
    using typename basic_descriptor_base::shape_type;
    using typename basic_descriptor_base::index_type;
    mapping_descriptor() = default;
    template<typename ShT, typename MapT_>
    mapping_descriptor(ShT&& shape__, MapT_&& index_map__):
        basic_descriptor_base{std::forward<ShT>(shape__)},
        index_map_{std::forward<MapT_>(index_map__)}
    {}
    index_type convert(const index_type& idx)const override{return index_map_[idx];}
    index_type convert(const shape_type& idx)const override{return index_map_[detail::convert_index(basic_descriptor_base::cstrides(),basic_descriptor_base::offset(),idx)];}
private:
    map_type index_map_;
};

}   //end of namespace gtensor

#endif