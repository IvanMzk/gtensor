#ifndef DESCRIPTOR_BASE_HPP_
#define DESCRIPTOR_BASE_HPP_

#include <stdexcept>
#include <string>
#include <sstream>
#include <numeric>
#include "libdivide_helper.hpp"

namespace gtensor{

namespace detail{

/*
* create strides
* parameters: shape
*/
template<typename ShT>
inline ShT make_strides(const ShT& shape, typename ShT::value_type min_stride = ShT::value_type(1)){
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

template<typename ShT>
inline auto make_size(const ShT& shape){
    using index_type = typename ShT::value_type;
    if (shape.size() == 0){
        return index_type(0);
    }else{
        return std::accumulate(shape.begin(),shape.end(),index_type(1),std::multiplies<index_type>{});
    }
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
    template<typename> struct selector{using type = typename CfgT::shape_type;};    //native division
    template<> struct selector<config::mode_div_libdivide>{using type = libdivide_vector<typename CfgT::index_type>;};  //libdivide division
    using type = typename selector<typename CfgT::div_mode>::type;
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
    const auto& strides_div()const{return strides_div(CfgT::div_mode{});}
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
    virtual std::string to_str()const = 0;
};

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
    auto to_str()const{return detail::shape_to_str(shape());}

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


}   //end of namespace gtensor

#endif