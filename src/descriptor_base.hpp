#ifndef DESCRIPTOR_BASE_HPP_
#define DESCRIPTOR_BASE_HPP_

#include <stdexcept>
#include <string>
#include <sstream>
#include <numeric>
#include "libdivide_helper.hpp"

namespace gtensor{

class broadcast_exception : public std::runtime_error{
    public: broadcast_exception(const char* what):runtime_error(what){}
};

namespace detail{

template<typename Arg>
inline auto& vmax(const Arg& arg){
    return arg;
}
template<typename Arg, typename...Args>
inline auto& vmax(const Arg& arg, const Args&...args){
    return std::max(arg, vmax(args...));
}
template<typename Arg>
inline auto& vmin(const Arg& arg){
    return arg;
}
template<typename Arg, typename...Args>
inline auto& vmin(const Arg& arg, const Args&...args){
    return std::min(arg, vmin(args...));
}
/*
* create broadcast shape
* parameters: shapes to broadcast
* exception if shapes are not broadcastable
*/
template<typename ShT, typename...Ts>
inline auto broadcast_shape(const Ts&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    if (vmin(shapes.size()...) == 0){
        throw broadcast_exception("shapes are not broadcastable");
    }else{
        auto res = shape_type(vmax(shapes.size()...),index_type(0));
        broadcast_shape_helper(res, shapes...);
        return res;
    }
}
template<typename ShT, typename T, typename...Ts>
inline void broadcast_shape_helper(ShT& res, const T& shape, const Ts&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    auto res_it = res.end();
    auto shape_it = shape.end();
    auto shape_begin = shape.begin();
    while(shape_it!=shape_begin){
        const index_type& r{*--res_it};
        const index_type& s{*--shape_it};
        if (r==index_type(0) || r==index_type(1)){
            *res_it = s;
        }
        else if (s!=index_type(1) && s!=r){
            throw broadcast_exception("shapes are not broadcastable");
        }
    }
    broadcast_shape_helper(res, shapes...);
}
template<typename ShT>
inline void broadcast_shape_helper(ShT&){}

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
class descriptor_strides
{
    using shape_type = typename CfgT::shape_type;
    shape_type strides_;
    shape_type reset_strides_;
public:
    descriptor_strides() = default;
    descriptor_strides(const shape_type& shape__):
        strides_{detail::make_strides(shape__)},
        reset_strides_{detail::make_reset_strides(shape__,strides_)}
    {}
    const auto&  strides()const{return strides_;}
    const auto&  reset_strides()const{return reset_strides_;}
};

}   //end of namespace detail

//descriptor abstract interface
template<typename CfgT>
class descriptor_base{
public:
    using config_type = CfgT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    virtual index_type convert(const shape_type& idx)const = 0;
    virtual index_type convert(const index_type& idx)const = 0;
    virtual index_type dim()const = 0;
    virtual index_type size()const = 0;
    virtual index_type offset()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const shape_type& strides()const = 0;
    virtual const shape_type& reset_strides()const = 0;
    virtual const shape_type& cstrides()const = 0;
    virtual const shape_type& reset_cstrides()const = 0;
    virtual std::string to_str()const = 0;

    virtual const descriptor_with_libdivide<config_type>* as_descriptor_with_libdivide()const {return nullptr;}
};

//common implementation of descriptor
template<typename CfgT>
class descriptor_common
{
protected:
    using config_type = CfgT;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    auto size()const{return detail::make_size(shape(),strides());}
    auto dim()const{return shape_.size();}
    const auto& shape()const{return shape_;}
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
    shape_type shape_;
    detail::descriptor_strides<CfgT> strides_;
};


}   //end of namespace gtensor

#endif