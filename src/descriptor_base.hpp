#ifndef DESCRIPTOR_BASE_HPP_
#define DESCRIPTOR_BASE_HPP_

#include <stdexcept>
#include <string>
#include <sstream>
#include "libdivide_helper.hpp"

namespace gtensor{

class stensor_descriptor_exception : public std::runtime_error{
    public: stensor_descriptor_exception(const char* what):runtime_error(what){}
};

namespace detail{
/*
* in: shape elements in right order, max stride element first
* out: shape container filled with shape elements in reverse order, min stride element first
*/
template<typename ShT, typename...Dims>
ShT make_shape(const Dims&...dims){
    ShT res{};
    res.reserve(sizeof...(Dims));
    make_shape_(res,dims...);
    return res;
}
template<typename ShT, typename Dim, typename...Dims>
inline void make_shape_(ShT& res, const Dim& d, const Dims&...dims){
    make_shape_(res,dims...);
    res.push_back(d);
}
template<typename ShT>
inline void make_shape_(ShT& res){}

/*
* create strides
* parameters: shape
*/
template<typename ShT>
inline ShT make_strides(const ShT& shape, typename ShT::value_type min_stride = ShT::value_type(1)){
    using index_type = typename ShT::value_type;
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
inline auto make_size(const ShT& shape, const ShT& strides){
    using index_type = typename ShT::value_type;
    return shape.empty() ? index_type(0) : shape.front()*strides.front();
}
/*get size not taking strides into account*/
template<typename ShT>
inline auto make_size(const ShT& shape){
    if (shape.size() != 0){    
        ShT::value_type res{1};
        for(const auto& i:shape)
            res*=i;
        return res;
    }else{
        return ShT::value_type(0);
    }
}

template<typename ValT,  typename CfgT> 
class descriptor_strides
{
    using config_type = CfgT;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    shape_type strides_;
protected:
    descriptor_strides() = default;            
    descriptor_strides(const shape_type& shape__):
        strides_{detail::make_strides(shape__)}
    {}
    const auto&  strides()const{return strides_;}
};

}   //end of namespace detail


template<typename ValT, typename CfgT> 
class descriptor_base{    
    using config_type = CfgT;            
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
protected:
    using strides_div_type = typename detail::libdiv_strides_traits<CfgT>::type;    
public:   
    virtual index_type convert(const shape_type& idx)const = 0;
    virtual index_type convert(const index_type& idx)const = 0;
    virtual index_type dim()const = 0;
    virtual index_type size()const = 0;
    virtual index_type offset()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const shape_type& strides()const = 0;
    virtual const strides_div_type& strides_div()const = 0;
    virtual const shape_type& cstrides()const = 0;
    virtual std::string to_str()const = 0;
};

template<typename ValT, typename CfgT>
class basic_descriptor : detail::descriptor_strides<ValT,CfgT>    
{
    using base_strides = detail::descriptor_strides<ValT,CfgT>;
    using config_type = CfgT;
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
protected:
    shape_type shape_;    
    auto size()const{return detail::make_size(shape_,base_strides::strides());}
    auto dim()const{return shape_.size();}
    const auto& shape()const{return shape_;}
    const auto& strides()const{return base_strides::strides();}    
    std::string to_str()const{
        std::stringstream ss{};
        ss<<"("<<[&ss,this](){for(const auto& i : shape()){ss<<i<<",";} return ")";}();
        return ss.str();
    }        
public:
    basic_descriptor() = default;       
    basic_descriptor(const shape_type& shape__):
        base_strides{shape__},
        shape_{shape__}
    {}
    basic_descriptor(shape_type&& shape__):
        base_strides{shape__},
        shape_{std::move(shape__)}
    {}
};


}   //end of namespace gtensor

#endif