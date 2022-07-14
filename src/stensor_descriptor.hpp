#ifndef TENSOR_SLICE_HPP_
#define TENSOR_SLICE_HPP_

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
ShT make_strides(const ShT& shape, typename ShT::value_type min_stride = ShT::value_type(1)){
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

template<typename ValT,  template<typename> typename Cfg, typename Mode> class descriptor_strides;

template<typename ValT,  template<typename> typename Cfg> 
class descriptor_strides<ValT,Cfg,config::mode_div_libdivide>{
    using config_type = Cfg<ValT>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    shape_type strides_;
    detail::libdivide_vector<index_type> strides_libdivide_;
protected:
    descriptor_strides() = default;            
    descriptor_strides(const shape_type& shape__):
        strides_{detail::make_strides(shape__)},
        strides_libdivide_{detail::make_libdiv_vector_helper<libdivide::divider>(strides_)}
    {}
    const auto&  strides_libdivide()const{return strides_libdivide_;}
    const auto&  strides()const{return strides_;}
};

template<typename ValT,  template<typename> typename Cfg> 
class descriptor_strides<ValT,Cfg,config::mode_div_native>{
    using config_type = Cfg<ValT>;
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



template<typename ValT, template<typename> typename Cfg>
class stensor_descriptor : detail::descriptor_strides<ValT,Cfg,typename Cfg<ValT>::div_mode>{
    using base_strides = detail::descriptor_strides<ValT,Cfg,typename Cfg<ValT>::div_mode>;
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    shape_type shape_;
    index_type size_{detail::make_size(shape_,base_strides::strides())};
public:
    stensor_descriptor() = default;       
    stensor_descriptor(const shape_type& shape__):
        base_strides{shape__},
        shape_{shape__}
    {}
    stensor_descriptor(shape_type&& shape__):
        base_strides{shape__},
        shape_{std::move(shape__)}
    {}
    auto size()const{return size_;}
    auto dim()const{return shape_.size();}
    const auto& shape()const{return shape_;}
    const auto& strides()const{return base_strides::strides();}
    //template<typename C = config_type, std::enable_if_t<detail::is_mode_div_native<C> ,int> =0 >
    const auto& strides_libdivide()const{return base_strides::strides_libdivide();}
    std::string to_str()const{
        std::stringstream ss{};
        ss<<"("<<[&ss,this](){for(const auto& i : shape()){ss<<i<<",";} return ")";}();
        return ss.str();
    }        
};




}   //end of namespace gtensor
#endif