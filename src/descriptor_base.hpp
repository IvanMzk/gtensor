#ifndef DESCRIPTOR_BASE_HPP_
#define DESCRIPTOR_BASE_HPP_

#include <stdexcept>
#include <string>
#include <sstream>
#include "libdivide_helper.hpp"

namespace gtensor{

class broadcast_exception : public std::runtime_error{
    public: broadcast_exception(const char* what):runtime_error(what){}
};

namespace detail{

/*
* create broadcast shape
* parameters: shapes to broadcast
* exception if shapes are not broadcastable
*/
template<typename ShT>
inline ShT broadcast(const ShT& shape1, const ShT& shape2){
    using shape_type = ShT;
    using index_type = typename ShT::value_type;
    if (shape1.size() == 0 || shape2.size() == 0){
        throw broadcast_exception("shapes are not broadcastable");
    }else{
        bool b{shape1.size() < shape2.size()};
        const shape_type& shorter{ b ? shape1 : shape2};
        const shape_type& longer{b ? shape2 : shape1};
        shape_type res(longer.size());    
        auto shorter_begin{shorter.begin()};
        auto shorter_end{shorter.end()};
        auto longer_begin{longer.begin()};
        auto longer_end{longer.end()};
        auto res_end{res.end()};
        while(shorter_begin!=shorter_end){
            const index_type& i{*--shorter_end};
            const index_type& j{*--longer_end};
            if (i==index_type(1)){
                *--res_end = j;
            }
            else if (j==index_type(1) || i==j){
                *--res_end = i;
            }                    
            else{                        
                throw broadcast_exception("shapes are not broadcastable");
            }
        }
        while(longer_begin!=longer_end){
            *--res_end = *--longer_end;
        }
        return res;        
    }
}

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

template<typename ShT>
inline auto make_size(const ShT& shape){
    using index_type = typename ShT::value_type;
    if (shape.size() == 0){
        return index_type(0);
    }else{
        return std::accumulate(shape.begin(),shape.end(),index_type(1),std::multiplies<index_type>{});
    }
}

template<typename CfgT> 
class descriptor_strides
{    
    using shape_type = typename CfgT::shape_type;
    shape_type strides_;
protected:
    descriptor_strides() = default;            
    descriptor_strides(const shape_type& shape__):
        strides_{detail::make_strides(shape__)}
    {}
    const auto&  strides()const{return strides_;}
};

}   //end of namespace detail

//descriptor abstract interface
template<typename CfgT> 
class descriptor_base{        
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
public:   
    virtual index_type convert(const shape_type& idx)const = 0;
    virtual index_type convert(const index_type& idx)const = 0;
    virtual index_type dim()const = 0;
    virtual index_type size()const = 0;
    virtual index_type offset()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const shape_type& strides()const = 0;
    virtual const shape_type& cstrides()const = 0;
    virtual std::string to_str()const = 0;

    virtual const descriptor_with_libdivide<CfgT>* as_descriptor_with_libdivide()const {return nullptr;}
};

//common implementation of descriptor
template<typename CfgT>
class descriptor_common : 
    private detail::descriptor_strides<CfgT>    
{
    using base_strides = detail::descriptor_strides<CfgT>;
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
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
    descriptor_common() = default;       
    descriptor_common(const shape_type& shape__):
        base_strides{shape__},
        shape_{shape__}
    {}
    descriptor_common(shape_type&& shape__):
        base_strides{shape__},
        shape_{std::move(shape__)}
    {}
};


}   //end of namespace gtensor

#endif