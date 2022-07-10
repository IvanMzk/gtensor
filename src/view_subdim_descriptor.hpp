#ifndef VIEW_SUBDIM_DESCRIPTOR_HPP_
#define VIEW_SUBDIM_DESCRIPTOR_HPP_

#include <numeric>
#include "stensor_descriptor.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg, typename...PrevT> class view_subdim_descriptor;

/*not view of view subdim and reshape descriptor specialization*/
template<typename ValT, template<typename> typename Cfg> 
class view_subdim_descriptor<ValT, Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    shape_type shape_;
    index_type offset_;
    shape_type strides_{gtensor::detail::make_strides(shape_)};

public:
    template<typename ShT>
    view_subdim_descriptor(ShT&& shape__, index_type offset__):
        shape_{std::forward<ShT>(shape__)},
        offset_{offset__}
    {}

    index_type convert_by_prev(const index_type& idx)const{return idx;}
    index_type convert(const shape_type& idx)const{return convert_helper(idx);}
    index_type convert(const index_type& idx)const{return idx+offset_;}
    index_type dim()const{return shape_.size();}
    index_type size()const{return detail::make_size(shape_,strides_);}
    index_type offset()const{return offset_;}
    const shape_type& shape()const{return shape_;};
    const shape_type& strides()const{return strides_;};
    const shape_type& cstrides()const{return strides_;};
    std::string to_str()const{
        std::stringstream ss{};
        ss<<"("<<[&ss,this](){for(const auto& i : shape()){ss<<i<<",";} return ")";}();
        return ss.str();
    }

private:
    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), strides_.begin(), offset_);
    }    
};

/*view of view subdim and reshape descriptor specialization*/
template<typename ValT, template<typename> typename Cfg, typename PrevT> 
class view_subdim_descriptor<ValT, Cfg, PrevT> : view_subdim_descriptor<ValT,Cfg>{
    using base_type = view_subdim_descriptor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using prev_descriptor_type = PrevT;
    prev_descriptor_type prev_descriptor;
public:
    template<typename ShT, typename DtT>
    view_subdim_descriptor(ShT&& shape__, index_type offset__, DtT&& prev_descriptor__):
        base_type{std::forward<ShT>(shape__), offset__},
        prev_descriptor{std::forward<DtT>(prev_descriptor__)}
    {}
    
    index_type convert_by_prev(const index_type& idx)const{return prev_descriptor.convert(idx);}
    index_type convert(const shape_type& idx)const{return convert_by_prev(base_type::convert(idx));}
    index_type convert(const index_type& idx)const{return convert_by_prev(base_type::convert(idx));}
    using base_type::dim;
    using base_type::size;
    using base_type::offset;
    using base_type::shape;
    using base_type::strides;
    using base_type::cstrides;
    using base_type::to_str;
};


}   //end of namespace gtensor

#endif