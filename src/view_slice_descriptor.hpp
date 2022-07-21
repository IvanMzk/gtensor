#ifndef VIEW_SLICE_DESCRIPTOR_HPP_
#define VIEW_SLICE_DESCRIPTOR_HPP_

#include <numeric>
#include <memory>
#include "stensor_descriptor.hpp"
#include "view_descriptor_base.hpp"

namespace gtensor{

/*not view of view slice and transpose descriptor specialization*/
template<typename ValT, template<typename> typename Cfg> 
class view_slice_descriptor : 
    private detail::descriptor_strides<ValT,Cfg,typename Cfg<ValT>::div_mode>,
    public view_descriptor_base<ValT,Cfg>
{
    using base_strides = detail::descriptor_strides<ValT,Cfg,typename Cfg<ValT>::div_mode>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

    shape_type shape_;
    shape_type cstrides_;
    index_type offset_;    

public:
    template<typename ShT, typename StT>
    view_slice_descriptor(ShT&& shape__, StT&& cstrides__,  index_type offset__):
        base_strides{shape__},
        shape_{std::forward<ShT>(shape__)},
        cstrides_{std::forward<StT>(cstrides__)},
        offset_{offset__}
    {}
    
    index_type convert_by_prev(const index_type& idx)const override{return idx;}
    index_type convert(const shape_type& idx)const override{return convert_helper(idx);}
    index_type convert(const index_type& idx)const override{return convert_helper(idx);}
    index_type dim()const override{return shape().size();}
    index_type size()const override{return detail::make_size(shape(),strides());}
    index_type offset()const override{return offset_;}
    const shape_type& shape()const override{return shape_;};
    const shape_type& strides()const override{return base_strides::strides();};
    const shape_type& cstrides()const override{return cstrides_;};
    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"("<<[&ss,this](){for(const auto& i : shape()){ss<<i<<",";} return ")";}();
        return ss.str();
    }        

private:
    const auto& strides_libdivide()const{return base_strides::strides_libdivide();}
    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides_.begin(), offset_);
    }
    template<typename C = config_type, std::enable_if_t<detail::is_mode_div_libdivide<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi<shape_type>(strides_libdivide(), idx));
    }
    template<typename C = config_type, std::enable_if_t<detail::is_mode_div_native<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi(strides(), idx));
    }
};

/*view of view slice and transpose descriptor specialization*/
template<typename ValT, template<typename> typename Cfg> 
class view_view_slice_descriptor : public view_slice_descriptor<ValT,Cfg>
{
    using base_type = view_slice_descriptor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    std::shared_ptr<view_descriptor_base> prev_descriptor;
public:
    template<typename ShT, typename StT, typename DtT>
    view_view_slice_descriptor(ShT&& shape__, StT&& cstrides__,  index_type offset__, DtT&& prev_descriptor__):
        base_type{std::forward<ShT>(shape__), std::forward<StT>(cstrides__), offset__},
        prev_descriptor{std::forward<DtT>(prev_descriptor__)}
    {}
    
    index_type convert_by_prev(const index_type& idx)const{return prev_descriptor->convert(idx);}
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