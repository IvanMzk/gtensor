#ifndef VIEW_DESCRIPTOR_HPP_
#define VIEW_DESCRIPTOR_HPP_

#include <numeric>
#include "stensor_descriptor.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg, typename...PrevT> class view_slice_descriptor;

/*not view of view descriptor specialization*/
template<typename ValT, template<typename> typename Cfg> 
class view_slice_descriptor<ValT, Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using libdiv_vector_type = decltype(gtensor::detail::make_libdive_vector<config_type>(std::declval<shape_type>()));

    shape_type shape_;
    shape_type cstrides_;
    index_type offset_;
    shape_type strides_{gtensor::detail::make_strides(shape_)};
    libdiv_vector_type strides_libdiv_{detail::make_libdive_vector<config_type>(strides_)};

public:
    template<typename ShT, typename StT>
    view_slice_descriptor(ShT&& shape__, StT&& cstrides__,  index_type offset__):
        shape_{std::forward<ShT>(shape__)},
        cstrides_{std::forward<StT>(cstrides__)},
        offset_{offset__}
    {}

    index_type convert(const shape_type& idx)const{return convert_helper(idx);}
    index_type convert(const index_type& idx)const{return convert_helper(idx);}

private:
    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides_.begin(), offset_);
    }
    template<typename C = config_type, std::enable_if_t<detail::is_mode_div_libdivide<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi<shape_type>(strides_libdiv_, idx));
    }
    template<typename C = config_type, std::enable_if_t<detail::is_mode_div_native<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi(strides_, idx));
    }
};

/*view of view descriptor specialization*/
template<typename ValT, template<typename> typename Cfg, typename PrevT> 
class view_slice_descriptor<ValT, Cfg, PrevT> : view_slice_descriptor<ValT,Cfg>{
    using base_type = view_slice_descriptor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using prev_descriptor_type = PrevT;
    prev_descriptor_type prev_descriptor;
public:
    template<typename ShT, typename StT, typename DtT>
    view_slice_descriptor(ShT&& shape__, StT&& cstrides__,  index_type offset__, DtT&& prev_descriptor__):
        base_type{std::forward<ShT>(shape__), std::forward<StT>(cstrides__), offset__},
        prev_descriptor{std::forward<DtT>(prev_descriptor__)}
    {}

    index_type convert(const shape_type& idx)const{return prev_descriptor.convert(base_type::convert(idx));}
    index_type convert(const index_type& idx)const{return prev_descriptor.convert(base_type::convert(idx));}
};


}   //end of namespace gtensor

#endif