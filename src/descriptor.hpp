#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include "descriptor_base.hpp"

namespace gtensor{

template<typename CfgT>
class basic_descriptor : public descriptor_base<CfgT>
{
    descriptor_common<CfgT> impl_;
public:
    using typename descriptor_base::config_type;
    using typename descriptor_base::index_type;
    using typename descriptor_base::shape_type;
    using typename descriptor_base::strides_div_type;
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
    template<typename ShT, typename MapT>
    mapping_descriptor(ShT&& shape__, MapT&& index_map__):
        basic_descriptor_base{std::forward<ShT>(shape__)},
        index_map_{std::forward<MapT>(index_map__)}
    {}
    index_type convert(const index_type& idx)const override{return index_map_[idx];}
    index_type convert(const shape_type& idx)const override{return index_map_[detail::convert_index(basic_descriptor_base::cstrides(),basic_descriptor_base::offset(),idx)];}
private:
    map_type index_map_;
};

}   //end of namespace gtensor

#endif