#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include <numeric>
#include "descriptor_base.hpp"

namespace gtensor{

template<typename CfgT>
class basic_descriptor :
    public descriptor_base<CfgT>,
    private descriptor_common<CfgT>
{
public:
    using typename descriptor_base::shape_type;
    using typename descriptor_base::index_type;
    basic_descriptor() = default;
    template<typename ShT>
    basic_descriptor(ShT&& shape__):
        descriptor_common{std::forward<ShT>(shape__)}
    {}

    index_type dim()const override{return descriptor_common::dim();}
    index_type size()const override{return descriptor_common::size();}
    const shape_type& shape()const override{return descriptor_common::shape();}
    const shape_type& strides()const override{return descriptor_common::strides();}
    const shape_type& reset_strides()const override{return descriptor_common::reset_strides();}
    std::string to_str()const override{return descriptor_common::to_str();}

    index_type offset()const override{return index_type{0};}
    const shape_type& cstrides()const override{return strides();}
    const shape_type& reset_cstrides()const override{return reset_strides();}
    index_type convert(const index_type& idx)const override{return idx;}
    index_type convert(const shape_type& idx)const override{return convert_helper(idx);}
private:
    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides().begin(), index_type{0});
    }
};

template<typename CfgT>
class descriptor_with_libdivide :
    public basic_descriptor<CfgT>,
    private detail::collection_libdivide_extension<CfgT,typename CfgT::div_mode>
{
    using base_strides_libdivide = detail::collection_libdivide_extension<CfgT,typename CfgT::div_mode>;


public:
    using typename basic_descriptor::shape_type;
    using typename basic_descriptor::index_type;
    descriptor_with_libdivide() = default;
    template<typename ShT>
    descriptor_with_libdivide(ShT&& shape__):
        basic_descriptor{std::forward<ShT>(shape__)},
        base_strides_libdivide{basic_descriptor::strides()}
    {}

    const descriptor_with_libdivide* as_descriptor_with_libdivide()const {return this;}
    template<typename C=CfgT, std::enable_if_t<detail::is_mode_div_libdivide<C> ,int> =0 >
    const auto& strides_libdivide()const{return base_strides_libdivide::dividers_libdivide();}
    template<typename C=CfgT, std::enable_if_t<detail::is_mode_div_native<C> ,int> =0 >
    const auto& strides_libdivide()const{return basic_descriptor::strides();}
};

template<typename CfgT>
class descriptor_with_offset : public basic_descriptor<CfgT>
{
public:
    using typename basic_descriptor::index_type;
    using typename basic_descriptor::shape_type;
    descriptor_with_offset() = default;
    template<typename ShT>
    descriptor_with_offset(ShT&& shape__, index_type offset__):
        basic_descriptor{std::forward<ShT>(shape__)},
        offset_{offset__}
    {}
    index_type offset()const override{return offset_;}
    index_type convert(const shape_type& idx)const override{return convert_helper(idx);}
    index_type convert(const index_type& idx)const override{return idx+offset_;}
private:
    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides().begin(), offset_);
    }

    index_type offset_;
};

template<typename CfgT>
class converting_descriptor : public descriptor_with_libdivide<CfgT>
{
public:
    using typename descriptor_with_libdivide::index_type;
    using typename descriptor_with_libdivide::shape_type;
    template<typename ShT, typename StT>
    converting_descriptor(ShT&& shape__, StT&& cstrides__,  const index_type& offset__):
        descriptor_with_libdivide{std::forward<ShT>(shape__)},
        cstrides_{std::forward<StT>(cstrides__)},
        reset_cstrides_{detail::make_reset_strides(descriptor_with_libdivide::shape(),cstrides_)},
        offset_{offset__}
    {}
    index_type offset()const override{return offset_;}
    const shape_type& cstrides()const override{return cstrides_;}
    const shape_type& reset_cstrides()const override{return reset_cstrides_;}
    index_type convert(const shape_type& idx)const override{return convert_helper(idx);}
    index_type convert(const index_type& idx)const override{return convert_helper(idx);}
private:
    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides_.begin(), offset_);
    }
    template<typename C = CfgT, std::enable_if_t<detail::is_mode_div_libdivide<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi<shape_type>(strides_libdivide(), idx));
    }
    template<typename C = CfgT, std::enable_if_t<detail::is_mode_div_native<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi(strides(), idx));
    }

    shape_type cstrides_;
    shape_type reset_cstrides_;
    index_type offset_;
};


}   //end of namespace gtensor

#endif