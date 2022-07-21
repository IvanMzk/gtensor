#ifndef VIEW_DESCRIPTOR_BASE_HPP_
#define VIEW_DESCRIPTOR_BASE_HPP_

namespace gtensor{

template<typename ValT, template<typename> typename Cfg> 
class view_descriptor_base{    
    using config_type = Cfg<ValT>;            
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    
public:   
    virtual index_type convert_by_prev(const index_type& idx)const = 0;
    virtual index_type convert(const shape_type& idx)const = 0;
    virtual index_type convert(const index_type& idx)const = 0;
    virtual index_type dim()const = 0;
    virtual index_type size()const = 0;
    virtual index_type offset()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const shape_type& strides()const = 0;    
    virtual const shape_type& cstrides()const = 0;
    virtual std::string to_str()const = 0;        
};

}   //end of namespace gtensor

#endif