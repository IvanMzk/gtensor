#ifndef IMPL_TENSOR_BASE_HPP_
#define IMPL_TENSOR_BASE_HPP_

#include <string>
#include "forward_decl.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg>
class tensor_impl_base{
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
public:
    virtual ~tensor_impl_base(){}    
    virtual index_type size()const = 0;
    virtual index_type dim()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual std::string to_str()const = 0;
    virtual std::unique_ptr<walker_impl_base<ValT,Cfg>> create_walker()const = 0;
};

}   //end of namespace gtensor

#endif