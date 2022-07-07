#ifndef WALKER_FACTORY_HPP_
#define WALKER_FACTORY_HPP_

#include <memory>
#include "impl_swalker.hpp"


namespace gtensor{

template<typename ValT, template<typename> typename Cfg>
class walker_factory{
public:
    virtual std::unique_ptr<walker_impl_base<ValT, Cfg>> create_walker()const = 0;
};

template<typename ValT, template<typename> typename Cfg, typename ParamT>
class ewalker_factory{
    std::unique_ptr<walker_impl_base<ValT, Cfg>> create_walker(){
        return nullptr;
    }
};





}   //end of namespace gtensor


#endif