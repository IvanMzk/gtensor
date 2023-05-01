#ifndef FORWARD_DECL_HPP_
#define FORWARD_DECL_HPP_

namespace gtensor{

namespace config{
struct default_config;
}

template<typename Impl> class basic_tensor;
template<typename T, typename Config> class tensor;

template<typename Config, typename T> class tensor_factory;
class view_factory;
class combiner;
class reducer;

}   //end of namespace gtensor


#endif