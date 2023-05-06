#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <vector>
#include "storage_adapter.hpp"

namespace gtensor{
namespace config{

enum class div_modes : std::size_t {native, libdivide};
enum class engines : std::size_t {expression_template};

template<typename T, T M> struct tag{static constexpr T value = M;};

using mode_div_native = tag<div_modes, div_modes::native>;
using mode_div_libdivide = tag<div_modes, div_modes::libdivide>;
using engine_expression_template = tag<engines, engines::expression_template>;

struct default_config
{
    using engine = engine_expression_template;
    using div_mode = mode_div_libdivide;
    //using div_mode = mode_div_native;

    //data elements storage template
    //must provide random access interface
    template<typename T> using storage = std::vector<T>;
    //meta-data elements storage template i.e. shape, strides are specialization of shape
    //must provide random access interface
    template<typename T> using shape = std::vector<T>;
    //generally when public interface expected container parameter it may be any type providig usual container semantic and interface: iterators, aliases...
    //specialization of config_type::container uses as return type in public interface
    //it may be used by implementation as general purpose container
    template<typename T> using container = std::vector<T>;
};

template<typename Config, typename IdxT>
struct extended_config{

    using config_type = Config;
    using engine = typename config_type::engine;
    using div_mode = typename config_type::div_mode;
    template<typename T> using storage = typename config_type::template storage<T>;
    template<typename T> using shape = typename config_type::template shape<T>;
    template<typename T> using container = typename config_type::template container<T>;

    //index_type defines data elements address space:
    //e.g. shape and strides elements are of index_type
    //slice, reshape view subscripts are of index_type
    //must have semantic of signed integral type
    using index_type = IdxT;
    using shape_type = shape<index_type>;
    //used in indexed access to meta-data elements:
    //e.g. index of direction, dimensions number
    //transpose view subscripts are of dim_type, since they are directions indexes
    //must have semantic of integral type
    using dim_type = typename shape_type::size_type;
};
template<typename Config, typename T, typename=void> struct extend_config{
    static_assert(!std::is_void_v<T>);
    using type = extended_config<Config, typename Config::template storage<T>::difference_type>;
};
template<typename Config, typename T> struct extend_config<Config,T,std::void_t<typename Config::config_type>>{
    template<typename, typename> struct selector_;
    template<typename Dummy> struct selector_<std::true_type,Dummy>{
        using type = Config;
    };
    template<typename Dummy> struct selector_<std::false_type,Dummy>{
        using type = extended_config<typename Config::config_type, typename Config::template storage<T>::difference_type>;
    };
    using type = typename selector_<typename std::is_void<T>::type,void>::type;
};
template<typename Config, typename T> using extend_config_t = typename extend_config<Config,T>::type;
template<typename T, typename=void> constexpr bool is_extended_config_v = false;
template<typename T> constexpr bool is_extended_config_v<T,std::void_t<typename T::index_type, typename T::shape_type, typename T::dim_type>> = true;

}   //end of namespace config
}   //end of namespace gtensor

#endif