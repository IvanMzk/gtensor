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
    template<typename ValT> using storage = std::vector<ValT>;
    //meta-data elements storage template i.e. shape, strides are specialization of shape
    //must provide random access interface
    template<typename IdxT> using shape = std::vector<IdxT>;
    //generally when public interface expected container parameter it may be any type providig usual container semantic and interface: iterators, aliases...
    //specialization of config_type::container uses as return type in public interface
    //it may be used by implementation as general purpose container
    template<typename T> using container = std::vector<T>;
};

template<typename Config, typename IdxT>
struct extended_config : Config{
    //using storage_type = typename CfgT::template storage<ValT>;

    //index_type defines data elements address space:
    //e.g. shape and strides elements are of index_type
    //slice, reshape view subscripts are of index_type
    //must have semantic of signed integral type
    using index_type = IdxT;
    using shape_type = typename Config::template shape<index_type>;
    //used in indexed access to meta-data elements:
    //e.g. index of direction, dimensions number
    //transpose view subscripts are of dim_type, since they are directions indexes
    //must have semantic of integral type
    using dim_type = typename shape_type::size_type;
};
template<typename Config, typename ValT> using extend_config_t = extended_config<Config, typename Config::template storage<ValT>::difference_type>;

}   //end of namespace config
}   //end of namespace gtensor

#endif