#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <vector>
#include "forward_decl.hpp"
#include "integral_type.hpp"
#include "index_type_adapter.hpp"

namespace gtensor{
namespace config{

enum class div_modes : std::size_t {native, libdivide};
enum class engines : std::size_t {expression_template};

template<typename T, T M> struct tag{static constexpr T value = M;};

using mode_div_native = tag<div_modes, div_modes::native>;
using mode_div_libdivide = tag<div_modes, div_modes::libdivide>;
using engine_expression_template = tag<engines, engines::expression_template>;

struct default_config{
    using host_engine = engine_expression_template;
    using div_mode = mode_div_libdivide;
    //using div_mode = mode_div_native;

    //index_type defines data elements address space
    //used in indexed access to data elements:
    //  index_type must be convertible to storage<value_type>::iterator::difference_type
    //shape and strides elements are of index_type
    //slice, reshape, subdim, mapping view subscripts are of index_type
    //must have semantic of signed integral type
    //using index_type = std::int64_t;
    using index_type = integral<std::int64_t>;

    //data elements storage template
    //must provide random access interface
    //template<typename ValT> using storage = std::vector<ValT>;
    template<typename ValT> using storage = storage_adapter<std::vector<ValT>, index_type>;


    //meta-data elements storage type i.e. shape, strides are of shape_type
    //must provide random access interface
    using shape_type = std::vector<index_type>;

    //used in indexed access to meta-data elements e.g. index of direction, directions number
    //transpose view subscripts are of size_type, since they are directions indexes
    //must have semantic of integral type, may be unsigned
    //using size_type = std::int64_t;
    using size_type = std::size_t;
};

}   //end of namespace config
}   //end of namespace gtensor

#endif