#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <vector>
#include "forward_decl.hpp"

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
    using difference_type = std::int64_t;
    using index_type = difference_type;
    //using index_type = std::size_t;
    template<typename ValT> using storage = std::vector<ValT>;
    using shape_type = std::vector<index_type>;
};

}   //end of namespace config
}   //end of namespace gtensor

#endif