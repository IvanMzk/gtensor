#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <vector>
#include "uvector.h"
#include "forward_decl.hpp"

namespace gtensor{
namespace config{

enum class div_modes : std::size_t {native, libdivide};
enum class engines : std::size_t {expression_template,cuda};

template<typename T, T M> struct tag{static constexpr T value = M;};

using mode_div_native = tag<div_modes, div_modes::native>;
using mode_div_libdivide = tag<div_modes, div_modes::libdivide>;

using engine_expression_template = tag<engines, engines::expression_template>;
using engine_cuda = tag<engines, engines::cuda>;

struct default_config{

    //using engine = engine_expression_template;
    using host_engine = engine_expression_template;
    using device_engine = engine_cuda;

    using div_mode = mode_div_libdivide;
    //using div_mode = mode_div_native;

    using difference_type = std::int64_t;
    using index_type = difference_type;
    //template<typename ValT> using storage = gtensor::detail::shareable_storage<std::vector<ValT>>;
    template<typename ValT> using storage = std::vector<ValT>;
    //template<typename ValT> using storage = trivial_type_vector::uvector<ValT>;
    //using storage_type = gtensor::detail::shareable_storage<std::vector<value_type>>;
    //using shape_type = trivial_type_vector::uvector<index_type>;
    using shape_type = std::vector<index_type>;
};

}   //end of namespace config
}   //end of namespace gtensor

#endif