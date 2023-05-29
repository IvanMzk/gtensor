#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <vector>
#include "storage_adapter.hpp"

namespace gtensor{
namespace config{

enum class div_modes : std::size_t {native, libdivide};
enum class engines : std::size_t {expression_template};
enum class orders : std::size_t {c,f};

//template<typename T, T M> struct tag{static constexpr T value = M;};

using mode_div_native = std::integral_constant<div_modes, div_modes::native>;
using mode_div_libdivide = std::integral_constant<div_modes, div_modes::libdivide>;
using engine_expression_template = std::integral_constant<engines, engines::expression_template>;
using c_order = std::integral_constant<orders, orders::c>;
using f_order = std::integral_constant<orders, orders::f>;

struct default_config
{
    using engine = engine_expression_template;
    using div_mode = mode_div_libdivide;
    //using div_mode = mode_div_native;

    //specify storage scheme of data elements - depricated
    //specify default traverse order of iterators
    //using order = c_order;
    using order = f_order;

    //data elements storage template
    //must provide at least storage(const difference_type& n) constructor, which constructs storage of size n
    //must provide at least subscript const operator or const iterator
    //template<typename T> using storage = std::vector<T>;
    template<typename T> using storage = storage_vector<T>;

    //meta-data elements storage template i.e. shape, strides are specialization of shape
    //must provide std::vector like interface
    template<typename T> using shape = std::vector<T>;

    //generally when public interface expected container parameter it may be any type providig usual container semantic and interface: iterators, aliases...
    //specialization of config_type::container uses as return type in public interface
    //it may be used by implementation as general purpose container
    //must provide std::vector like interface
    template<typename T> using container = std::vector<T>;

    //must provide at least index_map(const diference_type& n) constructor, which constructs map of size n
    //must provide subscript interface such that:
    //T& operator[](const U&), where T is index_type - type used to address data elements, U is index_map<T>::difference_type and static_assert(std::is_convertible_v<T,U>) must hold
    //index_type is defined in extended_config as storage<T>::difference_type, where T is data element type (value_type)
    //it means that static_asert(std::is_convertible_v<storage<T>::difference_type, index_map<storage<T>::difference_type>::difference_type>) must hold
    //index_map specialization is used in mapping_descriptor that is descriptor type of mapping_view
    //it is natural to use storage as index_map in general, but if storage is specific e.g. map to file system, these should differ
    template<typename T> using index_map = storage<T>;

};

template<typename Config, typename IdxT>
struct extended_config{

    using config_type = Config;
    using engine = typename config_type::engine;
    using div_mode = typename config_type::div_mode;
    using order = typename config_type::order;
    template<typename T> using storage = typename config_type::template storage<T>;
    template<typename T> using shape = typename config_type::template shape<T>;
    template<typename T> using container = typename config_type::template container<T>;
    template<typename T> using index_map = typename config_type::template index_map<T>;

    //index_type defines data elements address space:
    //e.g. shape and strides elements are of index_type
    //slice, reshape view subscripts are of index_type
    //must have semantic of signed integral type
    using index_type = IdxT;
    using shape_type = shape<index_type>;
    //used in indexed access to meta-data elements:
    //e.g. index of direction, dimensions number
    //transpose view subscripts are of dim_type, since they are directions indexes
    //must have semantic of signed integral type
    using dim_type = typename shape_type::difference_type;
    //index_map_type is used in mapping_descriptor that is descriptor type of mapping_view
    using index_map_type = index_map<index_type>;
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