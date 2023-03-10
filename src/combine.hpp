#ifndef COMBINE_HPP_
#define COMBINE_HPP_

#include <type_traits>
//#include

namespace gtensor{

namespace detail{

// template<typename T, typename...Ts>
// const auto& first_shape(const T& t, const Ts&...){
//     return t.
// }

template<typename ShT, typename SizeT>
auto make_stack_shape(const ShT& shape, const SizeT& direction, const typename ShT::value_type& tensors_number){
    using size_type = SizeT;
    using shape_type = ShT;
    size_type pdim = shape.size();
    if (pdim == size_type{0}){
        return shape_type{};
    }else{
        shape_type res(pdim+size_type{1});
        std::copy(shape.begin(), shape.begin()+direction, res.begin());
        std::copy(shape.begin()+direction, shape.end(), res.begin()+direction+size_type{1});
        res[direction] = tensors_number;
        return res;
    }
}

}   //end of namespace detail


//join tensors along new direction, tensors must have the same shape
template<typename SizeT, typename T, typename...Ts>
auto stack(const SizeT& direction, const T& t, const Ts&...ts){
    using config_type = typename T::config_type;
    using res_value_type = std::common_type_t<typename T::value_type, typename Ts::value_type...>;
    using res_impl_type = storage_tensor<typename detail::storage_engine_traits<typename config_type::host_engine,config_type,typename config_type::template storage<res_value_type>>::type>;
    if constexpr (sizeof...(Ts) == 0){

    }else{

    }
}


}   //end of namespace gtensor

#endif