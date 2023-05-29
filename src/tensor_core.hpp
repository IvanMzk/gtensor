#ifndef TENSOR_CORE_HPP_
#define TENSOR_CORE_HPP_

#include <type_traits>
#include <iterator>
#include "common.hpp"
#include "tensor_init_list.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"

namespace gtensor{

namespace detail{

template<typename T, typename=void> inline constexpr bool has_mutating_iterator_v = false;
template<typename T> inline constexpr bool has_mutating_iterator_v<T,std::void_t< std::enable_if_t<std::is_assignable_v<decltype(*std::declval<T&>().begin()), typename T::value_type> > > > = true;
template<typename T, typename=void> inline constexpr bool has_mutating_subscript_operator_v = false;
template<typename T> inline constexpr bool has_mutating_subscript_operator_v<T,std::void_t< std::enable_if_t<std::is_assignable_v<decltype(std::declval<T&>()[std::declval<typename T::difference_type&>()]), typename T::value_type> > > > = true;

//copy from range first,last to dst_first
//if n < distance(first,last), copy first n elements
//else copy whole range
template<typename It, typename IdxT, typename DstIt>
void copy_n(It first, It last, IdxT n, DstIt dst_first){
    using index_type = IdxT;
    if constexpr (
        std::is_convertible_v<typename std::iterator_traits<It>::iterator_category,std::random_access_iterator_tag> &&
        std::is_convertible_v<typename std::iterator_traits<It>::difference_type,index_type>
    ){
        auto d = static_cast<index_type>(std::distance(first,last));
        if (n < d){
            for(;n!=index_type{0}; --n,++dst_first,++first){
                *dst_first = *first;
            }
        }else{
            std::copy(first, last, dst_first);
        }
    }else{
        for(;n!=index_type{0} && first!=last;--n,static_cast<void>(++dst_first),static_cast<void>(++first)){
            *dst_first = *first;
        }
    }
}

}   //end of namespace detail

//storage core combine together data and meta-data
//owns descriptor
//owns storage of data elements of type T - Config::storage<T>
//Layout is storage scheme of data elements, may be config::c_order or config::f_order
template<typename Config, typename T, typename Layout>
class storage_core
{
    using descriptor_type = basic_descriptor<Config>;
    using storage_type = typename Config::template storage<T>;
    using layout_type = Layout;
public:
    using value_type = T;
    using config_type = Config;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using size_type = index_type;
    using difference_type = index_type;

    using order_type = layout_type;

    //if value_type is trivially copiable elements_ may be not initialized, depends on storage_type implementation
    template<typename ShT, std::enable_if_t<!std::is_same_v<ShT,storage_core>,int> =0>
    explicit storage_core(ShT&& shape):
        descriptor_(std::forward<ShT>(shape)),
        elements_(descriptor_.size())
    {}

    template<typename ShT>
    storage_core(ShT&& shape, const value_type& v):
        storage_core(std::forward<ShT>(shape), v, std::is_constructible<storage_type,index_type,value_type>{})
    {}

    template<typename Nested>
    explicit storage_core(std::initializer_list<Nested> init_data):
        descriptor_(detail::list_parse<dim_type,shape_type>(init_data)),
        elements_(descriptor_.size())
    {
        copy_from_list(init_data);
    }

    template<typename ShT, typename It>
    storage_core(ShT&& shape, It first, It last):
        storage_core(std::forward<ShT>(shape), first, last, std::conjunction<std::is_constructible<storage_type,It,It>, std::is_move_constructible<storage_type> >{})
    {}

    const descriptor_type& descriptor()const{return descriptor_;}

    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_iterator<Storage_>::value,int> =0>
    auto begin(){return elements_.begin();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_iterator<Storage_>::value,int> =0>
    auto end(){return elements_.end();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_iterator<const Storage_>::value,int> =0>
    auto begin()const{return elements_.begin();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_iterator<const Storage_>::value,int> =0>
    auto end()const{return elements_.end();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_reverse_iterator<Storage_>::value,int> =0>
    auto rbegin(){return elements_.rbegin();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_reverse_iterator<Storage_>::value,int> =0>
    auto rend(){return elements_.rend();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_reverse_iterator<const Storage_>::value,int> =0>
    auto rbegin()const{return elements_.rbegin();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_reverse_iterator<const Storage_>::value,int> =0>
    auto rend()const{return elements_.rend();}

    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_subscript_operator<Storage_>::value,int> =0>
    decltype(auto) operator[](index_type i){return elements_[i];}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_callable_subscript_operator<const Storage_>::value,int> =0>
    decltype(auto) operator[](index_type i)const{return elements_[i];}

    //inplace
    template<typename ShT>
    void resize(ShT&& shape){
        descriptor_ = descriptor_type{std::forward<ShT>(shape)};
        elements_.resize(descriptor_.size());
        elements_.shrink_to_fit();
    }
private:
    //size,value constructors
    //direct construction
    template<typename ShT>
    storage_core(ShT&& shape, const value_type& v, std::true_type):
        descriptor_(std::forward<ShT>(shape)),
        elements_(descriptor_.size(),v)
    {}
    //use fill
    template<typename ShT>
    storage_core(ShT&& shape, const value_type& v, std::false_type):
        descriptor_(std::forward<ShT>(shape)),
        elements_(descriptor_.size())
    {
        detail::fill(begin_(),end_(),v);
    }
    //from range constructors
    //try to construct directly from range and move
    template<typename ShT, typename It>
    storage_core(ShT&& shape, It& first, It& last, std::true_type):
        descriptor_(std::forward<ShT>(shape)),
        elements_(construct_from_range(descriptor_.size(), first, last, typename std::iterator_traits<It>::iterator_category{}))
    {}
    //no from range constructor or no move, use copy_n
    template<typename ShT, typename It>
    storage_core(ShT&& shape, It& first, It& last, std::false_type):
        descriptor_(std::forward<ShT>(shape)),
        elements_(descriptor_.size())
    {
        detail::copy_n(first,last,descriptor_.size(),begin_());
    }

    template<typename Nested>
    void copy_from_list(std::initializer_list<Nested> init_data){   //init_data always in c_layout
        detail::copy_from_list(init_data, begin_(), make_mapper(config::c_layout{}));
    }

    template<typename It>
    storage_type construct_from_range(const index_type& size, It& first, It& last){
        using it_difference_type = typename std::iterator_traits<It>::difference_type;
        if constexpr (detail::is_static_castable_v<it_difference_type,index_type>){
            auto d = static_cast<index_type>(std::distance(first,last));
            if(size == d){
               return storage_type{first,last};
            }else if(size<d){   //range is bigger than size
                if constexpr (detail::is_static_castable_v<index_type,it_difference_type>){
                    return storage_type{first,first+static_cast<it_difference_type>(size)};
                }else{
                    return construct_from_range(size, first, last, std::input_iterator_tag{});
                }
            }else{  //range is smaller than size
                return construct_from_range(size, first, last, std::input_iterator_tag{});
            }
        }else{
            return construct_from_range(size, first, last, std::input_iterator_tag{});
        }
    }

    template<typename It>
    storage_type construct_from_range(const index_type& size, It& first, It& last, std::input_iterator_tag){
        storage_type res(size);
        detail::copy_n(first,last,size,begin_(res));
        return res;
    }

    //parameter is layout of input
    auto make_mapper(config::c_layout){
        if constexpr (std::is_same_v<layout_type,config::c_layout>){    //input in my order
            return detail::trivial_mapper{};
        }else{  //mapper from c order to f order
            return [strides=detail::make_strides(descriptor_.shape(), config::c_layout{}), &cstrides=descriptor_.strides()](const auto& i){
                return detail::flat_to_flat(strides,cstrides,index_type{0},i,config::c_layout{});
            };
        }
    }
    auto make_mapper(config::f_layout){
        if constexpr (std::is_same_v<layout_type,config::f_layout>){    //input in my order
            return detail::trivial_mapper{};
        }else{  //mapper from f order to c order
            return [strides=detail::make_strides(descriptor_.shape(), config::f_layout{}), &cstrides=descriptor_.strides()](const auto& i){
                return detail::flat_to_flat(strides,cstrides,index_type{0},i,config::f_layout{});
            };
        }
    }

    auto begin_(storage_type& elements__){
        if constexpr (detail::has_mutating_iterator_v<storage_type>){
            return elements__.begin();
        }else if constexpr (detail::has_mutating_subscript_operator_v<storage_type>){
            using indexer_type = basic_indexer<storage_type&>;
            return indexer_iterator<config_type, indexer_type>{indexer_type{elements__},index_type{0}};
        }
    }
    auto end_(storage_type& elements__, index_type size__){
        if constexpr (detail::has_mutating_iterator_v<storage_type>){
            return elements__.end();
        }else if constexpr (detail::has_mutating_subscript_operator_v<storage_type>){
            using indexer_type = basic_indexer<storage_type&>;
            return indexer_iterator<config_type, indexer_type>{indexer_type{elements__},size__};
        }
    }
    auto begin_(){return begin_(elements_);}
    auto end_(){return end_(elements_,descriptor_.size());}

    descriptor_type descriptor_;
    storage_type elements_;
};

//view core owns its parent and provide data accessor to its data
//Descriptor depends on kind of view
//Parent is type of view parent(origin) i.e. it is basic_tensor specialization
template<typename Config, typename Descriptor, typename Parent>
class view_core
{
    using descriptor_type = Descriptor;
    using parent_type = Parent;
public:
    using config_type = Config;
    using value_type = typename Parent::value_type;

    template<typename Descriptor_, typename Parent_>
    view_core(Descriptor_&& descriptor__, Parent_&& parent__):
        descriptor_{std::forward<Descriptor_>(descriptor__)},
        parent_{std::forward<Parent_>(parent__)}
    {}

    const descriptor_type& descriptor()const{return descriptor_;}
    auto create_indexer()const{return create_indexer_helper(*this);}
    auto create_indexer(){return create_indexer_helper(*this);}
private:
    template<typename U>
    static auto create_indexer_helper(U& instance){
        return basic_indexer<decltype(instance.parent_.create_indexer()), descriptor_type>{
            instance.parent_.create_indexer(),
            static_cast<const descriptor_type&>(instance.descriptor_)
        };
    }
    descriptor_type descriptor_;
    parent_type parent_;
};

}   //end of namespace gtensor

#endif