#ifndef TENSOR_IMPLEMENTATION_HPP_
#define TENSOR_IMPLEMENTATION_HPP_

#include <type_traits>
#include "common.hpp"
#include "tensor_init_list.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"

namespace gtensor{

namespace detail{

GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(descriptor, decltype(std::declval<T&>().descriptor())(T::*)(), has_descriptor);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(descriptor, decltype(std::declval<const T&>().descriptor())(T::*)()const, has_descriptor_const);

GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(operator[], decltype(std::declval<T&>()[std::declval<typename T::size_type>()])(T::*)(typename T::size_type), has_subscript_operator);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(operator[], decltype(std::declval<const T&>()[std::declval<typename T::size_type>()])(T::*)(typename T::size_type)const, has_subscript_operator_const);

GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(create_indexer, decltype(std::declval<T&>().create_indexer())(T::*)(), has_create_indexer);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(create_indexer, decltype(std::declval<const T&>().create_indexer())(T::*)()const, has_create_indexer_const);

GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(create_walker, decltype(std::declval<T&>().create_walker())(T::*)(), has_create_walker);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(create_walker, decltype(std::declval<const T&>().create_walker())(T::*)()const, has_create_walker_const);

GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(begin, decltype(std::declval<T&>().begin())(T::*)(), has_begin);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(end, decltype(std::declval<T&>().end())(T::*)(), has_end);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(rbegin, decltype(std::declval<T&>().rbegin())(T::*)(), has_rbegin);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(rend, decltype(std::declval<T&>().rend())(T::*)(), has_rend);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(begin, decltype(std::declval<const T&>().begin())(T::*)()const, has_begin_const);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(end, decltype(std::declval<const T&>().end())(T::*)()const, has_end_const);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(rbegin, decltype(std::declval<const T&>().rbegin())(T::*)()const, has_rbegin_const);
GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(rend, decltype(std::declval<const T&>().rend())(T::*)()const, has_rend_const);

template<typename T> inline constexpr bool has_iterator_v = has_begin<T>()&&has_end<T>();
template<typename T> inline constexpr bool has_const_iterator_v = has_begin_const<T>()&&has_end_const<T>();
template<typename T> inline constexpr bool has_reverse_iterator_v = has_rbegin<T>()&&has_rend<T>();
template<typename T> inline constexpr bool has_const_reverse_iterator_v = has_rbegin_const<T>()&&has_rend_const<T>();

template<typename T> inline constexpr bool is_iterable_v = has_iterator_v<T>||has_const_iterator_v<T>;
template<typename T> inline constexpr bool is_subscriptable_v = has_subscript_operator<T>()||has_subscript_operator_const<T>();

template<typename T> inline constexpr bool is_random_access_iterator_v = std::is_convertible_v<typename std::iterator_traits<T>::iterator_category, std::random_access_iterator_tag>;

template<typename T, typename=void> inline constexpr bool has_mutating_iterator_v = false;
template<typename T> inline constexpr bool has_mutating_iterator_v<T,std::void_t< std::enable_if_t<std::is_assignable_v<decltype(*std::declval<T&>().begin()), typename T::value_type> > > > = true;

template<typename T, typename=void> inline constexpr bool has_mutating_subscript_operator_v = false;
template<typename T> inline constexpr bool has_mutating_subscript_operator_v<T,std::void_t< std::enable_if_t<std::is_assignable_v<decltype(std::declval<T&>()[std::declval<typename T::difference_type&>()]), typename T::value_type> > > > = true;

template<typename T> struct subscript_operator_result{
    template<typename,typename> struct selector_;
    template<typename Dummy> struct selector_<std::true_type,Dummy>{using type = decltype(std::declval<T&>()[std::declval<typename T::difference_type&>()]);};
    template<typename Dummy> struct selector_<std::false_type,Dummy>{using type = void;};
    using type = typename selector_<typename has_subscript_operator<T>::type,void>::type;
};
template<typename T> using subscript_operator_result_t = typename subscript_operator_result<T>::type;

template<typename T> struct subscript_operator_const_result{
    template<typename,typename> struct selector_;
    template<typename Dummy> struct selector_<std::true_type,Dummy>{using type = decltype(std::declval<const T&>()[std::declval<typename T::difference_type&>()]);};
    template<typename Dummy> struct selector_<std::false_type,Dummy>{using type = void;};
    using type = typename selector_<typename has_subscript_operator_const<T>::type,void>::type;
};
template<typename T> using subscript_operator_const_result_t = typename subscript_operator_const_result<T>::type;


//create indexer using available data accessors
template<typename T>
inline auto create_indexer(T& t){
    if constexpr (has_create_indexer<T>::value){
        return t.create_indexer();
    }else if constexpr (has_subscript_operator<T>::value){
        return gtensor::basic_indexer<T&>{t};
    }else if constexpr (has_create_walker<T>::value){
        //need walker indexer adapter
        //return ...
    }else if constexpr (has_iterator_v<T>){
        //need iterator indexer adapter
        //retrun ...
    }else{
        return;
    }
}
template<typename T>
inline auto create_const_indexer(const T& t){
    if constexpr (has_create_indexer_const<T>::value){
        return t.create_indexer();
    }else if constexpr (has_subscript_operator_const<T>::value){
        return gtensor::basic_indexer<const T&>{t};
    }else if constexpr (has_create_walker_const<T>::value){
        //need walker indexer adapter
        //return ...
    }else if constexpr (has_const_iterator_v<T>){
        //need iterator indexer adapter
        //retrun ...
    }else{
        return;
    }
}

//create walker using avilable data accessors
template<typename Config, typename T, typename Descriptor>
inline auto create_walker(T& t, const Descriptor& descriptor){
    if constexpr(has_create_walker<T>::value){
        return t.create_walker();
    }else if constexpr (has_create_indexer<T>::value){
        return gtensor::walker<Config, decltype(t.create_indexer())>{descriptor.adapted_strides(),descriptor.reset_strides(),descriptor.offset(),t.create_indexer(),descriptor.dim()};
    }else if constexpr (has_subscript_operator<T>::value){
        using indexer_type = gtensor::basic_indexer<T&>;
        return gtensor::walker<Config, indexer_type>{descriptor.adapted_strides(),descriptor.reset_strides(),descriptor.offset(),indexer_type{t},descriptor.dim()};
    }else if constexpr (has_iterator_v<T>){
        //need iterator indexer adapter
        //return...
    }else{
        return;
    }
}
template<typename Config, typename T, typename Descriptor>
inline auto create_const_walker(const T& t, const Descriptor& descriptor){
    if constexpr(has_create_walker_const<T>::value){
        return t.create_walker();
    }else if constexpr (has_create_indexer_const<T>::value){
        return gtensor::walker<Config, decltype(t.create_indexer())>{descriptor.adapted_strides(),descriptor.reset_strides(),descriptor.offset(),t.create_indexer(),descriptor.dim()};
    }else if constexpr (has_subscript_operator_const<T>::value){
        using indexer_type = gtensor::basic_indexer<const T&>;
        return gtensor::walker<Config, indexer_type>{descriptor.adapted_strides(),descriptor.reset_strides(),descriptor.offset(),indexer_type{t},descriptor.dim()};
    }else if constexpr (has_const_iterator_v<T>){
        //need iterator indexer adapter
        //return...
    }else{
        return;
    }
}

//create iterator using avilable data accessors
template<typename Config, typename T, typename Descriptor>
inline auto begin(T& t, const Descriptor& descriptor){
    using index_type = typename Config::index_type;
    if constexpr (has_iterator_v<T>){
        return t.begin();
    }else if constexpr (has_create_indexer<T>::value){
        return gtensor::indexer_iterator<Config, decltype(t.create_indexer())>{t.create_indexer(), index_type{0}};
    }else if constexpr (has_subscript_operator<T>::value){
        using indexer_type = gtensor::basic_indexer<T&>;
        return gtensor::indexer_iterator<Config, indexer_type>{indexer_type{t}, index_type{0}};
    }else if constexpr(has_create_walker<T>::value){
        return gtensor::walker_iterator<Config, decltype(t.create_walker())>{t.create_walker(), descriptor.shape(), descriptor.strides_div(), index_type{0}};
    }else{
        return;
    }
}
template<typename Config, typename T, typename Descriptor>
inline auto end(T& t, const Descriptor& descriptor){
    if constexpr (has_iterator_v<T>){
        return t.end();
    }else if constexpr (has_create_indexer<T>::value){
        return gtensor::indexer_iterator<Config, decltype(t.create_indexer())>{t.create_indexer(), descriptor.size()};
    }else if constexpr (has_subscript_operator<T>::value){
        using indexer_type = gtensor::basic_indexer<T&>;
        return gtensor::indexer_iterator<Config, indexer_type>{indexer_type{t}, descriptor.size()};
    }else if constexpr(has_create_walker<T>::value){
        return gtensor::walker_iterator<Config, decltype(t.create_walker())>{t.create_walker(), descriptor.shape(), descriptor.strides_div(), descriptor.size()};
    }else{
        return;
    }
}
template<typename Config, typename T, typename Descriptor>
inline auto begin_const(const T& t, const Descriptor& descriptor){
    using index_type = typename Config::index_type;
    if constexpr (has_const_iterator_v<T>){
        return t.begin();
    }else if constexpr (has_create_indexer_const<T>::value){
        return gtensor::indexer_iterator<Config, decltype(t.create_indexer())>{t.create_indexer(), index_type{0}};
    }else if constexpr (has_subscript_operator_const<T>::value){
        using indexer_type = gtensor::basic_indexer<const T&>;
        return gtensor::indexer_iterator<Config, indexer_type>{indexer_type{t}, index_type{0}};
    }else if constexpr(has_create_walker_const<T>::value){
        return gtensor::walker_iterator<Config, decltype(t.create_walker())>{t.create_walker(), descriptor.shape(), descriptor.strides_div(), index_type{0}};
    }else{
        return;
    }
}
template<typename Config, typename T, typename Descriptor>
inline auto end_const(const T& t, const Descriptor& descriptor){
    if constexpr (has_const_iterator_v<T>){
        return t.end();
    }else if constexpr (has_create_indexer_const<T>::value){
        return gtensor::indexer_iterator<Config, decltype(t.create_indexer())>{t.create_indexer(), descriptor.size()};
    }else if constexpr (has_subscript_operator_const<T>::value){
        using indexer_type = gtensor::basic_indexer<const T&>;
        return gtensor::indexer_iterator<Config, indexer_type>{indexer_type{t}, descriptor.size()};
    }else if constexpr(has_create_walker_const<T>::value){
        return gtensor::walker_iterator<Config, decltype(t.create_walker())>{t.create_walker(), descriptor.shape(), descriptor.strides_div(), descriptor.size()};
    }else{
        return;
    }
}

//create reverse iterator using avilable data accessors
template<typename Config, typename T, typename Descriptor>
inline auto rbegin(T& t, const Descriptor& descriptor){
    if constexpr (has_reverse_iterator_v<T>){
        return t.rbegin();
    }else if constexpr (has_create_indexer<T>::value){
        return gtensor::reverse_indexer_iterator<Config, decltype(t.create_indexer())>{t.create_indexer(), descriptor.size()};
    }else if constexpr (has_subscript_operator<T>::value){
        using indexer_type = gtensor::basic_indexer<T&>;
        return gtensor::reverse_indexer_iterator<Config, indexer_type>{indexer_type{t}, descriptor.size()};
    }else if constexpr(has_create_walker<T>::value){
        return gtensor::reverse_walker_iterator<Config, decltype(t.create_walker())>{t.create_walker(), descriptor.shape(), descriptor.strides_div(), descriptor.size()};
    }else{
        return;
    }
}
template<typename Config, typename T, typename Descriptor>
inline auto rend(T& t, const Descriptor& descriptor){
    using index_type = typename Config::index_type;
    if constexpr (has_reverse_iterator_v<T>){
        return t.rend();
    }else if constexpr (has_create_indexer<T>::value){
        return gtensor::reverse_indexer_iterator<Config, decltype(t.create_indexer())>{t.create_indexer(), index_type{0}};
    }else if constexpr (has_subscript_operator<T>::value){
        using indexer_type = gtensor::basic_indexer<T&>;
        return gtensor::reverse_indexer_iterator<Config, indexer_type>{indexer_type{t}, index_type{0}};
    }else if constexpr(has_create_walker<T>::value){
        return gtensor::reverse_walker_iterator<Config, decltype(t.create_walker())>{t.create_walker(), descriptor.shape(), descriptor.strides_div(), index_type{0}};
    }else{
        return;
    }
}



}   //end of namespace detail


//Engine must provide interface to access data and meta-data:
//descriptor() method for meta-data
//create_indexer() or create_walker() or both for data, if both is proveded type alias tag Engine::data_accessor_tag is used to select accessor
//if Engine provide iterators they are used, if not iterators are made using selected data accessor i.e. indexer or walker
template<typename Engine>
class tensor_implementation
{
    //check meta-data and data access intefaces
    static_assert(detail::has_descriptor<Engine>::value);
    //static_assert(detail::has_subscript_operator_v<Engine> || detail::has_indexer_v<Engine> || detail::has_walker_v<Engine> || detail::has_random_access_iterator_v<Engine>);
    using engine_type = Engine;
public:
    using value_type = typename engine_type::value_type;
    using config_type = typename engine_type::config_type;
    using dim_type = typename engine_type::dim_type;
    using index_type = typename engine_type::index_type;
    using shape_type = typename engine_type::shape_type;

    template<typename...Args>
    tensor_implementation(Args&&...args):
        engine_{std::forward<Args>(args)...}
    {}

    tensor_implementation(const tensor_implementation&) = delete;
    tensor_implementation& operator=(const tensor_implementation&) = delete;
    tensor_implementation(tensor_implementation&&) = delete;
    tensor_implementation& operator=(tensor_implementation&&) = delete;

    const auto& descriptor()const{return engine_.descriptor();}
    index_type size()const{return descriptor().size();}
    dim_type dim()const{return descriptor().dim();}
    const shape_type& shape()const{return descriptor().shape();}
    const shape_type& strides()const{return descriptor().strides();}

private:
    engine_type engine_;
};

//storage engine, owns storage of data elements of type T
//storage type is specialization of Config::storage
template<typename Config, typename T>
class storage_engine
{
    using descriptor_type = basic_descriptor<Config>;
    using storage_type = typename Config::template storage<T>;
public:
    using value_type = T;
    using config_type = Config;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using size_type = index_type;
    using difference_type = index_type;

    //if value_type is trivially copiable elements_ may be not initialized, depends on storage_type implementation
    template<typename ShT>
    explicit storage_engine(ShT&& shape):
        descriptor_{std::forward<ShT>(shape)},
        elements_(descriptor_.size())
    {}

    template<typename ShT>
    storage_engine(ShT&& shape, const value_type& v):
        storage_engine(std::forward<ShT>(shape), v, std::is_constructible<storage_type,index_type,value_type>{})
    {}

    template<typename Nested>
    storage_engine(std::initializer_list<Nested> init_data):
        descriptor_{detail::list_parse<dim_type,shape_type>(init_data)},
        elements_(descriptor_.size())
    {
        detail::fill_from_list(init_data, begin_());
    }

    template<typename ShT, typename It>
    storage_engine(ShT&& shape, It first, It last):
        storage_engine(std::forward<ShT>(shape), first, last, std::conjunction<std::is_constructible<storage_type,It,It>, std::is_move_constructible<storage_type> >{})
    {}

    const descriptor_type& descriptor()const{return descriptor_;}

    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_iterator_v<Storage_>,int> =0>
    auto begin(){return elements_.begin();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_iterator_v<Storage_>,int> =0>
    auto end(){return elements_.end();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_const_iterator_v<Storage_>,int> =0>
    auto begin()const{return elements_.begin();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_const_iterator_v<Storage_>,int> =0>
    auto end()const{return elements_.end();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_reverse_iterator_v<Storage_>,int> =0>
    auto rbegin(){return elements_.rbegin();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_reverse_iterator_v<Storage_>,int> =0>
    auto rend(){return elements_.rend();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_const_reverse_iterator_v<Storage_>,int> =0>
    auto rbegin()const{return elements_.rbegin();}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_const_reverse_iterator_v<Storage_>,int> =0>
    auto rend()const{return elements_.rend();}

    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_subscript_operator<Storage_>::value,int> =0>
    detail::subscript_operator_result_t<storage_type> operator[](index_type i){return elements_[i];}
    template<typename Storage_ = storage_type, std::enable_if_t<detail::has_subscript_operator_const<Storage_>::value,int> =0>
    detail::subscript_operator_const_result_t<storage_type> operator[](index_type i)const{return elements_[i];}

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
    storage_engine(ShT&& shape, const value_type& v, std::true_type):
        descriptor_{std::forward<ShT>(shape)},
        elements_(descriptor_.size(),v)
    {}
    //use fill
    template<typename ShT>
    storage_engine(ShT&& shape, const value_type& v, std::false_type):
        descriptor_{std::forward<ShT>(shape)},
        elements_(descriptor_.size())
    {
        std::fill(begin_(),end_(),v);
    }
    //from range constructors
    //try to construct directly from range
    template<typename ShT, typename It>
    storage_engine(ShT&& shape, It first, It last, std::true_type):
        descriptor_{std::forward<ShT>(shape)},
        elements_(construct_from_range(descriptor_.size(), first, last, typename std::iterator_traits<It>::iterator_category{}))
    {}
    //no from range constructor, use fill
    template<typename ShT, typename It>
    storage_engine(ShT&& shape, It& first, It& last, std::false_type):
        descriptor_{std::forward<ShT>(shape)},
        elements_(descriptor_.size())
    {
        fill_from_range(descriptor_.size(), first, last, begin_(), end_(), typename std::iterator_traits<It>::iterator_category{});
    }

    template<typename It, typename DstIt>
    void fill_from_range(index_type size, It& first, It& last, DstIt dst_first, DstIt dst_last, std::random_access_iterator_tag){
        auto d = static_cast<index_type>(std::distance(first,last));
        if (size < d){
            for(;dst_first!=dst_last; ++dst_first,++first){
                *dst_first = *first;
            }
        }else{
            std::copy(first, last, dst_first);
        }
    }

    template<typename It, typename DstIt>
    void fill_from_range(index_type, It& first, It& last, DstIt dst_first, DstIt dst_last, std::input_iterator_tag){
        for(;dst_first!=dst_last && first!=last; ++dst_first,++first){
            *dst_first = *first;
        }
    }

    template<typename It>
    storage_type construct_from_range(index_type size, It& first, It& last, std::random_access_iterator_tag){
        using it_difference_type = typename std::iterator_traits<It>::difference_type;
        auto d = static_cast<index_type>(std::distance(first,last));
        if(size == d){
            return storage_type{first,last};
        }else if(size<d){
            return storage_type{first,first+static_cast<it_difference_type>(size)};
        }else{
            return construct_from_range(size, first, last, std::input_iterator_tag{});
        }
    }

    template<typename It>
    storage_type construct_from_range(index_type size, It& first, It& last, std::input_iterator_tag){
        storage_type res(size);
        fill_from_range(size,first,last,begin_(res),end_(res,size),std::input_iterator_tag{});
        return res;
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


}   //end of namespace gtensor

#endif