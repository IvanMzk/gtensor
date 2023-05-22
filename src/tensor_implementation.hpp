#ifndef TENSOR_IMPLEMENTATION_HPP_
#define TENSOR_IMPLEMENTATION_HPP_

#include <type_traits>
#include <iterator>
#include "common.hpp"
#include "tensor_init_list.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"

namespace gtensor{

namespace detail{

#define GENERATE_HAS_CALLABLE_METHOD(call_expression,trait_name)\
template<typename,typename=void> struct trait_name : std::false_type{};\
template<typename T> struct trait_name<T,std::void_t<decltype(std::declval<T>().call_expression)>> : std::true_type{};

GENERATE_HAS_CALLABLE_METHOD(operator[](std::declval<typename T::size_type>()), has_callable_subscript_operator_size_type);
GENERATE_HAS_CALLABLE_METHOD(operator[](std::declval<typename T::difference_type>()), has_callable_subscript_operator_difference_type);
GENERATE_HAS_CALLABLE_METHOD(operator[](std::declval<typename T::index_type>()), has_callable_subscript_operator_index_type);
GENERATE_HAS_CALLABLE_METHOD(create_indexer(), has_callable_create_indexer);
GENERATE_HAS_CALLABLE_METHOD(create_walker(std::declval<typename T::dim_type>()), has_callable_create_walker);
GENERATE_HAS_CALLABLE_METHOD(begin(), has_callable_begin);
GENERATE_HAS_CALLABLE_METHOD(end(), has_callable_end);
GENERATE_HAS_CALLABLE_METHOD(rbegin(), has_callable_rbegin);
GENERATE_HAS_CALLABLE_METHOD(rend(), has_callable_rend);

template<typename T> using has_callable_subscript_operator = std::disjunction<
    has_callable_subscript_operator_difference_type<T>,
    has_callable_subscript_operator_size_type<T>,
    has_callable_subscript_operator_index_type<T>
>;

template<typename T> using has_callable_iterator = std::conjunction<has_callable_begin<T>,has_callable_end<T>>;
template<typename T> using has_callable_reverse_iterator = std::conjunction<has_callable_rbegin<T>,has_callable_rend<T>>;

template<typename T, typename=void> inline constexpr bool has_mutating_iterator_v = false;
template<typename T> inline constexpr bool has_mutating_iterator_v<T,std::void_t< std::enable_if_t<std::is_assignable_v<decltype(*std::declval<T&>().begin()), typename T::value_type> > > > = true;
template<typename T, typename=void> inline constexpr bool has_mutating_subscript_operator_v = false;
template<typename T> inline constexpr bool has_mutating_subscript_operator_v<T,std::void_t< std::enable_if_t<std::is_assignable_v<decltype(std::declval<T&>()[std::declval<typename T::difference_type&>()]), typename T::value_type> > > > = true;


//create indexer
template<typename Core, typename Descriptor>
inline auto create_indexer(Core& t, const Descriptor& descriptor){
    if constexpr (has_callable_create_indexer<Core>::value){
        return t.create_indexer();
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        return gtensor::basic_indexer<Core&>{t};
    }else if constexpr (has_callable_create_walker<Core>::value){
        return gtensor::walker_indexer<decltype(t.create_walker(descriptor.dim()))>{descriptor.strides_div() ,t.create_walker(descriptor.dim())};
    }else if constexpr (has_callable_iterator<Core>::value){
        return gtensor::iterator_indexer<decltype(t.begin())>{t.begin()};
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");;
    }
}

//create walker
template<typename Core, typename Descriptor, typename DimT>
inline auto create_walker(Core& t, const Descriptor& descriptor, const DimT& max_dim){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr(has_callable_create_walker<Core>::value){
        return t.create_walker(max_dim);
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::walker<config_type, decltype(t.create_indexer())>{descriptor.adapted_strides(),descriptor.reset_strides(),index_type{0},t.create_indexer(),max_dim};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::walker<config_type, indexer_type>{descriptor.adapted_strides(),descriptor.reset_strides(),index_type{0},indexer_type{t},max_dim};
    }else if constexpr (has_callable_iterator<Core>::value){
        using indexer_type = gtensor::iterator_indexer<decltype(t.begin())>;
        return gtensor::walker<config_type, indexer_type>{descriptor.adapted_strides(),descriptor.reset_strides(),index_type{0},indexer_type{t.begin()},max_dim};
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}

//create iterator
template<typename Core, typename Descriptor>
inline auto begin(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (has_callable_iterator<Core>::value){
        return t.begin();
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::indexer_iterator<config_type, decltype(t.create_indexer())>{t.create_indexer(), index_type{0}};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::indexer_iterator<config_type, indexer_type>{indexer_type{t}, index_type{0}};
    }else if constexpr(has_callable_create_walker<Core>::value){
        return gtensor::walker_iterator<config_type, decltype(t.create_walker(descriptor.dim()))>{
            t.create_walker(descriptor.dim()),
            descriptor.shape(),
            descriptor.strides_div(),
            index_type{0}
        };
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}
template<typename Core, typename Descriptor>
inline auto end(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    if constexpr (has_callable_iterator<Core>::value){
        return t.end();
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::indexer_iterator<config_type, decltype(t.create_indexer())>{t.create_indexer(), descriptor.size()};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::indexer_iterator<config_type, indexer_type>{indexer_type{t}, descriptor.size()};
    }else if constexpr(has_callable_create_walker<Core>::value){
        return gtensor::walker_iterator<config_type, decltype(t.create_walker(descriptor.dim()))>{
            t.create_walker(descriptor.dim()),
            descriptor.shape(),
            descriptor.strides_div(),
            descriptor.size()
        };
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}

//create reverse iterator
template<typename It>
inline std::reverse_iterator<It> create_reverse_iterator(It it){
    return std::reverse_iterator<It>{std::move(it)};
}
template<typename...Ts>
inline auto create_reverse_iterator(indexer_iterator<Ts...> it){
    return gtensor::reverse_iterator_generic<indexer_iterator<Ts...>>{std::move(it)};
}
template<typename...Ts>
inline auto create_reverse_iterator(walker_iterator<Ts...> it){
    return gtensor::reverse_iterator_generic<walker_iterator<Ts...>>{std::move(it)};
}
template<typename Core, typename Descriptor>
inline auto rbegin(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    if constexpr (has_callable_reverse_iterator<Core>::value){
        return t.rbegin();
    }else if constexpr (has_callable_iterator<Core>::value){
        return create_reverse_iterator(t.end());
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::reverse_indexer_iterator<config_type, decltype(t.create_indexer())>{t.create_indexer(), descriptor.size()};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::reverse_indexer_iterator<config_type, indexer_type>{indexer_type{t}, descriptor.size()};
    }else if constexpr(has_callable_create_walker<Core>::value){
        return gtensor::reverse_walker_iterator<config_type, decltype(t.create_walker(descriptor.dim()))>{
            t.create_walker(descriptor.dim()),
            descriptor.shape(),
            descriptor.strides_div(),
            descriptor.size()
        };
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}
template<typename Core, typename Descriptor>
inline auto rend(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (has_callable_reverse_iterator<Core>::value){
        return t.rend();
    }else if constexpr (has_callable_iterator<Core>::value){
        return create_reverse_iterator(t.begin());
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::reverse_indexer_iterator<config_type, decltype(t.create_indexer())>{t.create_indexer(), index_type{0}};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::reverse_indexer_iterator<config_type, indexer_type>{indexer_type{t}, index_type{0}};
    }else if constexpr(has_callable_create_walker<Core>::value){
        return gtensor::reverse_walker_iterator<config_type, decltype(t.create_walker(descriptor.dim()))>{
            t.create_walker(descriptor.dim()),
            descriptor.shape(),
            descriptor.strides_div(),
            index_type{0}
        };
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}

//create broadcast iterator
template<typename Core, typename Descriptor, typename ShT, typename IdxT>
inline auto create_broadcast_iterator(Core& t, const Descriptor& descriptor, ShT&& shape, const IdxT& pos){
    using config_type = typename Core::config_type;
    using dim_type = typename config_type::dim_type;
    dim_type max_dim = std::max(descriptor.dim(), detail::make_dim(shape));
    auto strides_div = make_strides_div<config_type>(shape);
    return broadcast_iterator<config_type, decltype(create_walker(t,descriptor,max_dim))>{
        create_walker(t,descriptor,max_dim),
        std::forward<ShT>(shape),
        std::move(strides_div),
        pos
    };
}
template<typename Core, typename Descriptor, typename ShT>
inline auto begin_broadcast(Core& t, const Descriptor& descriptor, ShT&& shape){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    return create_broadcast_iterator(t, descriptor, std::forward<ShT>(shape),index_type{0});
}
template<typename Core, typename Descriptor, typename ShT>
inline auto end_broadcast(Core& t, const Descriptor& descriptor, ShT&& shape){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    index_type size = make_size(shape);
    return create_broadcast_iterator(t, descriptor, std::forward<ShT>(shape),size);
}

//create reverse broadcast iterator
//non const
template<typename Core, typename Descriptor, typename ShT, typename IdxT>
inline auto create_reverse_broadcast_iterator(Core& t, const Descriptor& descriptor, ShT&& shape, const IdxT& pos){
    using config_type = typename Core::config_type;
    using dim_type = typename config_type::dim_type;
    dim_type max_dim = std::max(descriptor.dim(), detail::make_dim(shape));
    auto strides_div = make_strides_div<config_type>(shape);
    return reverse_broadcast_iterator<config_type, decltype(create_walker(t,descriptor,max_dim))>{
        create_walker(t,descriptor,max_dim),
        std::forward<ShT>(shape),
        std::move(strides_div),
        pos
    };
}
template<typename Core, typename Descriptor, typename ShT>
inline auto rbegin_broadcast(Core& t, const Descriptor& descriptor, ShT&& shape){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    index_type size = make_size(shape);
    return create_reverse_broadcast_iterator(t, descriptor, std::forward<ShT>(shape),size);
}
template<typename Core, typename Descriptor, typename ShT>
inline auto rend_broadcast(Core& t, const Descriptor& descriptor, ShT&& shape){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    return create_reverse_broadcast_iterator(t, descriptor, std::forward<ShT>(shape),index_type{0});
}

}   //end of namespace detail

//Core must provide interface to access data and meta-data:
//descriptor() method for meta-data
//create_indexer() or create_walker() or both for data
//if Core provide iterators they are used, if not iterators are made using selected data accessor i.e. indexer or walker
template<typename Core>
class tensor_implementation
{
    using core_type = Core;
public:
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    template<typename...> struct forward_args : std::true_type{};
    template<typename U> struct forward_args<U> : std::bool_constant<!std::is_same_v<U,tensor_implementation>>{};

    template<typename...Args, std::enable_if_t<forward_args<Args...>::value,int> =0>
    explicit tensor_implementation(Args&&...args):
        core_(std::forward<Args>(args)...)
    {}

    tensor_implementation(const tensor_implementation&) = delete;
    tensor_implementation& operator=(const tensor_implementation&) = delete;
    tensor_implementation(tensor_implementation&&) = delete;
    tensor_implementation& operator=(tensor_implementation&&) = delete;

    //meta-data interface
    const auto& descriptor()const{
        return core_.descriptor();
    }
    index_type size()const{
        return descriptor().size();
    }
    bool empty()const{
        return size() == index_type{0};
    }
    dim_type dim()const{
        return descriptor().dim();
    }
    const shape_type& shape()const{
        return descriptor().shape();
    }
    const shape_type& strides()const{
        return descriptor().strides();
    }

    //data interface
    auto begin(){
        return detail::begin(core_,descriptor());
    }
    auto end(){
        return detail::end(core_,descriptor());
    }
    auto rbegin(){
        return detail::rbegin(core_,descriptor());
    }
    auto rend(){
        return detail::rend(core_,descriptor());
    }
    auto create_indexer(){
        return detail::create_indexer(core_,descriptor());
    }
    auto create_walker(dim_type max_dim){
        return detail::create_walker(core_,descriptor(),max_dim);
    }
    auto create_walker(){
        return create_walker(dim());
    }
    template<typename Container>
    auto begin(Container&& shape){
        return detail::begin_broadcast(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Container>
    auto end(Container&& shape){
        return detail::end_broadcast(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Container>
    auto rbegin(Container&& shape){
        return detail::rbegin_broadcast(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Container>
    auto rend(Container&& shape){
        return detail::rend_broadcast(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }

    //const data interface
    auto begin()const{
        return detail::begin(core_,descriptor());
    }
    auto end()const{
        return detail::end(core_,descriptor());
    }
    auto rbegin()const{
        return detail::rbegin(core_,descriptor());
    }
    auto rend()const{
        return detail::rend(core_,descriptor());
    }
    auto create_indexer()const{
        return detail::create_indexer(core_,descriptor());
    }
    auto create_walker(dim_type max_dim)const{
        return detail::create_walker(core_,descriptor(),max_dim);
    }
    auto create_walker()const{
        return create_walker(dim());
    }
    template<typename Container>
    auto begin(Container&& shape)const{
        return detail::begin_broadcast(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Container>
    auto end(Container&& shape)const{
        return detail::end_broadcast(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Container>
    auto rbegin(Container&& shape)const{
        return detail::rbegin_broadcast(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Container>
    auto rend(Container&& shape)const{
        return detail::rend_broadcast(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }

private:
    core_type core_;
};

//storage core combine together data and meta-data
//owns descriptor
//owns storage of data elements of type T - Config::storage<T>
template<typename Config, typename T>
class storage_core
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
        detail::fill_from_list(init_data, begin_());
    }

    template<typename ShT, typename It>
    storage_core(ShT&& shape, It first, It last):
        storage_core(std::forward<ShT>(shape), first, last, std::conjunction<std::is_constructible<storage_type,It,It>, std::is_move_constructible<storage_type> >{})
    {
        using it_difference_type = typename std::iterator_traits<It>::difference_type;
        static_assert(detail::is_static_castable_v<it_difference_type,index_type>);
    }

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
    storage_core(ShT&& shape, It first, It last, std::true_type):
        descriptor_(std::forward<ShT>(shape)),
        elements_(construct_from_range(descriptor_.size(), first, last, typename std::iterator_traits<It>::iterator_category{}))
    {}
    //no from range constructor or no move, use fill_from_range
    template<typename ShT, typename It>
    storage_core(ShT&& shape, It& first, It& last, std::false_type):
        descriptor_(std::forward<ShT>(shape)),
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
        }else if(size<d){   //range is bigger than size
            if constexpr (detail::is_static_castable_v<index_type,it_difference_type>){
                return storage_type{first,first+static_cast<it_difference_type>(size)};
            }else{
                return construct_from_range(size, first, last, std::input_iterator_tag{});
            }
        }else{  //range is smaller than size
            return construct_from_range(size, first, last, std::input_iterator_tag{});
        }
    }

    template<typename It>
    storage_type construct_from_range(index_type size, It& first, It& last, std::input_iterator_tag){
        storage_type res(size);
        fill_from_range(size,first,last,begin_(res),end_(res,size),typename std::iterator_traits<It>::iterator_category{});
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