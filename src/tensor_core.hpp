#ifndef TENSOR_CORE_HPP_
#define TENSOR_CORE_HPP_

#include <type_traits>
#include <iterator>
#include "common.hpp"
#include "init_list_helper.hpp"
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
    using descriptor_type = basic_descriptor<Config,Layout>;
    using storage_type = typename Config::template storage<T>;
public:
    using order = Layout;
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
        std::fill(begin_(),end_(),v);
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
    void copy_from_list(std::initializer_list<Nested> init_data){   //init_data always in c_order
        if constexpr (std::is_same_v<order,config::c_order>){    //no map needed
            detail::copy_from_list(init_data, begin_());
        }else{  //mapper from c order to f order
            auto index_mapper = [this](const auto& idx){
                return detail::flat_to_flat<config::c_order>(descriptor_.strides_div(config::c_order{}),descriptor_.strides(),index_type{0},idx);
            };
            detail::copy_from_list(init_data, begin_(), index_mapper);
        }
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
//view descriptor type depends on kind of view
//Parent is type of view parent(origin) i.e. it is basic_tensor specialization
template<typename Parent>
class transpose_view_core
{
    using parent_type = Parent;
    using descriptor_type = transpose_descriptor<typename Parent::config_type,typename Parent::order>;
public:
    using config_type = typename parent_type::config_type;
    using order = typename parent_type::order;
    using value_type = typename parent_type::value_type;
    using dim_type = typename parent_type::dim_type;

    template<typename AxesMap, typename ShT, typename Parent_>
    transpose_view_core(AxesMap&& axes_map__, ShT&& shape__, Parent_&& parent__):
        descriptor_{std::forward<AxesMap>(axes_map__),std::forward<ShT>(shape__)},
        parent_{std::forward<Parent_>(parent__)}
    {}

    const descriptor_type& descriptor()const{return descriptor_;}
    auto create_walker(const dim_type& max_dim)const{return create_walker_helper(*this, max_dim);}
    auto create_walker(const dim_type& max_dim){return create_walker_helper(*this, max_dim);}
    auto create_walker()const{return create_walker_helper(*this);}
    auto create_walker(){return create_walker_helper(*this);}
private:
    template<typename U>
    static auto create_walker_helper(U& instance){
        using parent_walker_type = decltype(instance.parent_.create_walker());
        using walker_type = mapping_axes_walker<trivial_view_walker<parent_walker_type>>;
        return walker_type{
            instance.descriptor_.axes_map(),
            instance.parent_.create_walker()
        };
    }
    template<typename U>
    static auto create_walker_helper(U& instance, const dim_type& max_dim){
        using parent_walker_type = decltype(instance.parent_.create_walker());
        using walker_type = axes_correction_walker<mapping_axes_walker<trivial_view_walker<parent_walker_type>>>;
        return walker_type{
            max_dim,
            instance.descriptor_.axes_map(),
            instance.parent_.create_walker()
        };
    }
    descriptor_type descriptor_;
    parent_type parent_;
};

template<typename Parent>
class slice_view_core
{
    using parent_type = Parent;
    using descriptor_type = descriptor_w_scale<descriptor_w_offset<transpose_descriptor<typename Parent::config_type,typename Parent::order>>>;
public:
    using config_type = typename parent_type::config_type;
    using order = typename parent_type::order;
    using value_type = typename parent_type::value_type;
    using dim_type = typename parent_type::dim_type;
    using shape_type = typename parent_type::shape_type;

    template<typename Scale, typename Offset, typename AxesMap, typename ShT, typename Parent_>
    slice_view_core(Scale&& scale__, Offset&& offset__, AxesMap&& axes_map__, ShT&& shape__, Parent_&& parent__):
        descriptor_{std::forward<Scale>(scale__),std::forward<Offset>(offset__),std::forward<AxesMap>(axes_map__),std::forward<ShT>(shape__)},
        parent_{std::forward<Parent_>(parent__)}
    {}

    const descriptor_type& descriptor()const{return descriptor_;}
    auto create_walker(const dim_type& max_dim)const{return create_walker_helper(*this, max_dim);}
    auto create_walker(const dim_type& max_dim){return create_walker_helper(*this, max_dim);}
    auto create_walker()const{return create_walker_helper(*this);}
    auto create_walker(){return create_walker_helper(*this);}
private:
    template<typename U>
    static auto create_walker_helper(U& instance){
        using parent_walker_type = decltype(instance.parent_.create_walker());
        using walker_type = resetting_walker<scaling_walker<mapping_axes_walker<offsetting_walker<trivial_view_walker<parent_walker_type>>>>>;
        return walker_type{
            instance.descriptor_.shape(),
            instance.descriptor_.scale(),
            instance.descriptor_.axes_map(),
            instance.descriptor_.offset(),
            instance.parent_.create_walker()
        };
    }
    template<typename U>
    static auto create_walker_helper(U& instance, const dim_type& max_dim){
        using parent_walker_type = decltype(instance.parent_.create_walker());
        using walker_type = axes_correction_walker<resetting_walker<scaling_walker<mapping_axes_walker<offsetting_walker<trivial_view_walker<parent_walker_type>>>>>>;
        return walker_type{
            max_dim,
            instance.descriptor_.shape(),
            instance.descriptor_.scale(),
            instance.descriptor_.axes_map(),
            instance.descriptor_.offset(),
            instance.parent_.create_walker()
        };
    }
    descriptor_type descriptor_;
    parent_type parent_;
};

template<typename Parent>
class subdim_view_core
{
    using parent_type = Parent;
    using descriptor_type = descriptor_w_offset<transpose_descriptor<typename Parent::config_type,typename Parent::order>>;
public:
    using config_type = typename parent_type::config_type;
    using order = typename parent_type::order;
    using value_type = typename parent_type::value_type;
    using dim_type = typename parent_type::dim_type;
    using shape_type = typename parent_type::shape_type;

    template<typename Offset, typename AxesMap, typename ShT, typename Parent_>
    subdim_view_core(Offset&& offset__, AxesMap&& axes_map__, ShT&& shape__, Parent_&& parent__):
        descriptor_{std::forward<Offset>(offset__),std::forward<AxesMap>(axes_map__),std::forward<ShT>(shape__)},
        parent_{std::forward<Parent_>(parent__)}
    {}

    const descriptor_type& descriptor()const{return descriptor_;}
    auto create_walker(const dim_type& max_dim)const{return create_walker_helper(*this, max_dim);}
    auto create_walker(const dim_type& max_dim){return create_walker_helper(*this, max_dim);}
    auto create_walker()const{return create_walker_helper(*this);}
    auto create_walker(){return create_walker_helper(*this);}
private:
    template<typename U>
    static auto create_walker_helper(U& instance){
        using parent_walker_type = decltype(instance.parent_.create_walker());
        using walker_type = resetting_walker<mapping_axes_walker<offsetting_walker<trivial_view_walker<parent_walker_type>>>>;
        return walker_type{
            instance.descriptor_.shape(),
            instance.descriptor_.axes_map(),
            instance.descriptor_.offset(),
            instance.parent_.create_walker()
        };
    }
    template<typename U>
    static auto create_walker_helper(U& instance, const dim_type& max_dim){
        using parent_walker_type = decltype(instance.parent_.create_walker());
        using walker_type = axes_correction_walker<resetting_walker<mapping_axes_walker<offsetting_walker<trivial_view_walker<parent_walker_type>>>>>;
        return walker_type{
            max_dim,
            instance.descriptor_.shape(),
            instance.descriptor_.axes_map(),
            instance.descriptor_.offset(),
            instance.parent_.create_walker()
        };
    }
    descriptor_type descriptor_;
    parent_type parent_;
};

template<typename Parent>
class mapping_view_core
{
    using parent_config_type = typename Parent::config_type;
    using descriptor_type = basic_descriptor<parent_config_type,typename Parent::order>;
    using parent_type = Parent;
    using index_map_type = typename parent_config_type::template index_map<typename parent_config_type::index_type>;
public:
    using config_type = parent_config_type;
    using order = typename Parent::order;
    using value_type = typename Parent::value_type;
    using index_type = typename Parent::index_type;

    template<typename IndexMap, typename ShT, typename Parent_>
    mapping_view_core(IndexMap&& index_map__, ShT&& shape__, Parent_&& parent__):
        index_map_{std::forward<IndexMap>(index_map__)},
        descriptor_{std::forward<ShT>(shape__)},
        parent_{std::forward<Parent_>(parent__)}
    {}

    const descriptor_type& descriptor()const{return descriptor_;}
    auto create_indexer()const{return create_indexer_helper(*this);}
    auto create_indexer(){return create_indexer_helper(*this);}
private:
    struct index_mapper{
        const index_map_type* map;
        index_type operator()(const index_type& idx)const{
            return (*map)[idx];
        }
    };

    template<typename U>
    static auto create_indexer_helper(U& instance){
        auto a = instance.parent_.template traverse_order_adapter<order>();
        using parent_indexer_type = decltype(a.create_indexer());
        return basic_indexer<parent_indexer_type,index_mapper>{
            a.create_indexer(),
            index_mapper{&instance.index_map_}
        };
    }
    index_map_type index_map_;
    descriptor_type descriptor_;
    parent_type parent_;
};

template<typename Order, typename Parent>
class reshape_view_core
{
    using descriptor_type = basic_descriptor<typename Parent::config_type,Order>;
    using parent_type = Parent;
public:
    using order = Order;
    using config_type = typename Parent::config_type;
    using value_type = typename Parent::value_type;
    using dim_type = typename config_type::dim_type;

    template<typename ShT, typename Parent_>
    reshape_view_core(ShT&& shape__, Parent_&& parent__):
        descriptor_{std::forward<ShT>(shape__)},
        parent_{std::forward<Parent_>(parent__)}
    {}
    //descriptor interface
    const descriptor_type& descriptor()const{return descriptor_;}
    //reshape view can use parent's data interface taking order into account
    //non const data interface
    auto begin(){
        return parent_.template traverse_order_adapter<order>().begin();
    }
    auto end(){
        return parent_.template traverse_order_adapter<order>().end();
    }
    auto rbegin(){
        return parent_.template traverse_order_adapter<order>().rbegin();
    }
    auto rend(){
        return parent_.template traverse_order_adapter<order>().rend();
    }
    template<typename Container>
    auto begin(Container&& shape){
        return parent_.template traverse_order_adapter<order>().begin(std::forward<Container>(shape));
    }
    template<typename Container>
    auto end(Container&& shape){
        return parent_.template traverse_order_adapter<order>().end(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rbegin(Container&& shape){
        return parent_.template traverse_order_adapter<order>().rbegin(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rend(Container&& shape){
        return parent_.template traverse_order_adapter<order>().rend(std::forward<Container>(shape));
    }
    auto create_indexer(){
        return parent_.template traverse_order_adapter<order>().create_indexer();
    }
    //const data interface
    auto begin()const{
        return parent_.template traverse_order_adapter<order>().begin();
    }
    auto end()const{
        return parent_.template traverse_order_adapter<order>().end();
    }
    auto rbegin()const{
        return parent_.template traverse_order_adapter<order>().rbegin();
    }
    auto rend()const{
        return parent_.template traverse_order_adapter<order>().rend();
    }
    template<typename Container>
    auto begin(Container&& shape)const{
        return parent_.template traverse_order_adapter<order>().begin(std::forward<Container>(shape));
    }
    template<typename Container>
    auto end(Container&& shape)const{
        return parent_.template traverse_order_adapter<order>().end(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rbegin(Container&& shape)const{
        return parent_.template traverse_order_adapter<order>().rbegin(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rend(Container&& shape)const{
        return parent_.template traverse_order_adapter<order>().rend(std::forward<Container>(shape));
    }
    auto create_indexer()const{
        return parent_.template traverse_order_adapter<order>().create_indexer();
    }
private:
    descriptor_type descriptor_;
    parent_type parent_;
};

}   //end of namespace gtensor
#endif