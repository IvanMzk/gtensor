#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "module_selector.hpp"
#include "slice.hpp"
#include "tensor_factory.hpp"
#include "view_factory.hpp"
#include "tensor_operators.hpp"

#define GTENSOR_TENSOR_REDUCE_METHOD(NAME,F)\
template<typename Axes>\
auto NAME(const Axes& axes, bool keep_dims = false)const{\
    return F(*this,axes,keep_dims);\
}\
template<typename DimT>\
auto NAME(std::initializer_list<DimT> axes, bool keep_dims = false)const{\
    return F(*this,axes,keep_dims);\
}\
auto NAME(bool keep_dims = false)const{\
    return F(*this,keep_dims);\
}

#define GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(NAME,F)\
template<typename Axes, typename Initial = gtensor::detail::no_value>\
auto NAME(const Axes& axes, bool keep_dims = false, const Initial& initial = Initial{})const{\
    return F(*this,axes,keep_dims,initial);\
}\
template<typename DimT, typename Initial = gtensor::detail::no_value>\
auto NAME(std::initializer_list<DimT> axes, bool keep_dims = false, const Initial& initial = Initial{})const{\
    return F(*this,axes,keep_dims,initial);\
}\
template<typename Initial = gtensor::detail::no_value>\
auto NAME(bool keep_dims = false, const Initial& initial = Initial{})const{\
    return F(*this,keep_dims,initial);\
}

#define GTENSOR_TENSOR_CUMULATE_METHOD(NAME,F)\
template<typename DimT>\
auto NAME(const DimT& axis)const{\
    return F(*this,axis);\
}\
auto NAME()const{\
    return F(*this);\
}

namespace gtensor{
template<typename T, typename Layout, typename Config> class tensor;

namespace detail{

//helper to change traverse order of tensor
template<typename Impl, typename TraverseOrder>
class traverse_order_adapter
{
    ASSERT_ORDER(TraverseOrder);
    Impl* impl_;
public:
    traverse_order_adapter(Impl& impl__):
        impl_{&impl__}
    {}

    auto begin()const{
        return impl_->template begin<TraverseOrder>();
    }
    auto end()const{
        return impl_->template end<TraverseOrder>();
    }
    auto rbegin()const{
        return impl_->template rbegin<TraverseOrder>();
    }
    auto rend()const{
        return impl_->template rend<TraverseOrder>();
    }
    template<typename Container>
    auto begin(Container&& shape)const{
        return impl_->template begin<TraverseOrder>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto end(Container&& shape)const{
        return impl_->template end<TraverseOrder>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rbegin(Container&& shape)const{
        return impl_->template rbegin<TraverseOrder>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rend(Container&& shape)const{
        return impl_->template rend<TraverseOrder>(std::forward<Container>(shape));
    }
    auto create_indexer()const{
        return impl_->template create_indexer<TraverseOrder>();
    }
};

}   //end of namespace detail

template<typename Impl>
class basic_tensor
{
    using view_factory_type = view_factory_selector_t<typename Impl::config_type>;
    using impl_type = Impl;
    std::shared_ptr<impl_type> impl_;
public:
    using order = typename impl_type::order;
    using config_type = typename impl_type::config_type;
    using traverse_order = typename config_type::order;
    using value_type = typename impl_type::value_type;
    using dim_type = typename impl_type::dim_type;
    using index_type = typename impl_type::index_type;
    using shape_type = typename impl_type::shape_type;
    using size_type = index_type;
    using difference_type = index_type;
    using slice_type = slice<index_type>;
    using slice_item_type = typename slice_type::slice_item_type;

    basic_tensor(const basic_tensor&) = default;
    basic_tensor(basic_tensor&&) = default;

    explicit basic_tensor(std::shared_ptr<impl_type>&& impl__):
        impl_{std::move(impl__)}
    {}

    //value assignment operator=
    basic_tensor& operator=(const basic_tensor& rhs)&{
        copy_assign_(rhs);
        return *this;
    }
    basic_tensor& operator=(basic_tensor&& rhs)&{
        move_assign_(std::move(rhs));
        return *this;
    }
    template<typename Rhs, std::enable_if_t<!std::is_convertible_v<std::remove_cv_t<std::remove_reference_t<Rhs>>*,basic_tensor*>,int> =0>
    basic_tensor& operator=(Rhs&& rhs)&{
        copy_assign_(std::forward<Rhs>(rhs));
        return *this;
    }
    //elementwise broadcast assignment operator=
    basic_tensor& operator=(const basic_tensor& rhs)&&{
        assign(rhs);
        return *this;
    }
    basic_tensor& operator=(basic_tensor&& rhs)&&{
        assign(std::move(rhs));
        return *this;
    }
    template<typename Rhs, std::enable_if_t<!std::is_convertible_v<std::remove_cv_t<std::remove_reference_t<Rhs>>*,basic_tensor*>,int> =0>
    basic_tensor& operator=(Rhs&& rhs)&&{
        assign(std::forward<Rhs>(rhs));
        return *this;
    }
    //elementwise broadcast assignment, impl of this never changes, shapes of this and rhs must be broadcastable, or rhs convertible to value_type
    template<typename Rhs>
    basic_tensor& assign(Rhs&& rhs){
        gtensor::assign(*this, std::forward<Rhs>(rhs));
        return *this;
    }
    //broadcast equality, returns basic_tensor of results of broadcast element-wise comparisons, shapes of this and other must be broadcastable
    template<typename U>
    auto equal(const basic_tensor<U>& other)const{
        return gtensor::equal(*this, other);
    }
    //resize inplace
    template<typename Container>
    void resize(Container&& new_shape){
        resize_(std::forward<Container>(new_shape));
    }
    void resize(std::initializer_list<index_type> new_shape){
        resize_(new_shape);
    }
    //swap
    void swap(basic_tensor& other){
        impl_.swap(other.impl_);
    }
    void swap(basic_tensor&& other){
        swap(other);
    }
    //makes tensor in specified layout by copying shape and elements from this
    template<typename T=value_type, typename Config=config_type, typename Order = config::c_order>
    auto copy(Order order = Order{})const{
        ASSERT_ORDER(Order);
        (void)(order);
        auto a = traverse_order_adapter(Order{});
        return tensor<T,Order,Config>(shape(),a.begin(),a.end());
    }
    //makes 1d tensor in specified layout by copying elements from this
    //this element's traverse order the same as specified layout
    template<typename Order = config::c_order>
    auto flatten(Order order = Order{})const{
        ASSERT_ORDER(Order);
        (void)(order);
        auto a = traverse_order_adapter(Order{});
        return tensor<value_type,Order,config_type>({size()},a.begin(),a.end());
    }
    //check is this and other are tensors and have the same implementation
    template<typename Other>
    bool is_same(const Other& other)const{
        if constexpr (std::is_convertible_v<Other*, basic_tensor*>){
            return impl_ == other.impl_;
        }else{
            return false;
        }
    }

    //meta-data interface
    const auto& descriptor()const{
        return impl().descriptor();
    }
    index_type size()const{
        return impl().size();
    }
    dim_type dim()const{
        return impl().dim();
    }
    const shape_type& shape()const{
        return impl().shape();
    }
    const shape_type& strides()const{
        return impl().strides();
    }
    //guaranteed to be empty after move construct from this
    bool empty()const{
        return static_cast<bool>(impl_) ? impl().empty() : true;
    }

    //data interface
    //trverse order of iterator and indexer is config_type::order
    auto begin(){
        return impl().template begin<traverse_order>();
    }
    auto end(){
        return impl().template end<traverse_order>();
    }
    auto rbegin(){
        return impl().template rbegin<traverse_order>();
    }
    auto rend(){
        return impl().template rend<traverse_order>();
    }
    template<typename Container>
    auto begin(Container&& shape){
        return impl().template begin<traverse_order>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto end(Container&& shape){
        return impl().template end<traverse_order>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rbegin(Container&& shape){
        return impl().template rbegin<traverse_order>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rend(Container&& shape){
        return impl().template rend<traverse_order>(std::forward<Container>(shape));
    }
    auto create_indexer(){
        return impl().template create_indexer<traverse_order>();
    }
    auto create_walker(dim_type max_dim){
        return impl().create_walker(max_dim);
    }
    auto create_walker(){
        return impl().create_walker();
    }
    //return data interface adapter with specified traverse order
    template<typename TraverseOrder>
    auto traverse_order_adapter(TraverseOrder){
        return detail::traverse_order_adapter<impl_type,TraverseOrder>{impl()};
    }
    //const data interface
    auto begin()const{
        return impl().template begin<traverse_order>();
    }
    auto end()const{
        return impl().template end<traverse_order>();
    }
    auto rbegin()const{
        return impl().template rbegin<traverse_order>();
    }
    auto rend()const{
        return impl().template rend<traverse_order>();
    }
    template<typename Container>
    auto begin(Container&& shape)const{
        return impl().template begin<traverse_order>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto end(Container&& shape)const{
        return impl().template end<traverse_order>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rbegin(Container&& shape)const{
        return impl().template rbegin<traverse_order>(std::forward<Container>(shape));
    }
    template<typename Container>
    auto rend(Container&& shape)const{
        return impl().template rend<traverse_order>(std::forward<Container>(shape));
    }
    auto create_indexer()const{
        return impl().template create_indexer<traverse_order>();
    }
    auto create_walker(dim_type max_dim)const{
        return impl().create_walker(max_dim);
    }
    auto create_walker()const{
        return impl().create_walker();
    }
    template<typename TraverseOrder>
    auto traverse_order_adapter(TraverseOrder)const{
        return detail::traverse_order_adapter<const impl_type,TraverseOrder>{impl()};
    }

    //reduce_slide_transform methods to perform along axes using custom functor
    //reduce along axes, axes may be container or scalar
    //f should be like [](auto first, auto last){...}, where first,last is range along axes
    //f should return scalar - first,last reduction result - that determines result's value_type
    template<typename Axes, typename F>
    auto reduce(const Axes& axes, F f, bool keep_dims=false)const{
        return reduce(*this, axes, f, keep_dims);
    }
    template<typename F>
    auto reduce(std::initializer_list<dim_type> axes, F f, bool keep_dims=false)const{
        return reduce(*this, axes, f, keep_dims);
    }
    //reduce like over flatten
    template<typename F>
    auto reduce(F f, bool keep_dims=false)const{
        return reduce_flatten(*this, f, keep_dims);
    }
    //slide along given axis, axis is scalar
    //as if sliding window of width window_size moves along axis with step window_step and each window reduction result is stored to destination range
    //f should be like [](auto first, auto last, auto dfirst, auto dlast){...} where first,last is range along axis, dfirst,dlast range along corresponding result axis
    //dlast-dfirst equals to (axis_size - window_size)/window_step + 1
    //result's value_type may be specified by explicit specialization of R
    template<typename R=value_type, typename F>
    auto slide(const dim_type& axis, F f, const index_type& window_size, const index_type& window_step)const{
        return slide<R>(*this, axis, f, window_size, window_step);
    }
    //slide like over flatten
    template<int> auto slide_flatten(); //dummy declaration
    template<typename R=value_type, typename F>
    auto slide(F f, const index_type& window_size, const index_type& window_step)const{
        return slide_flatten<R>(*this, f, window_size, window_step);
    }
    //inplace tensor transform
    //f should be like [](auto first, auto last){...}, where first,last is range along axes
    template<typename F>
    void transform(const dim_type& axis, F f){
        transform(*this, axis, f);
    }

    //some methods that call corresponding free function of gtensor modules (tensor_math, statistic,...)
    //tensor_math
    GTENSOR_TENSOR_REDUCE_METHOD(all,all);
    GTENSOR_TENSOR_REDUCE_METHOD(any,any);
    GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(max,max);
    GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(min,min);
    GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(sum,sum);
    GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(prod,prod);
    GTENSOR_TENSOR_CUMULATE_METHOD(cumsum,cumsum);
    GTENSOR_TENSOR_CUMULATE_METHOD(cumprod,cumprod);
    //statistic
    GTENSOR_TENSOR_REDUCE_METHOD(ptp,ptp);
    GTENSOR_TENSOR_REDUCE_METHOD(mean,mean);
    GTENSOR_TENSOR_REDUCE_METHOD(median,median);
    GTENSOR_TENSOR_REDUCE_METHOD(var,var);
    GTENSOR_TENSOR_REDUCE_METHOD(std,std);
    //sort_search
    //inplace sort, should use gtensor::sort to sort copy
    template<typename DimT=dim_type, typename Comparator=std::less<void>>
    void sort(const DimT& axis=-1, const Comparator& comparator=Comparator{}){
        return transform(*this,axis,[&comparator](auto first, auto last){std::sort(first,last,comparator);});
    }
    template<typename DimT=dim_type, typename Comparator=std::less<void>>
    auto argsort(const DimT& axis=-1, const Comparator& comparator=Comparator{})const{
        return argsort(*this,axis,comparator);
    }
    GTENSOR_TENSOR_REDUCE_METHOD(argmax,argmax);
    GTENSOR_TENSOR_REDUCE_METHOD(argmin,argmin);
    auto nonzero()const{
        return nonzero(*this);
    }
    //indexing
    template<typename DimT, typename...Us>
    auto take(const basic_tensor<Us...>& indexes, const DimT& axis)const{
        return take(*this,indexes,axis);
    }
    template<typename...Us>
    auto take(const basic_tensor<Us...>& indexes)const{
        return take(*this,indexes);
    }

    //view construction operators and methods
    //slice view
    auto operator()(std::initializer_list<std::initializer_list<slice_item_type>> subs)const{
        return  create_view_(view_factory_type::create_slice_view(*this, subs));
    }

    template<typename Tensor, typename...Subs> struct enable_slice_view_variadic_ : std::conjunction<
        std::disjunction<
            std::is_convertible<Subs,typename Tensor::index_type>,
            std::is_convertible<Subs,typename Tensor::slice_type>
        >...
    >{};

    template<typename...Ts, typename...Subs, std::enable_if_t<enable_slice_view_variadic_<basic_tensor,Subs...>::value,int> = 0>
    auto operator()(const Subs&...subs)const{
        return create_view_(view_factory_type::create_slice_view(*this, subs...));
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_type_v<Container,slice_type>,int> = 0>
    auto operator()(const Container& subs)const{
        return create_view_(view_factory_type::create_slice_view(*this, subs));
    }
    //transpose view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,dim_type>...>,int> = 0 >
    auto transpose(const Subs&...subs)const{
        return create_view_(view_factory_type::create_transpose_view(*this, subs...));
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_type_v<Container,dim_type>,int> = 0>
    auto transpose(const Container& subs)const{
        return create_view_(view_factory_type::create_transpose_view(*this, subs));
    }
    auto transpose(std::initializer_list<dim_type> subs)const{
        return create_view_(view_factory_type::create_transpose_view(*this, subs));
    }
    //reshape view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    auto reshape(const Subs&...subs)const{
        return create_view_(view_factory_type::template create_reshape_view<config::c_order>(*this, subs...));
    }
    template<typename Container, typename Order = config::c_order, std::enable_if_t<detail::is_container_of_type_v<Container,index_type>,int> = 0 >
    auto reshape(const Container& subs, Order order = Order{})const{
        ASSERT_ORDER(Order);
        (void)order;
        return create_view_(view_factory_type::template create_reshape_view<Order>(*this, subs));
    }
    template<typename Order = config::c_order>
    auto reshape(std::initializer_list<index_type> subs, Order order = Order{})const{
        ASSERT_ORDER(Order);
        (void)order;
        return create_view_(view_factory_type::template create_reshape_view<Order>(*this, subs));
    }
    template<typename Order = config::c_order>
    auto ravel(Order order=Order{})const{
        return reshape({-1},order);
    }
    //mapping view
    template<typename...Subs> struct enable_index_mapping_view_variadic_ : std::conjunction<
        std::bool_constant<(sizeof...(Subs)>0)>,
        std::bool_constant<detail::is_tensor_of_type_v<Subs,index_type>>...
    >{};

    template<typename...Subs, std::enable_if_t<enable_index_mapping_view_variadic_<Subs...>::value,int> = 0 >
    auto operator()(const Subs&...subs)const{
        return create_view_(view_factory_type::create_index_mapping_view(*this, subs...));
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_tensor_of_type_v<Container,index_type>,int> =0>
    auto operator()(const Container& subs)const{
        return create_view_(view_factory_type::create_index_mapping_view(*this, subs));
    }
    template<typename Subs, std::enable_if_t<detail::is_bool_tensor_v<Subs> ,int> = 0 >
    auto operator()(const Subs& subs)const{
        return create_view_(view_factory_type::create_bool_mapping_view(*this, subs));
    }
private:
    impl_type& impl(){return *impl_.get();}
    const impl_type& impl()const{return *impl_.get();}
    template<typename Impl_>
    auto create_view_(std::shared_ptr<Impl_>&& impl__)const{
        return basic_tensor<Impl_>{std::move(impl__)};
    }
    template<typename Container>
    void resize_(Container&& new_shape){
        static_assert(std::is_convertible_v<tensor<value_type,order,config_type>*,basic_tensor*>,"can't resize view");
        const auto& shape_ = shape();
        if (!std::equal(shape_.begin(),shape_.end(),new_shape.begin(),new_shape.end())){
            auto a = traverse_order_adapter(order{});
            swap(tensor<value_type,order,config_type>{std::forward<Container>(new_shape),a.begin(),a.end()});
        }
    }

    template<typename Rhs>
    void copy_assign_(Rhs&& rhs){
        using Rhs_ = std::remove_cv_t<std::remove_reference_t<Rhs>>;
        if (is_same(rhs)){  //self assignment
            return;
        }
        if constexpr (std::is_convertible_v<tensor<value_type,order,config_type>*,basic_tensor*>){  //value assignment
            if constexpr (detail::is_tensor_v<Rhs_>){
                const auto& rhs_shape = rhs.shape();
                auto a = rhs.traverse_order_adapter(order{});
                if (shape() == rhs_shape){
                    std::copy(a.begin(),a.end(),traverse_order_adapter(order{}).begin());
                }else{
                    //swap(tensor<value_type,order,config_type>(rhs_shape,a.begin(),a.end()));
                    swap(rhs.template copy<value_type,config_type>(order{}));
                }
            }else if constexpr (std::is_convertible_v<Rhs_,value_type>){
                if (size() == index_type{1}){
                    *begin() = std::forward<Rhs>(rhs);
                }else{
                    swap(tensor<value_type,order,config_type>(rhs));
                }
            }else{
                static_assert(detail::always_false<Rhs>,"can't assign rhs: invalid rhs");
            }
        }else{
            static_assert(detail::always_false<basic_tensor>,"can't assign value to view");
        }
    }

    void move_assign_(basic_tensor&& rhs){
        if (is_same(rhs)){  //self assignment
            return;
        }
        if constexpr (std::is_convertible_v<tensor<value_type,order,config_type>*,basic_tensor*>){  //value assignment
            swap(rhs);
        }else{
            static_assert(detail::always_false<basic_tensor>,"can't assign value to view");
        }
    }
};

//tensor is basic_tensor with storage implementation and constructors
//T is type of element
//Layout is storage scheme of data elements, may be config::c_order or config::f_order
template<typename T, typename Layout = config::f_order, typename Config = config::extend_config_t<config::default_config,T>>
class tensor : public basic_tensor<typename tensor_factory_selector_t<Config,T,Layout>::result_type>
{
    static_assert(config::is_extended_config_v<Config>);
    using tensor_factory_type = tensor_factory_selector_t<Config,T,Layout>;
    using basic_tensor_base = basic_tensor<typename tensor_factory_type::result_type>;

    class forward_tag{
        struct private_tag{};
        forward_tag(private_tag){}  //make not default constructible
    public:
        static auto tag(){return forward_tag{private_tag{}};}
    };
    //this constructor should be used by all public constructors
    template<typename...Args>
    tensor(forward_tag, Args&&...args):
        basic_tensor_base(tensor_factory_type::create(std::forward<Args>(args)...))
    {}
public:
    using config_type = typename basic_tensor_base::config_type;
    using value_type = typename basic_tensor_base::value_type;
    using dim_type = typename basic_tensor_base::dim_type;
    using index_type = typename basic_tensor_base::index_type;
    using shape_type = typename basic_tensor_base::shape_type;
    using size_type = typename basic_tensor_base::size_type;
    using difference_type = typename basic_tensor_base::difference_type;

    tensor(const tensor&) = default;
    tensor(tensor&&) = default;

    //value assignment operator=
    tensor& operator=(const tensor& rhs)&{
        basic_tensor_base::operator=(rhs);
        return *this;
    }
    tensor& operator=(tensor&& rhs)&{
        basic_tensor_base::operator=(std::move(rhs));
        return *this;
    }
    template<typename Rhs, std::enable_if_t<!std::is_convertible_v<std::remove_cv_t<std::remove_reference_t<Rhs>>*,tensor*>,int> =0>
    tensor& operator=(Rhs&& rhs)&{
        basic_tensor_base::operator=(std::forward<Rhs>(rhs));
        return *this;
    }
    //elementwise broadcast assignment operator=
    tensor& operator=(const tensor& rhs)&&{
        std::move(*this).basic_tensor_base::operator=(rhs);
        return *this;
    }
    tensor& operator=(tensor&& rhs)&&{
        std::move(*this).basic_tensor_base::operator=(std::move(rhs));
        return *this;
    }
    template<typename Rhs, std::enable_if_t<!std::is_convertible_v<std::remove_cv_t<std::remove_reference_t<Rhs>>*,tensor*>,int> =0>
    tensor& operator=(Rhs&& rhs)&&{
        std::move(*this).basic_tensor_base::operator=(std::forward<Rhs>(rhs));
        return *this;
    }
    //broadcast assignment, impl of this never changes, shapes of this and rhs must be broadcastable, or rhs convertible to value_type
    template<typename Rhs>
    tensor& assign(Rhs&& rhs){
        basic_tensor_base::assign(std::forward<Rhs>(rhs));
        return *this;
    }
    //nested init_list constructors
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<U> init_data):tensor(forward_tag::tag(), init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<U>> init_data):tensor(forward_tag::tag(), init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<U>>> init_data):tensor(forward_tag::tag(), init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> init_data):tensor(forward_tag::tag(), init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>> init_data):tensor(forward_tag::tag(), init_data){}
    //default constructor makes empty 1-d tensor
    tensor():
        tensor(forward_tag::tag(), shape_type{0})
    {}
    //0-dim tensor constructor (aka tensor-scalar)
    explicit tensor(const value_type& value__):
        tensor(forward_tag::tag(), shape_type{}, value__)
    {}
    //init list shape and value
    template<typename IdxT>
    tensor(std::initializer_list<IdxT> shape__, const value_type& value__):
        tensor(forward_tag::tag(), shape__, value__)
    {}
    //init list shape and range
    template<typename IdxT, typename It>
    tensor(std::initializer_list<IdxT> shape__, It begin__, It end__):
        tensor(forward_tag::tag(), shape__, begin__, end__)
    {}

    template<typename Container> struct disable_forward_container : std::disjunction<
        std::is_convertible<Container,value_type>,
        std::is_convertible<Container,tensor>
    >{};
    //container shape, disambiguate with 0-dim constructor, copy,move constructor
    template<typename Container, std::enable_if_t<!disable_forward_container<std::remove_cv_t<std::remove_reference_t<Container>>>::value,int> =0>
    explicit tensor(Container&& shape__):
        tensor(forward_tag::tag(), std::forward<Container>(shape__))
    {}
    //container shape and value
    template<typename Container>
    tensor(Container&& shape__, const value_type& value__):
        tensor(forward_tag::tag(), std::forward<Container>(shape__), value__)
    {}
    //container shape and range
    template<typename Container, typename It>
    tensor(Container&& shape__, It begin__, It end__):
        tensor(forward_tag::tag(), std::forward<Container>(shape__), begin__, end__)
    {}
};

template<typename T>
void swap(basic_tensor<T>& u, basic_tensor<T>& v){
    u.swap(v);
}

}   //end of namespace gtensor

#undef GTENSOR_TENSOR_REDUCE_METHOD
#undef GTENSOR_TENSOR_REDUCE_INITIAL_METHOD
#undef GTENSOR_TENSOR_CUMULATE_METHOD
#endif