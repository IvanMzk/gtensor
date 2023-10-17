/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "module_selector.hpp"
#include "slice.hpp"
#include "tensor_factory.hpp"
#include "view_factory.hpp"
#include "tensor_operators.hpp"
#include "multithreading.hpp"

#define GTENSOR_TENSOR_REDUCE_METHOD(NAME,F)\
template<typename Policy, typename Axes, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>\
auto NAME(Policy policy, const Axes& axes, bool keep_dims = false)const{\
    return F(policy,*this,axes,keep_dims);\
}\
template<typename Policy, typename DimT, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>\
auto NAME(Policy policy, std::initializer_list<DimT> axes, bool keep_dims = false)const{\
    return F(policy,*this,axes,keep_dims);\
}\
template<typename Policy, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>\
auto NAME(Policy policy, bool keep_dims = false)const{\
    return F(policy,*this,keep_dims);\
}\
template<typename Axes, std::enable_if_t<!multithreading::is_policy_v<Axes>,int> =0>\
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
template<typename Policy, typename Axes, typename Initial = gtensor::detail::no_value, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>\
auto NAME(Policy policy, const Axes& axes, bool keep_dims = false, const Initial& initial = Initial{})const{\
    return F(policy,*this,axes,keep_dims,initial);\
}\
template<typename Policy, typename DimT, typename Initial = gtensor::detail::no_value, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>\
auto NAME(Policy policy, std::initializer_list<DimT> axes, bool keep_dims = false, const Initial& initial = Initial{})const{\
    return F(policy,*this,axes,keep_dims,initial);\
}\
template<typename Policy, typename Initial = gtensor::detail::no_value, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>\
auto NAME(Policy policy, bool keep_dims = false, const Initial& initial = Initial{})const{\
    return F(policy,*this,keep_dims,initial);\
}\
template<typename Axes, typename Initial = gtensor::detail::no_value, std::enable_if_t<!multithreading::is_policy_v<Axes>,int> =0>\
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
template<typename Policy, typename DimT, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>\
auto NAME(Policy policy, const DimT& axis)const{\
    return F(policy,*this,axis);\
}\
template<typename Policy, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>\
auto NAME(Policy policy)const{\
    return F(policy,*this);\
}\
template<typename DimT, std::enable_if_t<!multithreading::is_policy_v<DimT>,int> =0>\
auto NAME(const DimT& axis)const{\
    return F(*this,axis);\
}\
auto NAME()const{\
    return F(*this);\
}

#define GTENSOR_ADL_PROXY_METHOD(NAME,F)\
template<typename...Args>\
static auto NAME(Args&&...args){\
    return F(std::forward<Args>(args)...);\
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
    //data interface
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

    //trivial data interface
    auto begin_trivial()const{
        return impl_->template begin_trivial<TraverseOrder>();
    }
    auto end_trivial()const{
        return impl_->template end_trivial<TraverseOrder>();
    }
    auto rbegin_trivial()const{
        return impl_->template rbegin_trivial<TraverseOrder>();
    }
    auto rend_trivial()const{
        return impl_->template rend_trivial<TraverseOrder>();
    }
    auto create_trivial_indexer()const{
        return impl_->template create_trivial_indexer<TraverseOrder>();
    }
};

template<int> auto slide(); //dummy declaration
struct adl_proxy{
    template<typename ResT, typename...Args>
    static auto slide_(Args&&...args){
        return slide<ResT>(std::forward<Args>(args)...);
    }
    GTENSOR_ADL_PROXY_METHOD(reduce_range_,reduce_range);
    GTENSOR_ADL_PROXY_METHOD(reduce_binary_,reduce_binary);
    GTENSOR_ADL_PROXY_METHOD(transform_,transform);
    GTENSOR_ADL_PROXY_METHOD(all_,all);
    GTENSOR_ADL_PROXY_METHOD(any_,any);
    GTENSOR_ADL_PROXY_METHOD(max_,max);
    GTENSOR_ADL_PROXY_METHOD(min_,min);
    GTENSOR_ADL_PROXY_METHOD(sum_,sum);
    GTENSOR_ADL_PROXY_METHOD(prod_,prod);
    GTENSOR_ADL_PROXY_METHOD(cumsum_,cumsum);
    GTENSOR_ADL_PROXY_METHOD(cumprod_,cumprod);
    GTENSOR_ADL_PROXY_METHOD(ptp_,ptp);
    GTENSOR_ADL_PROXY_METHOD(mean_,mean);
    GTENSOR_ADL_PROXY_METHOD(median_,median);
    GTENSOR_ADL_PROXY_METHOD(var_,var);
    GTENSOR_ADL_PROXY_METHOD(stdev_,stdev);
    GTENSOR_ADL_PROXY_METHOD(argmax_,argmax);
    GTENSOR_ADL_PROXY_METHOD(argmin_,argmin);
    GTENSOR_ADL_PROXY_METHOD(argsort_,argsort);
    GTENSOR_ADL_PROXY_METHOD(nonzero_,nonzero);
    GTENSOR_ADL_PROXY_METHOD(take_,take);
};

template<typename T, typename Order, typename Config> struct copy_result{

    template<typename U, typename> struct selector_{using type = tensor<U,Order,Config>;};
    template<typename U> struct selector_<U,std::true_type>{
        using type = tensor<typename selector_<typename U::value_type,std::bool_constant<is_tensor_v<typename U::value_type>>>::type,Order,Config>;
    };
    using type = typename selector_<T,std::bool_constant<is_tensor_v<T>>>::type;
};
template<typename T, typename Order, typename Config> using copy_result_t = typename copy_result<T,Order,Config>::type;

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
    //broadcast equality,inequality returns basic_tensor of results of broadcast element-wise comparisons
    //other can be tensor or scalar, in case of tensor shapes of this and other must be broadcastable
    template<typename Other>
    auto equal(Other&& other)const{
        return gtensor::equal(*this, std::forward<Other>(other));
    }
    template<typename Other>
    auto not_equal(Other&& other)const{
        return gtensor::not_equal(*this, std::forward<Other>(other));
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
    template<typename T=value_type, typename Config=config_type, typename Order = order, typename Policy, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    auto copy(Policy policy, Order order_=Order{})const{
        ASSERT_ORDER(Order);
        using res_type = detail::copy_result_t<T,Order,Config>;
        res_type res(shape());
        auto a = traverse_order_adapter(order_);
        auto a_res = res.traverse_order_adapter(order_);
        if (is_trivial()){
            multithreading::copy(policy,a.begin_trivial(),a.end_trivial(),a_res.begin());
        }else{
            multithreading::copy(policy,a.begin(),a.end(),a_res.begin());
        }
        return res;
    }
    template<typename T=value_type, typename Config=config_type, typename Order = order, std::enable_if_t<!multithreading::is_policy_v<Order>,int> =0>
    auto copy(Order order_=Order{})const{
        return this->copy<T,Config,Order>(multithreading::exec_pol<1>{},order_);
    }
    //if this is view returns its copy, otherwise returns same tensor
    template<typename Policy>
    auto eval(Policy policy)const{
        if constexpr (std::is_convertible_v<tensor<value_type,order,config_type>*,basic_tensor*>){  //nothing to eval
            return *this;
        }else{
            return copy(policy);
        }
    }
    auto eval()const{
        return this->eval(multithreading::exec_pol<1>{});
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
    auto data(){
        return impl().data();
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
    auto data()const{
        return impl().data();
    }
    template<typename TraverseOrder>
    auto traverse_order_adapter(TraverseOrder)const{
        return detail::traverse_order_adapter<const impl_type,TraverseOrder>{impl()};
    }

    //trivial data interface
    auto begin_trivial(){
        return impl().template begin_trivial<traverse_order>();
    }
    auto end_trivial(){
        return impl().template end_trivial<traverse_order>();
    }
    auto rbegin_trivial(){
        return impl().template rbegin_trivial<traverse_order>();
    }
    auto rend_trivial(){
        return impl().template rend_trivial<traverse_order>();
    }
    auto create_trivial_indexer(){
        return impl().template create_trivial_indexer<traverse_order>();
    }
    auto create_trivial_walker(){
        return impl().create_trivial_walker();
    }

    //trivial const data interface
    auto begin_trivial()const{
        return impl().template begin_trivial<traverse_order>();
    }
    auto end_trivial()const{
        return impl().template end_trivial<traverse_order>();
    }
    auto rbegin_trivial()const{
        return impl().template rbegin_trivial<traverse_order>();
    }
    auto rend_trivial()const{
        return impl().template rend_trivial<traverse_order>();
    }
    auto create_trivial_indexer()const{
        return impl().template create_trivial_indexer<traverse_order>();
    }
    auto create_trivial_walker()const{
        return impl().create_trivial_walker();
    }

    bool is_trivial()const{
        return impl().is_trivial();
    }

    //reduce_slide_transform methods to perform along axes using custom functor
    //reduce along axes using range functor, axes may be container or scalar
    //f should be like [](auto first, auto last){...}, where first,last is range along axes
    //f should return scalar - first,last reduction result - that determines result's value_type
    //policy is specialization of multithreading::exec_pol
    template<typename Policy, typename Axes, typename RangeF, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    auto reduce_range(Policy policy, const Axes& axes, RangeF f, bool keep_dims=false)const{
        const bool any_order = false;
        return detail::adl_proxy::reduce_range_(policy,*this,axes,f,keep_dims,any_order);
    }
    template<typename Policy, typename RangeF, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    auto reduce_range(Policy policy, std::initializer_list<dim_type> axes, RangeF f, bool keep_dims=false)const{
        const bool any_order = false;
        return detail::adl_proxy::reduce_range_(policy,*this,axes,f,keep_dims,any_order);
    }
    template<typename Axes, typename RangeF, std::enable_if_t<!multithreading::is_policy_v<Axes>,int> =0>
    auto reduce_range(const Axes& axes, RangeF f, bool keep_dims=false)const{
        return this->reduce_range(multithreading::exec_pol<1>{},axes,f,keep_dims);
    }
    template<typename RangeF>
    auto reduce_range(std::initializer_list<dim_type> axes, RangeF f, bool keep_dims=false)const{
        return this->reduce_range(multithreading::exec_pol<1>{},axes,f,keep_dims);
    }
    //reduce like over flatten using range functor
    template<typename Policy, typename RangeF, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    auto reduce_range(Policy policy, RangeF f, bool keep_dims=false)const{
        return this->reduce_range(policy,detail::no_value{},f,keep_dims);
    }
    template<typename RangeF>
    auto reduce_range(RangeF f, bool keep_dims=false)const{
        return this->reduce_range(multithreading::exec_pol<1>{},detail::no_value{},f,keep_dims);
    }

    //reduce along axes using binary functor, axes may be container or scalar
    //f is binary reduce functor that operates on tensor's elements, like std::plus<void>
    //result tensor has value_type that is return type of f
    //initial must be such that expression f(initial,element) be valid or no_value
    //policy is specialization of multithreading::exec_pol
    template<typename Policy, typename Axes, typename BinaryF, typename Initial=detail::no_value, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    auto reduce_binary(Policy policy, const Axes& axes, BinaryF f, bool keep_dims=false, const Initial& initial=Initial{})const{
        return detail::adl_proxy::reduce_binary_(policy,*this,axes,f,keep_dims,initial);
    }
    template<typename Policy, typename BinaryF, typename Initial=detail::no_value, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    auto reduce_binary(Policy policy, std::initializer_list<dim_type> axes, BinaryF f, bool keep_dims=false, const Initial& initial=Initial{})const{
        return detail::adl_proxy::reduce_binary_(policy,*this,axes,f,keep_dims,initial);
    }
    template<typename Axes, typename BinaryF, typename Initial=detail::no_value, std::enable_if_t<!multithreading::is_policy_v<Axes>,int> =0>
    auto reduce_binary(const Axes& axes, BinaryF f, bool keep_dims=false, const Initial& initial=Initial{})const{
        return this->reduce_binary(multithreading::exec_pol<1>{},axes,f,keep_dims,initial);
    }
    template<typename BinaryF, typename Initial=detail::no_value>
    auto reduce_binary(std::initializer_list<dim_type> axes, BinaryF f, bool keep_dims=false, const Initial& initial=Initial{})const{
        return this->reduce_binary(multithreading::exec_pol<1>{},axes,f,keep_dims,initial);
    }
    //reduce like over flatten using binary functor
    template<typename Policy, typename BinaryF, typename Initial=detail::no_value, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    auto reduce_binary(Policy policy, BinaryF f, bool keep_dims=false, const Initial& initial=Initial{})const{
        return this->reduce_binary(policy,detail::no_value{},f,keep_dims,initial);
    }
    template<typename BinaryF, typename Initial=detail::no_value>
    auto reduce_binary(BinaryF f, bool keep_dims=false, const Initial& initial=Initial{})const{
        return this->reduce_binary(multithreading::exec_pol<1>{},detail::no_value{},f,keep_dims,initial);
    }

    //slide along given axis, axis is scalar
    //as if sliding window of width window_size moves along axis with step window_step and each window reduction result is stored to destination range
    //f should be like [](auto first, auto last, auto dfirst, auto dlast){...} where first,last is range along axis, dfirst,dlast range along corresponding result axis
    //dlast-dfirst equals to (axis_size - window_size)/window_step + 1
    //result's value_type may be specified by explicit specialization of R
    template<typename R=value_type, typename Policy, typename Axis, typename F>
    auto slide(Policy policy, const Axis& axis, F f, const index_type& window_size, const index_type& window_step)const{
        return detail::adl_proxy::slide_<R>(policy,*this,axis,f,window_size,window_step);
    }
    template<typename R=value_type, typename F>
    auto slide(const dim_type& axis, F f, const index_type& window_size, const index_type& window_step)const{
        return this->slide(multithreading::exec_pol<1>{},axis,f,window_size,window_step);
    }
    //slide like over flatten
    template<typename R=value_type, typename Policy, typename F>
    auto slide(Policy policy, F f, const index_type& window_size, const index_type& window_step)const{
        return this->slide(policy,detail::no_value{},f,window_size,window_step);
    }
    template<typename R=value_type, typename F>
    auto slide(F f, const index_type& window_size, const index_type& window_step)const{
        return this->slide(multithreading::exec_pol<1>{},detail::no_value{},f,window_size,window_step);
    }

    //inplace tensor transform along axes using range functor
    //f should be like [](auto first, auto last){...}, where first,last is range along axes
    template<typename Policy, typename RangeF>
    void transform(Policy policy, const dim_type& axis, RangeF f){
        detail::adl_proxy::transform_(policy,*this,axis,f);
    }
    template<typename RangeF>
    void transform(const dim_type& axis, RangeF f){
        this->transform(multithreading::exec_pol<1>{},axis,f);
    }

    //some methods that call corresponding free function of gtensor modules (tensor_math, statistic,...)
    //tensor_math
    GTENSOR_TENSOR_REDUCE_METHOD(all,detail::adl_proxy::all_);
    GTENSOR_TENSOR_REDUCE_METHOD(any,detail::adl_proxy::any_);
    GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(max,detail::adl_proxy::max_);
    GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(min,detail::adl_proxy::min_);
    GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(sum,detail::adl_proxy::sum_);
    GTENSOR_TENSOR_REDUCE_INITIAL_METHOD(prod,detail::adl_proxy::prod_);
    GTENSOR_TENSOR_CUMULATE_METHOD(cumsum,detail::adl_proxy::cumsum_);
    GTENSOR_TENSOR_CUMULATE_METHOD(cumprod,detail::adl_proxy::cumprod_);
    //statistic
    GTENSOR_TENSOR_REDUCE_METHOD(ptp,detail::adl_proxy::ptp_);
    GTENSOR_TENSOR_REDUCE_METHOD(mean,detail::adl_proxy::mean_);
    GTENSOR_TENSOR_REDUCE_METHOD(median,detail::adl_proxy::median_);
    GTENSOR_TENSOR_REDUCE_METHOD(var,detail::adl_proxy::var_);
    GTENSOR_TENSOR_REDUCE_METHOD(stdev,detail::adl_proxy::stdev_);
    //sort_search
    //inplace sort, should use gtensor::sort to get sorted copy
    template<typename Policy, typename DimT=dim_type, typename Comparator=std::less<void>, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    void sort(Policy policy, const DimT& axis=-1, const Comparator& comparator=Comparator{}){
        return detail::adl_proxy::transform_(policy,*this,axis,[&comparator](auto first, auto last){std::sort(first,last,comparator);});
    }
    template<typename DimT=dim_type, typename Comparator=std::less<void>, std::enable_if_t<!multithreading::is_policy_v<DimT>,int> =0>
    void sort(const DimT& axis=-1, const Comparator& comparator=Comparator{}){
        return this->sort(multithreading::exec_pol<1>{},axis,comparator);
    }
    template<typename Policy, typename DimT=dim_type, typename Comparator=std::less<void>, std::enable_if_t<multithreading::is_policy_v<Policy>,int> =0>
    auto argsort(Policy policy, const DimT& axis=-1, const Comparator& comparator=Comparator{})const{
        return detail::adl_proxy::argsort_(policy,*this,axis,comparator);
    }
    template<typename DimT=dim_type, typename Comparator=std::less<void>, std::enable_if_t<!multithreading::is_policy_v<DimT>,int> =0>
    auto argsort(const DimT& axis=-1, const Comparator& comparator=Comparator{})const{
        return this->argsort(multithreading::exec_pol<1>{},axis,comparator);
    }
    GTENSOR_TENSOR_REDUCE_METHOD(argmax,detail::adl_proxy::argmax_);
    GTENSOR_TENSOR_REDUCE_METHOD(argmin,detail::adl_proxy::argmin_);
    auto nonzero()const{
        return detail::adl_proxy::nonzero_(*this);
    }
    //indexing
    template<typename DimT, typename...Us>
    auto take(const basic_tensor<Us...>& indexes, const DimT& axis)const{
        return detail::adl_proxy::take_(*this,indexes,axis);
    }
    template<typename...Us>
    auto take(const basic_tensor<Us...>& indexes)const{
        return detail::adl_proxy::take_(*this,indexes);
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
template<typename T, typename Layout = config::c_order, typename Config = config::extend_config_t<config::default_config,T>>
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
    //construct tensor using shape that inferfed from given (nested) initializer list and using its elements
    //initializer list elements always considered to be in c_order
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
    //init list shape and value constructor
    //construct tensor of shape filled with value
    template<typename IdxT>
    tensor(std::initializer_list<IdxT> shape__, const value_type& value__):
        tensor(forward_tag::tag(), shape__, value__)
    {}
    //init list shape and range constructor
    //construct tensor of shape filled with values from iterator range
    //if tensor size n is smaller or equal than range - tensor initialized with first n range elements
    //if tensor size is greater than range - first tensor elements initialized with range, is rest tensor elements initialized dependes on underlaying storage
    template<typename IdxT, typename It, std::enable_if_t<detail::is_iterator_v<It>,int> =0>
    tensor(std::initializer_list<IdxT> shape__, It begin__, It end__):
        tensor(forward_tag::tag(), shape__, begin__, end__)
    {}

    template<typename Shape> struct disable_forward_shape : std::disjunction<
        std::is_convertible<Shape,value_type>,
        std::is_convertible<Shape,tensor>
    >{};
    //shape constructor, disambiguate with 0-dim constructor, copy,move constructor
    //construct tensor of shape, is it initialized dependes on underlaying storage
    //shape is container, in case of empty container 0dim tensor (tensor-scalar) is constructed
    template<typename Shape, std::enable_if_t<!disable_forward_shape<std::remove_cv_t<std::remove_reference_t<Shape>>>::value,int> =0>
    explicit tensor(Shape&& shape__):
        tensor(forward_tag::tag(), std::forward<Shape>(shape__))
    {}
    //shape and value constructor
    //construct tensor of shape filled with value
    //shape may be container or scalar, in case of scalar 1d tensor is constructed, in case of empty container 0dim tensor (tensor-scalar) is constructed
    template<typename Shape>
    tensor(Shape&& shape__, const value_type& value__):
        tensor(forward_tag::tag(), std::forward<Shape>(shape__), value__)
    {}
    //shape and range constructor
    //construct tensor of shape filled with values from iterator range
    //if tensor size n is smaller or equal than range - tensor initialized with first n range elements
    //if tensor size is greater than range - first tensor elements initialized with range, are rest tensor elements initialized dependes on underlaying storage
    //shape may be container or scalar, in case of scalar 1d tensor is constructed, in case of empty container 0dim tensor (tensor-scalar) is constructed
    template<typename Shape, typename It, std::enable_if_t<detail::is_iterator_v<It>,int> =0>
    tensor(Shape&& shape__, It begin__, It end__):
        tensor(forward_tag::tag(), std::forward<Shape>(shape__), begin__, end__)
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
#undef GTENSOR_ADL_PROXY_METHOD
#endif