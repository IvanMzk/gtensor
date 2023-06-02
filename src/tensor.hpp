#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <sstream>
#include "type_selector.hpp"
#include "tensor_factory.hpp"
#include "tensor_operators.hpp"
#include "slice.hpp"
#include "view_factory.hpp"
#include "reduce.hpp"

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
    //assignment
    basic_tensor& operator=(const basic_tensor& rhs){
        copy_assign_(rhs);
        return *this;
    }
    basic_tensor& operator=(basic_tensor&& rhs){
        move_assign_(std::move(rhs));
        return *this;
    }
    //converting assignment
    template<typename,typename> struct disable_forward_rhs0;
    template<typename Rhs> struct disable_forward_rhs0<std::false_type, Rhs> : std::negation<std::is_convertible<Rhs,value_type>>{};
    template<typename Rhs> struct disable_forward_rhs0<std::true_type, Rhs> : std::is_convertible<Rhs,basic_tensor>{};
    template<typename Rhs> struct disable_forward_rhs : disable_forward_rhs0<std::bool_constant<detail::is_tensor_v<Rhs>>, Rhs>{};

    template<typename Rhs, std::enable_if_t<!disable_forward_rhs<std::remove_cv_t<std::remove_reference_t<Rhs>>>::value,int> =0>
    basic_tensor& operator=(Rhs&& rhs){
        copy_assign_(std::forward<Rhs>(rhs));
        return *this;
    }
    //broadcast assignment, impl of this never changes, shapes of this and rhs must be broadcastable, or rhs convertible to value_type
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
    //resize
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
    //makes tensor by copying shape and elements from this
    auto copy()const{
        return tensor<value_type,order,config_type>(shape(),begin(),end());
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
    template<typename TraverseOrder>
    auto traverse_order_adapter(){
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
    auto traverse_order_adapter()const{
        return detail::traverse_order_adapter<const impl_type,TraverseOrder>{impl()};
    }
    //reduce
    template<typename Directions, typename F>
    auto reduce(const Directions& directions, F f, bool keep_dims=false)const{
        return gtensor::reduce(*this, directions, f, keep_dims);
    }
    template<typename F>
    auto slide(const dim_type& direction, F f, const index_type& window_size, const index_type& window_step)const{
        return gtensor::slide(*this, direction, f, window_size, window_step);
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
    //reshape view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    auto reshape(const Subs&...subs)const{
        return create_view_(view_factory_type::create_reshape_view(*this, subs...));
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_type_v<Container,index_type>,int> = 0 >
    auto reshape(const Container& subs)const{
        return create_view_(view_factory_type::create_reshape_view(*this, subs));
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
        static_assert(std::is_convertible_v<decltype(copy()),basic_tensor>,"can't resize view");
        const auto& shape_ = shape();
        if (!std::equal(shape_.begin(),shape_.end(),new_shape.begin(),new_shape.end())){
            swap(tensor<value_type,order,config_type>{std::forward<Container>(new_shape),begin(),end()});
        }
    }
    template<typename Rhs>
    void copy_assign_(Rhs&& rhs){
        using RhsT = std::remove_cv_t<std::remove_reference_t<Rhs>>;
        if (is_same(rhs)){  //self assignment
            return;
        }
        if constexpr (std::is_convertible_v<tensor<value_type,order,config_type>*,basic_tensor*>){  //value assignment
            if constexpr (detail::is_tensor_v<RhsT>){
                const auto& rhs_shape = rhs.shape();
                if (shape() == rhs_shape){
                    std::copy(rhs.begin(),rhs.end(),begin());
                }else{
                    swap(tensor<value_type,order,config_type>(rhs_shape,rhs.begin(),rhs.end()));
                }
            }else{
                if (size() == index_type{1}){
                    *begin() = std::forward<Rhs>(rhs);
                }else{
                    swap(tensor<value_type,order,config_type>(rhs));
                }
            }
        }else{  //broadcast assignment if possible, not compile otherwise
            assign(std::forward<Rhs>(rhs));
        }
    }
    void move_assign_(basic_tensor&& rhs){
        if (is_same(rhs)){  //self assignment
            return;
        }
        if constexpr (std::is_convertible_v<tensor<value_type,order,config_type>,basic_tensor>){  //value assignment
            swap(rhs);
        }else{  //broadcast assignment if possible, not compile otherwise
            assign(std::move(rhs));
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

    //assignment
    tensor& operator=(const tensor& rhs){
        basic_tensor_base::operator=(rhs);
        return *this;
    }
    tensor& operator=(tensor&& rhs){
        basic_tensor_base::operator=(std::move(rhs));
        return *this;
    }
    //converting assignment
    template<typename,typename> struct disable_forward_rhs0;
    template<typename Rhs> struct disable_forward_rhs0<std::false_type, Rhs> : std::negation<std::is_convertible<Rhs,value_type>>{};
    template<typename Rhs> struct disable_forward_rhs0<std::true_type, Rhs> : std::is_convertible<Rhs,tensor>{};
    template<typename Rhs> struct disable_forward_rhs : disable_forward_rhs0<std::bool_constant<detail::is_tensor_v<Rhs>>, Rhs>{};

    template<typename Rhs, std::enable_if_t<!disable_forward_rhs<std::remove_cv_t<std::remove_reference_t<Rhs>>>::value,int> =0>
    tensor& operator=(Rhs&& rhs){
        basic_tensor_base::operator=(std::forward<Rhs>(rhs));
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
    tensor(std::initializer_list<index_type> shape__, const value_type& value__):
        tensor(forward_tag::tag(), shape__, value__)
    {}
    //init list shape and range
    template<typename It>
    tensor(std::initializer_list<index_type> shape__, It begin__, It end__):
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

template<typename...Ts>
auto str(const basic_tensor<Ts...>& t){
    std::stringstream ss{};
    ss<<"{"<<detail::shape_to_str(t.shape())<<[&]{for(const auto& i:t){ss<<i<<" ";}; return "}";}();
    return ss.str();
}

template<typename T>
void swap(basic_tensor<T>& u, basic_tensor<T>& v){
    u.swap(v);
}

}   //end of namespace gtensor
#endif