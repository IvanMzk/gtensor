#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#pragma inline_depth(255)
#include <vector>
#include <iterator>

#include "catch.hpp"
#include "tensor.hpp"
#include "walker_base.hpp"
#include "iterator.hpp"
#include "test_config.hpp"

namespace benchmark_walker{
using gtensor::multiindex_iterator;
using gtensor::walker;
using gtensor::tensor;
using gtensor::evaluating_tensor;

template<typename ValT, template<typename> typename Cfg>
struct inline_walker_test_tensor : public tensor<ValT,Cfg>{
    using config_type = Cfg<ValT>;
    using iterator_type = multiindex_iterator<ValT,Cfg,walker<ValT,Cfg>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<inline_walker_test_tensor::config_type>::type;
    
    strides_type strides{gtensor::detail::make_dividers<inline_walker_test_tensor::config_type>(impl()->strides())};

    using tensor::tensor;
    inline_walker_test_tensor(const tensor& base):
        tensor{base}
    {}

    auto begin()const{return iterator_type{impl()->as_evaluating()->create_walker(),impl()->shape(), strides};}
    auto end()const{return iterator_type{impl()->as_evaluating()->create_walker(), impl()->shape(), strides, impl()->size()};}
};

template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return make_asymmetric_tree<Depth-1>(t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return make_symmetric_tree<Depth-1>(t2+t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

}   //end of namespace benchmark_walker

namespace noinline_evaluation{
using gtensor::walker;
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::tensor_base;
using gtensor::storage_walker_polymorphic;
using gtensor::evaluating_walker_polymorphic;
using gtensor::binary_operations::add;
using gtensor::walker_base;
using gtensor::multiindex_iterator;

//noinline walker maker interface
template<typename ValT, template<typename> typename Cfg>
class noinline_walker_maker
{
    virtual walker<ValT,Cfg> create_noinline_walker()const = 0;
public:
    auto create_walker()const{return create_noinline_walker();}    
};

template<typename ValT, template<typename> typename Cfg>
class noinline_storage_tensor : 
    public storage_tensor<ValT,Cfg>,    
    public noinline_walker_maker<ValT,Cfg>
{             
    walker<ValT,Cfg> create_noinline_walker()const override{return std::make_unique<storage_walker_polymorphic<ValT,Cfg>>(shape(),strides(),data());}
public:    
    using storage_tensor::storage_tensor;
};

template<typename ValT, template<typename> typename Cfg>
class test_tensor_noinline : public tensor<ValT, Cfg>
{    
    using storage_tensor_type = noinline_storage_tensor<ValT,Cfg>;
    using iterator_type = multiindex_iterator<ValT,Cfg,walker<ValT,Cfg>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<Cfg<ValT>>::type;

    strides_type strides{gtensor::detail::make_dividers<Cfg<ValT>>(impl()->strides())};

    template<typename Nested>
    test_tensor_noinline(std::initializer_list<Nested> init_data, int):        
        tensor(std::make_shared<storage_tensor_type>(init_data))
    {}
public:    
    test_tensor_noinline() = default;
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):test_tensor_noinline(init_data,0){}
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):test_tensor_noinline(init_data,0){}
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):test_tensor_noinline(init_data,0){}
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):test_tensor_noinline(init_data,0){}
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):test_tensor_noinline(init_data,0){}

    template<typename...Dims>
    test_tensor_noinline(const value_type& v, const Dims&...dims):        
        tensor(std::make_shared<storage_tensor_type>(v, dims...))
    {}

    test_tensor_noinline(std::shared_ptr<tensor_base<ValT,Cfg>>&& impl__):
        tensor(std::move(impl__))
    {}

    auto impl()const{return tensor::impl();}
    auto begin()const{return iterator_type{dynamic_cast<const noinline_walker_maker<ValT,Cfg>*>(impl()->impl().get())->create_walker(),impl()->shape(), strides};}
    auto end()const{return iterator_type{dynamic_cast<const noinline_walker_maker<ValT,Cfg>*>(impl()->impl().get())->create_walker(), impl()->shape(), strides, impl()->size()};}
    
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class noinline_evaluating_tensor : 
    public evaluating_tensor<ValT,Cfg,F,Ops...>,
    public noinline_walker_maker<ValT,Cfg>
{
    template<std::size_t...I>
    walker<ValT,Cfg> create_walker_helper(std::index_sequence<I...>)const{
        using walker_type = evaluating_walker_polymorphic<ValT,Cfg,F,decltype(dynamic_cast<noinline_walker_maker<ValT,Cfg>*>(std::declval<Ops>().get())->create_walker())...>;
        return std::make_unique<walker_type>(shape(),dynamic_cast<noinline_walker_maker<ValT,Cfg>*>(operand<I>().get())->create_walker()...);        
    }

    walker<ValT,Cfg> create_noinline_walker()const override{return create_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});}

public:
    using evaluating_tensor::evaluating_tensor;    
};

template<typename ValT1, typename ValT2, template<typename> typename Cfg>
static inline auto operator+(const test_tensor_noinline<ValT1, Cfg>& op1, const test_tensor_noinline<ValT2, Cfg>& op2){
    using operation_type = add;
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));
    using exp_operand1_type = std::shared_ptr<tensor_base<ValT1,Cfg>>;
    using exp_operand2_type = std::shared_ptr<tensor_base<ValT2,Cfg>>;
    using exp_type = noinline_evaluating_tensor<result_type, Cfg, operation_type, exp_operand1_type, exp_operand2_type>;
    return test_tensor_noinline<result_type,Cfg>{std::make_shared<exp_type>(op1.impl(),op2.impl())};
}

}   //end of namespace noinline_evaluation


namespace separate_evaluation{

using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::tensor;
using gtensor::tensor_base;
using gtensor::multiindex_iterator;
using gtensor::detail::tensor_kinds;
using gtensor::binary_operations::add;

enum class walking_walker_kinds{storage_walker, evaluating_walker};
template<typename IdxT, typename ShT> class storage_walking_walker;
template<typename IdxT, typename ShT> class evaluating_walking_walker;

template<typename IdxT, typename ShT>
class basic_walking_walker
{    
    using index_type = IdxT;
    using shape_type = ShT;
    using storage_walking_type = storage_walking_walker<index_type, shape_type>;
    using evaluating_walking_type = evaluating_walking_walker<index_type, shape_type>; 

    index_type dim_;
    gtensor::detail::shape_inverter<index_type,shape_type> shape_;
    walking_walker_kinds kind_;    

protected:
    auto dim()const{return dim_;}
    auto shape()const{return shape_;}
    auto kind()const{return kind_;}
    auto as_storage_walker(){return static_cast<storage_walking_type*>(this);}
    auto as_evaluating_walker(){return static_cast<evaluating_walking_type*>(this);}

    basic_walking_walker(walking_walker_kinds kind__, const shape_type& shape__):
        kind_{kind__},
        dim_{static_cast<index_type>(shape__.size())},
        shape_{shape__}        
    {}
public:
    virtual std::unique_ptr<basic_walking_walker> clone()const = 0;
    virtual ~basic_walking_walker(){}

    void walk(const index_type& direction, const index_type& steps){
        if (kind_ == walking_walker_kinds::storage_walker){
            static_cast<storage_walking_type*>(this)->walk(direction, steps);
        }else{
            static_cast<evaluating_walking_type*>(this)->walk(direction, steps);
        }
    }
    void step(const index_type& direction){
        if (kind_ == walking_walker_kinds::storage_walker){
            static_cast<storage_walking_type*>(this)->step(direction);
        }else{
            static_cast<evaluating_walking_type*>(this)->step(direction);
        }
    }
    void step_back(const index_type& direction){
        if (kind_ == walking_walker_kinds::storage_walker){
            static_cast<storage_walking_type*>(this)->step_back(direction);
        }else{
            static_cast<evaluating_walking_type*>(this)->step_back(direction);
        }
    }
    void reset(const index_type& direction){
        if (kind_ == walking_walker_kinds::storage_walker){
            static_cast<storage_walking_type*>(this)->reset(direction);
        }else{
            static_cast<evaluating_walking_type*>(this)->reset(direction);
        }
    }
    void reset(){
        if (kind_ == walking_walker_kinds::storage_walker){
            static_cast<storage_walking_type*>(this)->reset();
        }else{
            static_cast<evaluating_walking_type*>(this)->reset();
        }
    }
};

template<typename IdxT, typename ShT>
class storage_walking_walker : public basic_walking_walker<IdxT,ShT>
{    
    using index_type = IdxT;
    using shape_type = ShT;
        
    gtensor::detail::shape_inverter<index_type,shape_type> strides;
    index_type offset;
    index_type cursor_{offset};

    std::unique_ptr<basic_walking_walker> clone()const override{return std::make_unique<storage_walking_walker>(*this);}

public:
    storage_walking_walker(const shape_type& shape_, const shape_type& strides_, const index_type& offset_):
        basic_walking_walker{walking_walker_kinds::storage_walker, shape_},
        strides{strides_},
        offset{offset_}
    {}
    
    void walk(const index_type& direction, const index_type& steps){
        if (gtensor::detail::can_walk(direction, dim(), shape().element(direction))){
            cursor_+=steps*strides.element(direction);
        }   
    }
    void step(const index_type& direction){
        if (gtensor::detail::can_walk(direction, dim(), shape().element(direction))){
            cursor_+=strides.element(direction);
        }
    }
    void step_back(const index_type& direction){        
        if (gtensor::detail::can_walk(direction, dim(), shape().element(direction))){
            cursor_-=strides.element(direction);
        }            
    }
    void reset(const index_type& direction){
        if (gtensor::detail::can_walk(direction, dim(), shape().element(direction))){
            cursor_-=(shape().element(direction)-1)*strides.element(direction);
        }
    }
    void reset(){cursor_ = offset;}
    
    auto cursor()const{return cursor_;}
};

template<typename IdxT, typename ShT>
class evaluating_walking_walker : public basic_walking_walker<IdxT,ShT>
{    
    using basic_walker_type = basic_walking_walker<IdxT,ShT>;
    using index_type = IdxT;
    using shape_type = ShT;
    
    std::unique_ptr<basic_walker_type> w1;
    std::unique_ptr<basic_walker_type> w2;

    std::unique_ptr<basic_walking_walker> clone()const override{return std::make_unique<evaluating_walking_walker>(*this);}
    
public:
    evaluating_walking_walker(const shape_type& shape_, std::unique_ptr<basic_walker_type>&& w1_, std::unique_ptr<basic_walker_type>&& w2_):
        basic_walking_walker{walking_walker_kinds::evaluating_walker, shape_},
        w1{std::move(w1_)},
        w2{std::move(w2_)}
    {}

    evaluating_walking_walker(const evaluating_walking_walker& other):
        basic_walking_walker(other),
        w1{other.w1->clone()},
        w2{other.w2->clone()}
    {}
    
    void walk(const index_type& direction, const index_type& steps){
        if (gtensor::detail::can_walk(direction,dim(),shape().element(direction))){
            w1->walk(direction, steps);
            w2->walk(direction, steps);
        }
    }
    void step(const index_type& direction){
        if (gtensor::detail::can_walk(direction,dim(),shape().element(direction))){
            w1->step(direction);
            w2->step(direction);
        }
    }
    void step_back(const index_type& direction){
        if (gtensor::detail::can_walk(direction,dim(),shape().element(direction))){
            w1->step_back(direction);
            w2->step_back(direction);
        }
    }
    void reset(const index_type& direction){
        if (gtensor::detail::can_walk(direction,dim(),shape().element(direction))){
            w1->reset(direction);
            w2->reset(direction);
        }
    }
    void reset(){
        w1->reset();
        w2->reset();
    } 
};

template<typename ValT, template<typename> typename Cfg>
class evaluator_base
{       
public:        
    virtual ValT operator*() const = 0;
    virtual std::unique_ptr<evaluator_base> clone()const = 0;
};

template<typename ValT, template<typename> typename Cfg>
class evaluator{    
    using impl_base_type = evaluator_base<ValT, Cfg>;
    
    std::unique_ptr<impl_base_type> impl;    
public:    
    evaluator(std::unique_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}    
    evaluator(const evaluator& other):
        impl{other.impl->clone()}
    {}
    evaluator(evaluator&& other) = default;

    ValT operator*() const{return impl->operator*();}    
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Wks>
class evaluating_evaluator : public evaluator_base<ValT, Cfg>
{
    std::tuple<Wks...> walkers;
    F f{};            
    template<std::size_t...I>
    ValT deref_helper(std::index_sequence<I...>) const {return f(*std::get<I>(walkers)...);}
    std::unique_ptr<evaluator_base> clone()const override{return std::make_unique<evaluating_evaluator>(*this);}
public:
    evaluating_evaluator(Wks&&...walkers_):        
        walkers{std::move(walkers_)...}
    {}    
    ValT operator*() const override {return deref_helper(std::make_index_sequence<sizeof...(Wks)>{});}
};

template<typename ValT, template<typename> typename Cfg>
class storage_accessor
{   
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;
    const storage_walking_walker<index_type,shape_type>* walker;
    const ValT* data;
public:    
    storage_accessor(const storage_walking_walker<index_type,shape_type>* walker_, const ValT* data_):
        walker{walker_},
        data{data_}
    {}
    ValT operator*() const {return data[walker->cursor()];}
};

template<typename ValT, template<typename> typename Cfg>
class walker
{
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using basic_walking_type = basic_walking_walker<index_type,shape_type>;
    using evaluating_walking_type = evaluating_walking_walker<index_type, shape_type>;
    using evaluator_type = evaluator<ValT,Cfg>;

    std::unique_ptr<basic_walking_type> walking;
    evaluator_type evaluator;
    evaluating_walking_type* walking_impl{static_cast<evaluating_walking_type*>(walking.get())};

public:
    walker(std::unique_ptr<basic_walking_type>&& walking_, evaluator_type&& evaluator_):
        walking{std::move(walking_)},
        evaluator{std::move(evaluator_)}
    {}

    walker(walker&& other):
        walking{std::move(other.walking)},
        evaluator{std::move(other.evaluator)}
    {}
    
    walker(const walker& other):
        walking{other.walking->clone()},
        evaluator{other.evaluator}
    {}

    walker& walk(const index_type& direction, const index_type& steps){
        walking_impl->walk(direction,steps);
        return *this;
    }
    walker& step(const index_type& direction){
        walking_impl->step(direction);
        return *this;
    }
    walker& step_back(const index_type& direction){
        walking_impl->step_back(direction);
        return *this;
    }
    walker& reset(const index_type& direction){
        walking_impl->reset(direction);
        return *this;
    }
    walker& reset(){
        walking_impl->reset();
        return *this;
    }    
    value_type operator*() const{return *evaluator;}
};

//split walker maker interface
template<typename ValT, template<typename> typename Cfg>
class storage_split_walker_maker
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;
    using evaluator_type = storage_accessor<ValT,Cfg>;
    using basic_walking_type = basic_walking_walker<index_type,shape_type>;
    virtual std::pair<std::unique_ptr<basic_walking_type>, evaluator_type> create_storage_split_walker()const = 0;
public:
    auto create_split_walker()const{return create_storage_split_walker();}    
};
template<typename ValT, template<typename> typename Cfg>
class evaluating_split_walker_maker
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;
    using basic_walking_type = basic_walking_walker<index_type,shape_type>;
    using evaluating_walking_type = evaluating_walking_walker<index_type, shape_type>;    
    using evaluator_type = evaluator<ValT,Cfg>;
    virtual std::pair<std::unique_ptr<basic_walking_type>, evaluator_type> create_evaluating_split_walker()const = 0;    
public:
    auto create_split_walker()const{return create_evaluating_split_walker();}
    auto create_walker()const{
        auto split_walker = create_split_walker();        
        return walker<ValT,Cfg>{std::move(split_walker.first), std::move(split_walker.second)};
    }
};

//dispatcher of tensor interfaces to make walker
class dispatch_exception : public std::runtime_error{
    public: dispatch_exception(const char* what):runtime_error(what){}
};
template<typename ValT, template<typename> typename Cfg>
auto as_storage_split_walker_maker(const tensor_base<ValT,Cfg>& t){return dynamic_cast<const storage_split_walker_maker<ValT,Cfg>*>(&t);}
template<typename ValT, template<typename> typename Cfg>
auto as_evaluating_split_walker_maker(const tensor_base<ValT,Cfg>& t){return dynamic_cast<const evaluating_split_walker_maker<ValT,Cfg>*>(&t);}


class dispatcher{    
    template<typename FirstT, typename ValT2, template<typename> typename Cfg, typename F>
    static auto dispatch_second(F& f, const FirstT& first, const tensor_base<ValT2, Cfg>& second){
        if (second.tensor_kind() == tensor_kinds::storage_tensor)
        {
            return f(first, *as_storage_split_walker_maker(second));
        }
        else if (second.tensor_kind() == tensor_kinds::expression)
        {            
            return f(first, *as_evaluating_split_walker_maker(second));
        }        
        else
        {
            throw dispatch_exception("type is not supported by dispatcher");
        }
    }        
    template<typename ValT1, typename ValT2, template<typename> typename Cfg, typename F>
    static auto dispatch_first(F& f, const tensor_base<ValT1, Cfg>& first, const tensor_base<ValT2, Cfg>& second){
        if (first.tensor_kind() == tensor_kinds::storage_tensor)
        {
            return dispatch_second(f, *as_storage_split_walker_maker(first), second);
        }
        else if (first.tensor_kind() == tensor_kinds::expression)
        {            
            return dispatch_second(f, *as_evaluating_split_walker_maker(first), second);
        }                
        else
        {
            throw dispatch_exception("type is not supported by dispatcher");
        }
    }    
public:        
    template<typename ValT1, typename ValT2, template<typename> typename Cfg, typename F>
    static auto call(const F& f, const tensor_base<ValT1, Cfg>& first, const tensor_base<ValT2, Cfg>& second){
        return dispatch_first(f,first,second);
    }
};


template<typename ValT, template<typename> typename Cfg>
class test_storage_tensor : 
    public storage_tensor<ValT,Cfg>,    
    public storage_split_walker_maker<ValT,Cfg>
{             
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;    
    using basic_walking_type = basic_walking_walker<index_type,shape_type>;
    using storage_walking_type = storage_walking_walker<index_type, shape_type>;
    using storage_evaluator_type = storage_accessor<ValT,Cfg>;

    std::pair<std::unique_ptr<basic_walking_type>, storage_evaluator_type> create_storage_split_walker()const override{
        auto walking = new storage_walking_type{shape(),strides(),0};        
        return std::make_pair(std::unique_ptr<basic_walking_type>{walking},storage_evaluator_type{walking,data()});
    }
public:    
    using storage_tensor::storage_tensor;
};

template<typename ValT, template<typename> typename Cfg>
class test_tensor : public tensor<ValT, Cfg>
{    
    using storage_tensor_type = test_storage_tensor<ValT,Cfg>;
    using iterator_type = multiindex_iterator<ValT,Cfg,walker<ValT,Cfg>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<Cfg<ValT>>::type;

    strides_type strides{gtensor::detail::make_dividers<Cfg<ValT>>(impl()->strides())};

    template<typename Nested>
    test_tensor(std::initializer_list<Nested> init_data, int):        
        tensor(std::make_shared<storage_tensor_type>(init_data))
    {}
public:    
    test_tensor() = default;
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):test_tensor(init_data,0){}
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):test_tensor(init_data,0){}
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):test_tensor(init_data,0){}
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):test_tensor(init_data,0){}
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):test_tensor(init_data,0){}

    template<typename...Dims>
    test_tensor(const value_type& v, const Dims&...dims):        
        tensor(std::make_shared<storage_tensor_type>(v, dims...))
    {}

    test_tensor(std::shared_ptr<tensor_base<ValT,Cfg>>&& impl__):
        tensor(std::move(impl__))
    {}

    auto impl()const{return tensor::impl();}
    auto begin()const{return iterator_type{dynamic_cast<const evaluating_split_walker_maker<ValT,Cfg>*>(impl()->impl().get())->create_walker(),impl()->shape(), strides};}
    auto end()const{return iterator_type{dynamic_cast<const evaluating_split_walker_maker<ValT,Cfg>*>(impl()->impl().get())->create_walker(), impl()->shape(), strides, impl()->size()};}
    
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class test_evaluating_tensor : 
    public evaluating_tensor<ValT,Cfg,F,Ops...>,
    public evaluating_split_walker_maker<ValT,Cfg>
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;
    using basic_walking_type = basic_walking_walker<index_type,shape_type>;
    using evaluating_walking_type = evaluating_walking_walker<index_type, shape_type>;        
    using evaluator_type = evaluator<ValT,Cfg>;
    
    struct split_walker_maker{
        const shape_type& shape;
        split_walker_maker(const shape_type& shape_):
            shape{shape_}
        {}
        template<typename...Args>
        std::pair<std::unique_ptr<basic_walking_type>, evaluator_type> operator()(const Args&...args)const{
             return helper(args.create_split_walker()...);            
        }
        template<typename...Args>
        auto helper(Args&&...args)const{
            using evaluating_evaluator_type = evaluating_evaluator<ValT,Cfg,F,decltype(args.second)...>;
            return std::make_pair(
                std::unique_ptr<basic_walking_type>{new evaluating_walking_type{shape,std::move(args.first)...}}, 
                evaluator_type{std::make_unique<evaluating_evaluator_type>(std::move(args.second)...)}
            );
        }
    };

    std::pair<std::unique_ptr<basic_walking_type>, evaluator_type> create_evaluating_split_walker()const override{
        return create_split_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
    template<std::size_t...I>
    auto create_split_walker_helper(std::index_sequence<I...>)const{        
        return dispatcher::call(split_walker_maker{shape()},*operand<I>()...);
    }    

public:
    using evaluating_tensor::evaluating_tensor;    
};

template<typename ValT1, typename ValT2, template<typename> typename Cfg>
static inline auto operator+(const test_tensor<ValT1, Cfg>& op1, const test_tensor<ValT2, Cfg>& op2){
    using operation_type = add;
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));
    using exp_operand1_type = std::shared_ptr<tensor_base<ValT1,Cfg>>;
    using exp_operand2_type = std::shared_ptr<tensor_base<ValT2,Cfg>>;
    using exp_type = test_evaluating_tensor<result_type, Cfg, operation_type, exp_operand1_type, exp_operand2_type>;
    return test_tensor<result_type,Cfg>{std::make_shared<exp_type>(op1.impl(),op2.impl())};
}


}   //namespace separate_evaluation

namespace separate_evaluation_v1{

using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::tensor;
using gtensor::tensor_base;
using gtensor::multiindex_iterator;
using gtensor::detail::tensor_kinds;
using gtensor::binary_operations::add;

template<typename IdxT, typename ShT>
class storage_walking_walker
{    
    using index_type = IdxT;
    using shape_type = ShT;
    using shape_inverter_type = gtensor::detail::shape_inverter<index_type,shape_type>;
        
    index_type dim_;
    shape_inverter_type shape_;
    shape_inverter_type strides_;
    index_type offset_;
    index_type* cursor_;    

public:
    storage_walking_walker(const index_type& dim__, const shape_inverter_type& shape__, const shape_inverter_type& strides__, const index_type& offset__, index_type* cursor__):
        dim_{dim__},
        shape_{shape__},
        strides_{strides__},
        offset_{offset__},
        cursor_{cursor__}
    {}
    storage_walking_walker(const shape_inverter_type& shape__, const shape_inverter_type& strides__):
        shape_{shape__},
        strides_{strides__}
    {}
    
    void walk(const index_type& direction, const index_type& steps){
        if (gtensor::detail::can_walk(direction, dim_, shape_.element(direction))){
            *cursor_+=steps*strides_.element(direction);
        }   
    }
    void step(const index_type& direction){
        if (gtensor::detail::can_walk(direction, dim_, shape_.element(direction))){
            *cursor_+=strides_.element(direction);
        }
    }
    void step_back(const index_type& direction){        
        if (gtensor::detail::can_walk(direction, dim_, shape_.element(direction))){
            *cursor_-=strides_.element(direction);
        }            
    }
    void reset(const index_type& direction){
        if (gtensor::detail::can_walk(direction, dim_, shape_.element(direction))){
            *cursor_-=(shape_.element(direction)-1)*strides_.element(direction);
        }
    }
    void reset(){*cursor_ = offset;}    
};

template<typename ValT, template<typename> typename Cfg>
class storage_accessor
{   
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;
    
    index_type dim_;
    gtensor::detail::shape_inverter<index_type, shape_type> shape_;
    gtensor::detail::shape_inverter<index_type, shape_type> strides_;
    index_type offset_;
    const ValT* data_;
    index_type cursor_{offset_};

public:    
    storage_accessor(const shape_type& shape__, const shape_type& strides__, const index_type& offset__, const ValT* data__):
        dim_{static_cast<index_type>(shape__.size())},
        shape_{shape__},
        strides_{strides__},
        offset_{offset__},
        data_{data__}
    {}
    auto create_walker(){return storage_walking_walker<index_type,shape_type>{dim_,shape_,strides_,offset_,&cursor_};}
    //auto create_walker()const{return storage_walking_walker<index_type,shape_type>{shape_,strides_};}
    // auto dim()const{return dim_};
    // auto shape()const{return shape_};
    // auto strides()const{return strides_};
    // auto offset()const{return offset_};
    // auto cursor(){return &cursor_};
    ValT operator*() const {return data_[cursor_];}
};

template<typename ValT, template<typename> typename Cfg>
class evaluator_base
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;
public:        
    virtual ValT operator*() const = 0;
    virtual std::unique_ptr<evaluator_base> clone()const = 0;
    virtual void create_walkers_helper(std::vector<storage_walking_walker<index_type,shape_type>>& walkers) = 0;
    virtual std::size_t storages_number()const = 0;
};

template<typename ValT, template<typename> typename Cfg>
class evaluator{    
    using impl_base_type = evaluator_base<ValT, Cfg>;
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;

    std::unique_ptr<impl_base_type> impl_;
public:    
    evaluator(std::unique_ptr<impl_base_type>&& impl__):
        impl_{std::move(impl__)}
    {}    
    evaluator(const evaluator& other):
        impl_{other.impl_->clone()}
    {}
    evaluator(evaluator&& other) = default;
    
    ValT operator*() const{return impl_->operator*();}
    impl_base_type* impl()const{return impl_.get();}
    auto create_walkers(){
        std::vector<storage_walking_walker<index_type,shape_type>> walkers{};        
        walkers.reserve(impl()->storages_number());
        impl()->create_walkers_helper(walkers);
        return walkers;
    }
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...NodesT>
class evaluating_evaluator : public evaluator_base<ValT, Cfg>
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;

    std::size_t storages_number_;
    std::tuple<NodesT...> nodes;
    F f{};

    template<std::size_t...I>    
    ValT deref_helper(std::index_sequence<I...>) const {return f(*std::get<I>(nodes)...);}
    
    std::unique_ptr<evaluator_base> clone()const override{return std::make_unique<evaluating_evaluator>(*this);}
    
    template<typename V, typename ValT, template<typename> typename Cfg>
    void create_walker(V& v, storage_accessor<ValT,Cfg>& w){v.push_back(w.create_walker());}
    
    template<typename V, typename ValT, template<typename> typename Cfg>
    void create_walker(V& v, const evaluator<ValT,Cfg>& w){w.impl()->create_walkers_helper(v);}
    
    template<std::size_t...I>    
    void create_walkers_helper(std::vector<storage_walking_walker<index_type,shape_type>>& walkers, std::index_sequence<I...>){
        (create_walker(walkers, std::get<I>(nodes)),...);
    }
    
    void create_walkers_helper(std::vector<storage_walking_walker<index_type,shape_type>>& walkers) override{
        create_walkers_helper(walkers,std::make_index_sequence<sizeof...(NodesT)>{});
    }

    template<typename ValT, template<typename> typename Cfg>
    std::size_t storages_counter(const storage_accessor<ValT,Cfg>& w)const{return 1;}
    
    template<typename ValT, template<typename> typename Cfg>
    std::size_t storages_counter(const evaluator<ValT,Cfg>& w)const{return w.impl()->storages_number();}

    std::size_t storages_number()const override{return storages_number_;}

public:
    evaluating_evaluator(NodesT&&...nodes_):
        storages_number_{(storages_counter(nodes_)+...)},
        nodes{std::move(nodes_)...}
    {}    
    ValT operator*() const override {return deref_helper(std::make_index_sequence<sizeof...(NodesT)>{});}
};



template<typename ValT, template<typename> typename Cfg>
class walker
{
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using evaluator_type = evaluator<ValT,Cfg>;

    evaluator_type evaluator;
    std::vector<storage_walking_walker<index_type,shape_type>> walkers{evaluator.create_walkers()};

public:
    walker(evaluator_type&& evaluator_):        
        evaluator{std::move(evaluator_)}        
    {}

    walker(walker&& other):        
        evaluator{std::move(other.evaluator)}
    {}
    
    walker(const walker& other):        
        evaluator{other.evaluator}
    {}

    walker& walk(const index_type& direction, const index_type& steps){
        for (auto& walker : walkers)
        {walker.walk(direction,steps);}
        return *this;
    }
    walker& step(const index_type& direction){
        for (auto& walker : walkers)
        {walker.step(direction);}
        return *this;
    }
    walker& step_back(const index_type& direction){
        for (auto& walker : walkers)
        {walker.step_back(direction);}
        return *this;
    }
    walker& reset(const index_type& direction){
        for (auto& walker : walkers)
        {walker.reset(direction);}
        return *this;
    }
    walker& reset(){
        for (auto& walker : walkers)
        {walker.reset();}
        return *this;
    }    
    value_type operator*() const{return *evaluator;}
};

//split walker maker interface
template<typename ValT, template<typename> typename Cfg>
class storage_split_walker_maker
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;
    
    virtual storage_accessor<ValT,Cfg> create_storage_split_walker()const = 0;
public:
    auto create_split_walker()const{return create_storage_split_walker();}    
};
template<typename ValT, template<typename> typename Cfg>
class evaluating_split_walker_maker
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;

    virtual evaluator<ValT,Cfg> create_evaluating_split_walker()const = 0;    
public:
    auto create_split_walker()const{return create_evaluating_split_walker();}    
};

//dispatcher of tensor interfaces to make walker
class dispatch_exception : public std::runtime_error{
    public: dispatch_exception(const char* what):runtime_error(what){}
};
template<typename ValT, template<typename> typename Cfg>
auto as_storage_split_walker_maker(const tensor_base<ValT,Cfg>& t){return dynamic_cast<const storage_split_walker_maker<ValT,Cfg>*>(&t);}
template<typename ValT, template<typename> typename Cfg>
auto as_evaluating_split_walker_maker(const tensor_base<ValT,Cfg>& t){return dynamic_cast<const evaluating_split_walker_maker<ValT,Cfg>*>(&t);}



class dispatcher{    
    template<typename FirstT, typename ValT2, template<typename> typename Cfg, typename F>
    static auto dispatch_second(F& f, const FirstT& first, const tensor_base<ValT2, Cfg>& second){
        if (second.tensor_kind() == tensor_kinds::storage_tensor)
        {
            return f(first, *as_storage_split_walker_maker(second));
        }
        else if (second.tensor_kind() == tensor_kinds::expression)
        {            
            return f(first, *as_evaluating_split_walker_maker(second));
        }        
        else
        {
            throw dispatch_exception("type is not supported by dispatcher");
        }
    }        
    template<typename ValT1, typename ValT2, template<typename> typename Cfg, typename F>
    static auto dispatch_first(F& f, const tensor_base<ValT1, Cfg>& first, const tensor_base<ValT2, Cfg>& second){
        if (first.tensor_kind() == tensor_kinds::storage_tensor)
        {
            return dispatch_second(f, *as_storage_split_walker_maker(first), second);
        }
        else if (first.tensor_kind() == tensor_kinds::expression)
        {            
            return dispatch_second(f, *as_evaluating_split_walker_maker(first), second);
        }                
        else
        {
            throw dispatch_exception("type is not supported by dispatcher");
        }
    }    
public:        
    template<typename ValT1, typename ValT2, template<typename> typename Cfg, typename F>
    static auto call(const F& f, const tensor_base<ValT1, Cfg>& first, const tensor_base<ValT2, Cfg>& second){
        return dispatch_first(f,first,second);
    }
};


template<typename ValT, template<typename> typename Cfg>
class test_storage_tensor : 
    public storage_tensor<ValT,Cfg>,    
    public storage_split_walker_maker<ValT,Cfg>
{             
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;            

    storage_accessor<ValT,Cfg> create_storage_split_walker()const override{
        return storage_accessor<ValT,Cfg>{shape(),strides(),index_type{0},data()};
    }
public:    
    using storage_tensor::storage_tensor;
};

template<typename ValT, template<typename> typename Cfg>
class test_tensor : public tensor<ValT, Cfg>
{    
    using storage_tensor_type = test_storage_tensor<ValT,Cfg>;
    using iterator_type = multiindex_iterator<ValT,Cfg,walker<ValT,Cfg>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<Cfg<ValT>>::type;

    strides_type strides{gtensor::detail::make_dividers<Cfg<ValT>>(impl()->strides())};

    template<typename Nested>
    test_tensor(std::initializer_list<Nested> init_data, int):        
        tensor(std::make_shared<storage_tensor_type>(init_data))
    {}
public:    
    test_tensor() = default;
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):test_tensor(init_data,0){}
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):test_tensor(init_data,0){}
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):test_tensor(init_data,0){}
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):test_tensor(init_data,0){}
    test_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):test_tensor(init_data,0){}

    template<typename...Dims>
    test_tensor(const value_type& v, const Dims&...dims):        
        tensor(std::make_shared<storage_tensor_type>(v, dims...))
    {}

    test_tensor(std::shared_ptr<tensor_base<ValT,Cfg>>&& impl__):
        tensor(std::move(impl__))
    {}
    
    auto impl()const{return tensor::impl();}
    auto begin()const{
        return iterator_type{
            walker<ValT,Cfg>{dynamic_cast<const evaluating_split_walker_maker<ValT,Cfg>*>(impl()->impl().get())->create_split_walker()},
            impl()->shape(), 
            strides
        };
    }
    auto end()const{
        return iterator_type{
            walker<ValT,Cfg>{dynamic_cast<const evaluating_split_walker_maker<ValT,Cfg>*>(impl()->impl().get())->create_split_walker()},
            impl()->shape(),
            strides,
            impl()->size()
        };
    }    
};

template<typename ValT, template<typename> typename Cfg>
auto as_evaluating_split_walker_maker(const test_tensor<ValT,Cfg>& t){return dynamic_cast<const evaluating_split_walker_maker<ValT,Cfg>*>(t.impl()->impl().get());}

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class test_evaluating_tensor : 
    public evaluating_tensor<ValT,Cfg,F,Ops...>,
    public evaluating_split_walker_maker<ValT,Cfg>
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;    
    using evaluator_type = evaluator<ValT,Cfg>;
    
    struct split_walker_maker{
        template<typename...Args>
        evaluator_type operator()(const Args&...args)const{
            using evaluating_evaluator_type = evaluating_evaluator<ValT,Cfg,F,decltype(args.create_split_walker())...>;
            return evaluator_type{std::make_unique<evaluating_evaluator_type>(args.create_split_walker()...)};
        }        
    };

    template<std::size_t...I>
    auto create_split_walker_helper(std::index_sequence<I...>)const{return dispatcher::call(split_walker_maker{},*operand<I>()...);}    
    
    evaluator_type create_evaluating_split_walker()const override{ return create_split_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});}
    
public:
    using evaluating_tensor::evaluating_tensor;    
};

template<typename ValT1, typename ValT2, template<typename> typename Cfg>
static inline auto operator+(const test_tensor<ValT1, Cfg>& op1, const test_tensor<ValT2, Cfg>& op2){
    using operation_type = add;
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));
    using exp_operand1_type = std::shared_ptr<tensor_base<ValT1,Cfg>>;
    using exp_operand2_type = std::shared_ptr<tensor_base<ValT2,Cfg>>;
    using exp_type = test_evaluating_tensor<result_type, Cfg, operation_type, exp_operand1_type, exp_operand2_type>;
    return test_tensor<result_type,Cfg>{std::make_shared<exp_type>(op1.impl(),op2.impl())};
}


}   //namespace separate_evaluation_v1

namespace true_expression_template{
using gtensor::storage_tensor;
using gtensor::storage_walker;
using gtensor::evaluating_tensor;
using gtensor::evaluating_walker;
using gtensor::storage_walker_factory;
using gtensor::evaluating_walker_polymorphic;
using gtensor::storage_walker_polymorphic;
using gtensor::binary_operations::add;
using gtensor::multiindex_iterator;
using gtensor::basic_walker;

template<typename ValT, template<typename> typename Cfg>
class test_stensor : public storage_tensor<ValT,Cfg>
{ 
    using base_stensor = storage_tensor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

public:    
    test_stensor() = default;
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):base_stensor(init_data){}

    template<typename...Dims>
    test_stensor(const value_type& v, const Dims&...dims):
        base_stensor(v, dims...)
    {}    
    auto create_concrete_walker()const{return storage_walker<ValT, Cfg>{shape(),strides(),data()};}  
    bool is_trivial()const{return storage_tensor::is_trivial();}
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class test_etensor : public evaluating_tensor<ValT,Cfg,F,Ops...>
{
    template<std::size_t...I>
    auto create_concrete_walker_helper(std::index_sequence<I...>)const{        
        using walker_type = evaluating_walker<ValT,Cfg,F, decltype(std::declval<Ops>()->create_concrete_walker())...>;
        return walker_type{shape(),operand<I>()->create_concrete_walker()...};
    }    
public:
    using evaluating_tensor::evaluating_tensor;

    bool is_trivial()const{return evaluating_tensor::is_trivial();}
    
    auto create_concrete_walker()const{return create_concrete_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});}
    auto begin()const{
        using iterator_type = multiindex_iterator<ValT,Cfg,decltype(std::declval<test_etensor>().create_concrete_walker())>;
        return iterator_type{create_concrete_walker(),shape(),gtensor::detail::strides_div(concrete_descriptor())};
    }
    auto end()const{
        using iterator_type = multiindex_iterator<ValT,Cfg,decltype(std::declval<test_etensor>().create_concrete_walker())>;
        return iterator_type{create_concrete_walker(),shape(),gtensor::detail::strides_div(concrete_descriptor()),size()};
    }
};

template<typename ValT, template<typename> typename Cfg, typename ImplT = test_stensor<ValT,Cfg>>
class static_tensor{
    using base_stensor = storage_tensor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

    std::shared_ptr<ImplT> impl_; 

    template<typename Nested>
    static_tensor(std::initializer_list<Nested> init_data,int):
        impl_{new ImplT(init_data)}
    {}   

public:
    static_tensor() = default;
    static_tensor(std::shared_ptr<ImplT>&& impl__):
        impl_{std::move(impl__)}
    {}

    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):static_tensor(init_data,0){}

    template<typename...Dims>
    static_tensor(const value_type& v, const Dims&...dims):
        impl_{new ImplT(v, dims...)}
    {}
    auto impl()const{return impl_;}    
    const auto& shape()const{return impl()->shape();}
    auto create_concrete_walker()const{return impl()->create_concrete_walker();}
    auto begin()const{return impl()->begin();}
    auto end()const{return impl()->end();}
};

template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, template<typename> typename Cfg>
static inline auto operator+(const static_tensor<ValT1, Cfg, ImplT1>& op1, const static_tensor<ValT2, Cfg, ImplT2>& op2){
    using operation_type = add;
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));
    using operand1_type = std::shared_ptr<ImplT1>;
    using operand2_type = std::shared_ptr<ImplT2>;
    using exp_type = test_etensor<result_type, Cfg, operation_type, operand1_type, operand2_type>;
    return static_tensor<result_type,Cfg, exp_type>{std::make_shared<exp_type>(op1.impl(),op2.impl())};
}

}   //end of namespace true_expression_template

namespace true_expression_template_split{
using gtensor::storage_tensor;
using gtensor::storage_walker;
using gtensor::evaluating_tensor;
using gtensor::evaluating_walker;
using gtensor::storage_walker_factory;
using gtensor::evaluating_walker_polymorphic;
using gtensor::storage_walker_polymorphic;
using gtensor::binary_operations::add;
using gtensor::multiindex_iterator;
using gtensor::basic_walker;

template<typename IdxT, typename ShT>
class storage_walking_walker
{    
    using index_type = IdxT;
    using shape_type = ShT;
    using shape_inverter_type = gtensor::detail::shape_inverter<index_type,shape_type>;
        
    index_type dim_;
    shape_inverter_type shape_;
    shape_inverter_type strides_;
    index_type offset_;
    index_type* cursor_;    

public:
    storage_walking_walker(const index_type& dim__, const shape_inverter_type& shape__, const shape_inverter_type& strides__, const index_type& offset__, index_type* cursor__):
        dim_{dim__},
        shape_{shape__},
        strides_{strides__},
        offset_{offset__},
        cursor_{cursor__}
    {}
    storage_walking_walker(const shape_inverter_type& shape__, const shape_inverter_type& strides__):
        shape_{shape__},
        strides_{strides__}
    {}
    
    void walk(const index_type& direction, const index_type& steps){
        if (gtensor::detail::can_walk(direction, dim_, shape_.element(direction))){
            *cursor_+=steps*strides_.element(direction);
        }   
    }
    void step(const index_type& direction){
        if (gtensor::detail::can_walk(direction, dim_, shape_.element(direction))){
            *cursor_+=strides_.element(direction);
        }
    }
    void step_back(const index_type& direction){        
        if (gtensor::detail::can_walk(direction, dim_, shape_.element(direction))){
            *cursor_-=strides_.element(direction);
        }            
    }
    void reset(const index_type& direction){
        if (gtensor::detail::can_walk(direction, dim_, shape_.element(direction))){
            *cursor_-=(shape_.element(direction)-1)*strides_.element(direction);
        }
    }
    void reset(){*cursor_ = offset;}    
};

template<typename ValT, template<typename> typename Cfg>
class storage_accessor
{   
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;
    
    index_type dim_;
    gtensor::detail::shape_inverter<index_type, shape_type> shape_;
    gtensor::detail::shape_inverter<index_type, shape_type> strides_;
    index_type offset_;
    const ValT* data_;
    index_type cursor_{offset_};

public:    
    storage_accessor(const shape_type& shape__, const shape_type& strides__, const index_type& offset__, const ValT* data__):
        dim_{static_cast<index_type>(shape__.size())},
        shape_{shape__},
        strides_{strides__},
        offset_{offset__},
        data_{data__}
    {}
    auto create_walker(){return storage_walking_walker<index_type,shape_type>{dim_,shape_,strides_,offset_,&cursor_};}    
    ValT operator*() const {return data_[cursor_];}
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...NodesT>
class evaluating_evaluator
{
    using index_type = typename Cfg<ValT>::index_type;
    using shape_type = typename Cfg<ValT>::shape_type;

    std::size_t storages_number_;
    std::tuple<NodesT...> nodes;
    F f{};

    template<std::size_t...I>    
    ValT deref_helper(std::index_sequence<I...>) const {return f(*std::get<I>(nodes)...);}    
    
    template<typename V, typename...Ts>
    void create_walker(V& v, storage_accessor<Ts...>& w){v.push_back(w.create_walker());}
    
    template<typename V, typename...Ts>
    void create_walker(V& v, evaluating_evaluator<Ts...>& w){w.create_walkers_helper(v);}
    
    template<std::size_t...I>    
    void create_walkers_helper(std::vector<storage_walking_walker<index_type,shape_type>>& walkers, std::index_sequence<I...>){
        (create_walker(walkers, std::get<I>(nodes)),...);
    }
    
    template<typename...Ts>
    std::size_t storages_counter(const storage_accessor<Ts...>& w)const{return 1;}
    
    template<typename...Ts>
    std::size_t storages_counter(const evaluating_evaluator<Ts...>& w)const{return w.storages_number();}

public:
    evaluating_evaluator(NodesT&&...nodes_):
        storages_number_{(storages_counter(nodes_)+...)},
        nodes{std::move(nodes_)...}
    {}    
    
    ValT operator*() const {return deref_helper(std::make_index_sequence<sizeof...(NodesT)>{});}
    
    std::size_t storages_number()const{return storages_number_;}
    
    void create_walkers_helper(std::vector<storage_walking_walker<index_type,shape_type>>& walkers){
        create_walkers_helper(walkers,std::make_index_sequence<sizeof...(NodesT)>{});
    }

    auto create_walkers(){
        std::vector<storage_walking_walker<index_type,shape_type>> walkers{};        
        walkers.reserve(storages_number());
        create_walkers_helper(walkers);
        return walkers;
    }
};

template<typename ValT, template<typename> typename Cfg, typename EvalT>
class walker
{
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using evaluator_type = EvalT;

    evaluator_type evaluator;
    std::vector<storage_walking_walker<index_type,shape_type>> walkers{evaluator.create_walkers()};

public:
    walker(evaluator_type&& evaluator_):        
        evaluator{std::move(evaluator_)}        
    {}

    walker(walker&& other):        
        evaluator{std::move(other.evaluator)}
    {}
    
    walker(const walker& other):        
        evaluator{other.evaluator}
    {}

    walker& walk(const index_type& direction, const index_type& steps){
        for (auto& walker : walkers)
        {walker.walk(direction,steps);}
        return *this;
    }
    walker& step(const index_type& direction){
        for (auto& walker : walkers)
        {walker.step(direction);}
        return *this;
    }
    walker& step_back(const index_type& direction){
        for (auto& walker : walkers)
        {walker.step_back(direction);}
        return *this;
    }
    walker& reset(const index_type& direction){
        for (auto& walker : walkers)
        {walker.reset(direction);}
        return *this;
    }
    walker& reset(){
        for (auto& walker : walkers)
        {walker.reset();}
        return *this;
    }    
    value_type operator*() const{return *evaluator;}
};

template<typename ValT, template<typename> typename Cfg>
class test_stensor : public storage_tensor<ValT,Cfg>
{ 
    using base_stensor = storage_tensor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

public:    
    test_stensor() = default;
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):base_stensor(init_data){}

    template<typename...Dims>
    test_stensor(const value_type& v, const Dims&...dims):
        base_stensor(v, dims...)
    {}    
    auto create_concrete_walker()const{return storage_accessor<ValT, Cfg>{shape(), strides(), index_type{0}, data()};}  
    bool is_trivial()const{return storage_tensor::is_trivial();}
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class test_etensor : public evaluating_tensor<ValT,Cfg,F,Ops...>
{
    template<std::size_t...I>
    auto create_concrete_walker_helper(std::index_sequence<I...>)const{        
        using walker_type = evaluating_evaluator<ValT,Cfg,F, decltype(std::declval<Ops>()->create_concrete_walker())...>;
        return walker_type{operand<I>()->create_concrete_walker()...};
    }    
public:
    using evaluating_tensor::evaluating_tensor;

    bool is_trivial()const{return evaluating_tensor::is_trivial();}
    
    auto create_concrete_walker()const{return create_concrete_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});}
    auto begin()const{
        using walker_type = walker<ValT, Cfg, decltype(std::declval<test_etensor>().create_concrete_walker())>;
        using iterator_type = multiindex_iterator<ValT, Cfg, walker_type>;
        return iterator_type{walker_type{create_concrete_walker()},shape(),gtensor::detail::strides_div(concrete_descriptor())};
    }
    auto end()const{
        using walker_type = walker<ValT, Cfg, decltype(std::declval<test_etensor>().create_concrete_walker())>;
        using iterator_type = multiindex_iterator<ValT, Cfg, walker_type>;
        return iterator_type{walker_type{create_concrete_walker()},shape(),gtensor::detail::strides_div(concrete_descriptor()),size()};
    }
};

template<typename ValT, template<typename> typename Cfg, typename ImplT = test_stensor<ValT,Cfg>>
class static_tensor{
    using base_stensor = storage_tensor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

    std::shared_ptr<ImplT> impl_; 

    template<typename Nested>
    static_tensor(std::initializer_list<Nested> init_data,int):
        impl_{new ImplT(init_data)}
    {}   

public:
    static_tensor() = default;
    static_tensor(std::shared_ptr<ImplT>&& impl__):
        impl_{std::move(impl__)}
    {}

    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):static_tensor(init_data,0){}

    template<typename...Dims>
    static_tensor(const value_type& v, const Dims&...dims):
        impl_{new ImplT(v, dims...)}
    {}
    auto impl()const{return impl_;}    
    const auto& shape()const{return impl()->shape();}
    auto create_concrete_walker()const{return impl()->create_concrete_walker();}
    auto begin()const{return impl()->begin();}
    auto end()const{return impl()->end();}
};

template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, template<typename> typename Cfg>
static inline auto operator+(const static_tensor<ValT1, Cfg, ImplT1>& op1, const static_tensor<ValT2, Cfg, ImplT2>& op2){
    using operation_type = add;
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));
    using operand1_type = std::shared_ptr<ImplT1>;
    using operand2_type = std::shared_ptr<ImplT2>;
    using exp_type = test_etensor<result_type, Cfg, operation_type, operand1_type, operand2_type>;
    return static_tensor<result_type,Cfg, exp_type>{std::make_shared<exp_type>(op1.impl(),op2.impl())};
}

}   //end of namespace true_expression_template


TEST_CASE("test_benchmark_helper_classes","[benchmark_walker]"){
    using value_type = float;
    using gtensor::config::default_config;
    using gtensor::multiindex_iterator;
    using separate_evaluation::walker;
    
    using split_eval_tensor_type = separate_evaluation::test_tensor<value_type, default_config>;    
    using iterator_type = multiindex_iterator<value_type,default_config,walker<value_type,default_config>>;    
    


    SECTION("test_split_eval_tensor_v1_starages_number"){
        using split_eval_tensor_v1_type = separate_evaluation_v1::test_tensor<value_type, default_config>;
        using separate_evaluation_v1::as_evaluating_split_walker_maker;
        using test_type = std::tuple<std::size_t, std::size_t>;

        auto get_n = [](const auto& e){return as_evaluating_split_walker_maker(e)->create_split_walker().impl()->storages_number();};
        split_eval_tensor_v1_type t1{{1,2,3}};
        split_eval_tensor_v1_type t2{{1},{2},{3}};
        split_eval_tensor_v1_type t3{-2};

        auto test_data = GENERATE_REF(
            test_type{get_n(split_eval_tensor_v1_type{1,2,3} + split_eval_tensor_v1_type{1,2,3}), 2},
            test_type{get_n(t1+t1), 2},
            test_type{get_n(t1+t1+t1), 3},
            test_type{get_n(t1+(t1+t1)+t2), 4},
            test_type{get_n(t2+(t1+t1)+t3), 4}
        );

        auto storages_number = std::get<0>(test_data);
        auto expected_storages_number = std::get<1>(test_data);
        REQUIRE(storages_number == expected_storages_number);
    }
    
    SECTION("test_split_eval_tensor_v1_iterator"){
        using split_eval_tensor_v1_type = separate_evaluation_v1::test_tensor<value_type, default_config>;
        split_eval_tensor_v1_type t1{{1,2,3}};
        split_eval_tensor_v1_type t2{{1},{2},{3}};
        split_eval_tensor_v1_type t3{-2};
        split_eval_tensor_v1_type e = t2+t1+t2+t3;     
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }
    
    SECTION("test_split_eval_tensor_iterator"){

        split_eval_tensor_type t1{{1,2,3}};
        split_eval_tensor_type t2{{1},{2},{3}};
        split_eval_tensor_type t3{-2};
        split_eval_tensor_type e = t2+t1+t2+t3;
        // auto e_begin = e.begin();
        // auto e_end = e.end();        
        // for(const auto& i : e){
        //     std::cout<<i;
        // }
        
        //REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));                
    }

    SECTION("test_noinline_tensor_iterator"){
        using noinline_tensor_type = noinline_evaluation::test_tensor_noinline<value_type, default_config>;

        noinline_tensor_type t1{{1,2,3}};
        noinline_tensor_type t2{{1},{2},{3}};
        noinline_tensor_type t3{-2};
        noinline_tensor_type e = t2+t1+t2+t3;     
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }
    SECTION("test_partly_inline_tensor_iterator"){
        using partly_inline_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, default_config>;

        partly_inline_tensor_type t1{{1,2,3}};
        partly_inline_tensor_type t2{{1},{2},{3}};
        partly_inline_tensor_type t3{-2};
        partly_inline_tensor_type e = t2+t1+t2+t3;         
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }
    SECTION("test_full_inline_walker_iterator"){
        using full_inline_tensor_type = true_expression_template::static_tensor<value_type, default_config>;

        full_inline_tensor_type t1{1,2,3};
        full_inline_tensor_type t2{{1},{2},{3}};
        full_inline_tensor_type t3{-2};        
        auto e = t2+t1+t2+t3;        
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }
    SECTION("test_full_inline_split_walker_iterator"){
        using full_inline_split_tensor_type = true_expression_template_split::static_tensor<value_type, default_config>;

        full_inline_split_tensor_type t1{1,2,3};
        full_inline_split_tensor_type t2{{1},{2},{3}};
        full_inline_split_tensor_type t3{-2};        
        auto e = t2+t1+t2+t3;        
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }
}

TEMPLATE_TEST_CASE("benchmark_walker","[benchmark_walker]", gtensor::config::mode_div_native){
    using value_type = float;
    using config_type = test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl<value_type>;
    using shape_type = typename config_type::shape_type;    
    using tensor_type = gtensor::tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;        
    using noinline_tensor_type = noinline_evaluation::test_tensor_noinline<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using partly_inline_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using full_inline_tensor_type = true_expression_template::static_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using full_inline_split_tensor_type = true_expression_template_split::static_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using split_eval_tensor_type = separate_evaluation::test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>; 
    using split_eval_tensor_v1_type = separate_evaluation_v1::test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using benchmark_walker::make_asymmetric_tree;
    using benchmark_walker::make_symmetric_tree;
    
    auto iterate_without_deref = [](const auto& t){
        auto t_it = t.begin();
        auto t_end = t.end();
        std::size_t c{};
        while (t_it!=t_end){            
            ++c;            
            ++t_it;
        }
        return c;
    };
    
    auto iterate_with_deref = [](const auto& t){
        auto t_it = t.begin();
        auto t_end = t.end();
        std::size_t c{};
        while (t_it!=t_end){
            if (*t_it > 2){
                ++c;
            }
            ++t_it;
        }
        return c;
    };
    
    auto just_iterate_with_deref = [](auto& it_begin, auto& it_end){        
        std::size_t c{};        
        while (it_begin!=it_end){
            if (*it_begin > 2){
                ++c;
            }
            ++it_begin;
        }
        return c;
    };
    
    auto make_iterators = [](std::size_t n, const auto& t){
        return std::vector<std::pair<decltype(t.begin()), decltype(t.end())>>(n, std::make_pair(t.begin(), t.end()));
    };

    // shape_type shape1{1,1,3,1,5,1,7,1,9,1};
    // shape_type shape2{1,2,1,4,1,6,1,8,1,10};
    
    // shape_type shape1{1,2,1,4,1,6,1,8,1,10};
    // shape_type shape2{1,2,3,4,5,6,7,8,9,10};

    shape_type shape1{1, 10000};
    shape_type shape2{10,10000};
    
    // shape_type shape1{1,3000};
    // shape_type shape2{3000,1};
    
    // shape_type shape1{1,10000};
    // shape_type shape2{10000,1};

    static constexpr std::size_t tree_depth = 50;
    auto make_tree = [](const auto& t1, const auto& t2){return make_asymmetric_tree<tree_depth>(t1,t2);};

    full_inline_tensor_type t1_full(0, shape1);
    full_inline_tensor_type t2_full(0, shape2);
    auto e_full = make_tree(t1_full,t2_full);    
    
    full_inline_split_tensor_type t1_full_split(0, shape1);
    full_inline_split_tensor_type t2_full_split(0, shape2);
    auto e_full_split = make_tree(t1_full_split,t2_full_split);    
    
    split_eval_tensor_v1_type t1_split_v1(0, shape1);
    split_eval_tensor_v1_type t2_split_v1(0, shape2);
    split_eval_tensor_v1_type e_split_v1 = make_tree(t1_split_v1, t2_split_v1);
    
    tensor_type t1(0, shape1);
    tensor_type t2(0, shape2);
    partly_inline_tensor_type e_inline = make_tree(t1,t2);
    
    noinline_tensor_type t1_noinline(0, shape1);
    noinline_tensor_type t2_noinline(0, shape2);
    noinline_tensor_type e_noinline = make_tree(t1_noinline,t2_noinline);    
    
    // BENCHMARK_ADVANCED("full_inline_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
    //     auto v = make_iterators(meter.runs(),e_full);
    //     meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    // };
    // BENCHMARK_ADVANCED("split_v1_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
    //     auto v = make_iterators(meter.runs(),e_split_v1);
    //     meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    // };
    // BENCHMARK_ADVANCED("partly_inline_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
    //     auto v = make_iterators(meter.runs(),e_inline);
    //     meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    // };
    // BENCHMARK_ADVANCED("noinline_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
    //     auto v = make_iterators(meter.runs(),e_noinline);
    //     meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    // };
    
    BENCHMARK_ADVANCED("full_inline_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
        meter.measure([&iterate_with_deref, &e_full] { return iterate_with_deref(e_full); });
    };    
    BENCHMARK_ADVANCED("full_inline_split_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
        meter.measure([&iterate_with_deref, &e_full_split] { return iterate_with_deref(e_full_split); });
    };    
    BENCHMARK_ADVANCED("split_v1_inline_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
        meter.measure([&iterate_with_deref, &e_split_v1] { return iterate_with_deref(e_split_v1); });
    };
    BENCHMARK_ADVANCED("partly_inline_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
        meter.measure([&iterate_with_deref, &e_inline] { return iterate_with_deref(e_inline); });
    };
    // BENCHMARK_ADVANCED("noinline_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
    //     meter.measure([&iterate_with_deref, &e_noinline] { return iterate_with_deref(e_noinline); });
    // };    
    
    // BENCHMARK_ADVANCED("full_inline_iterator_construction_iteration_without_dereference")(Catch::Benchmark::Chronometer meter) {        
    //     meter.measure([&iterate_without_deref, &e_full] { return iterate_without_deref(e_full); });
    // };
    // BENCHMARK_ADVANCED("split_v1_inline_iterator_construction_iteration_without_dereference")(Catch::Benchmark::Chronometer meter) {        
    //     meter.measure([&iterate_without_deref, &e_split_v1] { return iterate_without_deref(e_split_v1); });
    // };
    // BENCHMARK_ADVANCED("partly_inline_iterator_construction_iteration_without_dereference")(Catch::Benchmark::Chronometer meter) {        
    //     meter.measure([&iterate_without_deref, &e_inline] { return iterate_without_deref(e_inline); });
    // };
    // BENCHMARK_ADVANCED("noinline_iterator_construction_iteration_without_dereference")(Catch::Benchmark::Chronometer meter) {        
    //     meter.measure([&iterate_without_deref, &e_noinline] { return iterate_without_deref(e_noinline); });
    // };    
}