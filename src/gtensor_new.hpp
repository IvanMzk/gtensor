#ifndef GTENSOR_NEW_HPP_
#define GTENSOR_NEW_HPP_

#include <memory>
#include "config.hpp"

template<typename> class basic_tensor;

template<typename Impl> class basic_tensor<std::shared_ptr<Impl>>{

};


//Impl may be storage implementation , view implementation, expression implementation
template<typename Impl>
class basic_tensor
{
    using impl_type = ImplT;
    std::shared_ptr<impl_type> impl_;

    using config_type = impl_type::config_type;
    using value_type = impl_type::value_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using slice_type = slice<index_type>;
    using slice_item_type = typename slice_type::slice_item_type;
    using size_type = index_type;
    using difference_type = index_type;

    //constructor is used by make_tensor static method to create tensor with explicitly specified implementation
    basic_tensor(std::shared_ptr<impl_type>&& impl__):
        impl_{std::move(impl__)}
    {}

    //no constructors to make tensor with storage implementation here

    //makes tensor with explicitly specified implementation type
    //Implementation must be convertible to tensor::impl_type
    template<typename Implementation, typename...Args>
    static tensor make_tensor(Args&&...args){
        return tensor(std::make_shared<Implementation>(std::forward<Args>(args)...));
    }


    //tensor interface
    //size()
    //shape()
    //...
    //reshape(){
    //  return basic_tensor<view_impl_base_type>(create_slice_view(...));
    //or
    //  return basic_tensor<decltype(create_slice_view(...))>(create_slice_view(...));
    //}


    //...
    //copy(){
    //  return basic_tensor{tensor_factory_selector<config_type,value_type>::create_tensor(shape(),begin(),end())};
    //or
    //  return create_tensor(shape(),begin(),end());    //free method select factory
    //or
    //  return tensor{shape(),begin(),end()};   //if no tensor factory
    //}
};


template<typename Config, typename...Ts>
class tensor_factory_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename Dummy, typename T> struct selector_<config::engine_expression_template,Dummy,T>
    {
        //using base_type =
        //using factory_type =
        //using factory_method =
        //using factory_function =
        //where factory must return convertible to base_type*, may be template, since base_type is known


        //tensor_implementation<storage_engine<Config,T>>;
    };
public:
    using type = typename selector_<typename config_type::engine, void, Ts...>::type;
    using base_type = typename selector_<typename config_type::engine, void, Ts...>::base_type;
};



//there is ONLY one storage implementation of tensor for given Config
template<typename Config, typename...Ts>
class tensor_implementation_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename Dummy, typename T> struct selector_<config::engine_expression_template,Dummy,T>
    {
        using type = tensor_implementation<storage_engine<Config,T>>;
        using base_type = type;
    };
public:
    using type = typename selector_<typename config_type::engine, void, Ts...>::type;
    using base_type = typename selector_<typename config_type::engine, void, Ts...>::base_type;
};

//or use factory_method and common_type


//tensor is basic_tensor adapter with storage implementation ONLY and constructors
//use private inheritance or composition
template<typename T, typename Config = gtensor::config::default_config>
class tensor
    //: basic_tensor<typename tensor_implementation_selector<Config,T>::base_type>
    //: basic_tensor<typename tensor_factory_selector<Config,T>::type::result_type>   //need base of storage implementation type here
    //: basic_tensor<decltype(create_tensor(???))>
{
    //using basic_tensor_base_type = ...
    //aliases
    //using basic_tensor_base_type::config_type
    //...

    //constructors to make tensor with storage implementation only
    //tensor(const shape_type&, const value_type&):
    //  basic_tensor_base_type{create_tensor<config_type, value_type>(...)}  //use copy constructor of basic_tensor, create_tensor should return ?? tensor<T,Config> type, which is convertible to basic_tensor<...>
    //  why not return basic_tensor<...>
    //{}
    //...

    //interface
    //using basic_tensor_type::size
    //using basic_tensor_type::shape
    //...
};

//or use alias
//in case of alias variant basic_tensor must have constructors to make tensor with storage implementation
//template<typename T, typename Config> using tensor = basic_tensor<impl_selector<T,Config>::type>;


//type_selector module is compile time factory implementation selector
//factory module interface uses type_selector to get right factory and call factory interface
//factory is tensor implementation selector

//factories intefaces should return pointer to implementation instance created by factory
//that pointer may be of dynamic type i.e. pointer to abstract base
//or it may of created implementation instance type

//factory make decision about tensor implementation type
//factory make decision about tensor implementation base type i.e. return pointer type
class some_view_factory
{

    //template<typename...Ts> using view_implementation = tensor_implementation<view_engine<Ts...>>;
    //view_implementation_base_type is same as view_implementation_type - it is behaviour of this specific factory

    //some_view_factory implementation
public:
    //some_view_factory interface
    //static auto create_slice_view(some_arguments){
    //  using slice_view_type = view_implementation<...>;
    //  return std::make_shared<slice_view_type>(...);
    //}

};

//module interface
//auto create_slice_view(some_arguments){
//  value_type, config_type get from view parent type
//  return factory_selector<...>::type::create_slice_view(...);
//}


//to make tensor with storage implementation value type and config must be proveded
//storage tensor implementation factory
template<typename Config, typename T>
class some_tensor_factory
{
    //template<typename...Ts> using storage_tensor_implementation = tensor_implementation<storage_engine<Ts...>>;
    //storage_tensor_implementation_base_type is same as storage_tensor_implementation_type - it is behaviour of this specific factory

    //using result_type = tensor_implementation<storage_engine<Config,T>>;

    //some_tensor_factory implementation
public:
    //some_tensor_factory interface
    //static auto create_tensor(shape_type&, value_type&){
    //  using storage_implementation_type = result_type;
    //  return std::static_pointer_cast<result_type>(std::make_shared<storage_implementation_type>(...));
    //or
    //  return
    //}

};

//tensor factory module interface
//template<typename Config, typename T, typename...Args>
//auto create_tensor(Args&&...args){
//  using result_type = ...
//  static_assert(that factory return result type or convertible to it)
//  return ...
//}

#endif