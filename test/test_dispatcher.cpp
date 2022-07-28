#include <typeindex>
#include "catch.hpp"
#include "tensor.hpp"
#include "impl_expression.hpp"
#include "dispatcher.hpp"


namespace test_dispatcher{
using gtensor::expression_impl;

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
struct test_expression_impl : public expression_impl<ValT,Cfg,F,Ops...>{
    using base_type = expression_impl<ValT,Cfg,F,Ops...>;
    bool is_storage_;
    test_expression_impl(bool is_storage__, Ops&...operands):
        base_type{operands...},
        is_storage_{is_storage__}
    {}
    bool is_storage()const override{return is_storage_;}
};

struct test_operation{
    template<typename T1, typename T2>
    auto operator()(const T1& t1, const T2& t2){
        return std::make_tuple(std::type_index{typeid(T1)}, std::type_index{typeid(T2)});
    }
};

}   //end of namespace test_dispatcher


TEST_CASE("test_dispatcher","[test_dispatcher]"){
    using value_type = float;
    using gtensor::binary_operations::add;
    using gtensor::stensor_impl;
    using gtensor::tensor_impl_base;
    using gtensor::config::default_config;
    using config_type = gtensor::config::default_config<value_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using tensor_impl_base_type = tensor_impl_base<value_type, default_config>;
    using stensor_impl_type = stensor_impl<value_type, default_config>;
    using test_dispatcher::test_expression_impl;
    using test_expression_impl_type = test_expression_impl<value_type, default_config, add, std::shared_ptr<tensor_impl_base_type>, std::shared_ptr<tensor_impl_base_type>>;    
    using gtensor::detail::tensor_kinds;
    using view_factory_type = gtensor::view_factory<value_type,default_config>;

    auto t1 = std::static_pointer_cast<tensor_impl_base_type>(std::shared_ptr<stensor_impl_type>{new stensor_impl_type{1,2,3}});
    auto t2 = std::static_pointer_cast<tensor_impl_base_type>(std::shared_ptr<stensor_impl_type>{new stensor_impl_type{{1},{2},{3}}});    

    auto e_trivial = std::static_pointer_cast<tensor_impl_base_type>(std::shared_ptr<test_expression_impl_type>(new test_expression_impl_type{false, t1,t1}));
    auto e = std::static_pointer_cast<tensor_impl_base_type>(std::shared_ptr<test_expression_impl_type>(new test_expression_impl_type{false, t1,t2}));
    auto ce_trivial = std::static_pointer_cast<tensor_impl_base_type>(std::shared_ptr<test_expression_impl_type>(new test_expression_impl_type{true, t1,t1}));
    auto ce = std::static_pointer_cast<tensor_impl_base_type>(std::shared_ptr<test_expression_impl_type>(new test_expression_impl_type{true, t1,t2}));

    auto v_of_t = view_factory_type::create_view_subdim(t2,shape_type{});
    auto v_of_e = view_factory_type::create_view_subdim(e,shape_type{});
    auto v_of_ce = view_factory_type::create_view_subdim(ce,shape_type{});
    auto v_of_v = view_factory_type::create_view_subdim(v_of_t,shape_type{});

    SECTION("test_inputs"){
        using test_type = std::tuple<std::shared_ptr<tensor_impl_base_type>&,tensor_kinds,bool,bool>;
        //tensor,expected_tensor_kind,expected_is_storage,expected_is_trivial
        auto test_data = GENERATE_REF(
            test_type{t1,tensor_kinds::storage_tensor,true,true},
            test_type{t2,tensor_kinds::storage_tensor,true,true},
            test_type{e_trivial,tensor_kinds::expression,false,true},
            test_type{e,tensor_kinds::expression,false,false},
            test_type{ce_trivial,tensor_kinds::expression,true,true},
            test_type{ce,tensor_kinds::expression,true,false},
            test_type{v_of_t,tensor_kinds::view,true,true},
            test_type{v_of_e,tensor_kinds::view,false,true},
            test_type{v_of_ce,tensor_kinds::view,true,true},
            test_type{v_of_v,tensor_kinds::view,false,true}
        );
        auto tensor = std::get<0>(test_data);
        auto expected_tensor_kind = std::get<1>(test_data);
        auto expected_is_storage = std::get<2>(test_data);
        auto expected_is_trivial = std::get<3>(test_data);
        REQUIRE(tensor->tensor_kind() == expected_tensor_kind);
        REQUIRE(tensor->is_storage() == expected_is_storage);
        REQUIRE(tensor->is_trivial() == expected_is_trivial);
    }

    SECTION("test_dispatcher"){
        using storage_type = gtensor::storage_tensor_impl_base<value_type, default_config>;
        using expression_type = gtensor::expression_impl_base<value_type, default_config>;
        using trivial_type = gtensor::trivial_impl_base<value_type, default_config>;
        using view_type = gtensor::view_impl_base<value_type, default_config>;
        using test_dispatcher::test_operation;
        using dispatcher_type = gtensor::detail::dispatcher<value_type,default_config>;
        using test_type = std::tuple<std::tuple<std::type_index,std::type_index>,std::tuple<std::type_index,std::type_index>>;

        auto s_s = std::make_tuple(std::type_index{typeid(storage_type)}, std::type_index{typeid(storage_type)});
        auto s_e = std::make_tuple(std::type_index{typeid(storage_type)}, std::type_index{typeid(expression_type)});
        auto s_t = std::make_tuple(std::type_index{typeid(storage_type)}, std::type_index{typeid(trivial_type)});
        auto s_v = std::make_tuple(std::type_index{typeid(storage_type)}, std::type_index{typeid(view_type)});
        auto e_s = std::make_tuple(std::type_index{typeid(expression_type)}, std::type_index{typeid(storage_type)});
        auto e_e = std::make_tuple(std::type_index{typeid(expression_type)}, std::type_index{typeid(expression_type)});
        auto e_t = std::make_tuple(std::type_index{typeid(expression_type)}, std::type_index{typeid(trivial_type)});
        auto e_v = std::make_tuple(std::type_index{typeid(expression_type)}, std::type_index{typeid(view_type)});
        auto t_s = std::make_tuple(std::type_index{typeid(trivial_type)}, std::type_index{typeid(storage_type)});
        auto t_e = std::make_tuple(std::type_index{typeid(trivial_type)}, std::type_index{typeid(expression_type)});
        auto t_t = std::make_tuple(std::type_index{typeid(trivial_type)}, std::type_index{typeid(trivial_type)});
        auto t_v = std::make_tuple(std::type_index{typeid(trivial_type)}, std::type_index{typeid(view_type)});
        auto v_s = std::make_tuple(std::type_index{typeid(view_type)}, std::type_index{typeid(storage_type)});
        auto v_e = std::make_tuple(std::type_index{typeid(view_type)}, std::type_index{typeid(expression_type)});
        auto v_t = std::make_tuple(std::type_index{typeid(view_type)}, std::type_index{typeid(trivial_type)});
        auto v_v = std::make_tuple(std::type_index{typeid(view_type)}, std::type_index{typeid(view_type)});
        
        test_operation f{};
        //result,expected_result
        auto test_data = GENERATE_REF(
            test_type{dispatcher_type::call(*t1,*t2,f),s_s},
            test_type{dispatcher_type::call(*t1,*e_trivial,f),s_t},
            test_type{dispatcher_type::call(*t1,*e,f),s_e},
            test_type{dispatcher_type::call(*t1,*ce_trivial,f),s_s},
            test_type{dispatcher_type::call(*t1,*ce,f),s_s},
            test_type{dispatcher_type::call(*t1,*v_of_t,f),s_s},
            test_type{dispatcher_type::call(*t1,*v_of_ce,f),s_s},
            test_type{dispatcher_type::call(*t1,*v_of_e,f),s_v},
            test_type{dispatcher_type::call(*t1,*v_of_v,f),s_v},
            
            test_type{dispatcher_type::call(*e_trivial,*t2,f),t_s},
            test_type{dispatcher_type::call(*e_trivial,*e_trivial,f),t_t},
            test_type{dispatcher_type::call(*e_trivial,*e,f),t_e},
            test_type{dispatcher_type::call(*e_trivial,*ce_trivial,f),t_s},
            test_type{dispatcher_type::call(*e_trivial,*ce,f),t_s},
            test_type{dispatcher_type::call(*e_trivial,*v_of_t,f),t_s},
            test_type{dispatcher_type::call(*e_trivial,*v_of_ce,f),t_s},
            test_type{dispatcher_type::call(*e_trivial,*v_of_e,f),t_v},
            test_type{dispatcher_type::call(*e_trivial,*v_of_v,f),t_v},
            
            test_type{dispatcher_type::call(*e,*t2,f),e_s},
            test_type{dispatcher_type::call(*e,*e_trivial,f),e_t},
            test_type{dispatcher_type::call(*e,*e,f),e_e},
            test_type{dispatcher_type::call(*e,*ce_trivial,f),e_s},
            test_type{dispatcher_type::call(*e,*ce,f),e_s},
            test_type{dispatcher_type::call(*e,*v_of_t,f),e_s},
            test_type{dispatcher_type::call(*e,*v_of_ce,f),e_s},
            test_type{dispatcher_type::call(*e,*v_of_e,f),e_v},
            test_type{dispatcher_type::call(*e,*v_of_v,f),e_v},
            
            test_type{dispatcher_type::call(*ce_trivial,*t2,f),s_s},
            test_type{dispatcher_type::call(*ce_trivial,*e_trivial,f),s_t},
            test_type{dispatcher_type::call(*ce_trivial,*e,f),s_e},
            test_type{dispatcher_type::call(*ce_trivial,*ce_trivial,f),s_s},
            test_type{dispatcher_type::call(*ce_trivial,*ce,f),s_s},
            test_type{dispatcher_type::call(*ce_trivial,*v_of_t,f),s_s},
            test_type{dispatcher_type::call(*ce_trivial,*v_of_ce,f),s_s},
            test_type{dispatcher_type::call(*ce_trivial,*v_of_e,f),s_v},
            test_type{dispatcher_type::call(*ce_trivial,*v_of_v,f),s_v},
            
            test_type{dispatcher_type::call(*ce,*t2,f),s_s},
            test_type{dispatcher_type::call(*ce,*e_trivial,f),s_t},
            test_type{dispatcher_type::call(*ce,*e,f),s_e},
            test_type{dispatcher_type::call(*ce,*ce_trivial,f),s_s},
            test_type{dispatcher_type::call(*ce,*ce,f),s_s},
            test_type{dispatcher_type::call(*ce,*v_of_t,f),s_s},
            test_type{dispatcher_type::call(*ce,*v_of_ce,f),s_s},
            test_type{dispatcher_type::call(*ce,*v_of_e,f),s_v},
            test_type{dispatcher_type::call(*ce,*v_of_v,f),s_v},
            
            test_type{dispatcher_type::call(*v_of_t,*t2,f),s_s},
            test_type{dispatcher_type::call(*v_of_t,*e_trivial,f),s_t},
            test_type{dispatcher_type::call(*v_of_t,*e,f),s_e},
            test_type{dispatcher_type::call(*v_of_t,*ce_trivial,f),s_s},
            test_type{dispatcher_type::call(*v_of_t,*ce,f),s_s},
            test_type{dispatcher_type::call(*v_of_t,*v_of_t,f),s_s},
            test_type{dispatcher_type::call(*v_of_t,*v_of_ce,f),s_s},
            test_type{dispatcher_type::call(*v_of_t,*v_of_e,f),s_v},
            test_type{dispatcher_type::call(*v_of_t,*v_of_v,f),s_v},
            
            test_type{dispatcher_type::call(*v_of_ce,*t2,f),s_s},
            test_type{dispatcher_type::call(*v_of_ce,*e_trivial,f),s_t},
            test_type{dispatcher_type::call(*v_of_ce,*e,f),s_e},
            test_type{dispatcher_type::call(*v_of_ce,*ce_trivial,f),s_s},
            test_type{dispatcher_type::call(*v_of_ce,*ce,f),s_s},
            test_type{dispatcher_type::call(*v_of_ce,*v_of_t,f),s_s},
            test_type{dispatcher_type::call(*v_of_ce,*v_of_ce,f),s_s},
            test_type{dispatcher_type::call(*v_of_ce,*v_of_e,f),s_v},
            test_type{dispatcher_type::call(*v_of_ce,*v_of_v,f),s_v},
            
            test_type{dispatcher_type::call(*v_of_e,*t2,f),v_s},
            test_type{dispatcher_type::call(*v_of_e,*e_trivial,f),v_t},
            test_type{dispatcher_type::call(*v_of_e,*e,f),v_e},
            test_type{dispatcher_type::call(*v_of_e,*ce_trivial,f),v_s},
            test_type{dispatcher_type::call(*v_of_e,*ce,f),v_s},
            test_type{dispatcher_type::call(*v_of_e,*v_of_t,f),v_s},
            test_type{dispatcher_type::call(*v_of_e,*v_of_ce,f),v_s},
            test_type{dispatcher_type::call(*v_of_e,*v_of_e,f),v_v},
            test_type{dispatcher_type::call(*v_of_e,*v_of_v,f),v_v},
            
            test_type{dispatcher_type::call(*v_of_v,*t2,f),v_s},
            test_type{dispatcher_type::call(*v_of_v,*e_trivial,f),v_t},
            test_type{dispatcher_type::call(*v_of_v,*e,f),v_e},
            test_type{dispatcher_type::call(*v_of_v,*ce_trivial,f),v_s},
            test_type{dispatcher_type::call(*v_of_v,*ce,f),v_s},
            test_type{dispatcher_type::call(*v_of_v,*v_of_t,f),v_s},
            test_type{dispatcher_type::call(*v_of_v,*v_of_ce,f),v_s},
            test_type{dispatcher_type::call(*v_of_v,*v_of_e,f),v_v},
            test_type{dispatcher_type::call(*v_of_v,*v_of_v,f),v_v}
        );

        auto result = std::get<0>(test_data);
        auto expected_result = std::get<1>(test_data);
        REQUIRE(result == expected_result);
    }
}