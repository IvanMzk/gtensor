#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "combine.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

namespace test_tmp{

template<typename T>
class subscriptable_storage
{
    using inner_storage_type = std::vector<T>;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = typename inner_storage_type::size_type;
    using difference_type = typename inner_storage_type::difference_type;
    subscriptable_storage(size_type n):
        impl_(n)
    {}
    decltype(auto) operator[](size_type i){return impl_[i];}
    decltype(auto) operator[](size_type i)const{return impl_[i];}
};

template<typename Order>
struct ddd{
    virtual void f(){
        using other = gtensor::detail::change_order_t<Order>;
        std::cout<<std::endl<<typeid(other).name();
    }
    // virtual void f(){
    //     using other = gtensor::detail::change_order_t<Order>;
    //     std::cout<<std::endl<<typeid(other).name();
    // }
};

}

TEST_CASE("test_tmp","[test_tmp]")
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::storage_core;
    using gtensor::basic_descriptor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_type = tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using shape_type = typename tensor_type::shape_type;
    using helpers_for_testing::apply_by_element;



    // auto ttt = [](const auto& t){
    //     auto layout = std::get<0>(t);
    //     using layout_type = decltype(layout);
    //     using other = std::decay_t<layout_type>;
    //     //using other = gtensor::detail::change_order_t<layout_type>;
    //     //using ddd_type = test_tmp::ddd<layout_type>;
    //     using ddd_type = test_tmp::ddd<other>;
    //     //using descriptor_type = basic_descriptor<config_type,layout_type>;
    //     //std::cout<<std::endl<<typeid(descriptor_type).name();
    //     //std::cout<<std::endl<<typeid(other).name();
    //     std::cout<<std::endl<<typeid(ddd_type).name();
    // };


    // auto t = std::make_tuple(c_order{}, shape_type{}, value_type{3}, shape_type{}, std::vector<value_type>{3});

    // auto layout = std::get<0>(t);
    // auto shape = std::get<1>(t);
    // auto value = std::get<2>(t);
    // auto expected_shape = std::get<3>(t);
    // auto expected_elements = std::get<4>(t);
    // using layout_type = decltype(layout);
    // using core_type = storage_core<config_type,value_type,layout_type>;
    // core_type core(shape,value);

    //0layout,1shape,2value,3expected_shape,4expected_elements
        auto test_data = std::make_tuple(
            //c_order
            std::make_tuple(c_order{}, shape_type{}, value_type{3}, shape_type{}, std::vector<value_type>{3}),
            std::make_tuple(c_order{}, shape_type{0}, value_type{0}, shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(c_order{}, shape_type{3,0}, value_type{2}, shape_type{3,0}, std::vector<value_type>{}),
            std::make_tuple(c_order{}, shape_type{1}, value_type{1}, shape_type{1}, std::vector<value_type>{1}),
            std::make_tuple(c_order{}, shape_type{5}, value_type{2}, shape_type{5}, std::vector<value_type>{2,2,2,2,2}),
            std::make_tuple(c_order{}, shape_type{2,3}, value_type{4}, shape_type{2,3}, std::vector<value_type>{4,4,4,4,4,4}),
            //f_order
            std::make_tuple(f_order{}, shape_type{}, value_type{3}, shape_type{}, std::vector<value_type>{3}),
            std::make_tuple(f_order{}, shape_type{0}, value_type{0}, shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(f_order{}, shape_type{3,0}, value_type{2}, shape_type{3,0}, std::vector<value_type>{}),
            std::make_tuple(f_order{}, shape_type{1}, value_type{1}, shape_type{1}, std::vector<value_type>{1}),
            std::make_tuple(f_order{}, shape_type{5}, value_type{2}, shape_type{5}, std::vector<value_type>{2,2,2,2,2}),
            std::make_tuple(f_order{}, shape_type{2,3}, value_type{4}, shape_type{2,3}, std::vector<value_type>{4,4,4,4,4,4})
        );
        auto test = [](const auto& t){
            auto order = std::get<0>(t);
            using order_type = decltype(order);
            auto shape = std::get<1>(t);
            auto value = std::get<2>(t);
            auto expected_shape = std::get<3>(t);
            auto expected_elements = std::get<4>(t);
            using descriptor_type = basic_descriptor<config_type, order_type>;
            auto descriptor = descriptor_type{shape};
            //std::cout<<std::endl<<typeid(descriptor_type).name();
        };
        apply_by_element(test,test_data);


        // auto test_data = std::make_tuple(
        //     //c_order
        //     std::make_tuple(c_order{}, shape_type{}, value_type{3}, shape_type{}, std::vector<value_type>{3}),
        //     std::make_tuple(c_order{}, shape_type{0}, value_type{0}, shape_type{0}, std::vector<value_type>{}),
        //     std::make_tuple(c_order{}, shape_type{3,0}, value_type{2}, shape_type{3,0}, std::vector<value_type>{}),
        //     std::make_tuple(c_order{}, shape_type{1}, value_type{1}, shape_type{1}, std::vector<value_type>{1}),
        //     std::make_tuple(c_order{}, shape_type{5}, value_type{2}, shape_type{5}, std::vector<value_type>{2,2,2,2,2}),
        //     std::make_tuple(c_order{}, shape_type{2,3}, value_type{4}, shape_type{2,3}, std::vector<value_type>{4,4,4,4,4,4}),
        //     //f_order
        //     std::make_tuple(f_order{}, shape_type{}, value_type{3}, shape_type{}, std::vector<value_type>{3}),
        //     std::make_tuple(f_order{}, shape_type{0}, value_type{0}, shape_type{0}, std::vector<value_type>{}),
        //     std::make_tuple(f_order{}, shape_type{3,0}, value_type{2}, shape_type{3,0}, std::vector<value_type>{}),
        //     std::make_tuple(f_order{}, shape_type{1}, value_type{1}, shape_type{1}, std::vector<value_type>{1}),
        //     std::make_tuple(f_order{}, shape_type{5}, value_type{2}, shape_type{5}, std::vector<value_type>{2,2,2,2,2}),
        //     std::make_tuple(f_order{}, shape_type{2,3}, value_type{4}, shape_type{2,3}, std::vector<value_type>{4,4,4,4,4,4})
        // );
        // auto test = [](const auto& t){
        //     auto layout_ = std::get<0>(t);
        //     auto shape = std::get<1>(t);
        //     auto value = std::get<2>(t);
        //     auto expected_shape = std::get<3>(t);
        //     auto expected_elements = std::get<4>(t);
        //     //using layout_type = std::tuple_element_t<0, std::decay_t<decltype(t)>>;
        //     using layout_type_ = decltype(layout_);
        //     using layout_type__ = gtensor::detail::change_order_t<layout_type_>;
        //     //std::cout<<std::endl<<helpers_for_testing::type_to_str<layout_type>();
        //     //using layout_type = f_order;

        //     std::cout<<std::endl<<typeid(layout_type__).name();
        //     //using core_type = storage_core<config_type,value_type,layout_type>;
        //     //using descriptor_type = basic_descriptor<config_type,decltype(layout)>;
        //     using descriptor_type = basic_descriptor<config_type,layout_type_>;
        //     std::cout<<std::endl<<typeid(descriptor_type).name();
        //     //std::cout<<std::endl<<typeid(core_type).name();
        //     // core_type core(shape,value);
        //     // auto result_shape = core.descriptor().shape();
        //     // REQUIRE(result_shape == expected_shape);
        // };
        // apply_by_element(test,test_data);


}