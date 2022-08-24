#include "catch.hpp"
#include "iterator.hpp"
#include "tensor.hpp"
#include "test_config.hpp"

namespace test_multiindex_iterator_{

using gtensor::tensor;
using gtensor::config::default_config;
using gtensor::multiindex_iterator;
using gtensor::evaluating_base;
using gtensor::walker;

template<typename ValT, typename CfgT>
struct test_tensor : public tensor<ValT,CfgT>{
    using base_type = tensor<ValT,CfgT>;    
    using expression_base = evaluating_base<ValT,CfgT>;
    using iterator_type = multiindex_iterator<ValT,CfgT,walker<ValT,CfgT>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<CfgT>::type;

    strides_type strides{gtensor::detail::make_dividers<CfgT>(impl()->strides())};

    using tensor::tensor;
    test_tensor(const base_type& base):
        base_type{base}
    {}
    
    auto begin()const{return iterator_type{impl()->as_evaluating()->create_walker(), impl()->shape(), strides};}
    auto end()const{return iterator_type{impl()->as_evaluating()->create_walker(), impl()->shape(), strides, impl()->size()};}
};

template<typename ValT, typename CfgT>
struct expression_maker{
    using value_type = ValT;
    using tensor_type = test_tensor<ValT,CfgT>;
    tensor_type operator()(){return tensor_type{2} * tensor_type{-1,-1,-1} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}} + tensor_type{5,5,5} - tensor_type{3} ;}
};


}   //end of namespace test_multiindex_iterator_



TEMPLATE_TEST_CASE("test_multiindex_iterator","[test_multiindex_iterator]",gtensor::config::mode_div_native, gtensor::config::mode_div_libdivide)
{
    using value_type = float;
    using emaker_type = test_multiindex_iterator_::expression_maker<value_type, test_config::config_div_mode_selector<TestType>::config_type>;
    SECTION("test_iter_deref"){
        using test_type = std::tuple<value_type, value_type>;
        //0deref,1expected_deref
        auto test_data = GENERATE(
            test_type{*emaker_type{}().begin(), value_type{1}},
            test_type{*emaker_type{}().begin().operator++(), value_type{2}},
            test_type{*emaker_type{}().begin().operator++().operator++(), value_type{3}},
            test_type{*emaker_type{}().begin().operator++().operator++().operator++(), value_type{4}},
            test_type{*emaker_type{}().begin().operator++().operator++().operator++().operator++(), value_type{5}},
            test_type{*emaker_type{}().begin().operator++().operator++().operator++().operator++().operator++(), value_type{6}},
            test_type{*emaker_type{}().begin().operator++().operator++().operator++().operator++().operator++().operator--(), value_type{5}},
            test_type{*emaker_type{}().begin().operator++().operator++().operator++().operator++().operator++().operator--().operator--(), value_type{4}},
            test_type{*emaker_type{}().begin().operator++().operator++().operator++().operator++().operator++().operator--().operator--().operator--(), value_type{3}},
            test_type{*emaker_type{}().begin().operator++().operator++().operator++().operator++().operator++().operator--().operator--().operator--().operator--(), value_type{2}},
            test_type{*emaker_type{}().begin().operator++().operator++().operator++().operator++().operator++().operator--().operator--().operator--().operator--().operator--(), value_type{1}},
            test_type{*emaker_type{}().end().operator--(), value_type{6}},
            test_type{*emaker_type{}().end().operator--().operator--(), value_type{5}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--(), value_type{4}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--().operator--(), value_type{3}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--().operator--().operator--(), value_type{2}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--().operator--().operator--().operator--(), value_type{1}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--().operator--().operator--().operator--().operator++(), value_type{2}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--().operator--().operator--().operator--().operator++().operator++(), value_type{3}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--().operator--().operator--().operator--().operator++().operator++().operator++(), value_type{4}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--().operator--().operator--().operator--().operator++().operator++().operator++().operator++(), value_type{5}},
            test_type{*emaker_type{}().end().operator--().operator--().operator--().operator--().operator--().operator--().operator++().operator++().operator++().operator++().operator++(), value_type{6}},
            test_type{emaker_type{}().begin()[0], value_type{1}},
            test_type{emaker_type{}().begin()[1], value_type{2}},
            test_type{emaker_type{}().begin()[2], value_type{3}},
            test_type{emaker_type{}().begin()[3], value_type{4}},
            test_type{emaker_type{}().begin()[4], value_type{5}},
            test_type{emaker_type{}().begin()[5], value_type{6}},
            test_type{emaker_type{}().end()[-1], value_type{6}},
            test_type{emaker_type{}().end()[-2], value_type{5}},
            test_type{emaker_type{}().end()[-3], value_type{4}},
            test_type{emaker_type{}().end()[-4], value_type{3}},
            test_type{emaker_type{}().end()[-5], value_type{2}},
            test_type{emaker_type{}().end()[-6], value_type{1}},        
            test_type{*(emaker_type{}().begin()+0), value_type{1}},
            test_type{*(emaker_type{}().begin()+1), value_type{2}},
            test_type{*(emaker_type{}().begin()+2), value_type{3}},
            test_type{*(emaker_type{}().begin()+3), value_type{4}},
            test_type{*(emaker_type{}().begin()+4), value_type{5}},
            test_type{*(emaker_type{}().begin()+5), value_type{6}},
            test_type{*(emaker_type{}().end()-1), value_type{6}},
            test_type{*(emaker_type{}().end()-2), value_type{5}},
            test_type{*(emaker_type{}().end()-3), value_type{4}},
            test_type{*(emaker_type{}().end()-4), value_type{3}},
            test_type{*(emaker_type{}().end()-5), value_type{2}},
            test_type{*(emaker_type{}().end()-6), value_type{1}}
        );
        auto deref = std::get<0>(test_data);
        auto expected_deref = std::get<1>(test_data);
        REQUIRE(deref == expected_deref);
    }
    SECTION("test_iter_cmp"){
        using test_type = std::tuple<bool,bool>;
        //0cmp_result,1expected_cmp_result
        auto test_data = GENERATE(
            test_type{emaker_type{}().begin() > emaker_type{}().end(), false},
            test_type{emaker_type{}().end() > emaker_type{}().begin(), true},
            test_type{emaker_type{}().begin() < emaker_type{}().end(), true},
            test_type{emaker_type{}().end() < emaker_type{}().begin(), false},
            test_type{emaker_type{}().end() > emaker_type{}().end(), false},
            test_type{emaker_type{}().begin() > emaker_type{}().begin(), false},
            test_type{emaker_type{}().end() < emaker_type{}().end(), false},
            test_type{emaker_type{}().begin() < emaker_type{}().begin(), false},
            test_type{emaker_type{}().end() >= emaker_type{}().end(), true},
            test_type{emaker_type{}().begin() >= emaker_type{}().begin(), true},
            test_type{emaker_type{}().end() <= emaker_type{}().end(), true},
            test_type{emaker_type{}().begin() <= emaker_type{}().begin(), true}
        );
        auto cmp_result = std::get<0>(test_data);
        auto expected_cmp_result = std::get<1>(test_data);
        REQUIRE(cmp_result == expected_cmp_result);
    }
}

