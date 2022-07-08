#include "catch.hpp"
#include "./catch2/trompeloeil.hpp"
#include "shareable_storage.hpp"
#include "uvector.h"
#include <vector>
#include "test_config.hpp"


TEMPLATE_PRODUCT_TEST_CASE("test_shareable_storage","[test_shareable_storage]", (std::vector, trivial_type_vector::uvector),(float)){
    using storage_impl_type = TestType;
    using value_type = typename storage_impl_type::value_type;
    using size_type = typename storage_impl_type::size_type;
    using sstorage_type = gtensor::detail::shareable_storage<storage_impl_type>;
    
    SECTION("default_construction"){
        sstorage_type sstorage{};
        REQUIRE(sstorage.size() == 0);
        REQUIRE(sstorage.empty());
    }
    SECTION("default_value_construction"){
        size_type n{11};
        sstorage_type sstorage(n);
        REQUIRE(sstorage.size() == n);
        REQUIRE(!sstorage.empty());
    }
    SECTION("n_value_construction"){
        size_type n{11};
        value_type v{1.1};
        sstorage_type sstorage(n,v);
        REQUIRE(sstorage.size() == n);
        REQUIRE(!sstorage.empty());
    }
    SECTION("range_construction"){
        value_type vec[] = {1,2,3,4,5,6,7,8,9};
        auto s = sizeof(vec);
        sstorage_type sstorage{vec,vec+s};
        REQUIRE(sstorage.size() == s);
        REQUIRE(!sstorage.empty());
        std::vector<value_type> vec1 = {1,2,3,4,5};
        auto s1 = vec1.size();
        sstorage_type sstorage1{vec1.begin(),vec1.end()};
        REQUIRE(sstorage1.size() == s1);
        REQUIRE(!sstorage1.empty());
    }
    SECTION("init_list_construction"){
        sstorage_type sstorage{1,2,3,4,5};
        REQUIRE(sstorage.size() == 5);
    }
    SECTION("storage_impl_type_construction"){
        storage_impl_type v{1,2,3,4,5};
        auto vsize{v.size()};
        sstorage_type sstorage{v};
        REQUIRE(sstorage.size() == vsize);
        sstorage_type sstorage_moved{std::move(v)};
        REQUIRE(sstorage.size() == vsize);
        REQUIRE(v.size() == 0);
    }    
    SECTION("operator=="){
        size_type n{11};        
        sstorage_type sstorage(n,2.2);
        sstorage_type sstorage1(n,1.1);
        sstorage_type sstorage2(n,1.1);
        REQUIRE(sstorage == sstorage);
        REQUIRE(sstorage != sstorage1);
        REQUIRE(sstorage1 == sstorage2);
    }
    SECTION("deep_copy_construction"){
        sstorage_type sstorage{1,2,3,4,5};
        auto size{sstorage.size()};
        REQUIRE(sstorage.use_count() == 1);
        auto sstorage_copy = sstorage.copy();
        REQUIRE(sstorage.use_count() == 1);
        REQUIRE(sstorage_copy.use_count() == 1);
        REQUIRE(sstorage == sstorage_copy);
    }
    SECTION("reference_semantic_copy_construction"){
        sstorage_type sstorage{1,2,3,4,5};
        auto size{sstorage.size()};
        REQUIRE(sstorage.use_count() == 1);
        sstorage_type sstorage_ref{sstorage};
        REQUIRE(sstorage.size() == size);
        REQUIRE(sstorage.use_count() == 2);
        REQUIRE(sstorage_ref.use_count() == 2);
        REQUIRE(sstorage_ref == sstorage);        
        REQUIRE(sstorage.data() == sstorage_ref.data());
    }
    SECTION("reference_semantic_copy_assignment"){
        sstorage_type sstorage{1,2,3,4,5};
        sstorage_type sstorage_other(10,1.1f);
        REQUIRE(sstorage.use_count() == 1);
        sstorage_other = sstorage;
        REQUIRE(sstorage.use_count() == 2);
        REQUIRE(sstorage_other.use_count() == 2);
        REQUIRE(sstorage_other == sstorage);        
        REQUIRE(sstorage.data() == sstorage_other.data());
    }
    SECTION("move_construction"){
        sstorage_type sstorage{1,2,3,4,5};
        sstorage_type sstorage_ref{sstorage};        
        sstorage_type sstorage_move = std::move(sstorage);
        REQUIRE(sstorage_move.use_count() == 2);
        REQUIRE(sstorage.empty());        
        REQUIRE(sstorage_ref == sstorage_move);
    }
    
    SECTION("move_assignment"){
        sstorage_type sstorage{1,2,3,4,5};
        sstorage_type sstorage_ref{sstorage};
        sstorage_type sstorage_other(10,1.1f);
        sstorage_other = std::move(sstorage);
        REQUIRE(sstorage_other.use_count() == 2);
        REQUIRE(sstorage.empty());
        REQUIRE(sstorage_ref == sstorage_other);
    }                     
    SECTION("destruction"){
        sstorage_type sstorage{10};
        REQUIRE(sstorage.use_count() == 1);
        {
            sstorage_type sstorage_ref{sstorage};
            REQUIRE(sstorage.use_count() == 2);
            REQUIRE(sstorage_ref.use_count() == 2);

        }
        REQUIRE(sstorage.use_count() == 1);
    }    
    SECTION("operator[]_non_const"){
        sstorage_type sstorage{1,2,3};
        REQUIRE( std::is_lvalue_reference_v<decltype(sstorage[std::declval<size_type>()])> );
        REQUIRE(!std::is_const_v<std::remove_reference_t<decltype(sstorage[std::declval<size_type>()])>>);
        REQUIRE(sstorage[0] == 1);
        sstorage[0] = 0;
        REQUIRE(sstorage[0] == 0);
    }
    SECTION("operator[]_const"){
        const sstorage_type& cref{1,2,3};
        REQUIRE(std::is_lvalue_reference_v<decltype(cref[std::declval<size_type>()])> );
        REQUIRE(std::is_const_v<std::remove_reference_t<decltype(cref[std::declval<size_type>()])>>);
        REQUIRE(cref[0] == 1);        
    }    
}